#!/usr/bin/env python3
# reevaluar_solo_ANP.py
# Re-evalúa el modelo SOLO sobre píxeles válidos del ANP (pendiente finita en la banda 2 del stack estático).
# El raster H x W es un rectángulo (bounding box); ~38 % de sus píxeles quedan fuera del polígono del ANP.
# Entrada : <RUN_DIR>/<sub>/ y_true_*.npy, y_pred_*_raw.npy (salida de reevaluar_completo.py)
# Salida  : <RUN_DIR>/evaluacion_ANP/  (arrays a tamaño completo con Platt reajustado SOLO con píxeles ANP
#           + valid_mask.npy + escenas_test.csv)  y un resumen en pantalla y en resumen_ANP.txt
# Uso: python reevaluar_solo_ANP.py [--sub evaluacion_abril]
import os, argparse, shutil, json
import numpy as np, rasterio
from rasterio.enums import Resampling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default="/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2")
ap.add_argument("--sub", default="evaluacion_abril")
ap.add_argument("--base", default="/lustre/home/kgalvan/cnn")
a = ap.parse_args()
SRC = os.path.join(a.run_dir, a.sub); OUT = os.path.join(a.run_dir, "evaluacion_ANP"); os.makedirs(OUT, exist_ok=True)
Y_DIR = f"{a.base}/y_binaria/GEE_Incendios_Mensual"
STATIC = f"{a.base}/x/Topographic_DL_Dataset/version1/static_features_LaPrimavera_30m_V20.tif"
PIXHA = 0.09

y_ref = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))[0]
with rasterio.open(os.path.join(Y_DIR, y_ref)) as s: H, W = s.height, s.width
PX = H * W
with rasterio.open(STATIC) as s:
    slope = s.read(2, out_shape=(H, W), resampling=Resampling.bilinear).astype(np.float32)
    valid = np.isfinite(slope)
    if s.nodata is not None: valid &= slope != s.nodata
mask = valid.reshape(-1); np.save(os.path.join(OUT, "valid_mask.npy"), mask)
print(f"Raster {H}x{W} = {PX:,} px = {PX*PIXHA:,.0f} ha | válidos (ANP) = {mask.sum():,} px = {mask.sum()*PIXHA:,.1f} ha ({100*mask.mean():.1f} %)")

def carga(split):
    y = np.load(os.path.join(SRC, f"y_true_{split}.npy")).astype(np.int8)
    p = np.load(os.path.join(SRC, f"y_pred_{split}_raw.npy")).astype(np.float32)
    n = len(y) // PX; assert n * PX == len(y), "longitud no múltiplo de H*W"
    m = np.tile(mask, n)
    return y, p, m, n
yv, pv, mv, nv = carga("val"); yt, pt, mt, nt = carga("test")
for nom, y, m in (("val", yv, mv), ("test", yt, mt)):
    print(f"{nom}: positivos totales={int(y.sum()):,} | dentro ANP={int(y[m].sum()):,} | FUERA del ANP={int(y[~m].sum()):,}")

lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
lr.fit(pv[mv].reshape(-1, 1), yv[mv].astype(int))
cv = lr.predict_proba(pv.reshape(-1, 1))[:, 1].astype(np.float32)
ct = lr.predict_proba(pt.reshape(-1, 1))[:, 1].astype(np.float32)
for n_, arr in (("y_true_val", yv), ("y_true_test", yt), ("y_pred_val_calibrated", cv), ("y_pred_test_calibrated", ct)):
    np.save(os.path.join(OUT, n_ + ".npy"), arr)
shutil.copy2(os.path.join(SRC, "escenas_test.csv"), os.path.join(OUT, "escenas_test.csv"))
if os.path.exists(os.path.join(SRC, "escenas_val.csv")): shutil.copy2(os.path.join(SRC, "escenas_val.csv"), OUT)

yv_, cv_, yt_, ct_ = yv[mv], cv[mv], yt[mt], ct[mt]
P, R, T = precision_recall_curve(yv_, cv_); f2 = 5*P*R/(4*P+R+1e-12); thr = float(T[np.argmax(f2[:-1])])
tn, fp, fn, tp = confusion_matrix(yt_, (ct_ >= thr).astype(np.int8), labels=[0, 1]).ravel()
L = []
L.append(f"ANP: {mask.sum():,} px = {mask.sum()*PIXHA:,.1f} ha | escenas val={nv} test={nt}")
L.append(f"N observaciones ANP: val={len(yv_):,} test={len(yt_):,} total={len(yv_)+len(yt_):,}")
L.append(f"Prevalencia ANP: val={yv_.mean()*100:.3f}%  test={yt_.mean()*100:.3f}%")
L.append(f"VAL 2023 (ANP): AUC-ROC={roc_auc_score(yv_,cv_):.4f}  AUC-PR={average_precision_score(yv_,cv_):.4f}")
L.append(f"TEST 2024 (ANP): AUC-ROC={roc_auc_score(yt_,ct_):.4f}  AUC-PR={average_precision_score(yt_,ct_):.4f}  baseline={yt_.mean():.4f}  razón={average_precision_score(yt_,ct_)/yt_.mean():.2f}x")
L.append(f"Umbral F2 (val ANP) = {thr:.5f}")
r = tp/(tp+fn); p = tp/(tp+fp)
L.append(f"TEST: recall={r:.4f} precision={p:.4f} F2={5*p*r/(4*p+r):.4f} FPR={fp/(fp+tn):.4f} TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")
L.append(f"Área marcada 'flagged' media por escena = {(tp+fp)/nt*PIXHA:,.0f} ha ({100*(tp+fp)/(tp+fp+fn+tn):.1f} % del ANP)")
txt = "\n".join(L); print("\n" + txt); open(os.path.join(OUT, "resumen_ANP.txt"), "w").write(txt + "\n")
print(f"\nGuardado en {OUT}\nSiguiente paso: python generar_todas_las_figuras.py --sub evaluacion_ANP")
