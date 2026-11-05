#!/usr/bin/env python3
# generar_figuras_articulo.py
# Corre en una sola pasada:
#   1) umbral_validacion.py  -> umbral elegido en 2023, métricas 2024 y Fig. 3 (desempeño)
#   2) figuras_suplementarias.py -> Figs. S1, S2, S3
#   3) AUC-ROC / AUC-PR de la validación completa (2023) para los valores amarillos del texto
# y copia TODO a una carpeta nueva con nombres listos para el artículo.
#
# Debe estar en la misma carpeta que umbral_validacion.py y figuras_suplementarias.py
# (/lustre/home/kgalvan/cnn). Uso:
#   python generar_figuras_articulo.py
#   python generar_figuras_articulo.py --fecha 2024-04-26 --out /lustre/home/kgalvan/cnn/figuras_articulo
import os, sys, shutil, subprocess, argparse, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN0 = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default=RUN0)
ap.add_argument("--sub", default="evaluacion_abril")
ap.add_argument("--fecha", default="2024-04-26")
ap.add_argument("--base", default="/lustre/home/kgalvan/cnn")
ap.add_argument("--out", default=None, help="carpeta de salida (por defecto <RUN>/figuras_articulo_final)")
a = ap.parse_args()

EVAL = os.path.join(a.run_dir, a.sub)
OUT = a.out or os.path.join(a.run_dir, "figuras_articulo_final")
os.makedirs(OUT, exist_ok=True)
TMP = os.path.join(OUT, "_intermedios")
os.makedirs(TMP, exist_ok=True)

def correr(cmd, titulo):
    print(f"\n=== {titulo} ===", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"ERROR en '{titulo}' (código {r.returncode}). Revisa el mensaje de arriba.")

# 1) Fig. 3 + umbral
correr([sys.executable, os.path.join(HERE, "umbral_validacion.py"), EVAL], "Umbral (2023) y figura de desempeño")

# 2) Figs. S1-S3
correr([sys.executable, os.path.join(HERE, "figuras_suplementarias.py"),
        "--run-dir", a.run_dir, "--sub", a.sub, "--fecha", a.fecha, "--base", a.base,
        "--out", os.path.join(TMP, "supl")], "Figuras suplementarias")

# 3) AUC de la validación completa
from sklearn.metrics import roc_auc_score, average_precision_score
yv = np.load(os.path.join(EVAL, "y_true_val.npy")).astype(np.int8)
sv = np.load(os.path.join(EVAL, "y_pred_val_calibrated.npy"))
auc_v, pr_v = roc_auc_score(yv, sv), average_precision_score(yv, sv)
yt = np.load(os.path.join(EVAL, "y_true_test.npy")).astype(np.int8)
st = np.load(os.path.join(EVAL, "y_pred_test_calibrated.npy"))
auc_t, pr_t = roc_auc_score(yt, st), average_precision_score(yt, st)
resumen = (f"VALIDACIÓN 2023 (80 escenas completas): AUC-ROC = {auc_v:.4f}  AUC-PR = {pr_v:.4f}  prevalencia = {yv.mean()*100:.3f}%\n"
           f"PRUEBA 2024 (70 escenas):              AUC-ROC = {auc_t:.4f}  AUC-PR = {pr_t:.4f}  prevalencia = {yt.mean()*100:.3f}%\n"
           f"Diferencia AUC-ROC val - test = {auc_v-auc_t:+.4f}\n")
print("\n=== AUC de validación completa ===\n" + resumen)

# 4) Copiar con nombres finales
def cp(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT, dst)); print("  ->", dst)
    else:
        print("  (falta)", src)

for ext in ("png", "pdf"):
    cp(os.path.join(EVAL, f"fig_performance.{ext}"), f"Fig3_performance.{ext}")
    cp(os.path.join(TMP, "supl", f"figS1_training_history.{ext}"), f"FigS1_training_history.{ext}")
    cp(os.path.join(TMP, "supl", f"figS2_monthly_burned_area.{ext}"), f"FigS2_monthly_burned_area.{ext}")
    cp(os.path.join(TMP, "supl", f"figS3_slope_vs_probability.{ext}"), f"FigS3_slope_vs_probability.{ext}")
cp(os.path.join(EVAL, "metricas_umbral_validacion.json"), "metricas_umbral_validacion.json")
with open(os.path.join(OUT, "resumen_metricas.txt"), "w") as f:
    f.write(resumen)
    try:
        m = json.load(open(os.path.join(EVAL, "metricas_umbral_validacion.json")))
        f.write("\n" + json.dumps(m, indent=2, ensure_ascii=False))
    except Exception:
        pass
print(f"\nLISTO. Todo en: {OUT}\n(copia los PNG a tu carpeta 'ariculo-2'; puedes borrar _intermedios)")