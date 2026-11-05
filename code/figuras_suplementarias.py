#!/usr/bin/env python3
# figuras_suplementarias.py
# Genera las figuras suplementarias (inglés, 300 dpi, PNG + PDF):
#   figS1_training_history        (antes Fig. 4: historial de entrenamiento)
#   figS2_monthly_burned_area     (antes Fig. 3: área quemada mensual 2020-2024)
#   figS3_slope_vs_probability    (antes Fig. 10: pendiente vs. mapa de probabilidad)
# Uso:  python figuras_suplementarias.py [--fecha 2024-04-27] [--sub evaluacion_abril]
# Salida: <RUN_DIR>/figuras_suplementarias/
import os, re, sys, argparse, warnings
import numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
from datetime import datetime
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default="/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2")
ap.add_argument("--sub", default="evaluacion_abril", help="subcarpeta con las predicciones de prueba")
ap.add_argument("--fecha", default="2024-04-27", help="fecha (YYYY-MM-DD) de la escena de la Fig. S3; se usa la más cercana")
ap.add_argument("--base", default="/lustre/home/kgalvan/cnn")
ap.add_argument("--static-tif", default=None, help="por defecto: version1 (stack de abril)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

RUN = args.run_dir
OUT = args.out or os.path.join(RUN, "figuras_suplementarias")
os.makedirs(OUT, exist_ok=True)
Y_DIR = f"{args.base}/y_binaria/GEE_Incendios_Mensual"
STATIC = args.static_tif or f"{args.base}/x/Topographic_DL_Dataset/version1/static_features_LaPrimavera_30m_V20.tif"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})
C_TRAIN, C_VAL, C_MARK, C_SPRING = "#4C78A8", "#E45756", "#333333", "#F2D8A7"

def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nombre}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  guardada", os.path.join(OUT, nombre + ".png"))

def parse_fecha(fn):
    b = os.path.basename(fn)
    for pat in [r"(\d{4})-(\d{2})-(\d{2})", r"(\d{4})_(\d{2})_(\d{2})"]:
        m = re.search(pat, b)
        if m:
            try: return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError: pass
    for pat in [r"(\d{4})-(\d{2})(?!-\d{2})", r"(\d{4})(\d{2})(?!\d)"]:
        m = re.search(pat, b)
        if m:
            try: return datetime(int(m.group(1)), int(m.group(2)), 1)
            except ValueError: pass
    return None

# ======================= S1: historial de entrenamiento =======================
print("S1: historial de entrenamiento")
h = pd.read_csv(os.path.join(RUN, "history.csv"))
ep = np.arange(1, len(h) + 1)
col_auc = "val_roc_auc" if "val_roc_auc" in h else [c for c in h if c.startswith("val_") and "auc" in c][0]
best = int(np.argmax(h[col_auc].values)) + 1
print(f"  epocas={len(h)}  mejor epoca (max {col_auc})={best}  valor={h[col_auc].max():.4f}")
paneles = [("loss", "val_loss", "Focal Tversky loss", "(a)"),
           ("roc_auc", "val_roc_auc", "ROC-AUC", "(b)"),
           ("pr_auc", "val_pr_auc", "PR-AUC", "(c)"),
           ("recall", "val_recall", "Recall", "(d)")]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
for ax, (tr, va, titulo, letra) in zip(axs.ravel(), paneles):
    if tr in h: ax.plot(ep, h[tr], color=C_TRAIN, lw=1.4, label="Training")
    if va in h: ax.plot(ep, h[va], color=C_VAL, lw=1.4, label="Validation")
    ax.axvline(best, color=C_MARK, ls="--", lw=1)
    ax.set_title(f"{letra} {titulo}", loc="left"); ax.set_ylabel(titulo)
    ax.grid(alpha=0.25, lw=0.5)
for ax in axs[1]: ax.set_xlabel("Epoch")
axs[0, 0].legend(frameon=False, loc="best")
axs[0, 1].annotate(f"epoch {best}", xy=(best, axs[0, 1].get_ylim()[0]), xytext=(4, 6),
                   textcoords="offset points", fontsize=8, color=C_MARK)
fig.tight_layout()
guardar(fig, "figS1_training_history")

# ======================= S2: área quemada mensual =======================
print("S2: área quemada mensual (a partir de las máscaras mensuales)")
filas = []
for f in sorted(os.listdir(Y_DIR)):
    if not f.endswith(".tif"): continue
    dt = parse_fecha(f)
    if dt is None: continue
    with rasterio.open(os.path.join(Y_DIR, f)) as s:
        a = s.read(1); px_ha = abs(s.res[0] * s.res[1]) / 1e4
    filas.append({"anio": dt.year, "mes": dt.month, "ha": float((a > 0).sum()) * px_ha})
m = pd.DataFrame(filas).groupby(["anio", "mes"], as_index=False)["ha"].sum()
m = m[(m.anio >= 2020) & (m.anio <= 2024)].sort_values(["anio", "mes"]).reset_index(drop=True)
tot = m.ha.sum(); prim = m[m.mes.isin([3, 4, 5])].ha.sum()
print(f"  total 2020-2024 = {tot:,.0f} ha | marzo-mayo = {prim:,.0f} ha ({100*prim/tot:.2f} %)")
print(f"  pico mensual = {m.ha.max():,.0f} ha en {int(m.loc[m.ha.idxmax(),'anio'])}-{int(m.loc[m.ha.idxmax(),'mes']):02d}")
print("  por año (ha):", {int(a): round(v) for a, v in m.groupby('anio').ha.sum().items()})
x = np.arange(len(m))
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for i, r in m.iterrows():
    if r.mes in (3, 4, 5): ax.axvspan(i - 0.5, i + 0.5, color=C_SPRING, lw=0, zorder=0)
ax.bar(x, m.ha, color=C_VAL, width=0.8, zorder=2)
ticks = [i for i, r in m.iterrows() if r.mes == 1]
ax.set_xticks(ticks); ax.set_xticklabels([str(int(m.loc[i, "anio"])) for i in ticks])
ax.set_xlim(-0.5, len(m) - 0.5); ax.set_ylabel("Burned area (ha)")
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.bar([], [], color=C_SPRING, label="March–May (spring fire season)")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
guardar(fig, "figS2_monthly_burned_area")

# ======================= S3: pendiente vs. probabilidad =======================
print("S3: pendiente vs. mapa de probabilidad calibrada")
PXF = None
y_ref = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))[0]
with rasterio.open(os.path.join(Y_DIR, y_ref)) as s:
    H, W = s.height, s.width
PX = H * W
esc = pd.read_csv(os.path.join(RUN, args.sub, "escenas_test.csv"))
esc["f"] = pd.to_datetime(esc["fecha_ultima_imagen"])
objetivo = pd.Timestamp(args.fecha)
k = int((esc["f"] - objetivo).abs().values.argmin())
print(f"  escena elegida: orden {k}  fecha de la última imagen = {esc.loc[k,'fecha_ultima_imagen']} (pedida {args.fecha})")
P = np.load(os.path.join(RUN, args.sub, "y_pred_test_calibrated.npy"), mmap_mode="r")
prob = np.asarray(P[k * PX:(k + 1) * PX]).reshape(H, W).astype(np.float32)
with rasterio.open(STATIC) as s:
    slope = s.read(2, out_shape=(H, W), resampling=Resampling.bilinear).astype(np.float32)   # banda 2 = slope
    valid = np.isfinite(slope)
    if s.nodata is not None: valid &= slope != s.nodata
slope = np.where(valid, slope, np.nan)
prob = np.where(valid, prob, np.nan)
vmax = float(np.nanpercentile(prob, 99.5))
cmap_p = plt.get_cmap("magma_r")
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.3))
im0 = axs[0].imshow(slope, cmap="Greys", vmin=0, vmax=np.nanpercentile(slope, 99))
axs[0].set_title("(A) Slope (°)", loc="left")
cb0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02); cb0.ax.tick_params(labelsize=7)
im1 = axs[1].imshow(prob, cmap=cmap_p, vmin=0, vmax=vmax)
axs[1].set_title("(B) Calibrated probability", loc="left")
cb1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02); cb1.ax.tick_params(labelsize=7)
axs[2].imshow(slope, cmap="Greys", vmin=0, vmax=np.nanpercentile(slope, 99), alpha=0.9)
ov = np.ma.masked_invalid(prob)
axs[2].imshow(ov, cmap=cmap_p, vmin=0, vmax=vmax, alpha=0.55)
axs[2].set_title("(C) Composite", loc="left")
for a in axs:
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
fig.suptitle(f"Acquisition of {esc.loc[k,'fecha_ultima_imagen']}", fontsize=8, y=0.02, color="#555555")
fig.tight_layout()
guardar(fig, "figS3_slope_vs_probability")
print("\nListo. Archivos en", OUT)
