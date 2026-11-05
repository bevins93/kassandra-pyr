#!/usr/bin/env python3
# ============================================================
# mapas_riesgo_v18.py
# Genera todos los mapas de riesgo para ConvLSTM_UNet_V18_E50
#
# Incluye:
#   1. Mapas de probabilidad (escala fija y percentil p99)
#   2. Mapas estacionales con categorización precautoria
#   3. Análisis de sensibilidad de umbrales
#   4. Tabla de área por categoría y estación
#   5. CSVs con métricas de área
#
# Uso:
#   python3 mapas_riesgo_v19.py
# ============================================================
import os, re, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import rasterio
from rasterio.enums import Resampling
from datetime import datetime
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================
RUN_DIR  = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
Y_DIR    = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Mensual"
HLS_DIR  = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos"
OUT_DIR  = RUN_DIR
PIXEL_M  = 30.0
PIXEL_HA = (PIXEL_M ** 2) / 10_000   # 0.09 ha/px

NOTEBOOK_NAME = "ConvLSTM_UNet_V19_E60"

# Estaciones Jalisco
ESTACIONES = {
    "Primavera\n(Mar-May)": {"meses": {3, 4, 5},   "color": "#b71c1c"},
    "Verano\n(Jun-Ago)":    {"meses": {6, 7, 8},   "color": "#1565c0"},
    "Otono\n(Sep-Nov)":     {"meses": {9, 10, 11}, "color": "#e65100"},
    "Invierno\n(Dic-Feb)":  {"meses": {12, 1, 2},  "color": "#4a148c"},
}

CATEGORY_NAMES  = ["Muy alto", "Alto", "Medio", "Bajo", "Muy bajo"]
CATEGORY_COLORS = ["#b71c1c", "#e65100", "#f9a825", "#558b2f", "#1a237e"]
CMAP_RIESGO = mcolors.LinearSegmentedColormap.from_list(
    "fire_risk", ["#1a237e","#558b2f","#f9a825","#e65100","#b71c1c"], N=256)
CMAP5 = mcolors.ListedColormap(CATEGORY_COLORS)

# ============================================================
# FUNCIONES
# ============================================================
def parse_fecha(fn):
    base = os.path.basename(fn)
    for pat in [r"(\d{4})-(\d{2})-(\d{2})", r"(\d{4})_(\d{2})_(\d{2})"]:
        m = re.search(pat, base)
        if m:
            try: return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except: pass
    for pat in [r"(\d{4})-(\d{2})(?!-\d{2})", r"(\d{4})(\d{2})(?!\d)"]:
        m = re.search(pat, base)
        if m:
            try: return datetime(int(m.group(1)), int(m.group(2)), 1)
            except: pass
    return None

def get_target_shape():
    tifs = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))
    with rasterio.open(os.path.join(Y_DIR, tifs[0])) as src:
        return src.height, src.width

def umbrales_propios(prob_2d):
    flat = prob_2d.flatten()
    flat = flat[np.isfinite(flat) & (flat > 1e-8)]
    if len(flat) < 100:
        return {"Muy alto": np.inf, "Alto": np.inf, "Medio": np.inf, "Bajo": np.inf}
    return {
        "Muy alto": float(np.percentile(flat, 95)),
        "Alto":     float(np.percentile(flat, 85)),
        "Medio":    float(np.percentile(flat, 65)),
        "Bajo":     float(np.percentile(flat, 40)),
    }

def categorizar(prob_2d, thrs):
    # FIX: iterar de Bajo→Muy alto para que el más restrictivo gane
    cat = np.full(prob_2d.shape, 4, dtype=np.uint8)
    for level, key in enumerate(["Bajo", "Medio", "Alto", "Muy alto"]):
        thr = thrs.get(key, np.inf)
        if np.isfinite(thr):
            cat[prob_2d >= thr] = 3 - level
    return cat

def tabla_area(cat, pixel_ha=PIXEL_HA):
    total = cat.size
    rows = []
    for i, (nm, co) in enumerate(zip(CATEGORY_NAMES, CATEGORY_COLORS)):
        px = int(np.sum(cat == i))
        rows.append({"categoria": nm, "color": co,
                     "pixeles": px,
                     "hectareas": round(px * pixel_ha, 1),
                     "pct_area": round(px / total * 100, 2)})
    return pd.DataFrame(rows)

def area_vs_umbral(prob_2d, n_puntos=250):
    flat  = prob_2d.flatten()
    total = len(flat)
    thrs  = np.linspace(float(flat.min()), float(np.percentile(flat, 99.5)), n_puntos)
    rows  = []
    for thr in thrs:
        px_in = int(np.sum(flat >= thr))
        rows.append({
            "umbral":       round(float(thr), 7),
            "px_alerta":    px_in,
            "ha_alerta":    round(px_in * PIXEL_HA, 1),
            "pct_alerta":   round(px_in / total * 100, 3),
            "px_excluido":  total - px_in,
            "ha_excluido":  round((total - px_in) * PIXEL_HA, 1),
            "pct_excluido": round((total - px_in) / total * 100, 3),
        })
    return pd.DataFrame(rows)

def leer_ndvi_estacion(meses, H, W):
    for f in sorted(os.listdir(HLS_DIR)):
        if not f.endswith(".tif"): continue
        dt = parse_fecha(f)
        if dt and dt.year == 2024 and dt.month in meses:
            try:
                with rasterio.open(os.path.join(HLS_DIR, f)) as src:
                    ndvi = src.read(1, out_shape=(H, W),
                                    resampling=Resampling.nearest).astype(np.float32)
                    return np.clip(ndvi, 0, 1)
            except: continue
    return np.zeros((H, W))

# ============================================================
# CARGAR DATOS
# ============================================================
print("=" * 60)
print(f"  Mapas de Riesgo — {NOTEBOOK_NAME}")
print("=" * 60)

print("\n[1] Cargando arrays...")
y_pred_cal  = np.load(os.path.join(RUN_DIR, "y_pred_test_calibrated.npy"))
y_true_test = np.load(os.path.join(RUN_DIR, "y_true_test.npy"))
y_pred_val_cal = np.load(os.path.join(RUN_DIR, "y_pred_val_calibrated.npy"))
y_true_val  = np.load(os.path.join(RUN_DIR, "y_true_val.npy"))
THRESHOLD   = float(np.load(os.path.join(RUN_DIR, "best_threshold.npy")))

H, W = get_target_shape()
TOTAL_PX = H * W
print(f"  Shape: {H}×{W} | {TOTAL_PX:,} px = {TOTAL_PX*PIXEL_HA:,.0f} ha")
print(f"  Threshold: {THRESHOLD:.5f}")
print(f"  y_pred min={y_pred_cal.min():.5f}  med={np.median(y_pred_cal):.5f}  max={y_pred_cal.max():.5f}")

# ============================================================
# RECONSTRUIR SERIE TEMPORAL
# ============================================================
print("\n[2] Reconstruyendo serie temporal test 2024...")
n_steps = len(y_pred_cal) // TOTAL_PX
y_pred_cal  = y_pred_cal[:n_steps * TOTAL_PX].reshape(n_steps, H, W)
y_true_test = y_true_test[:n_steps * TOTAL_PX].reshape(n_steps, H, W)
print(f"  Timesteps: {n_steps}")

# Fechas
hls_2024 = sorted(f for f in os.listdir(HLS_DIR)
                  if f.endswith(".tif") and "2024" in f)
fechas = []
for f in hls_2024[:n_steps]:
    dt = parse_fecha(f)
    fechas.append(dt if dt else datetime(2024, (len(fechas) % 12) + 1, 1))
while len(fechas) < n_steps:
    fechas.append(datetime(2024, (len(fechas) % 12) + 1, 1))
fechas = fechas[:n_steps]
print(f"  Fechas: {fechas[0].strftime('%Y-%m')} → {fechas[-1].strftime('%Y-%m')}")

# ============================================================
# MAPAS ESTACIONALES
# ============================================================
print("\n[3] Calculando mapas estacionales...")
est_prob = {}; est_gt = {}; est_n = {}; est_fec = {}
for estacion, cfg in ESTACIONES.items():
    idx = [i for i, dt in enumerate(fechas) if dt.month in cfg["meses"]]
    if idx:
        est_prob[estacion] = y_pred_cal[idx].mean(axis=0)
        est_gt[estacion]   = y_true_test[idx].max(axis=0)
    else:
        est_prob[estacion] = np.zeros((H, W))
        est_gt[estacion]   = np.zeros((H, W))
    est_n[estacion]   = len(idx)
    est_fec[estacion] = [fechas[i] for i in idx]
    nm = estacion.split("\n")[0]
    print(f"  {nm:<12}: n={len(idx):2d}  "
          f"min={est_prob[estacion].min():.5f}  "
          f"med={np.median(est_prob[estacion]):.5f}  "
          f"max={est_prob[estacion].max():.5f}")

prob_anual = y_pred_cal.mean(axis=0)
gt_anual   = y_true_test.max(axis=0)
thrs_anual = umbrales_propios(prob_anual)

print(f"\n  Umbrales adaptativos (mapa anual):")
for k, v in thrs_anual.items():
    pct = float(np.mean(prob_anual >= v) * 100)
    print(f"    {k:<10}: {v:.6f}  ({pct:.1f}% del área)")

archivos = []

# ============================================================
# PLOT 1: MAPAS DE PROBABILIDAD (fija + percentil p99)
# ============================================================
print("\n[4] Mapas de probabilidad (escala fija y percentil)...")

# Seleccionar 4 casos representativos del test
def seleccionar_casos():
    stats = []
    for i in range(n_steps):
        pos = int(y_true_test[i].sum())
        stats.append({"idx": i, "pos": pos,
                      "fecha": fechas[i], "mes": fechas[i].month})
    df = pd.DataFrame(stats)
    activos = df[df["pos"] > 0]
    if len(activos) == 0: activos = df
    casos = {
        "mayor_incendio": int(activos.loc[activos["pos"].idxmax(), "idx"]),
        "caso_tipico":    int(activos.iloc[(activos["pos"] -
                              activos["pos"].median()).abs().argmin()]["idx"]),
        "menor_incendio": int(activos.loc[activos["pos"].idxmin(), "idx"]),
    }
    inactivos = df[df["pos"] == 0]
    casos["sin_incendio"] = (int(inactivos.sample(1, random_state=42).iloc[0]["idx"])
                             if len(inactivos) > 0 else casos["caso_tipico"])
    return casos

CASOS = seleccionar_casos()
CASO_LABELS = {
    "mayor_incendio": "Mayor incendio real",
    "caso_tipico":    "Caso típico",
    "menor_incendio": "Menor incendio",
    "sin_incendio":   "Sin incendio (negativo)",
}

for sufijo, usar_pct, pct_val in [("_fija", False, None), ("_percentil", True, 99)]:
    try:
        fig = plt.figure(figsize=(20, 22))
        titulo_extra = f"escala percentil p{pct_val}" if usar_pct else "escala fija [0,1]"
        fig.suptitle(
            f"Mapas de Riesgo ({titulo_extra}) — {NOTEBOOK_NAME}\n"
            f"X(t)→Y(t+1) | Threshold={THRESHOLD:.4f} (calibrado)",
            fontsize=13, fontweight="bold", y=0.98)
        gs3 = gridspec.GridSpec(4, 4, figure=fig, hspace=0.40, wspace=0.15)

        for row, (caso_key, caso_label) in enumerate(CASO_LABELS.items()):
            idx    = CASOS[caso_key]
            prob_m = y_pred_cal[idx]
            gt_m   = y_true_test[idx]
            fecha  = fechas[idx]
            pos_real = int(gt_m.sum())
            pred_med = float(prob_m.mean())
            pred_p99 = float(np.percentile(prob_m, 99))
            vmax_map = (max(pred_p99, THRESHOLD * 1.5) if usar_pct else 1.0)

            # NDVI del mismo mes
            ndvi = leer_ndvi_estacion({fecha.month}, H, W)

            ax0 = fig.add_subplot(gs3[row, 0])
            ax0.imshow(ndvi, cmap="RdYlGn", vmin=0, vmax=1)
            ax0.set_title(f"NDVI\n{fecha.strftime('%Y-%m-%d')}", fontsize=8)
            ax0.set_ylabel(caso_label, fontsize=8, fontweight="bold",
                           rotation=90, labelpad=5)
            ax0.axis("off")

            ax1 = fig.add_subplot(gs3[row, 1])
            im1 = ax1.imshow(prob_m, cmap=CMAP_RIESGO, vmin=0, vmax=vmax_map)
            subtit = (f"Prob. Y(t+1) [cal]\nmed={pred_med:.4f} p99={pred_p99:.4f}"
                      f"\nvmax=p{pct_val}={vmax_map:.4f}" if usar_pct
                      else f"Prob. Y(t+1) [cal]\nmed={pred_med:.4f}")
            ax1.set_title(subtit, fontsize=7)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            ax1.axis("off")

            ax2 = fig.add_subplot(gs3[row, 2])
            ax2.imshow(gt_m, cmap="hot", vmin=0, vmax=1)
            ax2.set_title(f"Ground Truth Y(t+1)\n{pos_real:,} px", fontsize=8)
            plt.colorbar(fig.add_subplot(gs3[row, 2]).images[0]
                         if False else ax2.imshow(gt_m, cmap="hot", vmin=0, vmax=1),
                         ax=ax2, fraction=0.046, pad=0.04)
            ax2.axis("off")

            ax3 = fig.add_subplot(gs3[row, 3])
            ax3.imshow(prob_m, cmap=CMAP_RIESGO, vmin=0, vmax=vmax_map, alpha=0.8)
            if pos_real > 0:
                ax3.contour(gt_m, levels=[0.5], colors=["cyan"], linewidths=1.5)
            y_bin_m = (prob_m >= THRESHOLD).astype(float)
            if y_bin_m.sum() > 0:
                ax3.contour(y_bin_m, levels=[0.5], colors=["yellow"],
                            linewidths=1.0, linestyles="--")
            ax3.set_title("Overlay\ncyan=GT  --yellow=thr", fontsize=7)
            ax3.axis("off")

        leyenda = [
            mpatches.Patch(color="#b71c1c", label="Alto"),
            mpatches.Patch(color="#e65100", label="Medio-alto"),
            mpatches.Patch(color="#f9a825", label="Medio"),
            mpatches.Patch(color="#00695c", label="Bajo"),
            mpatches.Patch(color="#1a237e", label="Muy bajo"),
        ]
        fig.legend(handles=leyenda, loc="lower center", ncol=5,
                   fontsize=8, bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

        fname = f"mapas_riesgo{sufijo}.png"
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close("all")
        archivos.append(fname)
        print(f"  ✅ {fname}")
    except Exception as e:
        print(f"  ❌ mapas{sufijo}: {e}")
        import traceback; traceback.print_exc()

# ============================================================
# PLOT 2: MAPAS ESTACIONALES CON CATEGORIZACIÓN PRECAUTORIA
# ============================================================
print("\n[5] Mapas estacionales...")
try:
    fig1 = plt.figure(figsize=(24, 24))
    fig1.suptitle(
        f"Mapa de Riesgo de Incendio — Serie Estacional 2024\n"
        f"{NOTEBOOK_NAME}  |  Categorización Precautoria Adaptativa  |  ANP Jalisco",
        fontsize=13, fontweight="bold", y=0.99)
    gs1 = gridspec.GridSpec(4, 5, figure=fig1,
                            hspace=0.32, wspace=0.10,
                            left=0.06, right=0.97, top=0.94, bottom=0.07)
    col_titles = ["NDVI (t)", "Probabilidad\nY(t+1) [cal]",
                  "Riesgo categorizado\n(umbral adaptativo)",
                  "Ground Truth\n+ overlay prob.",
                  "Área por\ncategoría (%)"]
    for ci, ct in enumerate(col_titles):
        ax_h = fig1.add_subplot(gs1[0, ci])
        ax_h.set_title(ct, fontsize=9, fontweight="bold"); ax_h.axis("off")

    for row, (estacion, cfg) in enumerate(ESTACIONES.items()):
        prob   = est_prob[estacion]
        gt     = est_gt[estacion]
        thrs_e = umbrales_propios(prob) if est_n[estacion] > 0 else thrs_anual
        cat    = categorizar(prob, thrs_e)
        df_a   = tabla_area(cat)
        n_m    = est_n[estacion]
        pos_gt = int(gt.sum())
        ndvi   = leer_ndvi_estacion(cfg["meses"], H, W)
        vmax_p = max(float(np.percentile(prob, 99)), prob.min() + 1e-6)

        ax0 = fig1.add_subplot(gs1[row, 0])
        ax0.imshow(ndvi, cmap="RdYlGn", vmin=0, vmax=1)
        ax0.set_ylabel(estacion, fontsize=9, fontweight="bold",
                       color=cfg["color"], rotation=90, labelpad=4)
        ax0.set_xticks([]); ax0.set_yticks([])
        if n_m > 0:
            meses_str = " ".join(sorted({dt.strftime("%b") for dt in est_fec[estacion]}))
            ax0.set_xlabel(f"n={n_m} meses\n{meses_str}", fontsize=7)
        else:
            ax0.set_xlabel("sin datos 2024", fontsize=7, color="gray")

        ax1 = fig1.add_subplot(gs1[row, 1])
        ax1.imshow(prob, cmap=CMAP_RIESGO, vmin=prob.min(), vmax=vmax_p)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_xlabel(f"min={prob.min():.4f}  med={np.median(prob):.4f}\n"
                       f"p99={np.percentile(prob, 99):.4f}", fontsize=7)
        if pos_gt > 0:
            ax1.contour(gt, levels=[0.5], colors=["cyan"], linewidths=1.0)

        ax2 = fig1.add_subplot(gs1[row, 2])
        ax2.imshow(cat, cmap=CMAP5, vmin=0, vmax=4)
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlabel(f"MA>={thrs_e['Muy alto']:.5f}\nA>={thrs_e['Alto']:.5f}", fontsize=7)
        if pos_gt > 0:
            ax2.contour(gt, levels=[0.5], colors=["white"], linewidths=1.2)

        ax3 = fig1.add_subplot(gs1[row, 3])
        ax3.imshow(prob, cmap=CMAP_RIESGO, vmin=prob.min(), vmax=vmax_p, alpha=0.8)
        if pos_gt > 0:
            ax3.contourf(gt, levels=[0.5, 1.5], colors=["white"], alpha=0.5)
            ax3.contour(gt, levels=[0.5], colors=["cyan"], linewidths=1.5)
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_xlabel(f"GT: {pos_gt:,} px ({pos_gt*PIXEL_HA:,.0f} ha)", fontsize=7)

        ax4 = fig1.add_subplot(gs1[row, 4])
        ax4.set_xlim(0, 105); ax4.set_ylim(-0.6, 4.6)
        ax4.set_yticks(range(5))
        ax4.set_yticklabels(CATEGORY_NAMES, fontsize=7)
        ax4.invert_yaxis(); ax4.set_xlabel("% área ANP", fontsize=7)
        ax4.grid(axis="x", alpha=0.25)
        for i, r in df_a.iterrows():
            ax4.barh(i, r["pct_area"], color=r["color"], alpha=0.88, height=0.65)
            if r["pct_area"] >= 3:
                ax4.text(r["pct_area"] + 0.5, i,
                         f"{r['pct_area']:.1f}%\n{r['hectareas']:,.0f} ha",
                         va="center", fontsize=7)
            elif r["pct_area"] > 0.3:
                ax4.text(r["pct_area"] / 2, i, f"{r['pct_area']:.0f}%",
                         va="center", ha="center", fontsize=6,
                         color="white", fontweight="bold")

    patches_leg = [mpatches.Patch(color=c, label=n)
                   for n, c in zip(CATEGORY_NAMES, CATEGORY_COLORS)]
    patches_leg.append(mpatches.Patch(color="cyan", label="Ground truth (contorno)"))
    fig1.legend(handles=patches_leg, loc="lower center", ncol=6,
                fontsize=8, bbox_to_anchor=(0.5, 0.01), framealpha=0.95,
                title="Categorías de riesgo precautorio (p95/p85/p65/p40 del mapa estacional)",
                title_fontsize=8)

    out = os.path.join(OUT_DIR, "mapas_riesgo_estacional.png")
    fig1.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    archivos.append("mapas_riesgo_estacional.png")
    print("  ✅ mapas_riesgo_estacional.png")
except Exception as e:
    print(f"  ❌ estacional: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# PLOT 3: ANÁLISIS DE SENSIBILIDAD
# ============================================================
print("\n[6] Análisis de sensibilidad...")
try:
    df_sens = area_vs_umbral(prob_anual, n_puntos=300)
    thr_vals = list(thrs_anual.values())
    thr_keys = list(thrs_anual.keys())
    colors_v  = ["#b71c1c", "#e65100", "#f9a825", "#558b2f"]

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
    fig2.suptitle(
        f"Análisis de Sensibilidad — Umbrales Precautorios\n"
        f"{NOTEBOOK_NAME}  |  Mapa Anual Promedio 2024  |  ANP Jalisco",
        fontsize=12, fontweight="bold")

    ax = axes2[0, 0]
    ax.plot(df_sens["umbral"], df_sens["pct_alerta"],
            color="#b71c1c", lw=2.5, label="% en alerta (>= umbral)")
    ax.plot(df_sens["umbral"], df_sens["pct_excluido"],
            color="#1565c0", lw=2, ls="--", label="% excluido (< umbral)")
    for tv, tk, co in zip(thr_vals, thr_keys, colors_v):
        pct_v = float(df_sens.loc[(df_sens["umbral"] - tv).abs().idxmin(), "pct_alerta"])
        ax.axvline(tv, color=co, lw=1.8, alpha=0.85)
        ax.text(tv + 0.0001, pct_v + 5, f"{tk.split()[0]}\n{pct_v:.0f}%",
                fontsize=7.5, color=co, fontweight="bold")
    ax.set_xlabel("Umbral de probabilidad calibrada")
    ax.set_ylabel("% del área ANP")
    ax.set_title("Área en alerta vs umbral", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes2[0, 1]
    ax.fill_between(df_sens["umbral"], df_sens["ha_alerta"] / 1000, alpha=0.2, color="#b71c1c")
    ax.plot(df_sens["umbral"], df_sens["ha_alerta"] / 1000, color="#b71c1c", lw=2)
    for tv, co in zip(thr_vals, colors_v):
        ha_v = float(df_sens.loc[(df_sens["umbral"] - tv).abs().idxmin(), "ha_alerta"])
        ax.axvline(tv, color=co, lw=1.5, alpha=0.8)
        ax.text(tv, ha_v / 1000 + 0.2, f"{ha_v/1000:.1f}k", fontsize=7, color=co)
    ax.set_xlabel("Umbral de probabilidad calibrada")
    ax.set_ylabel("Miles de hectáreas en alerta")
    ax.set_title("Hectáreas en alerta", fontsize=10); ax.grid(alpha=0.3)

    ax = axes2[0, 2]
    dx = np.diff(df_sens["umbral"].values)
    dy = -np.diff(df_sens["ha_alerta"].values)
    with np.errstate(divide="ignore", invalid="ignore"):
        tasa = np.where(dx > 0, dy / dx, 0)
    ax.plot(df_sens["umbral"].values[:-1], tasa / 1000, color="#e65100", lw=1.8)
    ax.fill_between(df_sens["umbral"].values[:-1], tasa / 1000,
                    where=tasa / 1000 > 0, alpha=0.2, color="#e65100")
    for tv, co in zip(thr_vals, colors_v):
        ax.axvline(tv, color=co, lw=1, alpha=0.6)
    ax.set_xlabel("Umbral de probabilidad calibrada")
    ax.set_ylabel("ha excluidas / unidad umbral (×10³)")
    ax.set_title("Tasa marginal de exclusión", fontsize=10); ax.grid(alpha=0.3)

    esquemas = [
        ("Solo top 5%\n(Muy alto)",
         {"Muy alto": thrs_anual["Muy alto"], "Alto": np.inf, "Medio": np.inf, "Bajo": np.inf}),
        ("Dos niveles\n(top 5% + 15%)",
         {"Muy alto": thrs_anual["Muy alto"], "Alto": thrs_anual["Alto"],
          "Medio": np.inf, "Bajo": np.inf}),
        ("Cinco categorías\n(esquema completo)", thrs_anual),
    ]
    for ci, (titulo, thrs_demo) in enumerate(esquemas):
        ax = axes2[1, ci]
        cat_demo = categorizar(prob_anual, thrs_demo)
        ax.imshow(cat_demo, cmap=CMAP5, vmin=0, vmax=4)
        if gt_anual.sum() > 0:
            ax.contour(gt_anual, levels=[0.5], colors=["white"], linewidths=1.2)
        ax.set_title(titulo, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        df_demo = tabla_area(cat_demo)
        lbl_parts = [f"{r['categoria']}: {r['pct_area']:.0f}% ({r['hectareas']:,.0f}ha)"
                     for _, r in df_demo.iterrows() if r["pct_area"] > 0.5]
        ax.set_xlabel("\n".join(lbl_parts[:3]), fontsize=7)

    patches2 = [mpatches.Patch(color=c, label=n)
                for n, c in zip(CATEGORY_NAMES, CATEGORY_COLORS)]
    fig2.legend(handles=patches2, loc="lower center", ncol=5,
                fontsize=8, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    fig2.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = os.path.join(OUT_DIR, "analisis_sensibilidad.png")
    fig2.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    archivos.append("analisis_sensibilidad.png")
    print("  ✅ analisis_sensibilidad.png")
except Exception as e:
    print(f"  ❌ sensibilidad: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# PLOT 4: TABLA DE ÁREA ESTACIONAL
# ============================================================
print("\n[7] Tabla de área estacional...")
try:
    fig3, axes3 = plt.subplots(1, 2, figsize=(22, 9))
    fig3.suptitle(
        f"Área por Categoría de Riesgo — Desglose Estacional 2024\n"
        f"{NOTEBOOK_NAME}  |  ANP Jalisco",
        fontsize=12, fontweight="bold")

    ax = axes3[0]
    est_cortas = [e.split("\n")[0] for e in ESTACIONES.keys()]
    x = np.arange(len(est_cortas))
    bottoms = np.zeros(len(est_cortas))

    for cat_i, (nm, co) in enumerate(zip(CATEGORY_NAMES, CATEGORY_COLORS)):
        vals = []
        for estacion in ESTACIONES.keys():
            cat_e = categorizar(est_prob[estacion], thrs_anual)
            df_e  = tabla_area(cat_e)
            vals.append(df_e.iloc[cat_i]["pct_area"])
        ax.bar(x, vals, 0.6, bottom=bottoms, color=co, label=nm, alpha=0.9)
        for xi, val, bot in zip(x, vals, bottoms):
            if val >= 5:
                ax.text(xi, bot + val / 2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
            elif val > 1.5:
                ax.text(xi, bot + val / 2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottoms += np.array(vals)

    ax.set_xticks(x); ax.set_xticklabels(est_cortas, fontsize=10)
    ax.set_ylabel("% del área ANP"); ax.set_ylim(0, 108)
    ax.set_title("Distribución de categorías por estación\n(umbrales anuales p95/p85/p65/p40)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    umbral_txt = (f"Umbrales (pct mapa anual):\n"
                  f"  Muy alto  >= {thrs_anual['Muy alto']:.5f}\n"
                  f"  Alto      >= {thrs_anual['Alto']:.5f}\n"
                  f"  Medio     >= {thrs_anual['Medio']:.5f}\n"
                  f"  Bajo      >= {thrs_anual['Bajo']:.5f}")
    ax.text(0.01, 0.99, umbral_txt, transform=ax.transAxes,
            fontsize=7.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    # Tabla numérica
    ax2 = axes3[1]; ax2.axis("off")
    cols_tbl = ["Categoría", "Umbral"] + est_cortas + ["Anual"]
    rows_tbl = []
    cat_anual_m = categorizar(prob_anual, thrs_anual)
    df_anual_a  = tabla_area(cat_anual_m)
    thrs_str = [f">={thrs_anual['Muy alto']:.5f}", f">={thrs_anual['Alto']:.5f}",
                f">={thrs_anual['Medio']:.5f}",    f">={thrs_anual['Bajo']:.5f}",
                f"<{thrs_anual['Bajo']:.5f}"]
    for ci_r in range(5):
        row = [CATEGORY_NAMES[ci_r], thrs_str[ci_r]]
        for estacion in ESTACIONES.keys():
            cat_e = categorizar(est_prob[estacion], thrs_anual)
            df_e  = tabla_area(cat_e)
            r     = df_e.iloc[ci_r]
            row.append(f"{r['pct_area']:.1f}%\n{r['hectareas']:,.0f} ha"
                       if r["pct_area"] > 0 else "—")
        r_a = df_anual_a.iloc[ci_r]
        row.append(f"{r_a['pct_area']:.1f}%\n{r_a['hectareas']:,.0f} ha"
                   if r_a["pct_area"] > 0 else "—")
        rows_tbl.append(row)

    tbl = ax2.table(cellText=rows_tbl, colLabels=cols_tbl,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 2.5)
    for ci_r in range(5):
        rgba = mcolors.to_rgba(CATEGORY_COLORS[ci_r], alpha=0.22)
        for cj in range(len(cols_tbl)):
            tbl[ci_r + 1, cj].set_facecolor(rgba)
    for cj in range(len(cols_tbl)):
        tbl[0, cj].set_facecolor("#37474f")
        tbl[0, cj].set_text_props(color="white", fontweight="bold")
    ax2.set_title("% y ha por categoría y estación", fontsize=9, pad=12)

    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "tabla_area_estacional.png")
    fig3.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    archivos.append("tabla_area_estacional.png")
    print("  ✅ tabla_area_estacional.png")
except Exception as e:
    print(f"  ❌ tabla: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# CSVs
# ============================================================
print("\n[8] Exportando CSVs...")

# CSV 1: área por categoría x estación
rows_c1 = []
for estacion in list(ESTACIONES.keys()) + ["Anual"]:
    prob_e = est_prob[estacion] if estacion != "Anual" else prob_anual
    n_e    = est_n.get(estacion, n_steps) if estacion != "Anual" else n_steps
    thrs_e = umbrales_propios(prob_e)
    cat_e  = categorizar(prob_e, thrs_e)
    df_e   = tabla_area(cat_e)
    for _, r in df_e.iterrows():
        nm_cat = r["categoria"]
        thr_v  = thrs_e.get(nm_cat, None)
        rows_c1.append({
            "estacion":   estacion.replace("\n", " "),
            "n_meses":    n_e,
            "categoria":  nm_cat,
            "umbral":     f">= {thr_v:.6f}" if thr_v and np.isfinite(thr_v)
                          else f"< {thrs_e.get('Bajo', 0):.6f}",
            "pixeles":    r["pixeles"],
            "hectareas":  r["hectareas"],
            "pct_area":   r["pct_area"],
        })
csv1 = os.path.join(OUT_DIR, "area_categorias.csv")
pd.DataFrame(rows_c1).to_csv(csv1, index=False)
archivos.append("area_categorias.csv")
print(f"  ✅ area_categorias.csv")

# CSV 2: sensibilidad umbral
csv2 = os.path.join(OUT_DIR, "sensibilidad_umbral.csv")
df_sens.to_csv(csv2, index=False)
archivos.append("sensibilidad_umbral.csv")
print(f"  ✅ sensibilidad_umbral.csv")

# ============================================================
# RESUMEN FINAL
# ============================================================
print(f"\n{'='*60}")
print(f"  {NOTEBOOK_NAME}")
print(f"  ANP: {TOTAL_PX:,} px = {TOTAL_PX*PIXEL_HA:,.0f} ha")
print(f"  Threshold: {THRESHOLD:.5f}")
print(f"\n  Umbrales adaptativos (mapa anual):")
for k, v in thrs_anual.items():
    pct = float(np.mean(prob_anual >= v) * 100)
    ha  = pct / 100 * TOTAL_PX * PIXEL_HA
    print(f"    {k:<10}: {v:.6f}  →  {pct:.1f}% = {ha:,.0f} ha")

print(f"\n  Distribución anual:")
cat_anual_final = categorizar(prob_anual, thrs_anual)
df_final = tabla_area(cat_anual_final)
for _, r in df_final.iterrows():
    if r["pct_area"] > 0:
        bar = "#" * int(r["pct_area"] / 2)
        print(f"  {r['categoria']:<12} {bar:<30} {r['pct_area']:>5.1f}%  {r['hectareas']:>8,.0f} ha")

print(f"\n  Archivos generados: {len(archivos)}")
for a in archivos: print(f"    ✅ {a}")
print(f"{'='*60}")