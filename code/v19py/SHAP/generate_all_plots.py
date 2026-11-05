#!/usr/bin/env python3
# ============================================================
# shap_kassandra_unified.py — KASSANDRA-PYR (XAI Analysis)
# Script Unificado: Dinámico + Estático + NALCMS
# Incluye: Integrated Gradients para las 3 ramas del modelo
# NALCMS con nombres de categoría desde CSV de lookup
# ============================================================

import os
import re
import glob
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import Patch
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("[WARN] rasterio no disponible")

# ============================================================
# CONFIGURACIÓN PRINCIPAL  ← edita aquí
# ============================================================

STEPS       = 30   # Pasos para Integrated Gradients
MAX_SEQS    = 2    # Secuencias a analizar (1–N)

RUN_DIR     = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
MODEL_PATH  = os.path.join(RUN_DIR, "best_model.keras")
OUT_DIR     = os.path.join(RUN_DIR, f"shap_unified_n{STEPS}")

HLS_DIR     = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos"
CLIMATE_DIR = "/lustre/home/kgalvan/cnn/x/Terraclimate_DL_Dataset"
STATIC_TIF  = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
Y_DIR       = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Mensual"

# Ruta al CSV de lookup de categorías NALCMS

SEQ_LEN = 3

# ── Nombres de variables por rama ──────────────────────────
NOMBRES_DIN = [
    "NDVI", "NBR", "NDWI",
    "mask_NDVI", "mask_NBR", "mask_NDWI",
    "TempMin", "TempMax", "Viento",
    "Precip", "PDSI", "Evapo"
]

NOMBRES_STATIC = [
    "Elevación", "Pendiente", "Aspect_Cos", "Aspect_Sin",
    "TPI", "Rugosidad", "Densidad_Pob", "Frac_Ag", "Cobertura_Arb"
]

# ── Triángulo ecológico del fuego ────────────────────────────
FIRE_TRIANGLE = {
    "Combustible": {"canales": ["NDVI", "NBR", "NDWI"],                         "color": "#2E7D32"},
    "Clima":       {"canales": ["TempMin", "TempMax", "Viento", "Precip", "PDSI", "Evapo"], "color": "#1565C0"},
    "Topografía":  {"canales": ["elevation", "slope", "aspect_cos", "aspect_sin", "tpi", "roughness"], "color": "#6A1B9A"},
    "Ignición":    {"canales": ["pop_density", "ag_fraction", "nalcms"],         "color": "#B71C1C"},
    "Máscaras":    {"canales": ["mask_NDVI", "mask_NBR", "mask_NDWI"],           "color": "#546E7A"},
}

# ── Paleta por rama ─────────────────────────────────────────
COLOR_DIN  = "#2E7D32"
COLOR_STAT = "#1565C0"
COLOR_NALC = "#B71C1C"

# ── Estilo Nature ───────────────────────────────────────────
rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 8,
    'figure.dpi'       : 300,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth'   : 0.8,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '-',
    'grid.linewidth'   : 0.5,
    'lines.linewidth'  : 1.0,
})

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOOKUP DE NOMBRES NALCMS (hardcodeado desde categorical_lookup_V20.csv)
# ============================================================

# Mapeo: índice one-hot (0-based) → nombre legible
# Fuente: categorical_lookup_V20.csv, band = land_cover_NALCMS_30m
#   value (raster, 1-based) → oh_idx = value - 1
# Filas MODIS incluidas como clases adicionales (índices 11+)
_NALCMS_LOOKUP = {
    0 : "Water",
    1 : "Evergreen Forest",
    2 : "Mixed Forest",
    3 : "Deciduous Forest",
    4 : "Shrubland",
    5 : "Herbaceous",
    6 : "Cropland",
    7 : "Barren",
    8 : "Developed",
    9 : "Snow / Ice",
    10: "Cloud Shadow",
    # Clases MODIS adicionales (value 12-14 → oh_idx 11-13)
    11: "MODIS: Cropland",
    12: "MODIS: Cropland Nat. Veg",
    13: "MODIS: Urban",
    14: "MODIS: Mixed Forest",
}

def build_nalcms_names(n_classes):
    """
    Devuelve una lista de nombres para los n_classes canales one-hot de NALCMS.
    Usa el lookup hardcodeado; para índices fuera del rango usa 'NALCMS_C{i}'.
    """
    names = [_NALCMS_LOOKUP.get(i, f"NALCMS_C{i}") for i in range(n_classes)]
    print(f"  ✓ NALCMS lookup: {n_classes} clases ({sum(i in _NALCMS_LOOKUP for i in range(n_classes))} con nombre)")
    return names


# ============================================================
# FUNCIONES AUXILIARES COMUNES
# ============================================================

def parse_fecha_robusta(filename):
    base = os.path.basename(filename)
    patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
        (r'(\d{4})_(\d{2})_(\d{2})', '%Y_%m_%d'),
        (r'(\d{8})',                  '%Y%m%d'),
        (r'(\d{4})-(\d{2})(?!\d)',    '%Y-%m'),
        (r'(\d{4})_(\d{2})(?!\d)',    '%Y_%m'),
        (r'(\d{4})(\d{2})(?!\d)',     '%Y%m'),
    ]
    for pattern, fmt in patterns:
        m = re.search(pattern, base)
        if m:
            try:
                if fmt in ['%Y-%m', '%Y_%m', '%Y%m']:
                    return datetime(int(m.group(1)), int(m.group(2)), 15)
                return datetime.strptime('-'.join(m.groups()), fmt.replace('_', '-'))
            except (ValueError, IndexError):
                continue
    return None


def gelu(x):
    return x * tf.sigmoid(1.702 * x)

def match_shape(inputs):
    up, sk = inputs
    return tf.image.resize(up, [tf.shape(sk)[1], tf.shape(sk)[2]], method="bilinear")

def resize_gate(inputs):
    g, s = inputs
    return tf.image.resize(g, [tf.shape(s)[1], tf.shape(s)[2]], method="bilinear")

def resize_static(inputs):
    sf, ref = inputs
    return tf.image.resize(sf, [tf.shape(ref)[1], tf.shape(ref)[2]], method="bilinear")

def focal_tversky_loss(alpha=0.80, beta=0.20, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.clip_by_value(tf.reshape(y_pred, [-1]), 1e-7, 1 - 1e-7)
        tp  = tf.reduce_sum(y_true * y_pred)
        fp  = tf.reduce_sum((1 - y_true) * y_pred)
        fn  = tf.reduce_sum(y_true * (1 - y_pred))
        tvi = (tp + 1e-7) / (tp + alpha * fn + beta * fp + 1e-7)
        return tf.reduce_mean(tf.pow(1.0 - tvi, gamma))
    return loss

CUSTOM_DICT = {
    "gelu"         : gelu,
    "match_shape"  : match_shape,
    "resize_gate"  : resize_gate,
    "resize_static": resize_static,
    "loss"         : focal_tversky_loss(),
}

def asignar_vertice(canal):
    for vertice, datos in FIRE_TRIANGLE.items():
        if canal in datos["canales"]:
            return vertice
    return "Otros"


# ============================================================
# LECTURA Y PREPROCESAMIENTO GIS
# ============================================================

def read_and_fix(path, out_shape_hw=None, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        if out_shape_hw and (src.height, src.width) != out_shape_hw:
            data = src.read(
                out_shape=(src.count, out_shape_hw[0], out_shape_hw[1]),
                resampling=resampling)
        else:
            data = src.read()
        return np.transpose(data, (1, 2, 0)).astype(np.float32)


def sanitize_array(arr, clip_val=10.0):
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -clip_val, clip_val).astype(np.float32)


def procesar_hls_con_mascara(raw_hw_c):
    nube   = raw_hw_c[:, :, 3] if raw_hw_c.shape[-1] > 3 else raw_hw_c[:, :, -1]
    valido  = (np.isfinite(nube) & (nube > 0.5)).astype(np.float32)
    nublado = ~(np.isfinite(nube) & (nube > 0.5))
    res = []
    for ci in [0, 1, 2]:
        canal = np.clip(
            np.where(np.isfinite(raw_hw_c[:, :, ci]), raw_hw_c[:, :, ci], 0.0),
            0.0, 1.0)
        canal[nublado] = 0.0
        res.append(canal)
    for _ in [0, 1, 2]:
        res.append(valido.copy())
    return np.stack(res, axis=-1).astype(np.float32)


def normalizar_tc(arr_hwc):
    TC_RANGES = {
        0: (-30.0, 50.0), 1: (-20.0, 60.0), 2: (0.0, 15.0),
        3: (0.0, 500.0),  4: (-5.0, 5.0),   5: (0.0, 200.0),
    }
    LOG1P_TC = {3, 5}
    out = np.zeros_like(arr_hwc)
    for i, (vmin, vmax) in TC_RANGES.items():
        canal = arr_hwc[:, :, i].copy()
        if i in LOG1P_TC:
            canal = np.log1p(np.maximum(canal, 0.0))
            vmin, vmax = 0.0, np.log1p(vmax)
        out[:, :, i] = np.clip((canal - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return out.astype(np.float32)


def _leer_tc(dt, target_shape):
    clim_map = {}
    if os.path.exists(CLIMATE_DIR):
        for f in sorted(os.listdir(CLIMATE_DIR)):
            if f.endswith(".tif"):
                mdt = parse_fecha_robusta(f)
                if mdt:
                    clim_map[mdt.strftime("%Y-%m-%d")] = f
                    clim_map[mdt.strftime("%Y-%m")]    = f
    for key in [dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")]:
        if key in clim_map:
            raw = read_and_fix(
                os.path.join(CLIMATE_DIR, clim_map[key]),
                target_shape, Resampling.bilinear)
            tc = np.stack([raw[:, :, i] for i in range(min(6, raw.shape[-1]))], axis=-1)
            return normalizar_tc(sanitize_array(tc))
    return np.zeros((*target_shape, 6), np.float32)


def load_static_features(model, tif_path, target_shape):
    if not RASTERIO_OK or not os.path.exists(tif_path):
        return None, None
    try:
        try:
            expected_sn_dim = model.inputs[2].shape[-1]
        except Exception:
            expected_sn_dim = 19
        with rasterio.open(tif_path) as src:
            data = src.read(
                out_shape=(src.count, target_shape[0], target_shape[1]),
                resampling=Resampling.bilinear).astype(np.float32)
        all_bands  = np.transpose(data, (1, 2, 0))
        static_cont = sanitize_array(all_bands[:, :, :9], clip_val=1.0)
        nalcms_oh   = np.zeros((*target_shape, expected_sn_dim), dtype=np.float32)
        if all_bands.shape[-1] > 9:
            nalcms_raw = np.clip(
                all_bands[:, :, 9].astype(np.int32), 0, expected_sn_dim - 1)
            for c in range(expected_sn_dim):
                nalcms_oh[:, :, c] = (nalcms_raw == c).astype(np.float32)
        return static_cont, nalcms_oh
    except Exception as e:
        print(f"    [WARN] Error estáticos: {e}")
        return None, None


# ============================================================
# CONSTRUCCIÓN DE SECUENCIAS
# ============================================================

def build_sequences(model, hls_files, y_files, ventana_dias=60):
    H, W = model.inputs[0].shape[2], model.inputs[0].shape[3]
    target_shape = (H, W)
    print(f"    Shape: {H}×{W} | Ventana: ±{ventana_dias}d")

    static_cont, nalcms_oh = load_static_features(model, STATIC_TIF, target_shape)
    if static_cont is None:
        try:
            expected_sn_dim = model.inputs[2].shape[-1]
        except Exception:
            expected_sn_dim = 19
        static_cont = np.zeros((*target_shape, 9),              dtype=np.float32)
        nalcms_oh   = np.zeros((*target_shape, expected_sn_dim), dtype=np.float32)

    y_dates = sorted(set(
        parse_fecha_robusta(yf) for yf in y_files if parse_fecha_robusta(yf)))
    if y_dates:
        print(f"    Fechas Y: {y_dates[0].strftime('%Y-%m')} → {y_dates[-1].strftime('%Y-%m')}")

    sequences = []
    for i in range(len(hls_files) - SEQ_LEN):
        last_hls  = hls_files[i + SEQ_LEN - 1]
        ref_date  = parse_fecha_robusta(last_hls)
        if ref_date is None:
            continue

        best_y_file, best_delta = None, ventana_dias + 1
        for yf in y_files:
            ydt = parse_fecha_robusta(yf)
            if ydt:
                delta = abs((ydt - ref_date).days)
                if delta < best_delta:
                    best_delta, best_y_file = delta, yf

        if best_y_file and best_delta <= ventana_dias:
            try:
                seq_imgs = []
                for j in range(SEQ_LEN):
                    fi  = hls_files[i + j]
                    dt  = parse_fecha_robusta(fi)
                    if dt is None:
                        raise ValueError(f"Fecha no parseable: {fi}")
                    hls_frame = procesar_hls_con_mascara(read_and_fix(fi, target_shape))
                    tc_frame  = _leer_tc(dt, target_shape)
                    seq_imgs.append(
                        sanitize_array(np.concatenate([hls_frame, tc_frame], axis=-1)))

                y_map = np.zeros((H, W), dtype=np.float32)
                try:
                    ym    = read_and_fix(best_y_file, target_shape, Resampling.nearest)[:, :, 0]
                    y_map = (ym > 0).astype(np.float32)
                except Exception:
                    pass

                sequences.append({
                    "X_dyn"        : np.stack(seq_imgs, axis=0),
                    "X_sc"         : static_cont,
                    "X_sn"         : nalcms_oh,
                    "y_map"        : y_map,
                    "hls_file"     : last_hls,
                    "date"         : ref_date,
                    "n_fire_pixels": int(np.sum(y_map > 0.5)),
                })
            except Exception:
                pass

    total_fire   = sum(s["n_fire_pixels"] for s in sequences)
    seqs_fire    = sum(1 for s in sequences if s["n_fire_pixels"] > 0)
    print(f"    Secuencias: {len(sequences)} | Con incendios: {seqs_fire} | Píxeles fuego: {total_fire:,}")
    return sequences


# ============================================================
# CÁLCULO SHAP — INTEGRATED GRADIENTS (3 RAMAS)
# ============================================================

def compute_shap_all_branches(model, sequences, n_steps=30, max_seqs=2):
    """
    Integrated Gradients simultáneo para las 3 ramas del modelo.

    Las tres ramas usan la MISMA lógica de normalización y promedio espacial
    completo (H×W), de modo que shap_din_mean, shap_stat y shap_nalc son
    directamente comparables para calcular la contribución relativa por rama.

    Retorna:
      shap_dyn      : [N_px, T, 1, 1, C_din]  — por píxel muestreado (figs 1-6)
      shap_din_mean : [C_din]                  — promedio espacial completo (figs 7-9)
      shap_stat     : [C_stat]                 — promedio espacial completo
      shap_nalc     : [C_nalc]                 — promedio espacial completo
      labels        : [N_px]
      raw_dyn       : [N_px, T, 1, 1, C_din]
      hls_list      : [N_px]
    """
    print(f"\nCalculando SHAP (steps={n_steps}, seqs={max_seqs})...")

    shap_vals_list, labels_list, raw_dyn_list, hls_list = [], [], [], []

    # Acumuladores espaciales completos — misma lógica para las 3 ramas
    n_din  = sequences[0]["X_dyn"].shape[-1]
    n_stat = sequences[0]["X_sc"].shape[-1]
    n_nalc = sequences[0]["X_sn"].shape[-1]
    shap_din_acc  = np.zeros(n_din)
    shap_stat_acc = np.zeros(n_stat)
    shap_nalc_acc = np.zeros(n_nalc)
    count_seqs    = 0

    alphas = np.linspace(0.01, 1.0, n_steps)

    for seq_idx, seq in enumerate(sequences[:max_seqs]):
        print(f"  Mapa {seq_idx + 1}/{max_seqs}...")

        xd = np.expand_dims(seq["X_dyn"], axis=0)   # [1, T, H, W, C_din]
        xs = np.expand_dims(seq["X_sc"],  axis=0)   # [1, H, W, C_stat]
        xn = np.expand_dims(seq["X_sn"],  axis=0)   # [1, H, W, C_nalc]
        y_map = seq["y_map"]
        H, W  = y_map.shape

        baseline_dyn = np.zeros_like(xd)
        baseline_sc  = np.zeros_like(xs)
        baseline_sn  = np.zeros_like(xn)

        grads_accum_d = np.zeros_like(xd)
        grads_accum_s = np.zeros_like(xs)
        grads_accum_n = np.zeros_like(xn)

        for alpha in alphas:
            xd_interp = tf.cast(baseline_dyn + alpha * (xd - baseline_dyn), tf.float32)
            xs_interp = tf.cast(baseline_sc  + alpha * (xs - baseline_sc),  tf.float32)
            xn_interp = tf.cast(baseline_sn  + alpha * (xn - baseline_sn),  tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(xd_interp)
                tape.watch(xs_interp)
                tape.watch(xn_interp)
                preds    = model([xd_interp, xs_interp, xn_interp], training=False)
                loss_val = tf.reduce_mean(preds)

            g_d = tape.gradient(loss_val, xd_interp)
            g_s = tape.gradient(loss_val, xs_interp)
            g_n = tape.gradient(loss_val, xn_interp)
            del tape

            if g_d is not None: grads_accum_d += g_d.numpy()
            if g_s is not None: grads_accum_s += g_s.numpy()
            if g_n is not None: grads_accum_n += g_n.numpy()

        # ── Rama dinámica ────────────────────────────────────────
        # IG normalizado para píxeles (figs 1-6): normaliza por máx global
        ig_full = (xd[0] - baseline_dyn[0]) * (grads_accum_d[0] / n_steps)
        ch_max  = np.abs(ig_full).max(axis=(0, 1, 2), keepdims=True) + 1e-8
        ig_norm = ig_full / ch_max             # [T, H, W, C_din]

        # Importancia espacial completa (figs 7-9): misma lógica que ramas estáticas
        # Normaliza cada canal por su propio máximo → promedio sobre T, H, W
        ch_max_d = np.abs(grads_accum_d[0]).max(axis=(0, 1, 2), keepdims=True) + 1e-8
        val_d    = np.abs(grads_accum_d[0] / ch_max_d).mean(axis=(0, 1, 2))   # [C_din]

        # ── Ramas estáticas ──────────────────────────────────────
        # Normaliza cada canal por su propio máximo → promedio sobre H, W
        ch_max_s = np.abs(grads_accum_s[0]).max(axis=(0, 1), keepdims=True) + 1e-8
        val_s    = np.abs(grads_accum_s[0] / ch_max_s).mean(axis=(0, 1))      # [C_stat]

        ch_max_n = np.abs(grads_accum_n[0]).max(axis=(0, 1), keepdims=True) + 1e-8
        val_n    = np.abs(grads_accum_n[0] / ch_max_n).mean(axis=(0, 1))      # [C_nalc]

        shap_din_acc  += val_d
        shap_stat_acc += val_s
        shap_nalc_acc += val_n
        count_seqs    += 1

        # ── Muestreo de píxeles (fuego y no-fuego) — para figs 1-6 ─
        pos_idx = np.where(y_map.flatten() > 0.5)[0]
        neg_idx = np.where(y_map.flatten() < 0.5)[0]
        np.random.seed(42)
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        for idx in pos_idx[:30]:
            r, c = divmod(idx, W)
            shap_vals_list.append(ig_norm[:, r, c, :])
            labels_list.append(1)
            raw_dyn_list.append(seq["X_dyn"][:, r, c, :])
            hls_list.append(seq["hls_file"])

        for idx in neg_idx[:30]:
            r, c = divmod(idx, W)
            shap_vals_list.append(ig_norm[:, r, c, :])
            labels_list.append(0)
            raw_dyn_list.append(seq["X_dyn"][:, r, c, :])
            hls_list.append(seq["hls_file"])

    if not shap_vals_list:
        return None

    shap_dyn      = np.array(shap_vals_list)[:, :, np.newaxis, np.newaxis, :]
    raw_dyn       = np.array(raw_dyn_list)[:, :, np.newaxis, np.newaxis, :]
    labels        = np.array(labels_list)
    shap_din_mean = shap_din_acc  / count_seqs   # [C_din]  — espacial completo
    shap_stat     = shap_stat_acc / count_seqs   # [C_stat] — espacial completo
    shap_nalc     = shap_nalc_acc / count_seqs   # [C_nalc] — espacial completo

    print(f"  ✓ {len(labels)} píxeles | Rango SHAP dyn: [{shap_dyn.min():.4f}, {shap_dyn.max():.4f}]")
    print(f"  Importancia espacial — din: {shap_din_mean.sum():.4f} | "
          f"stat: {shap_stat.sum():.4f} | nalc: {shap_nalc.sum():.4f}")
    return shap_dyn, shap_din_mean, shap_stat, shap_nalc, labels, raw_dyn, hls_list


# ============================================================
# FIGURA 1 — Importancia global dinámica por vértice ecológico
# ============================================================

def plot_fig01_shap_summary_bar(shap_dyn, labels, out_dir):
    print("  [Fig 01] Importancia global dinámica...")

    shap_2d  = np.abs(shap_dyn).mean(axis=(1, 2, 3))   # [N, C_din]
    pos_mask = labels == 1
    neg_mask = labels == 0
    mean_pos = shap_2d[pos_mask].mean(axis=0) if pos_mask.any() else np.zeros(len(NOMBRES_DIN))
    mean_neg = shap_2d[neg_mask].mean(axis=0) if neg_mask.any() else np.zeros(len(NOMBRES_DIN))

    colors = [
        FIRE_TRIANGLE.get(asignar_vertice(n), {"color": "#999999"})["color"]
        for n in NOMBRES_DIN
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    w = 0.32
    x = np.arange(len(NOMBRES_DIN))

    ax.barh(x + w / 2, mean_pos, w, color=colors, alpha=0.85,
            label='Fire pixels', zorder=2)
    ax.barh(x - w / 2, mean_neg, w, color=colors, alpha=0.35,
            label='Non-fire pixels', hatch='///', edgecolor='gray', zorder=2)

    ax.set_yticks(x)
    ax.set_yticklabels(NOMBRES_DIN)
    ax.set_xlabel('Mean absolute SHAP value', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    for i, v in enumerate(mean_pos):
        if v > 1e-4:
            ax.text(v + max(mean_pos) * 0.015, i, f'{v:.3f}', va='center', fontsize=8)

    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig01_shap_summary.png"))
    plt.close()


# ============================================================
# FIGURA 2 — Dependencia SHAP (top 4 variables dinámicas)
# ============================================================

def plot_fig02_shap_dependence(shap_dyn, X_dyn_flat, labels, out_dir, top_n=4):
    print("  [Fig 02] Dependencia SHAP...")

    shap_mean_t = shap_dyn[:, :, 0, 0, :].mean(axis=1)      # [N, C_din]
    X_mean_t    = X_dyn_flat[:, :, 0, 0, :].mean(axis=1)    # [N, C_din]
    top_idx     = np.argsort(np.abs(shap_mean_t).mean(axis=0))[::-1][:top_n]
    pos_mask    = labels == 1
    neg_mask    = labels == 0

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
    for idx, canal_idx in enumerate(top_idx):
        ax     = axes.flatten()[idx]
        c_name = NOMBRES_DIN[canal_idx]
        x_vals = X_mean_t[:, canal_idx]
        s_vals = shap_mean_t[:, canal_idx]

        if pos_mask.sum() > 0:
            ax.scatter(x_vals[pos_mask], s_vals[pos_mask],
                       c='#d32f2f', alpha=0.7, s=20, label='Fire',
                       zorder=3, edgecolors='white', linewidth=0.3)
        if neg_mask.sum() > 0:
            n_show  = min(100, neg_mask.sum())
            neg_sel = np.random.choice(np.where(neg_mask)[0], n_show, replace=False)
            ax.scatter(x_vals[neg_sel], s_vals[neg_sel],
                       c='#1976d2', alpha=0.5, s=15, label='Non-fire',
                       zorder=2, edgecolors='none')

        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_title(c_name, fontweight='bold')
        ax.set_xlabel('Normalized feature value')
        ax.set_ylabel('SHAP value')
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig02_shap_dependence.png"))
    plt.close()


# ============================================================
# FIGURA 3 — Beeswarm dinámico
# ============================================================

def plot_fig03_shap_beeswarm(shap_dyn, X_dyn_flat, out_dir):
    print("  [Fig 03] Beeswarm dinámico...")

    shap_mean = shap_dyn[:, :, 0, 0, :].mean(axis=1)    # [N, C_din]
    X_mean    = X_dyn_flat[:, :, 0, 0, :].mean(axis=1)  # [N, C_din]
    order     = np.argsort(np.abs(shap_mean).mean(axis=0))[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    np.random.seed(0)
    for idx, feat_idx in enumerate(order):
        y_jitter = idx + np.random.uniform(-0.25, 0.25, shap_mean.shape[0])
        sc = ax.scatter(
            shap_mean[:, feat_idx], y_jitter,
            c=X_mean[:, feat_idx], cmap='RdYlBu_r',
            alpha=0.6, s=25, edgecolors='gray', linewidth=0.2, zorder=2)

    ax.set_yticks(range(len(NOMBRES_DIN)))
    ax.set_yticklabels([NOMBRES_DIN[i] for i in order])
    ax.set_xlabel('SHAP value (impact on model output)', fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.02)
    cbar.set_label('Feature value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig03_shap_beeswarm.png"))
    plt.close()


# ============================================================
# FIGURA 4 — Matriz de interacciones
# ============================================================

def plot_fig04_shap_interactions(model, sequences, out_dir):
    print("  [Fig 04] Matriz de interacciones...")

    seq = sequences[0]
    xd  = np.expand_dims(seq["X_dyn"], axis=0)
    xs  = tf.cast(np.expand_dims(seq["X_sc"], axis=0), tf.float32)
    xn  = tf.cast(np.expand_dims(seq["X_sn"], axis=0), tf.float32)
    H, W = seq["y_map"].shape

    n_features   = len(NOMBRES_DIN)
    interactions = np.zeros((n_features, n_features))
    np.random.seed(42)
    pixels = np.random.choice(H * W, min(50, H * W), replace=False)

    for px in pixels:
        r, c = divmod(px, W)
        with tf.GradientTape(persistent=True) as tape:
            xd_tf = tf.cast(xd, tf.float32)
            tape.watch(xd_tf)
            preds = model([xd_tf, xs, xn], training=False)
            loss  = preds[0, r, c, 0]
        grads = tape.gradient(loss, xd_tf)
        del tape
        if grads is not None:
            g      = grads[0, :, r, c, :].numpy().mean(axis=0)
            interactions += np.outer(np.abs(g), np.abs(g))

    interactions /= len(pixels)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(interactions, cmap='viridis', aspect='auto', vmin=0,
                   extent=[-0.5, n_features - 0.5, n_features - 0.5, -0.5])
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(NOMBRES_DIN, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(NOMBRES_DIN, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Interaction strength', fontsize=9)
    ax.set_title('SHAP Feature Interactions', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig04_shap_interactions.png"))
    plt.close()


# ============================================================
# FIGURA 5 — Mapas espaciales de atribuciones SHAP
# ============================================================

def plot_fig05_shap_spatial(model, sequences, out_dir):
    print("  [Fig 05] Mapas espaciales SHAP...")

    seq = sequences[0]
    xd  = np.expand_dims(seq["X_dyn"], axis=0)
    xs  = tf.cast(np.expand_dims(seq["X_sc"], axis=0), tf.float32)
    xn  = tf.cast(np.expand_dims(seq["X_sn"], axis=0), tf.float32)

    baseline     = np.zeros_like(xd)
    grads_accum  = np.zeros_like(xd)

    for alpha in np.linspace(0.1, 1.0, 10):
        xd_interp = tf.cast(baseline + alpha * (xd - baseline), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xd_interp)
            preds = model([xd_interp, xs, xn], training=False)
            loss  = tf.reduce_mean(preds)
        grad = tape.gradient(loss, xd_interp)
        if grad is not None:
            grads_accum += grad.numpy()

    ig_full = (xd[0] - baseline[0]) * (grads_accum[0] / 10)
    ig_mean = ig_full.mean(axis=0)   # [H, W, C_din]

    key_vars  = ['NDVI', 'PDSI', 'TempMax', 'Viento']
    var_idx   = [NOMBRES_DIN.index(v) for v in key_vars]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    for i, (ax, vi) in enumerate(zip(axes.flatten()[:4], var_idx)):
        sm   = ig_mean[:, :, vi]
        vlim = np.percentile(np.abs(sm), 95)
        im   = ax.imshow(sm, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='auto')
        ax.set_title(NOMBRES_DIN[vi], fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)

    ax_fire = axes[1, 0]
    im = ax_fire.imshow(seq["y_map"], cmap='Reds', aspect='auto')
    ax_fire.set_title('Fire map', fontweight='bold')
    ax_fire.axis('off')
    plt.colorbar(im, ax=ax_fire, fraction=0.046, pad=0.04, shrink=0.8)

    ax_ndvi = axes[1, 1]
    im = ax_ndvi.imshow(seq["X_dyn"][-1, :, :, 0], cmap='RdYlGn',
                        aspect='auto', vmin=0, vmax=1)
    ax_ndvi.set_title('NDVI (t)', fontweight='bold')
    ax_ndvi.axis('off')
    plt.colorbar(im, ax=ax_ndvi, fraction=0.046, pad=0.04, shrink=0.8)

    axes[1, 2].axis('off')

    plt.suptitle('Spatial SHAP Attributions', fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig05_shap_spatial.png"))
    plt.close()


# ============================================================
# FIGURA 6 — Evolución temporal de importancia
# ============================================================

def plot_fig06_shap_temporal(shap_dyn, labels, hls_list, out_dir):
    print("  [Fig 06] Evolución temporal...")

    dates = [parse_fecha_robusta(f) or datetime(2021, 1, 1) for f in hls_list]
    shap_abs = np.abs(shap_dyn[:, :, 0, 0, :]).mean(axis=1)   # [N, C_din]
    months   = [d.month for d in dates]
    uniq_m   = sorted(set(months))

    monthly = np.array([
        shap_abs[np.array(months) == m].mean(axis=0)
        if (np.array(months) == m).sum() > 0 else np.zeros(len(NOMBRES_DIN))
        for m in uniq_m
    ])

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax  = plt.subplots(figsize=(8, 5))
    x        = range(len(uniq_m))
    bottom   = np.zeros(len(uniq_m))
    colors   = plt.cm.RdYlGn(np.linspace(0, 1, len(NOMBRES_DIN)))

    for i in range(len(NOMBRES_DIN)):
        ax.bar(x, monthly[:, i], bottom=bottom,
               label=NOMBRES_DIN[i], color=colors[i], alpha=0.8, edgecolor='none')
        bottom += monthly[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m - 1] for m in uniq_m])
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Mean absolute SHAP', fontweight='bold')
    ax.set_title('Temporal evolution of feature importance', fontweight='bold', pad=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7,
              frameon=True, fancybox=False, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig06_shap_temporal.png"))
    plt.close()


# ============================================================
# FIGURA 7 — Importancia global completa (Dinámica + Estática + NALCMS)
# ============================================================

def plot_fig07_global_all(shap_din_mean, shap_stat, shap_nalc,
                          nombres_nalcms, out_dir):
    print("  [Fig 07] Importancia global (todas las variables)...")

    all_names  = NOMBRES_DIN + NOMBRES_STATIC + nombres_nalcms
    all_values = np.concatenate([shap_din_mean, shap_stat, shap_nalc])
    colors     = (
        [COLOR_DIN]  * len(shap_din_mean) +
        [COLOR_STAT] * len(shap_stat)     +
        [COLOR_NALC] * len(shap_nalc)
    )
    order = np.argsort(all_values)[::-1]

    fig, ax = plt.subplots(figsize=(9, max(8, len(all_names) * 0.28)))
    y = np.arange(len(all_names))
    ax.barh(y, all_values[order],
            color=[colors[i] for i in order], edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels([all_names[i] for i in order], fontsize=8)
    ax.set_xlabel('Mean Normalized SHAP Value', fontweight='bold')
    ax.set_title('Global Feature Importance (All Variables)', fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    legend_elements = [
        Patch(facecolor=COLOR_DIN,  label='Dinámica (Clima/Vegetación)'),
        Patch(facecolor=COLOR_STAT, label='Estática (Topografía/Humano)'),
        Patch(facecolor=COLOR_NALC, label='NALCMS (Uso de Suelo)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig07_global_importance_all.png"))
    plt.close()


# ============================================================
# FIGURA 08a — Barra apilada de contribución por rama
# ============================================================

def _branch_pcts(shap_din_mean, shap_stat, shap_nalc):
    """Calcula porcentajes de contribución por rama."""
    sums  = np.array([shap_din_mean.sum(), shap_stat.sum(), shap_nalc.sum()])
    total = sums.sum()
    return sums / total * 100


def plot_fig08a_branch_bar(shap_din_mean, shap_stat, shap_nalc, out_dir):
    print("  [Fig 08a] Barra apilada por rama...")

    pcts = _branch_pcts(shap_din_mean, shap_stat, shap_nalc)
    labs = ['Dinámica\n(Clima/Veg)', 'Estática\n(Topo/Humano)', 'NALCMS\n(Uso Suelo)']
    cols = [COLOR_DIN, COLOR_STAT, COLOR_NALC]

    # Figura más alta para dar espacio a la leyenda horizontal debajo
    fig, ax = plt.subplots(figsize=(10, 3.2))
    left = 0.0
    for pct, col, lab in zip(pcts, cols, labs):
        ax.barh([0], [pct], left=[left], color=col, height=0.55, label=lab)
        ax.text(left + pct / 2, 0, f'{pct:.1f}%',
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Contribución Relativa (%)', fontweight='bold')
    ax.set_title('Contribución Relativa por Rama del Modelo', fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    # Leyenda horizontal centrada debajo de la barra
    labs_flat = [l.replace('\n', ' ') for l in labs]
    legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(cols, labs_flat)]
    ax.legend(handles=legend_elements,
              loc='upper center', bbox_to_anchor=(0.5, -0.28),
              ncol=3, fontsize=9, frameon=True, fancybox=False,
              edgecolor='#cccccc', handlelength=1.2, handleheight=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(out_dir, "fig08a_branch_bar.png"))
    plt.close()


# ============================================================
# FIGURA 08b — Pie chart de contribución por rama
# ============================================================

def plot_fig08b_branch_pie(shap_din_mean, shap_stat, shap_nalc, out_dir):
    print("  [Fig 08b] Pie chart por rama...")

    pcts = _branch_pcts(shap_din_mean, shap_stat, shap_nalc)
    labs = ['Dinámica\n(Clima/Veg)', 'Estática\n(Topo/Humano)', 'NALCMS\n(Uso Suelo)']
    cols = [COLOR_DIN, COLOR_STAT, COLOR_NALC]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        pcts, labels=labs, colors=cols,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10},
        wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'},
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(11)
    ax.set_title('Distribución de Importancia por Rama', fontweight='bold', pad=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig08b_branch_pie.png"))
    plt.close()


# ============================================================
# FIGURA 08c — Top-N por rama (3 columnas, sin barra apilada)
# ============================================================

def plot_fig08c_branch_summary(shap_din_mean, shap_stat, shap_nalc,
                                nombres_nalcms, out_dir, top_n=10):
    print("  [Fig 08c] Top-N por rama...")

    pcts = _branch_pcts(shap_din_mean, shap_stat, shap_nalc)
    cols = [COLOR_DIN, COLOR_STAT, COLOR_NALC]

    def top_rama(values, names, n):
        idx = np.argsort(values)[::-1][:n]
        return [names[i] for i in idx], values[idx]

    rama_data = [
        top_rama(shap_din_mean, NOMBRES_DIN,    top_n),
        top_rama(shap_stat,     NOMBRES_STATIC, top_n),
        top_rama(shap_nalc,     nombres_nalcms, top_n),
    ]
    rama_titles = [
        f'Rama Dinámica  ({pcts[0]:.1f}%)\nClima / Vegetación',
        f'Rama Estática  ({pcts[1]:.1f}%)\nTopografía / Humano',
        f'NALCMS  ({pcts[2]:.1f}%)\nUso de Suelo',
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(14, 7),
        gridspec_kw={"wspace": 0.52}
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.08)

    for ax, (names, values), title, color in zip(axes, rama_data, rama_titles, cols):
        y    = np.arange(len(names))
        bars = ax.barh(y, values, color=color, alpha=0.85, edgecolor='none')

        x_max = values.max() if values.max() > 0 else 1
        for bar, val in zip(bars, values):
            ax.text(val + x_max * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=7.5)

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel('Mean Norm. SHAP', fontweight='bold', fontsize=8)
        ax.set_title(title, fontweight='bold', fontsize=9, color=color, pad=8)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, x_max * 1.22)

    fig.suptitle('Top Variables por Rama del Modelo',
                 fontweight='bold', fontsize=11, y=0.97)

    plt.savefig(os.path.join(out_dir, "fig08c_branch_top10.png"), dpi=300)
    plt.close()



# ============================================================
# FIGURA 9 — Top 20 variables (todas las ramas)
# ============================================================

def plot_fig09_top20(shap_din_mean, shap_stat, shap_nalc,
                     nombres_nalcms, out_dir):
    print("  [Fig 09] Top 20 variables (todas las ramas)...")

    all_names  = NOMBRES_DIN + NOMBRES_STATIC + nombres_nalcms
    all_values = np.concatenate([shap_din_mean, shap_stat, shap_nalc])
    colors     = (
        [COLOR_DIN]  * len(shap_din_mean) +
        [COLOR_STAT] * len(shap_stat)     +
        [COLOR_NALC] * len(shap_nalc)
    )
    top20 = np.argsort(all_values)[::-1][:20]

    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(top20))
    ax.barh(y, all_values[top20],
            color=[colors[i] for i in top20], edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels([all_names[i] for i in top20], fontsize=9)
    ax.set_xlabel('Mean Normalized SHAP Value', fontweight='bold')
    ax.set_title('Top 20 Features', fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    legend_elements = [
        Patch(facecolor=COLOR_DIN,  label='Dinámica'),
        Patch(facecolor=COLOR_STAT, label='Estática'),
        Patch(facecolor=COLOR_NALC, label='NALCMS'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig09_top20_all.png"))
    plt.close()


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    print("\n" + "=" * 60)
    print(f"  KASSANDRA-PYR — SHAP Unificado (n_steps={STEPS})")
    print("=" * 60)

    # 1. Cargar modelo ──────────────────────────────────────
    print("\n[1] Cargando modelo...")
    with tf.keras.utils.custom_object_scope(CUSTOM_DICT):
        model = tf.keras.models.load_model(MODEL_PATH)
    print("  ✓ Modelo cargado")

    # Dimensiones de las ramas estáticas
    N_NALCMS = int(model.inputs[2].shape[-1])
    print(f"  Ramas: dinámica={len(NOMBRES_DIN)} | estática={len(NOMBRES_STATIC)} | NALCMS={N_NALCMS}")

    # 2. Nombres NALCMS (lookup hardcodeado) ─────────────────
    print("\n[2] Cargando nombres NALCMS...")
    nombres_nalcms = build_nalcms_names(N_NALCMS)
    print(f"  Nombres (primeros 5): {nombres_nalcms[:5]}")

    # 3. Seleccionar año con más incendios ────────────────────
    print("\n[3] Seleccionando año con más incendios...")
    all_hls = sorted(glob.glob(os.path.join(HLS_DIR, "*.tif")))
    years   = sorted(set(
        int(re.search(r'(20\d{2})', os.path.basename(f)).group(1))
        for f in all_hls if re.search(r'(20\d{2})', os.path.basename(f))
    ))

    year_stats = {}
    for year in years:
        y_files = [
            os.path.join(Y_DIR, f) for f in os.listdir(Y_DIR)
            if f.endswith(".tif") and str(year) in f
        ]
        total_px = sum(
            np.sum(rasterio.open(yf).read(1) > 0) for yf in y_files)
        year_stats[year] = total_px
        print(f"    {year}: {total_px:,} píxeles")

    best_year = max(year_stats, key=year_stats.get)
    print(f"  ✓ Año seleccionado: {best_year} ({year_stats[best_year]:,} px)")

    # 4. Cargar datos ─────────────────────────────────────────
    print(f"\n[4] Cargando datos {best_year}...")
    hls_files = sorted([f for f in all_hls if str(best_year) in os.path.basename(f)])
    y_files   = sorted([
        os.path.join(Y_DIR, f) for f in os.listdir(Y_DIR)
        if str(best_year) in f and f.endswith(".tif")
    ])

    sequences = build_sequences(model, hls_files, y_files)
    if not sequences:
        print("[ERROR] No hay secuencias válidas")
        return

    # 5. Filtrar secuencias con incendios ────────────────────
    print("\n[5] Filtrando secuencias con incendios...")
    fire_seqs = sorted(
        [(i, s) for i, s in enumerate(sequences) if s["n_fire_pixels"] > 0],
        key=lambda x: x[1]["n_fire_pixels"], reverse=True
    )
    if not fire_seqs:
        print("[ERROR] No hay secuencias con incendios")
        return

    print(f"  ✓ {len(fire_seqs)} secuencias con incendios")
    for k, (si, s) in enumerate(fire_seqs[:3]):
        print(f"    {k+1}. Seq {si}: {s['n_fire_pixels']:,} px | {os.path.basename(s['hls_file'])}")

    selected_seqs = [s for _, s in fire_seqs[:MAX_SEQS]]

    # 6. Calcular SHAP ────────────────────────────────────────
    print(f"\n[6] Calculando SHAP (n_steps={STEPS})...")
    result = compute_shap_all_branches(
        model, selected_seqs, n_steps=STEPS, max_seqs=MAX_SEQS)

    if result is None:
        print("[ERROR] Cálculo SHAP falló")
        return

    shap_dyn, shap_din_mean, shap_stat, shap_nalc, labels, raw_dyn, hls_list = result

    # Guardar arrays
    np.save(os.path.join(OUT_DIR, "shap_dyn.npy"),   shap_dyn)
    np.save(os.path.join(OUT_DIR, "shap_stat.npy"),  shap_stat)
    np.save(os.path.join(OUT_DIR, "shap_nalc.npy"),  shap_nalc)
    np.save(os.path.join(OUT_DIR, "shap_labels.npy"), labels)

    # 7. Reportar top variables ──────────────────────────────
    print("\n[7] Top 3 por rama:")
    top_din  = sorted(zip(NOMBRES_DIN,     shap_din_mean), key=lambda x: x[1], reverse=True)[:3]
    top_stat = sorted(zip(NOMBRES_STATIC,  shap_stat),     key=lambda x: x[1], reverse=True)[:3]
    top_nalc = sorted(zip(nombres_nalcms,  shap_nalc),     key=lambda x: x[1], reverse=True)[:3]
    print(f"  Dinámica : {top_din}")
    print(f"  Estática : {top_stat}")
    print(f"  NALCMS   : {top_nalc}")

    # 8. Generar figuras ──────────────────────────────────────
    print("\n[8] Generando figuras...")
    plot_fig01_shap_summary_bar(shap_dyn, labels, OUT_DIR)
    plot_fig02_shap_dependence(shap_dyn, raw_dyn, labels, OUT_DIR)
    plot_fig03_shap_beeswarm(shap_dyn, raw_dyn, OUT_DIR)
    plot_fig04_shap_interactions(model, selected_seqs, OUT_DIR)
    plot_fig05_shap_spatial(model, selected_seqs, OUT_DIR)
    plot_fig06_shap_temporal(shap_dyn, labels, hls_list, OUT_DIR)
    plot_fig07_global_all(shap_din_mean, shap_stat, shap_nalc, nombres_nalcms, OUT_DIR)
    plot_fig08a_branch_bar(shap_din_mean, shap_stat, shap_nalc, OUT_DIR)
    plot_fig08b_branch_pie(shap_din_mean, shap_stat, shap_nalc, OUT_DIR)
    plot_fig08c_branch_summary(shap_din_mean, shap_stat, shap_nalc, nombres_nalcms, OUT_DIR)
    plot_fig09_top20(shap_din_mean, shap_stat, shap_nalc, nombres_nalcms, OUT_DIR)

    print("\n" + "=" * 60)
    print(f"  ✓ ANÁLISIS COMPLETADO — {len(selected_seqs)} secuencias | {len(labels)} píxeles")
    print(f"  Resultados: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()