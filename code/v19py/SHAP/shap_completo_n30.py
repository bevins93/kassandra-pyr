#!/usr/bin/env python3
#!/usr/bin/env python3
# ============================================================
# shap_completo_n30.py — KASSANDRA-PYR (XAI Analysis)
# Análisis SHAP Integral para Artículo Científico (Steps=30)
# ============================================================

import os
import re
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─ Intentar importar rasterio ─────────────────────────────
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("[WARN] rasterio no disponible")

# ============================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================

# ── PARÁMETROS CRÍTICOS ──────────────────────────────────────
STEPS = 30  # ← Número de pasos para Integrated Gradients
# ============================================================

# ── RUTAS Y DIRECTORIOS ────────────────────────────────────
RUN_DIR     = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
MODEL_PATH  = os.path.join(RUN_DIR, "best_model.keras")
OUT_DIR     = os.path.join(RUN_DIR, f"shap_n{STEPS}")
os.makedirs(OUT_DIR, exist_ok=True)

HLS_DIR      = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos"
CLIMATE_DIR  = "/lustre/home/kgalvan/cnn/x/Terraclimate_DL_Dataset"
STATIC_TIF   = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
Y_DIR        = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Mensual"

# Parámetros
SEQ_LEN = 3
N_CANALES_DIN = 12

# ─ Nombres de canales ───────────────────────────────────────
NOMBRES_DIN = [
    "NDVI", "NBR", "NDWI",
    "mask_NDVI", "mask_NBR", "mask_NDWI",
    "TempMin", "TempMax", "Viento",
    "Precip", "PDSI", "Evapo"
]

# ─ Triángulo ecológico del fuego ────────────────────────────
FIRE_TRIANGLE = {
    "Combustible": {
        "canales": ["NDVI", "NBR", "NDWI"],
        "color": "#2E7D32"
    },
    "Clima": {
        "canales": ["TempMin", "TempMax", "Viento", "Precip", "PDSI", "Evapo"],
        "color": "#1565C0"
    },
    "Topografía": {
        "canales": ["elevation", "slope", "aspect_cos", "aspect_sin", "tpi", "roughness"],
        "color": "#6A1B9A"
    },
    "Ignición": {
        "canales": ["pop_density", "ag_fraction", "nalcms"],
        "color": "#B71C1C"
    },
    "Máscaras": {
        "canales": ["mask_NDVI", "mask_NBR", "mask_NDWI"],
        "color": "#546E7A"
    }
}

# ── Estilo Nature para gráficos ──────────────────────────────
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'figure.figsize': [8, 6]
})

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def parse_fecha_robusta(filename):
    """Parsea fechas de nombres de archivo con múltiples formatos."""
    base = os.path.basename(filename)
    
    patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
        (r'(\d{4})_(\d{2})_(\d{2})', '%Y_%m_%d'),
        (r'(\d{8})', '%Y%m%d'),
        (r'(\d{4})-(\d{2})(?!\d)', '%Y-%m'),
        (r'(\d{4})_(\d{2})(?!\d)', '%Y_%m'),
        (r'(\d{4})(\d{2})(?!\d)', '%Y%m'),
    ]
    
    for pattern, fmt in patterns:
        m = re.search(pattern, base)
        if m:
            try:
                if fmt in ['%Y-%m', '%Y_%m', '%Y%m']:
                    year, month = int(m.group(1)), int(m.group(2))
                    return datetime(year, month, 15)
                return datetime.strptime('-'.join(m.groups()), fmt.replace('_', '-'))
            except (ValueError, IndexError):
                continue
    return None

def gelu(x): return x * tf.sigmoid(1.702 * x)
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
        y_pred = tf.clip_by_value(tf.reshape(y_pred, [-1]), 1e-7, 1-1e-7)
        tp  = tf.reduce_sum(y_true * y_pred)
        fp  = tf.reduce_sum((1-y_true) * y_pred)
        fn  = tf.reduce_sum(y_true * (1-y_pred))
        tvi = (tp+1e-7) / (tp + alpha*fn + beta*fp + 1e-7)
        return tf.reduce_mean(tf.pow(1.0-tvi, gamma))
    return loss

custom_dict = {
    "gelu": gelu, "match_shape": match_shape, "resize_gate": resize_gate, 
    "resize_static": resize_static, "loss": focal_tversky_loss()
}

def asignar_vertice(canal):
    """Clasifica un canal en su vértice ecológico."""
    for vertice, datos in FIRE_TRIANGLE.items():
        if canal in datos["canales"]:
            return vertice
    return "Otros"

# ============================================================
# CARGA DE DATOS GIS
# ============================================================

def read_and_fix(path, out_shape_hw=None, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        if out_shape_hw and (src.height, src.width) != out_shape_hw:
            data = src.read(out_shape=(src.count, out_shape_hw[0], out_shape_hw[1]), resampling=resampling)
        else: data = src.read()
        return np.transpose(data, (1, 2, 0)).astype(np.float32)

def sanitize_array(arr, clip_val=10.0):
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -clip_val, clip_val).astype(np.float32)

def procesar_hls_con_mascara(raw_hw_c):
    CANALES_HLS_IDX = [0, 1, 2]
    nube = raw_hw_c[:, :, 3] if raw_hw_c.shape[-1] > 3 else raw_hw_c[:, :, -1]
    valido = (np.isfinite(nube) & (nube > 0.5)).astype(np.float32)
    nublado = ~(np.isfinite(nube) & (nube > 0.5))
    res = []
    for ci in CANALES_HLS_IDX:
        canal = np.clip(np.where(np.isfinite(raw_hw_c[:, :, ci]), raw_hw_c[:, :, ci], 0.0), 0.0, 1.0)
        canal[nublado] = 0.0
        res.append(canal)
    for _ in CANALES_HLS_IDX: res.append(valido.copy())
    return np.stack(res, axis=-1).astype(np.float32)

def normalizar_tc(arr_hwc):
    TC_RANGES = {0: (-30.0, 50.0), 1: (-20.0, 60.0), 2: (0.0, 15.0), 
                 3: (0.0, 500.0), 4: (-5.0, 5.0), 5: (0.0, 200.0)}
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
                    clim_map[mdt.strftime("%Y-%m")] = f
                    
    for key in [dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")]:
        if key in clim_map:
            raw = read_and_fix(os.path.join(CLIMATE_DIR, clim_map[key]), target_shape, Resampling.bilinear)
            tc = np.stack([raw[:, :, i] for i in range(min(6, raw.shape[-1]))], axis=-1)
            return normalizar_tc(sanitize_array(tc))
    return np.zeros((*target_shape, 6), np.float32)

def load_static_features(model, tif_path, target_shape):
    """Carga features estáticas usando el modelo para inferir dimensiones."""
    if not RASTERIO_OK or not os.path.exists(tif_path): return None, None
    try:
        try: 
            expected_sn_dim = model.inputs[2].shape[-1]
        except: 
            expected_sn_dim = 19
            
        with rasterio.open(tif_path) as src:
            data = src.read(out_shape=(src.count, target_shape[0], target_shape[1]), 
                          resampling=Resampling.bilinear).astype(np.float32)
        all_bands = np.transpose(data, (1, 2, 0))
        static_cont = sanitize_array(all_bands[:, :, :9], clip_val=1.0)
        
        nalcms_oh = np.zeros((*target_shape, expected_sn_dim), dtype=np.float32)
        if all_bands.shape[-1] > 9:
            nalcms_raw = np.clip(all_bands[:, :, 9].astype(np.int32), 0, expected_sn_dim - 1)
            for c in range(expected_sn_dim):
                nalcms_oh[:, :, c] = (nalcms_raw == c).astype(np.float32)
        return static_cont, nalcms_oh
    except Exception as e:
        print(f"    [WARN] Error estáticos: {e}")
        return None, None

def build_sequences_test_debug(model, hls_files, y_files, ventana_dias=60):
    """Versión con debug exhaustivo del matching temporal."""
    H, W = model.inputs[0].shape[2], model.inputs[0].shape[3]
    target_shape = (H, W)
    print(f"    Shape: {H}×{W} | Ventana: ±{ventana_dias}d")
    
    static_cont, nalcms_oh = load_static_features(model, STATIC_TIF, target_shape)
    if static_cont is None:
        try: expected_sn_dim = model.inputs[2].shape[-1]
        except: expected_sn_dim = 19
        static_cont = np.zeros((*target_shape, 9), dtype=np.float32)
        nalcms_oh = np.zeros((*target_shape, expected_sn_dim), dtype=np.float32)

    print(f"    Indexando {len(y_files)} archivos Y...")
    y_dates = sorted(set(parse_fecha_robusta(yf) for yf in y_files if parse_fecha_robusta(yf)))
    if y_dates:
        print(f"    Fechas Y: {y_dates[0].strftime('%Y-%m')} a {y_dates[-1].strftime('%Y-%m')}")
    
    sequences = []
    matched_count = 0
    
    for i in range(len(hls_files) - SEQ_LEN):
        last_hls = hls_files[i + SEQ_LEN - 1]
        ref_date = parse_fecha_robusta(last_hls)
        
        if ref_date is None: continue
        
        best_y_file = None
        best_delta = ventana_dias + 1
        
        for yf in y_files:
            ydt = parse_fecha_robusta(yf)
            if ydt:
                delta = abs((ydt - ref_date).days)
                if delta < best_delta:
                    best_delta = delta
                    best_y_file = yf
        
        if best_y_file and best_delta <= ventana_dias:
            try:
                seq_imgs = []
                for j in range(SEQ_LEN):
                    fi = hls_files[i+j]
                    dt = parse_fecha_robusta(fi)
                    if dt is None: raise ValueError(f"Fecha no parseable: {fi}")
                    
                    hls_frame = procesar_hls_con_mascara(read_and_fix(fi, target_shape))
                    tc_frame = _leer_tc(dt, target_shape)
                    full_frame = sanitize_array(np.concatenate([hls_frame, tc_frame], axis=-1))
                    seq_imgs.append(full_frame)
                
                y_map = np.zeros((H, W), dtype=np.float32)
                try:
                    ym = read_and_fix(best_y_file, target_shape, Resampling.nearest)[:, :, 0]
                    y_map = (ym > 0).astype(np.float32)
                except Exception: pass
                
                n_fire = int(np.sum(y_map > 0.5))
                
                sequences.append({
                    "X_dyn": np.stack(seq_imgs, axis=0),
                    "X_sc": static_cont, "X_sn": nalcms_oh,
                    "y_map": y_map, "hls_file": last_hls,
                    "n_fire_pixels": n_fire
                })
                matched_count += 1
            except Exception: pass
    
    total_fire_px = sum(s["n_fire_pixels"] for s in sequences)
    seq_with_fires = sum(1 for s in sequences if s["n_fire_pixels"] > 0)
    print(f"    Secuencias: {matched_count} | Con incendios: {seq_with_fires} | Píxeles: {total_fire_px:,}")
    
    return sequences

def build_sequences_test(model, hls_files, y_files):
    return build_sequences_test_debug(model, hls_files, y_files, ventana_dias=60)

# ============================================================
# CÁLCULO SHAP (CORREGIDO)
# ============================================================

def compute_shap_attributions(model, sequences, n_steps=30, max_seqs=2):
    """Integrated Gradients con diagnóstico completo."""
    print(f"\nCalculando atribuciones (steps={n_steps}, seqs={max_seqs})...")
    
    shap_vals_list, labels_list, raw_dyn_list, hls_list = [], [], [], []
    
    for seq_idx, seq in enumerate(sequences[:max_seqs]):
        print(f"  Mapa {seq_idx+1}/{max_seqs}...")
        
        xd = np.expand_dims(seq["X_dyn"], axis=0)
        xs = np.expand_dims(seq["X_sc"], axis=0)
        xn = np.expand_dims(seq["X_sn"], axis=0)
        y_map = seq["y_map"]
        H, W = y_map.shape
        
        baseline_dyn = np.zeros_like(xd)
        grads_accum = np.zeros_like(xd)
        alphas = np.linspace(0.01, 1.0, n_steps)
        
        # ✅ CORRECCIÓN: Convertir inputs estáticos a Tensores UNA vez antes del bucle
        xs_tf = tf.cast(xs, tf.float32)
        xn_tf = tf.cast(xn, tf.float32)
        
        for step, alpha in enumerate(alphas):
            xd_interp = baseline_dyn + alpha * (xd - baseline_dyn)
            xd_interp = tf.cast(xd_interp, tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(xd_interp)
                # ✅ CORRECCIÓN: Usar tensores (xs_tf, xn_tf) en la llamada al modelo
                preds = model([xd_interp, xs_tf, xn_tf], training=False)
                loss_val = tf.reduce_mean(preds)
            
            grad = tape.gradient(loss_val, xd_interp)
            if grad is not None:
                grads_accum += grad.numpy()
        
        ig_full = (xd[0] - baseline_dyn[0]) * (grads_accum[0] / n_steps)
        channel_max = np.abs(ig_full).max(axis=(0, 1, 2), keepdims=True) + 1e-8
        ig_normalized = ig_full / channel_max
        
        pos_idx = np.where(y_map.flatten() > 0.5)[0]
        neg_idx = np.where(y_map.flatten() < 0.5)[0]
        np.random.seed(42)
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        
        for idx in pos_idx[:30]:
            r, c = divmod(idx, W)
            shap_vals_list.append(ig_normalized[:, r, c, :])
            labels_list.append(1)
            raw_dyn_list.append(seq["X_dyn"][:, r, c, :])
            hls_list.append(seq["hls_file"])
            
        for idx in neg_idx[:30]:
            r, c = divmod(idx, W)
            shap_vals_list.append(ig_normalized[:, r, c, :])
            labels_list.append(0)
            raw_dyn_list.append(seq["X_dyn"][:, r, c, :])
            hls_list.append(seq["hls_file"])
    
    if not shap_vals_list: return None
    
    shap_vals = np.array(shap_vals_list)
    labels = np.array(labels_list)
    raw_dyn = np.array(raw_dyn_list)
    
    shap_dyn = shap_vals[:, :, np.newaxis, np.newaxis, :]
    raw_dyn_4d = raw_dyn[:, :, np.newaxis, np.newaxis, :]
    
    print(f"  ✓ Completado: {len(labels)} píxeles")
    print(f"  Rango SHAP: [{shap_dyn.min():.4f}, {shap_dyn.max():.4f}]")
    
    return shap_dyn, labels, raw_dyn_4d, hls_list

# ============================================================
# VISUALIZACIONES - ESTILO NATURE (CORREGIDAS)
# ============================================================

def plot_shap_summary_bar(shap_dyn, labels, out_dir):
    """Figura 1: Importancia global por vértice ecológico."""
    print("  [1/6] Barra de importancia global...")
    
    shap_2d = np.abs(shap_dyn).mean(axis=(1, 2, 3))
    pos_mask, neg_mask = labels == 1, labels == 0
    
    mean_pos = shap_2d[pos_mask].mean(axis=0) if pos_mask.any() else np.zeros(12)
    mean_neg = shap_2d[neg_mask].mean(axis=0) if neg_mask.any() else np.zeros(12)
    
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    w, x = 0.32, np.arange(len(NOMBRES_DIN))
    
    colors = [FIRE_TRIANGLE.get(asignar_vertice(n), {"color": "#999999"})["color"] for n in NOMBRES_DIN]
    
    ax.barh(x + w/2, mean_pos, w, color=colors, alpha=0.85, label='Fire pixels', zorder=2)
    ax.barh(x - w/2, mean_neg, w, color=colors, alpha=0.35, label='Non-fire pixels', 
            hatch='///', edgecolor='gray', zorder=2)
    
    ax.set_yticks(x)
    ax.set_yticklabels(NOMBRES_DIN)
    ax.set_xlabel('Mean absolute SHAP value', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    
    # Añadir valores numéricos
    for i, v in enumerate(mean_pos):
        if v > 1e-4:
            ax.text(v + max(mean_pos)*0.015, i, f'{v:.3f}', va='center', fontsize=8)
    
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig01_shap_summary.png"), 
                bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_shap_dependence(shap_dyn, X_dyn_flat, labels, out_dir, top_n=4):
    """Figura 2: Dependencia SHAP para variables top."""
    print("  [2/6] Dependencia SHAP...")
    
    shap_mean_t = shap_dyn[:, :, 0, 0, :].mean(axis=1)
    shap_abs_mean = np.abs(shap_mean_t).mean(axis=0)
    top_idx = np.argsort(shap_abs_mean)[::-1][:top_n]
    pos_mask = labels == 1
    
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
    X_mean_t = X_dyn_flat[:, :, 0, 0, :].mean(axis=1)
    
    for idx, canal_idx in enumerate(top_idx):
        ax = axes.flatten()[idx]
        c_name = NOMBRES_DIN[canal_idx]
        
        x_vals = X_mean_t[:, canal_idx]
        shap_vals = shap_mean_t[:, canal_idx]
        
        if pos_mask.sum() > 0:
            ax.scatter(x_vals[pos_mask], shap_vals[pos_mask], 
                      c='#d32f2f', alpha=0.7, s=20, label='Fire', zorder=3, edgecolors='white', linewidth=0.3)
        
        neg_mask = labels == 0
        if neg_mask.sum() > 0:
            n_show = min(100, neg_mask.sum())
            neg_idx = np.random.choice(np.where(neg_mask)[0], n_show, replace=False)
            ax.scatter(x_vals[neg_idx], shap_vals[neg_idx], 
                      c='#1976d2', alpha=0.5, s=15, label='Non-fire', zorder=2, edgecolors='none')
        
        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_title(c_name, fontweight='bold')
        ax.set_xlabel('Normalized feature value')
        ax.set_ylabel('SHAP value')
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig02_shap_dependence.png"), 
                bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_shap_beeswarm(shap_dyn, X_dyn_flat, labels, out_dir):
    """Figura 3: Beeswarm plot - Distribución de impactos SHAP."""
    print("  [3/6] Beeswarm plot...")
    
    shap_values = shap_dyn[:, :, 0, 0, :]
    X_values = X_dyn_flat[:, :, 0, 0, :]
    
    shap_mean = shap_values.mean(axis=1)
    X_mean = X_values.mean(axis=1)
    
    importance = np.abs(shap_mean).mean(axis=0)
    order = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    
    for idx, feat_idx in enumerate(order):
        feat_name = NOMBRES_DIN[feat_idx]
        shap_f = shap_mean[:, feat_idx]
        feat_val = X_mean[:, feat_idx]
        
        # Jitter para evitar solapamiento
        y_jitter = idx + np.random.uniform(-0.25, 0.25, len(shap_f))
        
        scatter = ax.scatter(shap_f, y_jitter, c=feat_val, 
                           cmap='RdYlBu_r', alpha=0.6, s=25, 
                           edgecolors='gray', linewidth=0.2, zorder=2)
    
    ax.set_yticks(range(len(NOMBRES_DIN)))
    ax.set_yticklabels([NOMBRES_DIN[i] for i in order])
    ax.set_xlabel('SHAP value (impact on model output)', fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.02)
    cbar.set_label('Feature value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig03_shap_beeswarm.png"), 
                bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_shap_interactions(model, sequences, out_dir):
    """Figura 4: Matriz de interacciones SHAP."""
    print("  [4/6] Matriz de interacciones...")
    
    seq = sequences[0]
    xd = np.expand_dims(seq["X_dyn"], axis=0)
    xs = np.expand_dims(seq["X_sc"], axis=0)
    xn = np.expand_dims(seq["X_sn"], axis=0)
    y_map = seq["y_map"]
    
    H, W = y_map.shape
    n_features = 12
    interactions = np.zeros((n_features, n_features))
    
    np.random.seed(42)
    n_pixels = min(50, H * W)
    pixels = np.random.choice(H * W, n_pixels, replace=False)
    
    # ✅ CORRECCIÓN: Convertir inputs estáticos a Tensores antes del bucle
    xs_tf = tf.cast(xs, tf.float32)
    xn_tf = tf.cast(xn, tf.float32)
    
    for px in pixels:
        r, c = divmod(px, W)
        
        with tf.GradientTape(persistent=True) as tape:
            xd_tf = tf.cast(xd, tf.float32)
            tape.watch(xd_tf)
            # ✅ CORRECCIÓN: Usar tensores (xs_tf, xn_tf)
            preds = model([xd_tf, xs_tf, xn_tf], training=False)
            loss = preds[0, r, c, 0]
        
        grads = tape.gradient(loss, xd_tf)
        if grads is not None:
            g = grads[0, :, r, c, :].numpy()
            g_mean = g.mean(axis=0)
            interactions += np.outer(np.abs(g_mean), np.abs(g_mean))
        
        del tape
    
    interactions /= n_pixels
    
    fig, ax = plt.subplots(figsize=(7, 6.5))
    
    mask = np.zeros_like(interactions)
    np.fill_diagonal(mask, 1)
    
    im = ax.imshow(interactions, cmap='viridis', aspect='auto', vmin=0, 
                  extent=[-0.5, 11.5, 11.5, -0.5])
    
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(NOMBRES_DIN, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(NOMBRES_DIN, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Interaction strength', fontsize=9)
    
    ax.set_title('SHAP Feature Interactions', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig04_shap_interactions.png"), 
                bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_shap_spatial_maps(model, sequences, out_dir):
    """Figura 5: Mapas espaciales de atribuciones SHAP."""
    print("  [5/6] Mapas espaciales...")
    
    seq = sequences[0]
    xd = np.expand_dims(seq["X_dyn"], axis=0)
    xs = np.expand_dims(seq["X_sc"], axis=0)
    xn = np.expand_dims(seq["X_sn"], axis=0)
    y_map = seq["y_map"]
    
    H, W = y_map.shape
    
    baseline = np.zeros_like(xd)
    alphas = np.linspace(0.1, 1.0, 10)
    grads_accum = np.zeros_like(xd)
    
    # ✅ CORRECCIÓN: Convertir inputs estáticos a Tensores antes del bucle
    xs_tf = tf.cast(xs, tf.float32)
    xn_tf = tf.cast(xn, tf.float32)
    
    for alpha in alphas:
        xd_interp = baseline + alpha * (xd - baseline)
        xd_interp = tf.cast(xd_interp, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(xd_interp)
            # ✅ CORRECCIÓN: Usar tensores (xs_tf, xn_tf)
            preds = model([xd_interp, xs_tf, xn_tf], training=False)
            loss = tf.reduce_mean(preds)
        
        grad = tape.gradient(loss, xd_interp)
        if grad is not None:
            grads_accum += grad.numpy()
    
    ig_full = (xd[0] - baseline[0]) * (grads_accum[0] / len(alphas))
    ig_mean = ig_full.mean(axis=0)
    
    key_vars = ['NDVI', 'PDSI', 'TempMax', 'Viento']
    var_indices = [NOMBRES_DIN.index(v) for v in key_vars]
    
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    
    for idx, (ax, var_idx) in enumerate(zip(axes[:2].flatten(), var_indices)):
        var_name = NOMBRES_DIN[var_idx]
        
        shap_map = ig_mean[:, :, var_idx]
        vlim = np.percentile(np.abs(shap_map), 95)
        
        im = ax.imshow(shap_map, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='auto')
        ax.set_title(var_name, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    
    # Mapa de incendios
    ax_fire = axes[1, 0]
    im = ax_fire.imshow(y_map, cmap='Reds', aspect='auto')
    ax_fire.set_title('Fire map', fontweight='bold')
    ax_fire.axis('off')
    plt.colorbar(im, ax=ax_fire, fraction=0.046, pad=0.04, shrink=0.8)
    
    # Mapa de NDVI de referencia
    ax_ndvi = axes[1, 1]
    im = ax_ndvi.imshow(seq["X_dyn"][-1, :, :, 0], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax_ndvi.set_title('NDVI (t)', fontweight='bold')
    ax_ndvi.axis('off')
    plt.colorbar(im, ax=ax_ndvi, fraction=0.046, pad=0.04, shrink=0.8)
    
    # Ocultar último panel
    axes[1, 2].axis('off')
    
    plt.suptitle('Spatial SHAP Attributions', fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig05_shap_spatial.png"), 
                bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_shap_temporal(shap_dyn, X_dyn_flat, labels, hls_list, out_dir):
    """Figura 6: Evolución temporal de importancia SHAP."""
    print("  [6/6] Evolución temporal...")
    
    dates = []
    for f in hls_list:
        dt = parse_fecha_robusta(f)
        dates.append(dt if dt else datetime(2021, 1, 1))
    
    # shap_dyn shape: [N, T, 1, 1, 12]
    # shap_dyn[:, :, 0, 0, :] -> [N, T, 12]
    # ✅ CORRECCIÓN: Usar axis=1 para promediar sobre el tiempo (T), obteniendo [N, 12]
    shap_abs = np.abs(shap_dyn[:, :, 0, 0, :]).mean(axis=1)
    
    months = [d.month for d in dates]
    unique_months = sorted(set(months))
    
    monthly_importance = []
    for month in unique_months:
        mask = np.array(months) == month
        if mask.sum() > 0:
            monthly_importance.append(shap_abs[mask].mean(axis=0))
        else:
            monthly_importance.append(np.zeros(12))
    
    monthly_importance = np.array(monthly_importance)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = range(len(unique_months))
    
    bottom = np.zeros(len(unique_months))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, 12))
    
    for i in range(12):
        ax.bar(x, monthly_importance[:, i], bottom=bottom, 
              label=NOMBRES_DIN[i], color=colors[i], alpha=0.8, edgecolor='none')
        bottom += monthly_importance[:, i]
    
    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m-1] for m in unique_months])
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Mean absolute SHAP', fontweight='bold')
    ax.set_title('Temporal evolution of feature importance', fontweight='bold', pad=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig06_shap_temporal.png"), 
                bbox_inches='tight', pad_inches=0.05)
    plt.close()


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    print("\n" + "="*60)
    print(f"  KASSANDRA-PYR — Análisis SHAP Integral (n_steps={STEPS})")
    print("="*60)
    
    # 1. Cargar modelo
    print("\n[1] Cargando modelo...")
    with tf.keras.utils.custom_object_scope(custom_dict):
        model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  ✓ Modelo cargado")
    
    # 2. Seleccionar mejor año
    print("\n[2] Seleccionando año con más incendios...")
    import glob
    all_hls = sorted(glob.glob(os.path.join(HLS_DIR, "*.tif")))
    years = sorted(set(int(re.search(r'(20\d{2})', os.path.basename(f)).group(1)) 
                      for f in all_hls if re.search(r'(20\d{2})', os.path.basename(f))))
    
    year_stats = {}
    for year in years:
        y_files = [os.path.join(Y_DIR, f) for f in os.listdir(Y_DIR) 
                   if f.endswith(".tif") and str(year) in f]
        total_px = sum(np.sum(rasterio.open(yf).read(1) > 0) for yf in y_files)
        year_stats[year] = total_px
        print(f"    {year}: {total_px:,} píxeles")
    
    best_year = max(year_stats, key=year_stats.get)
    print(f"  ✓ Año: {best_year} ({year_stats[best_year]:,} píxeles)")
    
    # 3. Cargar datos
    print(f"\n[3] Cargando datos {best_year}...")
    hls_files = sorted([f for f in all_hls if str(best_year) in os.path.basename(f)])
    y_files = sorted([os.path.join(Y_DIR, f) for f in os.listdir(Y_DIR) 
                      if str(best_year) in f and f.endswith(".tif")])
    
    sequences = build_sequences_test(model, hls_files, y_files)
    if not sequences:
        print("[ERROR] No hay secuencias válidas")
        return
    
    # 4. Filtrar secuencias con incendios
    print("\n[4] Filtrando secuencias con incendios...")
    fire_seqs = [(i, seq) for i, seq in enumerate(sequences) if seq["n_fire_pixels"] > 0]
    fire_seqs.sort(key=lambda x: x[1]["n_fire_pixels"], reverse=True)
    
    if not fire_seqs:
        print("[ERROR] No hay secuencias con incendios")
        return
    
    print(f"  ✓ {len(fire_seqs)} secuencias con incendios")
    print(f"  Top 3:")
    for idx, (seq_idx, seq) in enumerate(fire_seqs[:3]):
        print(f"    {idx+1}. Seq {seq_idx}: {seq['n_fire_pixels']:,} píxeles | {os.path.basename(seq['hls_file'])}")
    
    selected_seqs = [seq for _, seq in fire_seqs[:2]]
    hls_list = [seq["hls_file"] for seq in selected_seqs]
    
    # 5. Calcular SHAP
    print(f"\n[5] Calculando SHAP (n_steps={STEPS})...")
    result = compute_shap_attributions(model, selected_seqs, n_steps=STEPS, max_seqs=2)
    
    if result is None:
        print("[ERROR] Cálculo SHAP falló")
        return
    
    shap_dyn, labels, X_dyn_raw, hls_list_result = result
    
    np.save(os.path.join(OUT_DIR, "shap_values.npy"), shap_dyn)
    np.save(os.path.join(OUT_DIR, "shap_labels.npy"), labels)
    
    # 6. Generar visualizaciones
    print("\n[6] Generando figuras (estilo Nature)...")
    plot_shap_summary_bar(shap_dyn, labels, OUT_DIR)
    plot_shap_dependence(shap_dyn, X_dyn_raw, labels, OUT_DIR)
    plot_shap_beeswarm(shap_dyn, X_dyn_raw, labels, OUT_DIR)
    plot_shap_interactions(model, selected_seqs, OUT_DIR)
    plot_shap_spatial_maps(model, selected_seqs, OUT_DIR)
    plot_shap_temporal(shap_dyn, X_dyn_raw, labels, hls_list_result, OUT_DIR)
    
    print("\n" + "="*60)
    print(f"  ✓ ANÁLISIS COMPLETADO (n_steps={STEPS})")
    print(f"  Resultados: {OUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()