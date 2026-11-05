#!/usr/bin/env python3
# ============================================================
# temporal_evolution_clean.py — KASSANDRA-PYR (XAI Analysis)
# Figura 7b: Evolución Temporal SIN Máscaras de Nubes
# ============================================================

import os
import re
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from datetime import datetime
import warnings
import glob as glob_module
warnings.filterwarnings("ignore")

# ── Intentar importar rasterio ─────────────────────────────
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

# ============================================================
# CONFIGURACIÓN
# ============================================================
STEPS = 30
RUN_DIR     = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
MODEL_PATH  = os.path.join(RUN_DIR, "best_model.keras")
# Guardamos en una carpeta distinta para no mezclar
OUT_DIR     = os.path.join(RUN_DIR, "shap_temporal_evolution_clean")
os.makedirs(OUT_DIR, exist_ok=True)

HLS_DIR      = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos"
CLIMATE_DIR  = "/lustre/home/kgalvan/cnn/x/Terraclimate_DL_Dataset"
STATIC_TIF   = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
Y_DIR        = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Mensual"

SEQ_LEN = 3
N_CANALES_DIN = 12

# Lista completa de canales
ALL_NOMBRES_DIN = ["NDVI", "NBR", "NDWI", "mask_NDVI", "mask_NBR", "mask_NDWI",
                   "TempMin", "TempMax", "Viento", "Precip", "PDSI", "Evapo"]

# 🔧 VARIABLES A EXCLUIR DEL GRÁFICO
EXCLUDE_VARS = ["mask_NDVI", "mask_NBR", "mask_NDWI"]

# Filtrar nombres para el gráfico
NOMBRES_DIN = [n for n in ALL_NOMBRES_DIN if n not in EXCLUDE_VARS]
# Índices originales para extraer los datos correctos
INDICES_TO_PLOT = [i for i, n in enumerate(ALL_NOMBRES_DIN) if n not in EXCLUDE_VARS]

# ── Estilo Nature ───────────────────────────────────────────
rcParams.update({
    'font.family': 'sans-serif', # Usamos sans-serif para evitar warnings en HPC
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans'],
    'font.size': 10, 'axes.linewidth': 0.8,
    'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

# ============================================================
# FUNCIONES AUXILIARES (Idénticas al script anterior)
# ============================================================

def parse_fecha_robusta(filename):
    base = os.path.basename(filename)
    patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
        (r'(\d{4})_(\d{2})_(\d{2})', '%Y_%m_%d'),
        (r'(\d{8})', '%Y%m%d'),
        (r'(\d{4})-(\d{2})(?!\d)', '%Y-%m'),
    ]
    for pattern, fmt in patterns:
        m = re.search(pattern, base)
        if m:
            try:
                if fmt in ['%Y-%m']: return datetime(int(m.group(1)), int(m.group(2)), 15)
                return datetime.strptime('-'.join(m.groups()), fmt.replace('_', '-'))
            except: continue
    return None

def gelu(x): return x * tf.sigmoid(1.702 * x)
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
    "gelu": gelu, "loss": focal_tversky_loss(),
    "match_shape": lambda x: tf.image.resize(x[0], [tf.shape(x[1])[1], tf.shape(x[1])[2]], method="bilinear"),
    "resize_gate": lambda x: tf.image.resize(x[0], [tf.shape(x[1])[1], tf.shape(x[1])[2]], method="bilinear"),
    "resize_static": lambda x: tf.image.resize(x[0], [tf.shape(x[1])[1], tf.shape(x[1])[2]], method="bilinear")
}

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
    nube = raw_hw_c[:, :, 3] if raw_hw_c.shape[-1] > 3 else raw_hw_c[:, :, -1]
    valido = (np.isfinite(nube) & (nube > 0.5)).astype(np.float32)
    nublado = ~(np.isfinite(nube) & (nube > 0.5))
    res = []
    for ci in [0, 1, 2]:
        canal = np.clip(np.where(np.isfinite(raw_hw_c[:, :, ci]), raw_hw_c[:, :, ci], 0.0), 0.0, 1.0)
        canal[nublado] = 0.0
        res.append(canal)
    for _ in [0, 1, 2]: res.append(valido.copy())
    return np.stack(res, axis=-1).astype(np.float32)

def normalizar_tc(arr_hwc):
    TC_RANGES = {0: (-30.0, 50.0), 1: (-20.0, 60.0), 2: (0.0, 15.0), 3: (0.0, 500.0), 4: (-5.0, 5.0), 5: (0.0, 200.0)}
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
        for f in os.listdir(CLIMATE_DIR):
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

def build_sequences_temporal(model, hls_files, y_files):
    H, W = model.inputs[0].shape[2], model.inputs[0].shape[3]
    target_shape = (H, W)
    
    try: expected_sn_dim = model.inputs[2].shape[-1]
    except: expected_sn_dim = 19
    
    static_cont = np.zeros((*target_shape, 9), dtype=np.float32)
    nalcms_oh = np.zeros((*target_shape, expected_sn_dim), dtype=np.float32)
    
    sequences = []
    for i in range(len(hls_files) - SEQ_LEN):
        last_hls = hls_files[i + SEQ_LEN - 1]
        ref_date = parse_fecha_robusta(last_hls)
        if not ref_date: continue
        
        best_y_file = None
        best_delta = 61
        for yf in y_files:
            ydt = parse_fecha_robusta(yf)
            if ydt:
                delta = abs((ydt - ref_date).days)
                if delta < best_delta:
                    best_delta = delta
                    best_y_file = yf
        
        if best_y_file and best_delta <= 60:
            try:
                seq_imgs = []
                for j in range(SEQ_LEN):
                    fi = hls_files[i+j]
                    hls_frame = procesar_hls_con_mascara(read_and_fix(fi, target_shape))
                    tc_frame = _leer_tc(parse_fecha_robusta(fi), target_shape)
                    seq_imgs.append(sanitize_array(np.concatenate([hls_frame, tc_frame], axis=-1)))
                
                y_map = np.zeros((H, W), dtype=np.float32)
                try:
                    ym = read_and_fix(best_y_file, target_shape, Resampling.nearest)[:, :, 0]
                    y_map = (ym > 0).astype(np.float32)
                except: pass
                
                sequences.append({
                    "X_dyn": np.stack(seq_imgs, axis=0), "y_map": y_map,
                    "hls_file": last_hls, "date": ref_date,
                    "n_fire_pixels": int(np.sum(y_map > 0.5))
                })
            except: pass
    return sequences

def compute_shap_attributions_temporal(model, sequences, static_cont, nalcms_oh, n_steps=30):
    print(f"  Calculando SHAP para {len(sequences)} secuencias...")
    monthly_shap = {}
    
    for seq in sequences:
        xd = np.expand_dims(seq["X_dyn"], axis=0)
        xs = np.expand_dims(static_cont, axis=0)
        xn = np.expand_dims(nalcms_oh, axis=0)
        y_map = seq["y_map"]
        H, W = y_map.shape
        
        baseline_dyn = np.zeros_like(xd)
        grads_accum = np.zeros_like(xd)
        alphas = np.linspace(0.01, 1.0, n_steps)
        
        xs_tf = tf.cast(xs, tf.float32)
        xn_tf = tf.cast(xn, tf.float32)
        
        for alpha in alphas:
            xd_interp = baseline_dyn + alpha * (xd - baseline_dyn)
            xd_interp = tf.cast(xd_interp, tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(xd_interp)
                preds = model([xd_interp, xs_tf, xn_tf], training=False)
                loss_val = tf.reduce_mean(preds)
            
            grad = tape.gradient(loss_val, xd_interp)
            if grad is not None:
                grads_accum += grad.numpy()
        
        ig_full = (xd[0] - baseline_dyn[0]) * (grads_accum[0] / n_steps)
        channel_max = np.abs(ig_full).max(axis=(0, 1, 2), keepdims=True) + 1e-8
        ig_normalized = ig_full / channel_max
        
        # Promedio absoluto sobre el mapa [12]
        mean_abs_shap = np.abs(ig_normalized).mean(axis=(0, 1, 2))
        
        month = seq["date"].month
        if month not in monthly_shap:
            monthly_shap[month] = []
        monthly_shap[month].append(mean_abs_shap)
        
    return monthly_shap

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    print("\n" + "="*60)
    print("  GENERANDO FIGURA 7b: EVOLUCIÓN TEMPORAL (SIN MÁSCARAS)")
    print("="*60)
    
    # 1. Cargar modelo
    print("\n[1] Cargando modelo...")
    with tf.keras.utils.custom_object_scope(custom_dict):
        model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Cargar datos 2021
    print("\n[2] Cargando datos 2021...")
    all_hls = sorted(glob_module.glob(os.path.join(HLS_DIR, "*.tif")))
    hls_2021 = [f for f in all_hls if "2021" in os.path.basename(f)]
    y_2021 = sorted([os.path.join(Y_DIR, f) for f in os.listdir(Y_DIR) if "2021" in f and f.endswith(".tif")])
    
    if not hls_2021:
        print("[ERROR] No hay datos 2021")
        return

    # 3. Cargar estáticas
    H, W = model.inputs[0].shape[2], model.inputs[0].shape[3]
    try: expected_sn_dim = model.inputs[2].shape[-1]
    except: expected_sn_dim = 19
    
    print("  Cargando features estáticos...")
    static_cont = np.zeros((H, W, 9), dtype=np.float32)
    nalcms_oh = np.zeros((H, W, expected_sn_dim), dtype=np.float32)
    try:
        if RASTERIO_OK and os.path.exists(STATIC_TIF):
            with rasterio.open(STATIC_TIF) as src:
                data = src.read(out_shape=(src.count, H, W), resampling=Resampling.bilinear).astype(np.float32)
            all_bands = np.transpose(data, (1, 2, 0))
            static_cont = sanitize_array(all_bands[:, :, :9], clip_val=1.0)
            if all_bands.shape[-1] > 9:
                nalcms_raw = np.clip(all_bands[:, :, 9].astype(np.int32), 0, expected_sn_dim - 1)
                for c in range(expected_sn_dim):
                    nalcms_oh[:, :, c] = (nalcms_raw == c).astype(np.float32)
    except Exception as e:
        print(f"  [WARN] Error cargando estáticos: {e}")

    # 4. Construir secuencias Y FILTRAR POR MES
    print("\n[3] Procesando secuencias para cubrir todos los meses...")
    sequences = build_sequences_temporal(model, hls_2021, y_2021)
    
    selected_seqs_by_month = {}
    for seq in sequences:
        m = seq["date"].month
        if m not in selected_seqs_by_month:
            selected_seqs_by_month[m] = None
        if seq["n_fire_pixels"] > 0:
            selected_seqs_by_month[m] = seq
        elif selected_seqs_by_month[m] is None:
            selected_seqs_by_month[m] = seq
            
    final_seqs = []
    months_covered = []
    for m in sorted(selected_seqs_by_month.keys()):
        if selected_seqs_by_month[m] is not None:
            final_seqs.append(selected_seqs_by_month[m])
            months_covered.append(m)
            
    print(f"  Meses cubiertos: {months_covered}")
    
    # 5. Calcular SHAP
    print("\n[4] Calculando SHAP...")
    monthly_results = compute_shap_attributions_temporal(model, final_seqs, static_cont, nalcms_oh, n_steps=STEPS)
    
    # 6. Plot (FILTRADO)
    print("\n[5] Generando Figura 7b (Sin Máscaras)...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    x = range(len(months_covered))
    
    bottom = np.zeros(len(months_covered))
    # Colores: Usamos un mapa que cubra bien las variables restantes
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(NOMBRES_DIN)))
    
    for idx, original_idx in enumerate(INDICES_TO_PLOT):
        var_name = ALL_NOMBRES_DIN[original_idx]
        heights = []
        for m in months_covered:
            if m in monthly_results:
                # Usamos el índice original para sacar el dato correcto del array de 12
                val = np.mean(monthly_results[m], axis=0)[original_idx]
                heights.append(val)
            else:
                heights.append(0)
        
        ax.bar(x, heights, bottom=bottom, label=var_name, color=colors[idx], alpha=0.85, edgecolor='none')
        bottom += heights
        
    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m-1] for m in months_covered], rotation=45)
    ax.set_xlabel('Month (2021)', fontweight='bold')
    ax.set_ylabel('Mean absolute SHAP', fontweight='bold')
    ax.set_title('Temporal evolution of feature importance (Excluding Cloud Masks)', fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig07b_temporal_evolution_no_masks.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Figura 7b guardada en: {OUT_DIR}/fig07b_temporal_evolution_no_masks.png")

if __name__ == "__main__":
    main()