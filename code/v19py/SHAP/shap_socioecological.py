#!/usr/bin/env python3
# ============================================================
# shap_socioecological.py — KASSANDRA-PYR
# Análisis SHAP Integral: Dinámico + Estático + Socioecológico
# CORREGIDO: Persistent GradientTape
# ============================================================

import os
import re
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
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
OUT_DIR     = os.path.join(RUN_DIR, "shap_socioecological")
os.makedirs(OUT_DIR, exist_ok=True)

HLS_DIR      = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos"
CLIMATE_DIR  = "/lustre/home/kgalvan/cnn/x/Terraclimate_DL_Dataset"
STATIC_TIF   = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
Y_DIR        = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Mensual"

SEQ_LEN = 3

# ── Nombres de variables por Rama ──────────────────────────
# Rama 1: Dinámicas (12)
NOMBRES_DIN = ["NDVI", "NBR", "NDWI", "mask_NDVI", "mask_NBR", "mask_NDWI",
               "TempMin", "TempMax", "Viento", "Precip", "PDSI", "Evapo"]

# Rama 2: Estáticas (9)
NOMBRES_STATIC = ["Elevación", "Pendiente", "Aspect_Cos", "Aspect_Sin", 
                  "TPI", "Rugosidad", "Densidad_Pob", "Frac_Ag", "Cobertura_Arb"]

# Colores por Rama
COLORS_DIN = "#2E7D32"   # Verde
COLORS_STAT = "#1565C0"  # Azul
COLORS_NALC = "#B71C1C"  # Rojo

# ── Estilo Nature ───────────────────────────────────────────
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10, 'axes.linewidth': 0.8,
    'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

# ============================================================
# FUNCIONES AUXILIARES
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
                if fmt in ['%Y-%m']: return __import__('datetime').datetime(int(m.group(1)), int(m.group(2)), 15)
                return __import__('datetime').datetime.strptime('-'.join(m.groups()), fmt.replace('_', '-'))
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
    except: expected_sn_dim = 15
    
    print("  Cargando features estáticos...")
    static_cont = np.zeros((*target_shape, 9), dtype=np.float32)
    nalcms_oh = np.zeros((*target_shape, expected_sn_dim), dtype=np.float32)
    
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
                    "X_dyn": np.stack(seq_imgs, axis=0), 
                    "X_sc": static_cont, 
                    "X_sn": nalcms_oh,
                    "y_map": y_map, "date": ref_date
                })
            except: pass
    return sequences

# ============================================================
# CÁLCULO SHAP PARA LAS 3 RAMAS (CORREGIDO)
# ============================================================

def compute_shap_all_branches(model, sequences, n_steps=30, max_seqs=1):
    print(f"\n  Calculando SHAP para 3 ramas (Steps={n_steps})...")
    
    # Inicializar acumuladores basados en la forma de la primera secuencia
    sample_seq = sequences[0]
    n_din = sample_seq["X_dyn"].shape[-1]       # 12
    n_stat = sample_seq["X_sc"].shape[-1]       # 9
    n_nalc = sample_seq["X_sn"].shape[-1]       # 15 o 19
    
    shap_din_acc = np.zeros(n_din)
    shap_stat_acc = np.zeros(n_stat)
    shap_nalc_acc = np.zeros(n_nalc)
    count = 0
    
    for seq_idx, seq in enumerate(sequences[:max_seqs]):
        print(f"    Procesando secuencia {seq_idx+1}/{max_seqs}...")
        
        xd = np.expand_dims(seq["X_dyn"], axis=0)
        xs = np.expand_dims(seq["X_sc"], axis=0)
        xn = np.expand_dims(seq["X_sn"], axis=0)
        
        baseline_dyn = np.zeros_like(xd)
        baseline_sc = np.zeros_like(xs)
        baseline_sn = np.zeros_like(xn)
        
        grads_accum_d = np.zeros_like(xd)
        grads_accum_s = np.zeros_like(xs)
        grads_accum_n = np.zeros_like(xn)
        
        alphas = np.linspace(0.01, 1.0, n_steps)
        
        for alpha in alphas:
            # Interpolación para las 3 ramas
            xd_interp = baseline_dyn + alpha * (xd - baseline_dyn)
            xs_interp = baseline_sc + alpha * (xs - baseline_sc)
            xn_interp = baseline_sn + alpha * (xn - baseline_sn)
            
            xd_interp = tf.cast(xd_interp, tf.float32)
            xs_interp = tf.cast(xs_interp, tf.float32)
            xn_interp = tf.cast(xn_interp, tf.float32)
            
            # ✅ CLAVE: persistent=True para múltiples gradientes
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(xd_interp)
                tape.watch(xs_interp)
                tape.watch(xn_interp)
                
                preds = model([xd_interp, xs_interp, xn_interp], training=False)
                loss_val = tf.reduce_mean(preds)
            
            # Calcular los 3 gradientes
            g_d = tape.gradient(loss_val, xd_interp)
            g_s = tape.gradient(loss_val, xs_interp)
            g_n = tape.gradient(loss_val, xn_interp)
            
            if g_d is not None: grads_accum_d += g_d.numpy()
            if g_s is not None: grads_accum_s += g_s.numpy()
            if g_n is not None: grads_accum_n += g_n.numpy()
            
            del tape # Liberar memoria de la cinta persistente
        
        # Normalización y promediado
        # Rama 1: [T, H, W, C] -> Promedio sobre T,H,W
        ch_max_d = np.abs(grads_accum_d[0]).max(axis=(0,1,2), keepdims=True) + 1e-8
        ig_d_norm = (grads_accum_d[0] / ch_max_d)
        val_d = np.abs(ig_d_norm).mean(axis=(0,1,2)) # [12]
        
        # Rama 2: [H, W, C] -> Promedio sobre H,W
        ch_max_s = np.abs(grads_accum_s[0]).max(axis=(0,1), keepdims=True) + 1e-8
        ig_s_norm = (grads_accum_s[0] / ch_max_s)
        val_s = np.abs(ig_s_norm).mean(axis=(0,1)) # [9]
        
        # Rama 3: [H, W, C] -> Promedio sobre H,W
        ch_max_n = np.abs(grads_accum_n[0]).max(axis=(0,1), keepdims=True) + 1e-8
        ig_n_norm = (grads_accum_n[0] / ch_max_n)
        val_n = np.abs(ig_n_norm).mean(axis=(0,1)) # [N]
        
        shap_din_acc += val_d
        shap_stat_acc += val_s
        shap_nalc_acc += val_n
        count += 1
        
    # Promediar sobre secuencias
    shap_din_acc /= count
    shap_stat_acc /= count
    shap_nalc_acc /= count
    
    return shap_din_acc, shap_stat_acc, shap_nalc_acc

# ============================================================
# VISUALIZACIÓN
# ============================================================

def plot_socioecological_importance(shap_din, shap_stat, shap_nalc, nalc_names, out_dir):
    print("\n  Generando Figura 8: Importancia Socioecológica...")
    
    # Combinar todo
    all_names = NOMBRES_DIN + NOMBRES_STATIC + nalc_names
    all_values = np.concatenate([shap_din, shap_stat, shap_nalc])
    
    # Colores
    colors = ([COLORS_DIN]*len(shap_din)) + ([COLORS_STAT]*len(shap_stat)) + ([COLORS_NALC]*len(shap_nalc))
    
    # Ordenar
    sorted_idx = np.argsort(all_values)[::-1]
    sorted_names = [all_names[i] for i in sorted_idx]
    sorted_values = all_values[sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(8, 12))
    
    y = np.arange(len(sorted_names))
    ax.barh(y, sorted_values, color=sorted_colors, edgecolor='none')
    
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Mean Normalized SHAP Value', fontweight='bold')
    ax.set_title('Global Feature Importance (Dynamic + Static + Socio-ecological)', fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS_DIN, label='Rama 1: Dinámica (Clima/Vegetación)'),
        Patch(facecolor=COLORS_STAT, label='Rama 2: Estática (Topografía/Humano)'),
        Patch(facecolor=COLORS_NALC, label='Rama 3: NALCMS (Uso de Suelo)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig08_socioecological_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico de Torta
    sum_din = np.sum(shap_din)
    sum_stat = np.sum(shap_stat)
    sum_nalc = np.sum(shap_nalc)
    total = sum_din + sum_stat + sum_nalc
    
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    sizes = [sum_din/total, sum_stat/total, sum_nalc/total]
    labels = ['Dinámica', 'Estática', 'NALCMS']
    colors_pie = [COLORS_DIN, COLORS_STAT, COLORS_NALC]
    
    ax2.pie(sizes, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    ax2.set_title('Contribución Relativa por Rama', fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(out_dir, "fig08b_branch_contribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("  ANÁLISIS SHAP SOCIOECOLÓGICO (3 RAMAS)")
    print("="*60)
    
    print("\n[1] Cargando modelo...")
    with tf.keras.utils.custom_object_scope(custom_dict):
        model = tf.keras.models.load_model(MODEL_PATH)
    
    print("\n[2] Cargando datos (Marzo 2021)...")
    import glob as glob_module
    all_hls = sorted(glob_module.glob(os.path.join(HLS_DIR, "*.tif")))
    hls_2021 = [f for f in all_hls if "2021" in os.path.basename(f)]
    y_2021 = sorted([os.path.join(Y_DIR, f) for f in os.listdir(Y_DIR) if "2021" in f and f.endswith(".tif")])
    
    sequences = build_sequences_temporal(model, hls_2021, y_2021)
    
    if not sequences:
        print("[ERROR] No se encontraron secuencias.")
        return

    # Seleccionar una secuencia representativa
    seq_with_fire = [s for s in sequences if np.sum(s["y_map"] > 0) > 0]
    if seq_with_fire:
        selected_seqs = [seq_with_fire[0]]
    else:
        selected_seqs = [sequences[0]]
        
    print(f"  Analizando {len(selected_seqs)} secuencia(s)...")
    
    # Definir nombres dinámicos para NALCMS
    N_NALCMS = model.inputs[2].shape[-1]
    NOMBRES_NALCMS = [f"NALCMS_C{i}" for i in range(N_NALCMS)]
    
    print("\n[3] Ejecutando SHAP Multirrama...")
    shap_din, shap_stat, shap_nalc = compute_shap_all_branches(model, selected_seqs, n_steps=STEPS, max_seqs=1)
    
    print("\n[4] Resultados Top 3 por Rama:")
    print(f"  Dinámica: {sorted(zip(NOMBRES_DIN, shap_din), key=lambda x: x[1], reverse=True)[:3]}")
    print(f"  Estática: {sorted(zip(NOMBRES_STATIC, shap_stat), key=lambda x: x[1], reverse=True)[:3]}")
    print(f"  NALCMS:   {sorted(zip(NOMBRES_NALCMS, shap_nalc), key=lambda x: x[1], reverse=True)[:3]}")
    
    print("\n[5] Generando Figuras...")
    plot_socioecological_importance(shap_din, shap_stat, shap_nalc, NOMBRES_NALCMS, OUT_DIR)
    
    print(f"\n  ✓ Guardado en: {OUT_DIR}")

if __name__ == "__main__":
    main()