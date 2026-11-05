#!/usr/bin/env python3
# ============================================================
# shap_gradcam_v19r2.py — KASSANDRA-PYR (XAI Analysis)
# Versión v3: Integrated Gradients Robusto + Diagnóstico
# ============================================================

import os
import json
import re
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Intentar importar rasterio ──────────────────────────────
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("[WARN] rasterio no disponible")

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUN_DIR     = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
MODEL_PATH  = os.path.join(RUN_DIR, "best_model.keras")
OUT_DIR     = os.path.join(RUN_DIR, "xai_shap_gradcam_v3")
os.makedirs(OUT_DIR, exist_ok=True)

HLS_DIR      = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos"
CLIMATE_DIR  = "/lustre/home/kgalvan/cnn/x/Terraclimate_DL_Dataset"
STATIC_TIF   = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
Y_DIR        = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Mensual"

# Parámetros
SEQ_LEN = 3
N_CANALES_DIN = 12

# ── Nombres de canales ───────────────────────────────────────
NOMBRES_DIN = [
    "NDVI", "NBR", "NDWI",
    "mask_NDVI", "mask_NBR", "mask_NDWI",
    "TempMin", "TempMax", "Viento",
    "Precip", "PDSI", "Evapo"
]

# ── Triángulo ecológico del fuego ────────────────────────────
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

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def parse_fecha_robusta(filename):
    """
    Parsea fechas de nombres de archivo con múltiples formatos.
    Retorna datetime o None.
    """
    base = os.path.basename(filename)
    
    # Patrones a intentar (en orden de prioridad)
    patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),      # 2021-06-15
        (r'(\d{4})_(\d{2})_(\d{2})', '%Y_%m_%d'),      # 2021_06_15
        (r'(\d{8})', '%Y%m%d'),                         # 20210615
        (r'(\d{4})-(\d{2})(?!\d)', '%Y-%m'),            # 2021-06 (mensual)
        (r'(\d{4})_(\d{2})(?!\d)', '%Y_%m'),            # 2021_06
        (r'(\d{4})(\d{2})(?!\d)', '%Y%m'),              # 202106
    ]
    
    for pattern, fmt in patterns:
        m = re.search(pattern, base)
        if m:
            try:
                # Para formatos mensuales, usar día 15 como referencia
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
    nube   = raw_hw_c[:, :, 3] if raw_hw_c.shape[-1] > 3 else raw_hw_c[:, :, -1]
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
        # Usar modelo para obtener dimensión correcta de NALCMS
        try: 
            expected_sn_dim = model.inputs[2].shape[-1]
        except: 
            expected_sn_dim = 19 # Fallback seguro
            
        with rasterio.open(tif_path) as src:
            data = src.read(out_shape=(src.count, target_shape[0], target_shape[1]), resampling=Resampling.bilinear).astype(np.float32)
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
    """
    Versión con debug exhaustivo del matching temporal.
    """
    H, W = model.inputs[0].shape[2], model.inputs[0].shape[3]
    target_shape = (H, W)
    print(f"    Shape espacial: {H}×{W} | Ventana de búsqueda: ±{ventana_dias} días")
    
    # Cargar features estáticas
    static_cont, nalcms_oh = load_static_features(model, STATIC_TIF, target_shape)
    if static_cont is None:
        try: expected_sn_dim = model.inputs[2].shape[-1]
        except: expected_sn_dim = 19
        static_cont = np.zeros((*target_shape, 9), dtype=np.float32)
        nalcms_oh = np.zeros((*target_shape, expected_sn_dim), dtype=np.float32)

    # Pre-procesar archivos Y: crear diccionario fecha -> ruta
    print(f"    Indexando {len(y_files)} archivos Y...")
    y_index = {}
    for yf in y_files:
        ydt = parse_fecha_robusta(yf)
        if ydt:
            key = ydt.strftime("%Y-%m-%d")
            y_index[key] = yf
            # También indexar por mes
            y_index[ydt.strftime("%Y-%m")] = yf
    print(f"    Archivos Y indexados: {len(y_index)} entradas únicas")
    
    # Debug: mostrar fechas Y disponibles
    y_dates = sorted(set(parse_fecha_robusta(yf) for yf in y_files if parse_fecha_robusta(yf)))
    if y_dates:
        print(f"    Fechas Y disponibles: {y_dates[0].strftime('%Y-%m')} a {y_dates[-1].strftime('%Y-%m')}")
    
    sequences = []
    matched_count = 0
    unmatched_count = 0
    
    print(f"\n    Procesando {len(hls_files)} archivos HLS...")
    
    for i in range(len(hls_files) - SEQ_LEN):
        # Fecha de referencia: última imagen de la secuencia
        last_hls = hls_files[i + SEQ_LEN - 1]
        ref_date = parse_fecha_robusta(last_hls)
        
        if ref_date is None:
            unmatched_count += 1
            continue
        
        # Buscar archivo Y más cercano
        best_y_file = None
        best_delta = ventana_dias + 1
        
        for yf in y_files:
            ydt = parse_fecha_robusta(yf)
            if ydt:
                delta = abs((ydt - ref_date).days)
                if delta < best_delta:
                    best_delta = delta
                    best_y_file = yf
        
        # Si encontramos un match dentro de la ventana
        if best_y_file and best_delta <= ventana_dias:
            try:
                # Construir secuencia dinámica
                seq_imgs = []
                for j in range(SEQ_LEN):
                    fi = hls_files[i+j]
                    dt = parse_fecha_robusta(fi)
                    if dt is None:
                        raise ValueError(f"Fecha no parseable: {fi}")
                    
                    hls_frame = procesar_hls_con_mascara(read_and_fix(fi, target_shape))
                    tc_frame = _leer_tc(dt, target_shape)
                    full_frame = sanitize_array(np.concatenate([hls_frame, tc_frame], axis=-1))
                    seq_imgs.append(full_frame)
                
                # Cargar mapa de incendios
                y_map = np.zeros((H, W), dtype=np.float32)
                try:
                    ym = read_and_fix(best_y_file, target_shape, Resampling.nearest)[:, :, 0]
                    y_map = (ym > 0).astype(np.float32)
                except Exception as e:
                    print(f"      [WARN] Error cargando Y {best_y_file}: {e}")
                
                n_fire = int(np.sum(y_map > 0.5))
                
                sequences.append({
                    "X_dyn": np.stack(seq_imgs, axis=0),
                    "X_sc": static_cont,
                    "X_sn": nalcms_oh,
                    "y_map": y_map,
                    "hls_file": last_hls,
                    "y_file": best_y_file,
                    "ref_date": ref_date,
                    "y_date": parse_fecha_robusta(best_y_file),
                    "delta_days": best_delta,
                    "n_fire_pixels": n_fire
                })
                
                matched_count += 1
                
                # Debug: mostrar primeras coincidencias
                if matched_count <= 3:
                    print(f"      ✓ Match: HLS={ref_date.strftime('%Y-%m-%d')} "
                          f"<-> Y={parse_fecha_robusta(best_y_file).strftime('%Y-%m-%d')} "
                          f"(Δ={best_delta}d) | Incendios: {n_fire}")
                
            except Exception as e:
                print(f"      [ERROR] Procesando secuencia {i}: {e}")
                unmatched_count += 1
        else:
            unmatched_count += 1
    
    print(f"\n    Resultados del matching:")
    print(f"      - Secuencias válidas: {matched_count}")
    print(f"      - Sin match temporal: {unmatched_count}")
    
    # Debug adicional: contar incendios en secuencias
    if sequences:
        total_fire_px = sum(s["n_fire_pixels"] for s in sequences)
        seq_with_fires = sum(1 for s in sequences if s["n_fire_pixels"] > 0)
        print(f"      - Píxeles de incendio totales: {total_fire_px:,}")
        print(f"      - Secuencias CON incendios: {seq_with_fires}/{len(sequences)}")
        
        if seq_with_fires == 0:
            print(f"\n    [ALERTA] Ninguna secuencia tiene incendios.")
            print(f"    Posibles causas:")
            print(f"      1. Los archivos Y están en formato mensual y no coinciden con fechas HLS diarias")
            print(f"      2. La ventana de {ventana_dias} días es muy estrecha")
            print(f"      3. Los incendios en Y están en resolución diferente a {H}x{W}")
    
    return sequences

# Alias para compatibilidad
def build_sequences_test(model, hls_files, y_files):
    return build_sequences_test_debug(model, hls_files, y_files, ventana_dias=60)

# ============================================================
# 4. INTEGRATED GRADIENTS V3 (ROBUSTO + DIAGNÓSTICO)
# ============================================================

def compute_unet_attributions_v3(model, sequences, n_steps=30, max_seqs=2):
    """
    Integrated Gradients simplificado y robusto con diagnóstico completo.
    """
    print(f"\n[IG v3] Calculando atribuciones en {min(len(sequences), max_seqs)} secuencias...")
    print(f"    Steps: {n_steps} | Baseline: ceros | Strategy: simple & robust")
    
    shap_vals_list = []
    labels_list = []
    raw_dyn_list = []
    hls_list = []
    
    for seq_idx, seq in enumerate(sequences[:max_seqs]):
        print(f"\n    --- Procesando mapa [{seq_idx+1}/{min(len(sequences), max_seqs)}] ---")
        
        xd = np.expand_dims(seq["X_dyn"], axis=0)  # [1, T, H, W, 12]
        xs = np.expand_dims(seq["X_sc"], axis=0)
        xn = np.expand_dims(seq["X_sn"], axis=0)
        y_map = seq["y_map"]
        
        H, W = y_map.shape
        
        # --- DIAGNÓSTICO 1: Inputs ---
        print(f"    [DIAG] Input shape: {xd.shape}")
        print(f"    [DIAG] Input range (min/max/mean): {xd.min():.4f} / {xd.max():.4f} / {xd.mean():.4f}")
        
        # Baseline = ceros (más seguro)
        baseline_dyn = np.zeros_like(xd)
        
        # --- DIAGNÓSTICO 2: Predicción inicial ---
        print(f"    [DIAG] Ejecutando Forward pass completo...")
        try:
            preds_initial = model([xd, xs, xn], training=False)
            print(f"    [DIAG] Predicciones (antes de IG): min={preds_initial.numpy().min():.6f}, max={preds_initial.numpy().max():.6f}, mean={preds_initial.numpy().mean():.6f}")
            
            if preds_initial.numpy().max() < 0.01:
                print("    [WARN] El modelo predice valores casi nulos. Los gradientes pueden ser 0.")
        except Exception as e:
            print(f"    [ERROR] Forward pass falló: {e}")
            return None

        # Calcular gradientes acumulados
        print(f"      Calculando {n_steps} pasos de interpolación...")
        grads_accum = np.zeros_like(xd)
        # Empezar en 0.01 para evitar alpha=0
        alphas = np.linspace(0.01, 1.0, n_steps) 
        
        for step, alpha in enumerate(alphas):
            xd_interp = baseline_dyn + alpha * (xd - baseline_dyn)
            xd_interp = tf.cast(xd_interp, tf.float32)
            xs_tf = tf.cast(xs, tf.float32)
            xn_tf = tf.cast(xn, tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(xd_interp)
                preds = model([xd_interp, xs_tf, xn_tf], training=False)
                # Simple: suma de predicciones
                loss_val = tf.reduce_mean(preds)
            
            grad = tape.gradient(loss_val, xd_interp)
            if grad is not None:
                grads_accum += grad.numpy()
            
            if (step + 1) % 10 == 0:
                print(f"        Step {step+1}/{n_steps}...")
        
        # --- DIAGNÓSTICO 3: Gradientes ---
        print(f"    [DIAG] Gradientes acumulados (sum): min={grads_accum.min():.6f}, max={grads_accum.max():.6f}")
        print(f"    [DIAG] Gradientes (abs mean): {np.abs(grads_accum).mean():.6f}")

        # Calcular IG: (Input - Baseline) * Gradiente Promedio
        ig_full = (xd[0] - baseline_dyn[0]) * (grads_accum[0] / n_steps)
        
        # Normalizar por canal
        channel_max = np.abs(ig_full).max(axis=(0, 1, 2), keepdims=True) + 1e-8
        ig_normalized = ig_full / channel_max
        
        # --- DIAGNÓSTICO 4: Resultados IG ---
        print(f"    [DIAG] Valores SHAP (IG) finales: min={ig_normalized.min():.6f}, max={ig_normalized.max():.6f}")
        print(f"    [DIAG] Valores SHAP (abs mean): {np.abs(ig_normalized).mean():.6f}")
        
        # Extraer píxeles
        pos_idx = np.where(y_map.flatten() > 0.5)[0]
        neg_idx = np.where(y_map.flatten() < 0.5)[0]
        np.random.seed(42)
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        
        n_pos = min(30, len(pos_idx))
        n_neg = min(30, len(neg_idx))
        print(f"      Extrayendo {n_pos} positivos + {n_neg} negativos...")
        
        for idx in pos_idx[:n_pos]:
            r, c = divmod(idx, W)
            shap_vals_list.append(ig_normalized[:, r, c, :])
            labels_list.append(1)
            raw_dyn_list.append(seq["X_dyn"][:, r, c, :])
            hls_list.append(seq["hls_file"])
            
        for idx in neg_idx[:n_neg]:
            r, c = divmod(idx, W)
            shap_vals_list.append(ig_normalized[:, r, c, :])
            labels_list.append(0)
            raw_dyn_list.append(seq["X_dyn"][:, r, c, :])
            hls_list.append(seq["hls_file"])
    
    if not shap_vals_list:
        print("    [WARN] No se encontraron píxeles válidos")
        return None
    
    shap_vals = np.array(shap_vals_list)
    labels = np.array(labels_list)
    raw_dyn = np.array(raw_dyn_list)
    
    # Reshape para compatibilidad con plotting
    shap_dyn = shap_vals[:, :, np.newaxis, np.newaxis, :]
    raw_dyn_4d = raw_dyn[:, :, np.newaxis, np.newaxis, :]
    
    # Estadísticas
    print(f"\n    ✓ Completado: {len(labels)} píxeles")
    print(f"    Rango SHAP: [{shap_dyn.min():.6f}, {shap_dyn.max():.6f}]")
    print(f"    Media |SHAP|: {np.abs(shap_dyn).mean():.6f}")
    
    for i, name in enumerate(NOMBRES_DIN):
        mean_abs = np.abs(shap_dyn[..., i]).mean()
        print(f"      {name:12s}: {mean_abs:.6f}")
    
    return shap_dyn, labels, raw_dyn_4d, hls_list


# ============================================================
# 5. VISUALIZACIONES
# ============================================================

def plot_shap_summary_bar_v3(shap_dyn, labels, out_dir):
    print("    Generando shap_summary_bar_v3.png...")
    
    shap_2d = np.abs(shap_dyn).mean(axis=(1, 2, 3))
    pos_mask, neg_mask = labels == 1, labels == 0
    
    mean_pos = shap_2d[pos_mask].mean(axis=0) if pos_mask.any() else np.zeros(12)
    mean_neg = shap_2d[neg_mask].mean(axis=0) if neg_mask.any() else np.zeros(12)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    w, x = 0.35, np.arange(len(NOMBRES_DIN))
    
    # Colores por vértice
    colors = []
    for n in NOMBRES_DIN:
        vertice = asignar_vertice(n)
        colors.append(FIRE_TRIANGLE.get(vertice, {"color": "#666"})["color"])
    
    ax.barh(x + w/2, mean_pos, w, color=colors, alpha=0.85, label="Incendio (+)")
    ax.barh(x - w/2, mean_neg, w, color=colors, alpha=0.35, label="No incendio (−)", hatch='//')
    
    ax.set_yticks(x)
    ax.set_yticklabels(NOMBRES_DIN, fontsize=10)
    ax.set_xlabel("Importancia media |XAI|", fontsize=11)
    ax.set_title("KASSANDRA-PYR — Atribución XAI Global", fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=FIRE_TRIANGLE[v]["color"], label=v) 
                      for v in FIRE_TRIANGLE.keys()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Añadir valores
    for i, v in enumerate(mean_pos):
        if v > 1e-6:
            ax.text(v + max(mean_pos)*0.01, i, f'{v:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_bar_v3.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_dependence_v3(shap_dyn, X_dyn_flat, labels, out_dir, top_n=4):
    print("    Generando shap_dependence_v3.png...")
    
    shap_mean_t = shap_dyn[:, :, 0, 0, :].mean(axis=1)
    shap_abs_mean = np.abs(shap_mean_t).mean(axis=0)
    top_idx = np.argsort(shap_abs_mean)[::-1][:top_n]
    pos_mask = labels == 1
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    X_mean_t = X_dyn_flat[:, :, 0, 0, :].mean(axis=1)
    
    for idx, canal_idx in enumerate(top_idx):
        ax = axes.flatten()[idx]
        c_name = NOMBRES_DIN[canal_idx]
        
        x_vals = X_mean_t[:, canal_idx]
        shap_vals = shap_mean_t[:, canal_idx]
        
        if pos_mask.sum() > 0:
            ax.scatter(x_vals[pos_mask], shap_vals[pos_mask], 
                      alpha=0.7, s=25, c='#d32f2f', label="Incendio (+)", edgecolors='white', linewidth=0.5)
        
        neg_mask = labels == 0
        if neg_mask.sum() > 0:
            n_show = min(100, neg_mask.sum())
            neg_idx = np.random.choice(np.where(neg_mask)[0], n_show, replace=False)
            ax.scatter(x_vals[neg_idx], shap_vals[neg_idx], 
                      alpha=0.4, s=15, c='#1976d2', label="No incendio (−)", edgecolors='none')
        
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title(f"{c_name}\n|SHAP|={shap_abs_mean[canal_idx]:.4f}", fontweight='bold', fontsize=11)
        ax.set_xlabel("Valor", fontsize=9)
        ax.set_ylabel("SHAP", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
    
    plt.suptitle("Dependencia SHAP - Top 4 Variables", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_dependence_v3.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 6. PIPELINE PRINCIPAL
# ============================================================

def main_v3():
    print("\n" + "="*60)
    print("   KASSANDRA-PYR — XAI v3 (Búsqueda automática de año)")
    print("="*60)
    
    print("\n[1] Cargando modelo ConvLSTM-UNet...")
    with tf.keras.utils.custom_object_scope(custom_dict):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"    OK: {MODEL_PATH}")
        except Exception as e:
            print(f"    ERROR cargando modelo: {e}")
            return

    # ============================================
    # [NEW] BUSCAR AÑO CON MÁS INCENDIOS
    # ============================================
    print("\n[2] Escaneando años disponibles...")
    import glob
    
    # Obtener todos los archivos HLS disponibles
    all_hls = sorted(glob.glob(os.path.join(HLS_DIR, "*.tif")))
    
    # Extraer años únicos
    years_available = set()
    for f in all_hls:
        m = re.search(r'(20\d{2})', os.path.basename(f))
        if m:
            years_available.add(int(m.group(1)))
    
    years_available = sorted(list(years_available))
    print(f"    Años disponibles: {years_available}")
    
    # Contar incendios por año
    year_stats = {}
    for year in years_available:
        y_files = [os.path.join(Y_DIR, f) for f in os.listdir(Y_DIR) 
                   if f.endswith(".tif") and str(year) in f]
        
        total_fire_pixels = 0
        for yf in y_files:
            try:
                with rasterio.open(yf) as src:
                    data = src.read(1)
                    total_fire_pixels += np.sum(data > 0)
            except:
                pass
        
        year_stats[year] = total_fire_pixels
        print(f"    {year}: {total_fire_pixels:,} píxeles de incendio")
    
    # Seleccionar año con más incendios (excluyendo 2024 si hay mejores opciones)
    best_year = max(year_stats, key=year_stats.get)
    
    if year_stats[best_year] == 0:
        print("\n    [ERROR] No se encontraron incendios en ningún año!")
        return
    
    print(f"\n    ✓ Año seleccionado: {best_year} ({year_stats[best_year]:,} píxeles)")
    
    # ============================================
    # CARGAR DATOS DEL AÑO SELECCIONADO
    # ============================================
    print(f"\n[3] Cargando datos de {best_year}...")
    hls_files = sorted([f for f in all_hls if str(best_year) in os.path.basename(f)])
    y_files = sorted([os.path.join(Y_DIR, f) for f in os.listdir(Y_DIR) 
                      if str(best_year) in f and f.endswith(".tif")])
    
    print(f"    Archivos HLS: {len(hls_files)}")
    print(f"    Archivos Y: {len(y_files)}")
    
    # Construir secuencias
    sequences = build_sequences_test(model, hls_files, y_files)
    if not sequences:
        print("[ERROR] No se encontraron secuencias")
        return
    
    # Filtrar secuencias CON incendios
    print("\n[4] Filtrando secuencias con incendios...")
    sequences_with_fires = []
    for i, seq in enumerate(sequences):
        n_fire_pixels = np.sum(seq["y_map"] > 0.5)
        if n_fire_pixels > 0:
            sequences_with_fires.append({
                "seq": seq,
                "n_pixels": int(n_fire_pixels),
                "idx": i,
                "date": os.path.basename(seq["hls_file"])
            })
    
    # Ordenar por número de píxeles
    sequences_with_fires.sort(key=lambda x: x["n_pixels"], reverse=True)
    
    print(f"    Secuencias con incendios: {len(sequences_with_fires)}/{len(sequences)}")
    
    if len(sequences_with_fires) > 0:
        print("\n    Top 5 secuencias:")
        for item in sequences_with_fires[:5]:
            print(f"      - Seq {item['idx']}: {item['n_pixels']:,} píxeles | {item['date']}")
        
        # Seleccionar las 2 mejores (o más si hay pocas)
        n_select = min(2, len(sequences_with_fires))
        selected_seqs = [item["seq"] for item in sequences_with_fires[:n_select]]
    else:
        print("    [WARN] No se encontraron secuencias con incendios.")
        return
    
    # ============================================
    # Calcular SHAP
    # ============================================
    print("\n[5] Calculando Integrated Gradients...")
    result = compute_unet_attributions_v3(model, selected_seqs, n_steps=30, max_seqs=2)
    
    if result is None:
        print("[ERROR] Cálculo SHAP falló")
        return
    
    shap_dyn, labels_shap, X_dyn_raw, hls_list = result
    
    # Guardar
    np.save(os.path.join(OUT_DIR, f"shap_values_{best_year}.npy"), shap_dyn)
    np.save(os.path.join(OUT_DIR, f"shap_labels_{best_year}.npy"), labels_shap)
    
    # Plots
    print("\n[6] Generando visualizaciones...")
    plot_shap_summary_bar_v3(shap_dyn, labels_shap, OUT_DIR)
    plot_shap_dependence_v3(shap_dyn, X_dyn_raw, labels_shap, OUT_DIR)
    
    print("\n" + "="*60)
    print(f"   ✓ ANÁLISIS XAI COMPLETADO (Año: {best_year})")
    print(f"   Resultados en: {OUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main_v3()