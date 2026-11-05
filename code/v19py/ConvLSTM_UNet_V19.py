#!/usr/bin/env python3
# ============================================================
# ConvLSTM_UNet_V19_E{EPOCHS}
# Split       : Leave-Year-Out (Train 2020+2021+2022 | Val 2023 | Test 2024)
# Arquitectura: V18 — Static Feature Branch sobre V16
#
# Cambios sobre V16:
#   [NEW-1] Rama estática separada (Static Feature Injection)
#           → 9 canales continuos normalizados + 4 NALCMS embedding
#           → Total estático: 13 canales [B, H, W, 13]
#           → Inyección en cada nivel del decoder antes de attention gates
#   [NEW-2] NALCMS embedding: one-hot → Conv2D(4, 1×1) aprendido
#   [NEW-3] Bandas usadas del TIF estático (descartando MODIS banda 7):
#           0=elevation, 1=slope, 2=aspect_cos, 3=aspect_sin,
#           4=tpi, 5=roughness, 6=pop_density,
#           8=ag_fraction_500m, 9=NALCMS(cat), 10=canopy_cover
#   [NEW-4] Rama dinámica idéntica a V16 (12 canales HLS + Terraclimate)
#   [NEW-5] Train incluye 2022 (3 años vs 2 años en V16)
#
# Fixes heredados de V16:
#   [FIX-1] ConvLSTM en serie
#   [FIX-2] Split temporal válido
#   [FIX-3] bias_init = Constant(-2.0)
#   [FIX-4] Monitor = val_roc_auc + patience=15
#   [FIX-5] LR = 5e-5 + ReduceLROnPlateau
#   [FIX-6] Platt scaling (LogisticRegression)
#   [FIX-7] batch_train = 8
# ============================================================
import os, re, json, math, warnings, multiprocessing
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import (
    ConvLSTM2D, Conv2D, Conv2DTranspose,
    BatchNormalization, SpatialDropout2D,
    Concatenate, MaxPooling2D, Activation,
    Input, Lambda
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, brier_score_loss,
    confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")

# ============================================================
# ARGUMENTOS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',   type=int, required=True)
parser.add_argument('--use-gpu',  action='store_true')
args = parser.parse_args()
EPOCHS = args.epochs

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

if args.use_gpu:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        DEVICE = "GPU"
    else:
        DEVICE = "CPU"
else:
    tf.config.set_visible_devices([], 'GPU')
    DEVICE = "CPU"

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
tf.config.threading.set_intra_op_parallelism_threads(N_CPUS)
tf.config.threading.set_inter_op_parallelism_threads(max(2, N_CPUS // 4))
print(f"TF {tf.__version__} | {DEVICE} | CPUs={N_CPUS}")
INICIO_TIEMPO = datetime.now()

# ============================================================
# RUTAS
# ============================================================
BASE_PATH    = "/lustre/home/kgalvan/cnn"
HLS_DIR      = f"{BASE_PATH}/x/GEE_Export_HLS_Mosaicos"
CLIMATE_DIR  = f"{BASE_PATH}/x/Terraclimate_DL_Dataset"
Y_DIR        = f"{BASE_PATH}/y_binaria/GEE_Incendios_Mensual"
STATIC_TIF   = f"{BASE_PATH}/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
STATIC_META  = f"{BASE_PATH}/x/Topographic_DL_Dataset/feature_metadata_V20.csv"
STATIC_CAT   = f"{BASE_PATH}/x/Topographic_DL_Dataset/categorical_lookup_V20.csv"
RESULTS_DIR  = f"{BASE_PATH}/resultados"
LOG_DIR      = f"{BASE_PATH}/logs_json"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

# ============================================================
# CONFIGURACIÓN DE BANDAS ESTÁTICAS
# ============================================================
# Bandas continuas del TIF V20 (índices base-0, descartando banda 7=MODIS)
STATIC_CONT_IDX  = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # 9 canales
STATIC_CAT_IDX   = [7]                              # NALCMS_30m
STATIC_CONT_NAMES = [
    "elevation", "slope", "aspect_cos", "aspect_sin",
    "tpi", "roughness", "pop_density", "ag_fraction_500m", "canopy_cover"
]
STATIC_CAT_NAMES = ["land_cover_NALCMS_30m"]

# Rangos de normalización para canales continuos
# (ajustar según estadísticas reales de la ANP Jalisco)
STATIC_NORM_RANGES = {
    "elevation":       (0.0,    3500.0),   # m NASADEM
    "slope":           (0.0,    60.0),     # grados
    "aspect_cos":      (-1.0,   1.0),      # ya normalizado
    "aspect_sin":      (-1.0,   1.0),      # ya normalizado
    "tpi":             (-50.0,  50.0),     # m, relativo
    "roughness":       (0.0,    100.0),    # m
    "pop_density":     (0.0,    50.0),     # personas/100m2 (log1p)
    "ag_fraction_500m":(0.0,   1.0),      # ya en [0,1]
    "canopy_cover":    (0.0,    100.0),    # %
}
LOG1P_STATIC = {"pop_density", "roughness"}  # aplicar log1p antes de normalizar

# NALCMS: número de clases (confirmar con categorical_lookup_V20.csv)
# Clases NALCMS estándar: 19. Si el lookup tiene menos, ajustar.
try:
    df_cat = pd.read_csv(STATIC_CAT)
    N_NALCMS_CLASSES = int(df_cat['value'].max()) + 1
    print(f"NALCMS clases detectadas: {N_NALCMS_CLASSES}")
except Exception:
    N_NALCMS_CLASSES = 19
    print(f"NALCMS: usando {N_NALCMS_CLASSES} clases por defecto")

NALCMS_EMBED_DIM = 4   # dimensión del embedding aprendido
N_STATIC_TOTAL   = len(STATIC_CONT_IDX) + NALCMS_EMBED_DIM  # 9 + 4 = 13

# ============================================================
# CANALES DINÁMICOS (igual que V16)
# ============================================================
CANALES_HLS_IDX  = [0, 1, 2]
CANALES_TC_IDX   = [0, 1, 2, 3, 4, 5]
N_CANALES_DIN    = 12
NOMBRES_DIN      = ["NDVI","NBR","NDWI",
                    "mask_NDVI","mask_NBR","mask_NDWI",
                    "TempMin","TempMax","Viento","Precip","PDSI","Evapo"]
TC_RANGES = {
    0: (-30.0, 50.0), 1: (-20.0, 60.0), 2: (0.0, 15.0),
    3: (0.0, 500.0),  4: (-5.0,  5.0),  5: (0.0, 200.0),
}
LOG1P_TC = {3, 5}

NOTEBOOK_NAME = f"ConvLSTM_UNet_V19_E{EPOCHS}"
ARCH          = "V19"
SEQ_LEN       = 3
BATCH_TRAIN   = 16
BATCH_VAL     = 4   # aumentado vs V16 para señal más estable
BATCH_TEST    = 1
POS_RATIO     = 0.22

POS_COUNT  = 164_034
NEG_COUNT  = 24_209_154
TOTAL_PX_TRAIN = POS_COUNT + NEG_COUNT

print(f"Experimento: {NOTEBOOK_NAME}")
print(f"Canales dinámicos: {N_CANALES_DIN} | Estáticos: {N_STATIC_TOTAL}")
print(f"Split V18: Train 2020+2021+2022 | Val 2023 | Test 2024")

# ============================================================
# SHAPE DE REFERENCIA
# ============================================================
y_ref = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))[0]
with rasterio.open(os.path.join(Y_DIR, y_ref)) as src:
    TARGET_SHAPE_HW = (src.height, src.width)
print(f"Target shape: {TARGET_SHAPE_HW}")

# ============================================================
# CARGA Y PREPROCESAMIENTO DEL TIF ESTÁTICO
# ============================================================
def cargar_static_features():
    """
    Carga el TIF estático V20 y retorna:
        static_cont: np.float32 [H, W, 9]   — canales continuos normalizados
        static_nalcms_onehot: np.float32 [H, W, N_NALCMS_CLASSES] — one-hot
    """
    print(f"\nCargando features estáticos: {STATIC_TIF}")
    with rasterio.open(STATIC_TIF) as src:
        all_bands = src.read(
            out_shape=(src.count, TARGET_SHAPE_HW[0], TARGET_SHAPE_HW[1]),
            resampling=Resampling.bilinear
        ).astype(np.float32)  # [11, H, W]
    all_bands = np.transpose(all_bands, (1, 2, 0))  # [H, W, 11]

    # --- Canales continuos ---
    cont = all_bands[:, :, STATIC_CONT_IDX]  # [H, W, 9]
    for ci, name in enumerate(STATIC_CONT_NAMES):
        band = cont[:, :, ci]
        band = np.where(np.isfinite(band), band, 0.0)
        if name in LOG1P_STATIC:
            band = np.log1p(np.maximum(band, 0.0))
            vmin = np.log1p(max(STATIC_NORM_RANGES[name][0], 0))
            vmax = np.log1p(STATIC_NORM_RANGES[name][1])
        else:
            vmin, vmax = STATIC_NORM_RANGES[name]
        cont[:, :, ci] = np.clip((band - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    static_cont = cont.astype(np.float32)

    # --- NALCMS one-hot ---
    nalcms_raw = all_bands[:, :, STATIC_CAT_IDX[0]]  # [H, W]
    nalcms_raw = np.where(np.isfinite(nalcms_raw), nalcms_raw, 0).astype(np.int32)
    nalcms_raw = np.clip(nalcms_raw, 0, N_NALCMS_CLASSES - 1)
    # One-hot: [H, W, N_NALCMS_CLASSES]
    H, W = TARGET_SHAPE_HW
    nalcms_oh = np.zeros((H, W, N_NALCMS_CLASSES), dtype=np.float32)
    for c in range(N_NALCMS_CLASSES):
        nalcms_oh[:, :, c] = (nalcms_raw == c).astype(np.float32)

    print(f"  static_cont     : {static_cont.shape}  "
          f"min={static_cont.min():.3f} max={static_cont.max():.3f}")
    print(f"  nalcms_onehot   : {nalcms_oh.shape}  "
          f"clases presentes={np.unique(nalcms_raw)}")
    return static_cont, nalcms_oh

STATIC_CONT, STATIC_NALCMS_OH = cargar_static_features()
# Expandir dim batch para broadcasting
STATIC_CONT_B    = STATIC_CONT[np.newaxis]       # [1, H, W, 9]
STATIC_NALCMS_B  = STATIC_NALCMS_OH[np.newaxis]  # [1, H, W, N_NALCMS]

# ============================================================
# FUNCIONES AUXILIARES (heredadas de V16)
# ============================================================
def parse_fecha(filename):
    base = os.path.basename(filename)
    for pat in [r"(\d{4})-(\d{2})-(\d{2})", r"(\d{4})_(\d{2})_(\d{2})"]:
        m = re.search(pat, base)
        if m:
            try: return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError: pass
    for pat in [r"(\d{4})-(\d{2})(?!-\d{2})", r"(\d{4})(\d{2})(?!\d)"]:
        m = re.search(pat, base)
        if m:
            try: return datetime(int(m.group(1)), int(m.group(2)), 1)
            except ValueError: pass
    return None

def read_and_fix(path, out_shape_hw=None, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        if out_shape_hw and (src.height, src.width) != out_shape_hw:
            data = src.read(out_shape=(src.count, out_shape_hw[0], out_shape_hw[1]),
                            resampling=resampling)
        else:
            data = src.read()
        return np.transpose(data, (1, 2, 0)).astype(np.float32)

def sanitize_array(arr, clip_val=10.0):
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -clip_val, clip_val).astype(np.float32)

def procesar_hls_con_mascara(raw_hw_c):
    nube   = raw_hw_c[:, :, 3]
    valido = (np.isfinite(nube) & (nube > 0.5)).astype(np.float32)
    nublado = ~(np.isfinite(nube) & (nube > 0.5))
    res = []
    for ci in CANALES_HLS_IDX:
        canal = np.where(np.isfinite(raw_hw_c[:, :, ci]), raw_hw_c[:, :, ci], 0.0)
        canal = np.clip(canal, 0.0, 1.0)
        canal[nublado] = 0.0
        res.append(canal)
    for _ in CANALES_HLS_IDX:
        res.append(valido.copy())
    return np.stack(res, axis=-1).astype(np.float32)

def normalizar_tc(arr_hwc):
    out = np.zeros_like(arr_hwc)
    for i, (vmin, vmax) in TC_RANGES.items():
        canal = arr_hwc[:, :, i].copy()
        if i in LOG1P_TC:
            canal = np.log1p(np.maximum(canal, 0.0))
            vmin, vmax = 0.0, np.log1p(vmax)
        out[:, :, i] = np.clip((canal - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return out.astype(np.float32)

# ============================================================
# REGISTRO
# ============================================================
def is_run_complete(run_dir):
    return all(os.path.exists(os.path.join(run_dir, f))
               for f in ["best_model.keras", "metricas.txt",
                         "y_true_test.npy", "y_pred_test.npy"])

def setup_run_dir(base=RESULTS_DIR, nombre=NOTEBOOK_NAME):
    run_root = os.path.join(base, nombre)
    os.makedirs(run_root, exist_ok=True)
    existing = sorted([d for d in os.listdir(run_root)
                       if d.startswith(f"{nombre}_r")
                       and d.split("_r")[-1].isdigit()])
    for d in existing:
        if not is_run_complete(os.path.join(run_root, d)):
            return os.path.join(run_root, d), int(d.split("_r")[-1])
    n = int(existing[-1].split("_r")[-1]) + 1 if existing else 1
    run_dir = os.path.join(run_root, f"{nombre}_r{n}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, n

RUN_DIR, RUN_NUM = setup_run_dir()
LOG_PATH = os.path.join(RUN_DIR, "registro.json")
print(f"\nRUN_DIR: {RUN_DIR}")

def actualizar_registro(path, **kwargs):
    reg = json.load(open(path)) if os.path.exists(path) else {}
    reg.update(kwargs)
    json.dump(reg, open(path, "w"), indent=2)

actualizar_registro(LOG_PATH,
    notebook=NOTEBOOK_NAME, version=ARCH,
    n_canales_din=N_CANALES_DIN, n_canales_static=N_STATIC_TOTAL,
    nalcms_clases=N_NALCMS_CLASSES, nalcms_embed_dim=NALCMS_EMBED_DIM,
    split="Train:2020+2021+2022 | Val:2023 | Test:2024",
    inicio=datetime.now().isoformat(), estado="iniciado")

# ============================================================
# GENERADOR V18
# ============================================================
class FireRiskGeneratorV18(tf.keras.utils.Sequence):
    """
    Generador V18: retorna (X_din, X_static_cont, X_static_nalcms), Y
    donde:
        X_din         : [B, T, H, W, 12]  — secuencia dinámica
        X_static_cont : [B, H, W, 9]      — features estáticos continuos
        X_static_nalcms: [B, H, W, N_NALCMS] — one-hot NALCMS
        Y             : [B, H, W, 1]
    """
    def __init__(self, hls_dir, climate_dir, y_dir,
                 batch_size=8, seq_len=3, shuffle=True,
                 pos_ratio=POS_RATIO, modo="train"):
        self.hls_dir      = hls_dir
        self.climate_dir  = climate_dir
        self.y_dir        = y_dir
        self.batch_size   = batch_size
        self.seq_len      = seq_len
        self.shuffle      = shuffle
        self.pos_ratio    = pos_ratio if modo == "train" else 1.0
        self.modo         = modo
        self.target_shape = TARGET_SHAPE_HW
        self.hls_files    = sorted(f for f in os.listdir(hls_dir) if f.endswith(".tif"))
        self.y_map        = self._map_dir(y_dir)
        self.clim_map     = self._map_dir(climate_dir)
        self.meses_activos = self._calcular_meses_activos()
        self.valid = []
        for i in range(len(self.hls_files) - seq_len):
            last_dt = parse_fecha(self.hls_files[i + seq_len - 1])
            next_dt = parse_fecha(self.hls_files[i + seq_len])
            if last_dt is None or next_dt is None: continue
            y_next = self._buscar_y_cercano(next_dt)
            if y_next:
                self.valid.append({
                    "seq_start": i, "y_file": y_next,
                    "y_activo": self.meses_activos.get(y_next, False),
                    "fecha": last_dt, "anio": last_dt.year, "mes": last_dt.month,
                })
        self.indexes = np.arange(len(self.valid))
        self.pos_idx = np.array([i for i, v in enumerate(self.valid) if v["y_activo"]])
        self.neg_idx = np.array([i for i, v in enumerate(self.valid) if not v["y_activo"]])
        print(f"[{modo}] seqs={len(self.valid)} pos={len(self.pos_idx)} neg={len(self.neg_idx)}")

    def _map_dir(self, dir_path):
        mapping = {}
        if not os.path.exists(dir_path): return mapping
        for f in sorted(os.listdir(dir_path)):
            if f.endswith(".tif"):
                dt = parse_fecha(f)
                if dt:
                    mapping[dt.strftime("%Y-%m-%d")] = f
                    mapping[dt.strftime("%Y-%m")]    = f
        return mapping

    def _calcular_meses_activos(self):
        activos = {}
        for yf in sorted(f for f in os.listdir(self.y_dir) if f.endswith(".tif")):
            try:
                with rasterio.open(os.path.join(self.y_dir, yf)) as src:
                    data = src.read(1, out_shape=(1, 64, 64),
                                    resampling=Resampling.nearest)
                    activos[yf] = bool(np.any(data > 0))
            except Exception:
                activos[yf] = False
        return activos

    def _buscar_y_cercano(self, dt, ventana_dias=3):
        for key in [dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")]:
            if key in self.y_map: return self.y_map[key]
        mejor, mejor_delta = None, ventana_dias + 1
        for yf in sorted(f for f in os.listdir(self.y_dir) if f.endswith(".tif")):
            ydt = parse_fecha(yf)
            if ydt:
                delta = abs((ydt - dt).days)
                if delta < mejor_delta:
                    mejor, mejor_delta = yf, delta
        return mejor if mejor_delta <= ventana_dias else None

    def _leer_hls(self, fname):
        return procesar_hls_con_mascara(
            read_and_fix(os.path.join(self.hls_dir, fname), self.target_shape))

    def _leer_tc(self, dt):
        for key in [dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")]:
            if key in self.clim_map:
                raw = read_and_fix(
                    os.path.join(self.climate_dir, self.clim_map[key]),
                    self.target_shape, Resampling.bilinear)
                tc = np.stack([raw[:, :, i] for i in CANALES_TC_IDX], axis=-1)
                return normalizar_tc(sanitize_array(tc))
        return np.zeros((*self.target_shape, len(CANALES_TC_IDX)), np.float32)

    def _leer_y(self, y_file):
        y_raw = read_and_fix(os.path.join(self.y_dir, y_file), self.target_shape)
        return (y_raw[:, :, 0] > 0).astype(np.float32)[..., None]

    def __len__(self):
        return math.ceil(len(self.valid) / self.batch_size)

    def _augment(self, X_seq, Y_map, static_c, static_n):
        flip_h = np.random.rand() < 0.5
        flip_v = np.random.rand() < 0.5
        rot_k  = np.random.choice([0, 1, 2, 3])
        shift  = np.random.uniform(-3, 3, 2).astype(int)

        def aug2d(arr):
            if flip_h: arr = np.fliplr(arr)
            if flip_v: arr = np.flipud(arr)
            arr = np.rot90(arr, k=rot_k, axes=(0, 1))
            if np.any(shift != 0):
                arr = np.roll(arr, (shift[0], shift[1]), axis=(0, 1))
            return arr

        X_aug = np.stack([aug2d(X_seq[t]) for t in range(self.seq_len)], axis=0)
        Y_aug = aug2d(Y_map)
        sc_aug = aug2d(static_c)
        sn_aug = aug2d(static_n)
        return X_aug, Y_aug, sc_aug, sn_aug

    def __getitem__(self, idx):
        n_pos = max(1, int(self.batch_size * self.pos_ratio))
        n_neg = self.batch_size - n_pos
        pos_sample = (np.random.choice(self.pos_idx,
                       size=min(n_pos, len(self.pos_idx)),
                       replace=len(self.pos_idx) < n_pos)
                      if len(self.pos_idx) > 0 else np.array([], dtype=int))
        neg_sample = np.random.choice(self.neg_idx,
                                      size=min(n_neg, len(self.neg_idx)),
                                      replace=True)
        batch_idx = np.concatenate([pos_sample, neg_sample])
        np.random.shuffle(batch_idx)

        X_din_batch, X_sc_batch, X_sn_batch, Y_batch = [], [], [], []
        expected_shape = (self.seq_len, *self.target_shape, N_CANALES_DIN)

        for i in batch_idx:
            item = self.valid[i]
            y    = self._leer_y(item["y_file"])
            try:
                seq = []
                for fi in self.hls_files[item["seq_start"]:item["seq_start"]+self.seq_len]:
                    dt    = parse_fecha(fi)
                    frame = sanitize_array(
                        np.concatenate([self._leer_hls(fi), self._leer_tc(dt)], axis=-1))
                    seq.append(frame)
                X_seq  = np.stack(seq, axis=0)
                sc     = STATIC_CONT.copy()    # [H, W, 9]
                sn     = STATIC_NALCMS_OH.copy()  # [H, W, N_NALCMS]
                if self.modo == "train":
                    X_seq, y, sc, sn = self._augment(X_seq, y, sc, sn)
                if X_seq.shape != expected_shape: continue
                X_din_batch.append(X_seq)
                X_sc_batch.append(sc)
                X_sn_batch.append(sn)
                Y_batch.append(y)
            except Exception as e:
                print(f"  ⚠ Error item {i}: {e}")
                continue

        if not X_din_batch:
            H, W = self.target_shape
            return (
                {
                    "input_dynamic":       np.zeros((1, *expected_shape), np.float32),
                    "input_static_cont":   np.zeros((1, H, W, len(STATIC_CONT_IDX)), np.float32),
                    "input_static_nalcms": np.zeros((1, H, W, N_NALCMS_CLASSES), np.float32),
                },
                np.zeros((1, H, W, 1), np.float32)
            )

        return (
            {
                "input_dynamic":       np.stack(X_din_batch).astype(np.float32),
                "input_static_cont":   np.stack(X_sc_batch).astype(np.float32),
                "input_static_nalcms": np.stack(X_sn_batch).astype(np.float32),
            },
            np.stack(Y_batch).astype(np.float32)
        )

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

# ============================================================
# BUILD GENERADORES
# ============================================================
print("\nConstruyendo generador base...")
_full = FireRiskGeneratorV18(HLS_DIR, CLIMATE_DIR, Y_DIR,
                             batch_size=1, shuffle=False,
                             pos_ratio=1.0, modo="full")

class _LightGen(FireRiskGeneratorV18):
    def __init__(self): pass

def make_gen(seqs, batch_size, shuffle, pos_ratio, modo):
    g = _LightGen()
    for attr in ["hls_dir","climate_dir","y_dir","hls_files",
                 "y_map","clim_map","meses_activos","target_shape","seq_len"]:
        setattr(g, attr, getattr(_full, attr))
    g.batch_size = batch_size
    g.shuffle    = shuffle
    g.pos_ratio  = pos_ratio if modo == "train" else 1.0
    g.modo       = modo
    g.valid      = seqs
    g.indexes    = np.arange(len(seqs))
    g.pos_idx    = np.array([i for i,v in enumerate(seqs) if v["y_activo"]])
    g.neg_idx    = np.array([i for i,v in enumerate(seqs) if not v["y_activo"]])
    return g

# Split V18: +2022 en train
ANIOS_TRAIN = {2020, 2021, 2022, 2023}
train_seqs = sorted([v for v in _full.valid if v["anio"] in ANIOS_TRAIN],
                    key=lambda v: v["fecha"])
val_seqs   = sorted([v for v in _full.valid if v["anio"] == 2023],
                    key=lambda v: v["fecha"])
test_seqs  = sorted([v for v in _full.valid if v["anio"] == 2024],
                    key=lambda v: v["fecha"])

print("\nSplit V18:")
for nm, seqs in [("Train 2020+2021+2022", train_seqs),
                 ("Val   2023",           val_seqs),
                 ("Test  2024",           test_seqs)]:
    n_act = sum(1 for v in seqs if v["y_activo"])
    print(f"  {nm}: {len(seqs)} seqs | activos={n_act}")

train_gen = make_gen(train_seqs, BATCH_TRAIN, True,  POS_RATIO,  "train")
val_gen   = make_gen(val_seqs,   BATCH_VAL,   False, 1.0,        "val")
test_gen  = make_gen(test_seqs,  BATCH_TEST,  False, 1.0,        "test")

# Verificar shapes
X_dict_s, Y_s = train_gen[0]
X_d  = X_dict_s["input_dynamic"]
X_sc = X_dict_s["input_static_cont"]
X_sn = X_dict_s["input_static_nalcms"]
print(f"\nBatch verificación:")
print(f"  X_din        : {X_d.shape}")
print(f"  X_static_cont: {X_sc.shape}")
print(f"  X_static_nalcms:{X_sn.shape}")
print(f"  Y            : {Y_s.shape}")

# ============================================================
# LOSS (heredada de V16)
# ============================================================
def gelu(x):
    return x * tf.sigmoid(1.702 * x)

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

# ============================================================
# ARQUITECTURA V18
# ============================================================
def conv_block(x, filters, name_prefix):
    x = Conv2D(filters, (3,3), padding="same", name=f"{name_prefix}_c1")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = Activation(gelu, name=f"{name_prefix}_g1")(x)
    x = Conv2D(filters, (3,3), padding="same", name=f"{name_prefix}_c2")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = Activation(gelu, name=f"{name_prefix}_g2")(x)
    return x

def attention_gate(skip, gate, filters, name_prefix):
    phi_skip = Conv2D(filters, (1,1), padding="same", use_bias=False,
                      name=f"{name_prefix}_as")(skip)
    phi_gate = Conv2D(filters, (1,1), padding="same", use_bias=False,
                      name=f"{name_prefix}_ag")(gate)
    def resize_gate(inputs):
        g, s = inputs
        return tf.image.resize(g, [tf.shape(s)[1], tf.shape(s)[2]], method="bilinear")
    phi_gate_up = Lambda(resize_gate, name=f"{name_prefix}_ar")([phi_gate, skip])
    psi   = Activation(gelu, name=f"{name_prefix}_apsi")(phi_skip + phi_gate_up)
    alpha = Conv2D(1, (1,1), padding="same", activation="sigmoid",
                   name=f"{name_prefix}_aalpha")(psi)
    return skip * alpha

def up_block_attention(x, skip, filters, name_prefix):
    x = Conv2DTranspose(filters, (2,2), strides=(2,2), padding="same",
                        name=f"{name_prefix}_up")(x)
    def match_shape(inputs):
        up, sk = inputs
        return tf.image.resize(up, [tf.shape(sk)[1], tf.shape(sk)[2]], method="bilinear")
    x    = Lambda(match_shape, name=f"{name_prefix}_ms")([x, skip])
    skip = attention_gate(skip, x, filters//2, name_prefix)
    x    = Concatenate(name=f"{name_prefix}_cat")([x, skip])
    x    = conv_block(x, filters, name_prefix)
    return x

def build_static_branch(inp_cont, inp_nalcms, name="static"):
    """
    Rama estática:
      inp_cont   : [B, H, W, 9]    — continuos normalizados
      inp_nalcms : [B, H, W, N_NALCMS] — one-hot NALCMS

    Retorna features a distintas escalas para inyectar en el decoder.
    """
    # NALCMS embedding: one-hot → 4 canales aprendidos
    nalcms_emb = Conv2D(NALCMS_EMBED_DIM, (1,1), padding="same",
                        activation="relu",
                        name=f"{name}_nalcms_emb")(inp_nalcms)

    # Concatenar continuos + embedding
    x = Concatenate(name=f"{name}_concat")([inp_cont, nalcms_emb])
    # [B, H, W, 9+4=13]

    # Procesar con Conv2D para extraer features espaciales
    x = Conv2D(32, (3,3), padding="same", name=f"{name}_c1")(x)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Activation(gelu, name=f"{name}_g1")(x)
    feat_full = x  # escala completa [B, H, W, 32]

    x = MaxPooling2D((2,2), name=f"{name}_pool1")(x)
    x = Conv2D(64, (3,3), padding="same", name=f"{name}_c2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = Activation(gelu, name=f"{name}_g2")(x)
    feat_half = x  # media escala [B, H/2, W/2, 64]

    return feat_full, feat_half

def build_model_v18(input_shape_din, n_cont_static, n_nalcms):
    bias_init = tf.keras.initializers.Constant(-2.0)

    # --- Inputs ---
    inp_din    = Input(shape=input_shape_din,       name="input_dynamic")
    inp_sc     = Input(shape=(*TARGET_SHAPE_HW, n_cont_static), name="input_static_cont")
    inp_sn     = Input(shape=(*TARGET_SHAPE_HW, n_nalcms),      name="input_static_nalcms")

    # --- Rama dinámica: ConvLSTM encoder (igual V16) ---
    x = ConvLSTM2D(16, (3,3), padding="same", return_sequences=True,
                   activation="tanh", recurrent_dropout=0.35,
                   name="convlstm_1")(inp_din)
    x = BatchNormalization(name="bn_lstm1")(x)
    x = ConvLSTM2D(32, (3,3), padding="same", return_sequences=False,
                   activation="tanh", recurrent_dropout=0.35,
                   name="convlstm_2")(x)
    x = BatchNormalization(name="bn_lstm2")(x)
    x = SpatialDropout2D(0.45, name="drop_lstm")(x)

    e1     = conv_block(x,  32, "enc1");  skip_1 = e1
    p1     = MaxPooling2D((2,2), name="pool1")(e1)
    e2     = conv_block(p1, 64, "enc2");  skip_2 = e2
    p2     = MaxPooling2D((2,2), name="pool2")(e2)
    bn_out = conv_block(p2, 128, "bottleneck")
    bn_out = SpatialDropout2D(0.45, name="drop_bn")(bn_out)

    # --- Rama estática ---
    sf_full, sf_half = build_static_branch(inp_sc, inp_sn, name="static")

    # --- Decoder con inyección estática ---
    # Nivel 2 (media escala): concatenar features estáticos de media escala
    d2 = up_block_attention(bn_out, skip_2, 64, "dec2")
    # Inyectar estático en media escala
    def resize_static(inputs):
        sf, ref = inputs
        return tf.image.resize(sf, [tf.shape(ref)[1], tf.shape(ref)[2]], method="bilinear")
    sf_half_resized = Lambda(resize_static, name="static_resize_half")([sf_half, d2])
    d2 = Concatenate(name="dec2_static_inject")([d2, sf_half_resized])
    d2 = Conv2D(64, (1,1), padding="same", activation="relu",
                name="dec2_static_reduce")(d2)

    # Nivel 1 (escala completa): concatenar features estáticos completos
    d1 = up_block_attention(d2, skip_1, 32, "dec1")
    sf_full_resized = Lambda(resize_static, name="static_resize_full")([sf_full, d1])
    d1 = Concatenate(name="dec1_static_inject")([d1, sf_full_resized])
    d1 = Conv2D(32, (1,1), padding="same", activation="relu",
                name="dec1_static_reduce")(d1)

    # --- Output ---
    out = Conv2D(1, (1,1), padding="same", activation="sigmoid",
                 bias_initializer=bias_init, name="fire_risk_map")(d1)

    return tf.keras.Model(
        inputs=[inp_din, inp_sc, inp_sn],
        outputs=out,
        name="ConvLSTM_UNet_V19")

# ============================================================
# CONSTRUCCIÓN Y COMPILACIÓN
# ============================================================
import gc
gc.collect()
tf.keras.backend.clear_session()

print("\nConstruyendo modelo V18...")
input_shape_din = X_d.shape[1:]  # (T, H, W, 12)
model = build_model_v18(input_shape_din,
                        n_cont_static=len(STATIC_CONT_IDX),
                        n_nalcms=N_NALCMS_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=focal_tversky_loss(alpha=0.80, beta=0.20, gamma=2.0),
    metrics=[
        tf.keras.metrics.AUC(curve="PR",  name="pr_auc"),
        tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ])

model.summary()
n_params = model.count_params()
print(f"\nParámetros: {n_params:,}")
actualizar_registro(LOG_PATH, estado="modelo_listo",
                    n_parametros=n_params,
                    input_shape_din=list(input_shape_din),
                    n_static_cont=len(STATIC_CONT_IDX),
                    n_nalcms_classes=N_NALCMS_CLASSES)

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_roc_auc", mode="max",
        patience=15, min_delta=0.0002,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(RUN_DIR, "best_model.keras"),
        monitor="val_roc_auc", mode="max",
        save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_roc_auc", mode="max",
        factor=0.5, patience=5,
        min_lr=1e-6, verbose=1),
    tf.keras.callbacks.CSVLogger(
        os.path.join(RUN_DIR, "training_log.csv"), append=True),
]

print("=" * 60)
print(f"Iniciando: {NOTEBOOK_NAME}")
print(f"Train:{len(train_gen)} batches | Val:{len(val_gen)} batches")
print(f"Input dinámico: {input_shape_din}")
print(f"Input estático: [{len(STATIC_CONT_IDX)} cont + {N_NALCMS_CLASSES} NALCMS → {NALCMS_EMBED_DIM} emb]")
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

history = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS, callbacks=callbacks, verbose=1)

df_hist = pd.DataFrame(history.history)
df_hist.to_csv(os.path.join(RUN_DIR, "history.csv"), index=False)
mejor_epoch = int(df_hist["val_roc_auc"].argmax())
print(f"\nMejor epoch: {mejor_epoch+1} | val_roc_auc={df_hist['val_roc_auc'].max():.4f}")

# ============================================================
# EVALUACIÓN
# ============================================================
def evaluar_gen_v18(modelo, gen, nombre):
    yt, yp = [], []
    print(f"  Evaluando {nombre} ({len(gen)} batches)...")
    for X_dict, Y_b in gen:
        preds = modelo.predict(X_dict, verbose=0)
        yt.append(Y_b.reshape(-1))
        yp.append(preds.reshape(-1))
    y_true = np.concatenate(yt).astype(np.float32)
    y_pred = np.concatenate(yp).astype(np.float32)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], np.clip(y_pred[mask], 1e-7, 1-1e-7)

print("\n" + "="*55 + "\nEVALUACIÓN V18\n" + "="*55)
y_true_val,  y_pred_val  = evaluar_gen_v18(model, val_gen,  "Val")
y_true_test, y_pred_test = evaluar_gen_v18(model, test_gen, "Test")

for arr, fn in [(y_true_val,  "y_true_val"),
                (y_pred_val,  "y_pred_val"),
                (y_true_test, "y_true_test"),
                (y_pred_test, "y_pred_test")]:
    np.save(os.path.join(RUN_DIR, f"{fn}.npy"), arr)

# ============================================================
# CALIBRACIÓN (Platt scaling — igual que V16 corregido)
# ============================================================
print("\n🔧 Calibración Platt scaling...")
_cal_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
_cal_lr.fit(y_pred_val.reshape(-1, 1), y_true_val.astype(int))
y_pred_val_cal  = _cal_lr.predict_proba(y_pred_val.reshape(-1,  1))[:, 1]
y_pred_test_cal = _cal_lr.predict_proba(y_pred_test.reshape(-1, 1))[:, 1]

class _PlattWrapper:
    def __init__(self, lr): self._lr = lr
    def predict_proba(self, X): return self._lr.predict_proba(X)
calibrated_clf = _PlattWrapper(_cal_lr)

np.save(os.path.join(RUN_DIR, "y_pred_val_calibrated.npy"),  y_pred_val_cal)
np.save(os.path.join(RUN_DIR, "y_pred_test_calibrated.npy"), y_pred_test_cal)

# ============================================================
# THRESHOLDS Y MÉTRICAS
# ============================================================
prec_v, rec_v, thr_v = precision_recall_curve(y_true_val, y_pred_val_cal)
f1_v   = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-8)
THR_F1 = float(thr_v[np.argmax(f1_v[:-1])])

fpr_t, tpr_t, thr_roc = roc_curve(y_true_test, y_pred_test_cal)
prec_t, rec_t, thr_pr = precision_recall_curve(y_true_test, y_pred_test_cal)
THR_RECALL_60 = float(thr_pr[np.argmax((rec_t[:-1] >= 0.60).astype(int))]) \
    if np.any(rec_t >= 0.60) else THR_F1
THR_FPR_005 = float(thr_roc[np.argmax((fpr_t <= 0.05).astype(int))]) \
    if np.any(fpr_t <= 0.05) else THR_F1

THRESHOLD = THR_F1
np.save(os.path.join(RUN_DIR, "best_threshold.npy"), THRESHOLD)
thresholds = {
    "thr_f1_optimo_cal":   THR_F1,
    "thr_recall_0.60_cal": THR_RECALL_60,
    "thr_fpr_0.05_cal":    THR_FPR_005,
    "threshold_activo":    THRESHOLD,
}
print(f"  Threshold activo: {THRESHOLD:.5f}")

def calc_metricas(y_true, y_pred, thr):
    y_bin = (y_pred >= thr).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0,1]).ravel()
    rec  = tp / (tp+fn+1e-8); prec = tp / (tp+fp+1e-8)
    f1   = 2*prec*rec / (prec+rec+1e-8)
    fpr  = fp / (fp+tn+1e-8)
    return {
        "AUC_ROC": float(roc_auc_score(y_true, y_pred)),
        "AUC_PR":  float(average_precision_score(y_true, y_pred)),
        "Recall": float(rec), "Precision": float(prec), "F1": float(f1),
        "Brier":  float(brier_score_loss(y_true, y_pred)),
        "FPR": float(fpr), "FNR": float(fn/(fn+tp+1e-8)),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Threshold": float(thr),
    }

m_val  = calc_metricas(y_true_val,  y_pred_val_cal,  THRESHOLD)
m_test = calc_metricas(y_true_test, y_pred_test_cal, THRESHOLD)

print(f"\n  {'Métrica':<12} {'Val':>10} {'Test':>10}")
for k in ["AUC_ROC","AUC_PR","Recall","Precision","F1","FPR"]:
    print(f"  {k:<12} {m_val.get(k,0):>10.4f} {m_test.get(k,0):>10.4f}")

# Guardar métricas
metricas_json = {
    "experimento": NOTEBOOK_NAME, "version": ARCH,
    "fecha": datetime.now().isoformat(),
    "split": {"train":"2020+2021+2022","val":"2023","test":"2024"},
    "config": {
        "seq_len": SEQ_LEN, "n_canales_din": N_CANALES_DIN,
        "canales_din": NOMBRES_DIN,
        "n_static_cont": len(STATIC_CONT_IDX),
        "static_cont_names": STATIC_CONT_NAMES,
        "nalcms_clases": N_NALCMS_CLASSES,
        "nalcms_embed_dim": NALCMS_EMBED_DIM,
        "n_static_total": N_STATIC_TOTAL,
        "batch_train": BATCH_TRAIN, "batch_val": BATCH_VAL,
        "pos_ratio": 0.22, "learning_rate": 2e-5,
        "optimizer": "Adam",
        "loss": "FocalTversky(alpha=0.80,beta=0.20,gamma=2.0)",
        "bias_init": -2.0, "monitor": "val_roc_auc",
        "patience": 15, "min_delta": 0.0002,
        "mejor_epoch": mejor_epoch+1, "epochs_config": EPOCHS,
        "n_parametros": n_params,
    },
    "thresholds": thresholds,
    "calibracion": {
        "metodo": "Platt scaling (LogisticRegression)",
        "ajustado_con": "val 2023",
        "threshold_activo": float(THRESHOLD),
    },
    "metricas": {
        "val_calibrado": m_val,
        "test_calibrado": m_test,
    },
    "confusion_test_calibrado": {
        "TP": m_test["TP"], "FP": m_test["FP"],
        "FN": m_test["FN"], "TN": m_test["TN"],
        "total_positivos": m_test["TP"]+m_test["FN"],
        "prevalencia_pct": round(
            (m_test["TP"]+m_test["FN"]) /
            (m_test["TP"]+m_test["FP"]+m_test["FN"]+m_test["TN"])*100, 4),
    },
}
with open(os.path.join(RUN_DIR, "metricas_completas.json"), "w") as f:
    json.dump(metricas_json, f, indent=2, ensure_ascii=False)

with open(os.path.join(RUN_DIR, "metricas.txt"), "w") as f:
    f.write(f"Experimento : {NOTEBOOK_NAME}\n")
    f.write(f"Fecha       : {datetime.now().isoformat()}\n")
    f.write(f"Threshold   : {THRESHOLD:.5f}\n")
    f.write(f"Canales din : {N_CANALES_DIN} | Estáticos: {N_STATIC_TOTAL}\n\n")
    f.write("=== VAL CALIBRADO ===\n")
    for k, v in m_val.items(): f.write(f"{k}: {v}\n")
    f.write("\n=== TEST CALIBRADO ===\n")
    for k, v in m_test.items(): f.write(f"{k}: {v}\n")

actualizar_registro(LOG_PATH, estado="completo",
                    metricas_val=m_val, metricas_test=m_test,
                    thresholds=thresholds,
                    fin=datetime.now().isoformat())

# ============================================================
# PLOTS COMPLETOS (historial + curvas + confusion + calibracion)
# ============================================================
print("\n🎨 Generando plots completos...")
import matplotlib.gridspec as gridspec_plots

CMAP_RIESGO = mcolors.LinearSegmentedColormap.from_list(
    "fire_risk",
    ["#0d0221","#1a237e","#00695c","#f9a825","#e65100","#b71c1c"], N=256)

archivos_generados = []

# --- 1. Historial ---
try:
    print("  [1/5] historial.png...")
    df_h = pd.read_csv(os.path.join(RUN_DIR, "history.csv"))
    mej  = int(df_h["val_roc_auc"].argmax())
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Historial — {NOTEBOOK_NAME}\nMejor epoch: {mej+1}  val_roc_auc={df_h['val_roc_auc'].max():.4f}",
                 fontsize=12, fontweight="bold")
    for (tk, vk, tit), ax in zip(
        [("loss","val_loss","Loss"),("pr_auc","val_pr_auc","PR-AUC"),
         ("recall","val_recall","Recall"),("roc_auc","val_roc_auc","ROC-AUC")],
        axes.flat):
        if tk in df_h: ax.plot(df_h[tk], label="Train", color="steelblue")
        if vk in df_h: ax.plot(df_h[vk], label="Val",   color="tomato")
        ax.axvline(mej, color="green", ls="--", alpha=0.7, label="Mejor")
        ax.set_title(tit); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "historial.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    archivos_generados.append("historial.png")
    print("    ✅ OK")
except Exception as e:
    print(f"    ❌ {e}")

# --- 2. Curvas ROC/PR ---
try:
    print("  [2/5] curvas_roc_pr.png...")
    fpr_r, tpr_r, _ = roc_curve(y_true_test, y_pred_test_cal)
    prec_p, rec_p, _ = precision_recall_curve(y_true_test, y_pred_test_cal)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Curvas — {NOTEBOOK_NAME} (calibradas)", fontsize=11, fontweight="bold")
    axes[0].plot(fpr_r, tpr_r, color="steelblue", lw=2,
                 label=f"AUC={roc_auc_score(y_true_test, y_pred_test_cal):.4f}")
    axes[0].plot([0,1],[0,1],"k--",alpha=0.4,label="Aleatorio")
    axes[0].set_title("ROC — Test"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[1].plot(rec_p, prec_p, color="tomato", lw=2,
                 label=f"AUC-PR={average_precision_score(y_true_test, y_pred_test_cal):.4f}")
    axes[1].axhline(y_true_test.mean(), color="gray", ls="--", alpha=0.6,
                    label=f"Baseline={y_true_test.mean():.4f}")
    axes[1].set_title("PR — Test"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "curvas_roc_pr.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    archivos_generados.append("curvas_roc_pr.png")
    print("    ✅ OK")
except Exception as e:
    print(f"    ❌ {e}")

# --- 3 y 4. Matrices de confusion ---
def plot_confusion_v18(y_true, y_pred, threshold, nombre, run_dir):
    try:
        y_bin = (y_pred >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0,1]).ravel()
        total = tn+fp+fn+tp
        rec  = tp/(tp+fn+1e-8); prec = tp/(tp+fp+1e-8)
        f1   = 2*prec*rec/(prec+rec+1e-8)
        fpr  = fp/(fp+tn+1e-8); fnr  = fn/(fn+tp+1e-8)

        fig = plt.figure(figsize=(18, 10))
        gs2 = gridspec_plots.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle(
            f"Confusion — {nombre} | {NOTEBOOK_NAME}\nThreshold={threshold:.4f} | "
            f"Positivos={tp+fn:,} ({(tp+fn)/total*100:.3f}%)",
            fontsize=12, fontweight="bold")

        ax1 = fig.add_subplot(gs2[0,0])
        cm_abs = np.array([[tn,fp],[fn,tp]])
        im1 = ax1.imshow(cm_abs, cmap="Blues", aspect="auto")
        for i, row_v in enumerate(cm_abs):
            for j, val in enumerate(row_v):
                color = "white" if val > cm_abs.max()*0.5 else "black"
                ax1.text(j,i,f"{val:,}",ha="center",va="center",fontsize=10,fontweight="bold",color=color)
        ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
        ax1.set_xticklabels(["Pred:No","Pred:Si"],fontsize=9)
        ax1.set_yticklabels(["Real:No","Real:Si"],fontsize=9)
        ax1.set_title("Valores absolutos",fontsize=10)
        plt.colorbar(im1,ax=ax1,fraction=0.046,pad=0.04)

        ax2 = fig.add_subplot(gs2[0,1])
        cm_norm = np.array([
            [tn/(tn+fp+1e-8), fp/(tn+fp+1e-8)],
            [fn/(fn+tp+1e-8), tp/(fn+tp+1e-8)]])
        etiq = [[f"{cm_norm[0,0]*100:.1f}%\nTNR",f"{cm_norm[0,1]*100:.1f}%\nFPR"],
                [f"{cm_norm[1,0]*100:.1f}%\nFNR",f"{cm_norm[1,1]*100:.1f}%\nTPR"]]
        im2 = ax2.imshow(cm_norm,cmap="Blues",vmin=0,vmax=1,aspect="auto")
        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i,j]>0.5 else "black"
                ax2.text(j,i,etiq[i][j],ha="center",va="center",fontsize=10,fontweight="bold",color=color)
        ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
        ax2.set_xticklabels(["Pred:No","Pred:Si"],fontsize=9)
        ax2.set_yticklabels(["Real:No","Real:Si"],fontsize=9)
        ax2.set_title("Normalizada por fila",fontsize=10)
        plt.colorbar(im2,ax=ax2,fraction=0.046,pad=0.04)

        ax3 = fig.add_subplot(gs2[0,2])
        mbar = {"Recall\n(TPR)":rec,"Precision":prec,"F1-Score":f1,"FPR":fpr,"FNR\n(Miss)":fnr}
        cols = ["#2196F3","#4CAF50","#FF9800","#F44336","#9C27B0"]
        bars = ax3.barh(list(mbar.keys()),list(mbar.values()),color=cols,alpha=0.85)
        for bar, val in zip(bars,mbar.values()):
            ax3.text(min(val+0.01,0.95),bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",fontsize=9)
        ax3.set_xlim(0,1.15); ax3.set_title("Metricas derivadas",fontsize=10)
        ax3.axvline(0.5,color="gray",ls="--",alpha=0.4); ax3.grid(axis="x",alpha=0.3)

        ax4 = fig.add_subplot(gs2[1,0:2])
        pred_pos = y_pred[y_true==1]; pred_neg = y_pred[y_true==0]
        idx_neg  = np.random.choice(len(pred_neg),size=min(50_000,len(pred_neg)),replace=False)
        ax4.hist(pred_neg[idx_neg],bins=80,alpha=0.5,color="steelblue",density=True,
                 label=f"Negativos (n={len(pred_neg):,})")
        ax4.hist(pred_pos,bins=80,alpha=0.7,color="tomato",density=True,
                 label=f"Positivos (n={len(pred_pos):,})")
        ax4.axvline(threshold,color="green",ls="--",lw=2,label=f"Threshold={threshold:.4f}")
        ax4.set_title("Distribucion predicciones calibradas",fontsize=9)
        ax4.set_xlabel("Probabilidad calibrada"); ax4.set_ylabel("Densidad")
        ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

        ax5 = fig.add_subplot(gs2[1,2]); ax5.axis("off")
        resumen = (f"RESUMEN — {nombre}\n{chr(9472)*28}\n"
                   f"Threshold   : {threshold:.4f}\n"
                   f"TP  : {tp:>12,}\nFP  : {fp:>12,}\n"
                   f"FN  : {fn:>12,}\nTN  : {tn:>12,}\n"
                   f"Recall      : {rec:.4f}\nPrecision   : {prec:.4f}\n"
                   f"F1          : {f1:.4f}\nFNR (miss)  : {fnr:.4f}\n"
                   f"FPR         : {fpr:.6f}\n"
                   f"Detecta {rec*100:.1f} de cada 100\nincendios reales")
        ax5.text(0.05,0.97,resumen,transform=ax5.transAxes,fontsize=8.5,
                 va="top",fontfamily="monospace",
                 bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.9))

        fname = f"confusion_matrix_{nombre.lower()}.png"
        plt.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
        plt.close("all")
        archivos_generados.append(fname)
        print(f"    ✅ OK {fname}")
    except Exception as e:
        print(f"    ❌ confusion_{nombre}: {e}")

print("  [3/5] confusion_matrix_val.png...")
plot_confusion_v18(y_true_val,  y_pred_val_cal,  THRESHOLD, "val",  RUN_DIR)
print("  [4/5] confusion_matrix_test.png...")
plot_confusion_v18(y_true_test, y_pred_test_cal, THRESHOLD, "test", RUN_DIR)

# --- 5. Comparacion crudo vs calibrado ---
try:
    print("  [5/5] comparacion_calibracion.txt...")
    m_val_crudo  = calc_metricas(y_true_val,  y_pred_val,  THRESHOLD)
    m_test_crudo = calc_metricas(y_true_test, y_pred_test, THRESHOLD)
    with open(os.path.join(RUN_DIR, "comparacion_calibracion.txt"), "w") as f:
        f.write(f"Comparacion crudo vs calibrado — {NOTEBOOK_NAME}\n")
        f.write(f"Threshold activo: {THRESHOLD:.4f}\n{'─'*60}\n")
        f.write(f"{'Metrica':<12} {'Val_crudo':>11} {'Val_cal':>11} {'Test_crudo':>12} {'Test_cal':>11}\n")
        for k in ["AUC_ROC","AUC_PR","Recall","Precision","F1","FPR","Brier"]:
            vc=m_val_crudo.get(k,0); vk=m_val.get(k,0)
            tc=m_test_crudo.get(k,0); tk=m_test.get(k,0)
            f.write(f"{k:<12} {vc:>11.4f} {vk:>11.4f} {tc:>12.4f} {tk:>11.4f}\n")
    archivos_generados.append("comparacion_calibracion.txt")
    print("    ✅ OK")
except Exception as e:
    print(f"    ❌ {e}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print(f"\n{'='*60}")
print(f"  COMPLETO: {NOTEBOOK_NAME}")
print(f"  Dir: {RUN_DIR}")
print(f"  Parametros: {n_params:,}")
print(f"  Canales: {N_CANALES_DIN} dinamicos + {N_STATIC_TOTAL} estaticos")
print(f"  Threshold: {THRESHOLD:.5f}")
print(f"\n  {'Metrica':<12} {'Val':>10} {'Test':>10}")
print(f"  {'─'*34}")
for k in ["AUC_ROC","AUC_PR","Recall","Precision","F1","FPR"]:
    status = "OK" if m_test.get(k,0) >= {"AUC_ROC":0.75,"AUC_PR":0.04,"Recall":0.50}.get(k,-1) else "  "
    print(f"  {status} {k:<12} {m_val.get(k,0):>10.4f} {m_test.get(k,0):>10.4f}")
print(f"\n  Archivos generados: {len(archivos_generados)}")
for a in archivos_generados: print(f"    - {a}")
print(f"{'='*60}")


# Historial
try:
    df_h = pd.read_csv(os.path.join(RUN_DIR, "history.csv"))
    mej  = int(df_h["val_roc_auc"].argmax())
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Historial — {NOTEBOOK_NAME}\n"
                 f"Mejor epoch: {mej+1}  val_roc_auc={df_h['val_roc_auc'].max():.4f}",
                 fontsize=12, fontweight="bold")
    for (tk, vk, tit), ax in zip(
        [("loss","val_loss","Loss"),("pr_auc","val_pr_auc","PR-AUC"),
         ("recall","val_recall","Recall"),("roc_auc","val_roc_auc","ROC-AUC")],
        axes.flat):
        if tk in df_h: ax.plot(df_h[tk], label="Train", color="steelblue")
        if vk in df_h: ax.plot(df_h[vk], label="Val",   color="tomato")
        ax.axvline(mej, color="green", linestyle="--", alpha=0.7)
        ax.set_title(tit); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "historial.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  ✅ historial.png")
except Exception as e:
    print(f"  ❌ historial: {e}")

# Curvas ROC/PR
try:
    fpr_r, tpr_r, _ = roc_curve(y_true_test, y_pred_test_cal)
    prec_p, rec_p, _ = precision_recall_curve(y_true_test, y_pred_test_cal)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Curvas — {NOTEBOOK_NAME} (calibradas)", fontsize=11, fontweight="bold")
    axes[0].plot(fpr_r, tpr_r, color="steelblue", lw=2,
                 label=f"AUC={roc_auc_score(y_true_test,y_pred_test_cal):.4f}")
    axes[0].plot([0,1],[0,1],"k--",alpha=0.4)
    axes[0].set_title("ROC — Test"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(rec_p, prec_p, color="tomato", lw=2,
                 label=f"AUC-PR={average_precision_score(y_true_test,y_pred_test_cal):.4f}")
    axes[1].axhline(y_true_test.mean(), color="gray", ls="--", alpha=0.6)
    axes[1].set_title("PR — Test"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "curvas_roc_pr.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  ✅ curvas_roc_pr.png")
except Exception as e:
    print(f"  ❌ curvas: {e}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print(f"\n{'='*60}")
print(f"  COMPLETO: {NOTEBOOK_NAME}")
print(f"  Dir: {RUN_DIR}")
print(f"  Parámetros: {n_params:,}")
print(f"  Canales: {N_CANALES_DIN} dinámicos + {N_STATIC_TOTAL} estáticos")
print(f"  Threshold: {THRESHOLD:.5f}")
print(f"  AUC-ROC test: {m_test['AUC_ROC']:.4f}")
print(f"  Recall  test: {m_test['Recall']:.4f}")
print(f"  Mejora esperada vs V16: +0.02 a +0.05 AUC-ROC")
print(f"{'='*60}")