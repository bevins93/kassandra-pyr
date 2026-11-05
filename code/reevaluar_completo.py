#!/usr/bin/env python3
# reevaluar_completo.py
# Reevalúa el modelo V19 sobre TODAS las secuencias de validación (2023) y prueba (2024),
# en orden y sin repetir escenas ni excluir meses sin incendio.
# Calibra con Platt scaling sobre la validación completa y guarda las predicciones en
#   <RUN_DIR>/evaluacion_completa/  (mismos nombres de archivo que usan los otros scripts).
# Uso:  python reevaluar_completo.py [--use-gpu]
# Luego: python umbral_validacion.py <RUN_DIR>/evaluacion_completa
import os, re, sys, argparse, warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from datetime import datetime
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--use-gpu", action="store_true")
ap.add_argument("--static-dir", default="/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/version1",
                help="carpeta con static_features_LaPrimavera_30m_V20.tif y categorical_lookup_V20.csv (por defecto, los de abril)")
ap.add_argument("--cfg", default="B", choices=["A", "B", "C", "cur"],
                help="A: cont[0-6,8,9] cat[7] | B: cont[0-6,8,10] cat[9] | C: cont[0-6,8,10] cat[7] | cur: TIF de 10 bandas")
ap.add_argument("--quick", type=int, default=0,
                help="N>0: solo predice N escenas de validacion y las compara con las originales; no guarda nada")
ap.add_argument("--out-name", default="evaluacion_completa")
ap.add_argument("--run-dir", default="/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2")
args = ap.parse_args()

import tensorflow as tf
from tensorflow.keras.layers import (ConvLSTM2D, Conv2D, Conv2DTranspose, BatchNormalization,
                                     SpatialDropout2D, Concatenate, MaxPooling2D, Activation, Input, Lambda)
from sklearn.linear_model import LogisticRegression
if not args.use_gpu:
    tf.config.set_visible_devices([], "GPU")

RUN_DIR = args.run_dir
OUT_DIR = os.path.join(RUN_DIR, args.out_name)
if not args.quick:
    os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Rutas y configuración (idénticas al script de entrenamiento) ----------------
BASE_PATH   = "/lustre/home/kgalvan/cnn"
HLS_DIR     = f"{BASE_PATH}/x/GEE_Export_HLS_Mosaicos"
CLIMATE_DIR = f"{BASE_PATH}/x/Terraclimate_DL_Dataset"
Y_DIR       = f"{BASE_PATH}/y_binaria/GEE_Incendios_Mensual"
STATIC_TIF  = f"{args.static_dir}/static_features_LaPrimavera_30m_V20.tif"
STATIC_CAT  = f"{args.static_dir}/categorical_lookup_V20.csv"

_CFGS = {"A": ([0, 1, 2, 3, 4, 5, 6, 8, 9], [7]),
         "B": ([0, 1, 2, 3, 4, 5, 6, 8, 10], [9]),
         "C": ([0, 1, 2, 3, 4, 5, 6, 8, 10], [7]),
         "cur": ([0, 1, 2, 3, 4, 5, 6, 8, 9], [7])}
STATIC_CONT_IDX, STATIC_CAT_IDX = _CFGS[args.cfg]
print(f"Config estatica {args.cfg}: continuas idx={STATIC_CONT_IDX} categorica idx={STATIC_CAT_IDX} | TIF={STATIC_TIF}")
STATIC_CONT_NAMES = ["elevation", "slope", "aspect_cos", "aspect_sin",
                     "tpi", "roughness", "pop_density", "ag_fraction_500m", "canopy_cover"]
STATIC_NORM_RANGES = {
    "elevation": (0.0, 3500.0), "slope": (0.0, 60.0), "aspect_cos": (-1.0, 1.0),
    "aspect_sin": (-1.0, 1.0), "tpi": (-50.0, 50.0), "roughness": (0.0, 100.0),
    "pop_density": (0.0, 50.0), "ag_fraction_500m": (0.0, 1.0), "canopy_cover": (0.0, 100.0),
}
LOG1P_STATIC = {"pop_density", "roughness"}
# El modelo guardado se entrenó con 15 clases NALCMS (kernel del embedding = 1x1x15x4).
N_NALCMS_CLASSES = 15
try:
    _csv_n = int(pd.read_csv(STATIC_CAT)["value"].max()) + 1
    print(f"AVISO: el archivo de categorías indica {_csv_n} clases; el modelo usa {N_NALCMS_CLASSES}")
except Exception as _e:
    print("No se pudo leer el archivo de categorías:", repr(_e)[:100])
NALCMS_EMBED_DIM = 4
CANALES_HLS_IDX = [0, 1, 2]
CANALES_TC_IDX  = [0, 1, 2, 3, 4, 5]
N_CANALES_DIN   = 12
TC_RANGES = {0: (-30.0, 50.0), 1: (-20.0, 60.0), 2: (0.0, 15.0),
             3: (0.0, 500.0), 4: (-5.0, 5.0), 5: (0.0, 200.0)}
LOG1P_TC = {3, 5}
SEQ_LEN = 3

y_ref = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))[0]
with rasterio.open(os.path.join(Y_DIR, y_ref)) as src:
    TARGET_SHAPE_HW = (src.height, src.width)
print("Target shape:", TARGET_SHAPE_HW, "| píxeles por escena:", TARGET_SHAPE_HW[0] * TARGET_SHAPE_HW[1])

# ---------------- Funciones auxiliares (copiadas del script de entrenamiento) ----------------
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
            print(f"  AVISO: remuestreo en {os.path.basename(path)} {(src.height, src.width)} -> {out_shape_hw}")
            data = src.read(out_shape=(src.count, out_shape_hw[0], out_shape_hw[1]), resampling=resampling)
        else:
            data = src.read()
        return np.transpose(data, (1, 2, 0)).astype(np.float32)

def sanitize_array(arr, clip_val=10.0):
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -clip_val, clip_val).astype(np.float32)

def procesar_hls_con_mascara(raw):
    nube = raw[:, :, 3]
    valido = (np.isfinite(nube) & (nube > 0.5)).astype(np.float32)
    nublado = ~(np.isfinite(nube) & (nube > 0.5))
    res = []
    for ci in CANALES_HLS_IDX:
        canal = np.where(np.isfinite(raw[:, :, ci]), raw[:, :, ci], 0.0)
        canal = np.clip(canal, 0.0, 1.0)
        canal[nublado] = 0.0
        res.append(canal)
    for _ in CANALES_HLS_IDX:
        res.append(valido.copy())
    return np.stack(res, axis=-1).astype(np.float32)

def normalizar_tc(arr):
    out = np.zeros_like(arr)
    for i, (vmin, vmax) in TC_RANGES.items():
        canal = arr[:, :, i].copy()
        if i in LOG1P_TC:
            canal = np.log1p(np.maximum(canal, 0.0)); vmin, vmax = 0.0, np.log1p(vmax)
        out[:, :, i] = np.clip((canal - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return out.astype(np.float32)

def cargar_static_features():
    with rasterio.open(STATIC_TIF) as src:
        allb = src.read(out_shape=(src.count, TARGET_SHAPE_HW[0], TARGET_SHAPE_HW[1]),
                        resampling=Resampling.bilinear).astype(np.float32)
    allb = np.transpose(allb, (1, 2, 0))
    cont = allb[:, :, STATIC_CONT_IDX]
    for ci, name in enumerate(STATIC_CONT_NAMES):
        band = np.where(np.isfinite(cont[:, :, ci]), cont[:, :, ci], 0.0)
        if name in LOG1P_STATIC:
            band = np.log1p(np.maximum(band, 0.0))
            vmin = np.log1p(max(STATIC_NORM_RANGES[name][0], 0)); vmax = np.log1p(STATIC_NORM_RANGES[name][1])
        else:
            vmin, vmax = STATIC_NORM_RANGES[name]
        cont[:, :, ci] = np.clip((band - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    nal = allb[:, :, STATIC_CAT_IDX[0]]
    nal = np.clip(np.where(np.isfinite(nal), nal, 0).astype(np.int32), 0, N_NALCMS_CLASSES - 1)
    H, W = TARGET_SHAPE_HW
    oh = np.zeros((H, W, N_NALCMS_CLASSES), dtype=np.float32)
    for c in range(N_NALCMS_CLASSES):
        oh[:, :, c] = (nal == c).astype(np.float32)
    return cont.astype(np.float32), oh

STATIC_CONT, STATIC_OH = cargar_static_features()
STATIC_CONT_B, STATIC_OH_B = STATIC_CONT[None], STATIC_OH[None]

def map_dir(dir_path):
    m = {}
    for f in sorted(os.listdir(dir_path)):
        if f.endswith(".tif"):
            dt = parse_fecha(f)
            if dt:
                m[dt.strftime("%Y-%m-%d")] = f
                m[dt.strftime("%Y-%m")] = f
    return m

hls_files = sorted(f for f in os.listdir(HLS_DIR) if f.endswith(".tif"))
y_map, clim_map = map_dir(Y_DIR), map_dir(CLIMATE_DIR)
y_files = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))

def buscar_y(dt, ventana=3):
    for key in [dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")]:
        if key in y_map: return y_map[key]
    mejor, md = None, ventana + 1
    for yf in y_files:
        ydt = parse_fecha(yf)
        if ydt:
            d = abs((ydt - dt).days)
            if d < md: mejor, md = yf, d
    return mejor if md <= ventana else None

valid = []
for i in range(len(hls_files) - SEQ_LEN):
    last_dt = parse_fecha(hls_files[i + SEQ_LEN - 1]); next_dt = parse_fecha(hls_files[i + SEQ_LEN])
    if last_dt is None or next_dt is None: continue
    y_next = buscar_y(next_dt)
    if y_next:
        valid.append({"seq_start": i, "y_file": y_next, "fecha": last_dt, "anio": last_dt.year, "mes": last_dt.month})
val_seqs  = sorted([v for v in valid if v["anio"] == 2023], key=lambda v: v["fecha"])
test_seqs = sorted([v for v in valid if v["anio"] == 2024], key=lambda v: v["fecha"])
print(f"Secuencias válidas: val 2023 = {len(val_seqs)} | test 2024 = {len(test_seqs)}")

def leer_hls(fname):
    return procesar_hls_con_mascara(read_and_fix(os.path.join(HLS_DIR, fname), TARGET_SHAPE_HW))

def leer_tc(dt):
    for key in [dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")]:
        if key in clim_map:
            raw = read_and_fix(os.path.join(CLIMATE_DIR, clim_map[key]), TARGET_SHAPE_HW, Resampling.bilinear)
            tc = np.stack([raw[:, :, i] for i in CANALES_TC_IDX], axis=-1)
            return normalizar_tc(sanitize_array(tc))
    return np.zeros((*TARGET_SHAPE_HW, len(CANALES_TC_IDX)), np.float32)

def leer_y(yf):
    return (read_and_fix(os.path.join(Y_DIR, yf), TARGET_SHAPE_HW)[:, :, 0] > 0).astype(np.int8)

# ---------------- Modelo (copiado del script de entrenamiento) ----------------
def gelu(x): return x * tf.sigmoid(1.702 * x)

def conv_block(x, filters, p):
    x = Conv2D(filters, (3, 3), padding="same", name=f"{p}_c1")(x)
    x = BatchNormalization(name=f"{p}_bn1")(x); x = Activation(gelu, name=f"{p}_g1")(x)
    x = Conv2D(filters, (3, 3), padding="same", name=f"{p}_c2")(x)
    x = BatchNormalization(name=f"{p}_bn2")(x); x = Activation(gelu, name=f"{p}_g2")(x)
    return x

def attention_gate(skip, gate, filters, p):
    ps = Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{p}_as")(skip)
    pg = Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"{p}_ag")(gate)
    def resize_gate(inputs):
        g, s = inputs
        return tf.image.resize(g, [tf.shape(s)[1], tf.shape(s)[2]], method="bilinear")
    pgu = Lambda(resize_gate, name=f"{p}_ar")([pg, skip])
    psi = Activation(gelu, name=f"{p}_apsi")(ps + pgu)
    alpha = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name=f"{p}_aalpha")(psi)
    return skip * alpha

def up_block_attention(x, skip, filters, p):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", name=f"{p}_up")(x)
    def match_shape(inputs):
        up, sk = inputs
        return tf.image.resize(up, [tf.shape(sk)[1], tf.shape(sk)[2]], method="bilinear")
    x = Lambda(match_shape, name=f"{p}_ms")([x, skip])
    skip = attention_gate(skip, x, filters // 2, p)
    x = Concatenate(name=f"{p}_cat")([x, skip])
    return conv_block(x, filters, p)

def build_static_branch(inp_cont, inp_nal, name="static"):
    emb = Conv2D(NALCMS_EMBED_DIM, (1, 1), padding="same", activation="relu", name=f"{name}_nalcms_emb")(inp_nal)
    x = Concatenate(name=f"{name}_concat")([inp_cont, emb])
    x = Conv2D(32, (3, 3), padding="same", name=f"{name}_c1")(x)
    x = BatchNormalization(name=f"{name}_bn1")(x); x = Activation(gelu, name=f"{name}_g1")(x)
    feat_full = x
    x = MaxPooling2D((2, 2), name=f"{name}_pool1")(x)
    x = Conv2D(64, (3, 3), padding="same", name=f"{name}_c2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x); x = Activation(gelu, name=f"{name}_g2")(x)
    return feat_full, x

def build_model(shape_din, n_cont, n_nal):
    bias_init = tf.keras.initializers.Constant(-2.0)
    inp_din = Input(shape=shape_din, name="input_dynamic")
    inp_sc = Input(shape=(*TARGET_SHAPE_HW, n_cont), name="input_static_cont")
    inp_sn = Input(shape=(*TARGET_SHAPE_HW, n_nal), name="input_static_nalcms")
    x = ConvLSTM2D(16, (3, 3), padding="same", return_sequences=True, activation="tanh",
                   recurrent_dropout=0.35, name="convlstm_1")(inp_din)
    x = BatchNormalization(name="bn_lstm1")(x)
    x = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=False, activation="tanh",
                   recurrent_dropout=0.35, name="convlstm_2")(x)
    x = BatchNormalization(name="bn_lstm2")(x)
    x = SpatialDropout2D(0.45, name="drop_lstm")(x)
    e1 = conv_block(x, 32, "enc1"); skip_1 = e1
    p1 = MaxPooling2D((2, 2), name="pool1")(e1)
    e2 = conv_block(p1, 64, "enc2"); skip_2 = e2
    p2 = MaxPooling2D((2, 2), name="pool2")(e2)
    bn_out = conv_block(p2, 128, "bottleneck")
    bn_out = SpatialDropout2D(0.45, name="drop_bn")(bn_out)
    sf_full, sf_half = build_static_branch(inp_sc, inp_sn, name="static")
    d2 = up_block_attention(bn_out, skip_2, 64, "dec2")
    def resize_static(inputs):
        sf, ref = inputs
        return tf.image.resize(sf, [tf.shape(ref)[1], tf.shape(ref)[2]], method="bilinear")
    d2 = Concatenate(name="dec2_static_inject")([d2, Lambda(resize_static, name="static_resize_half")([sf_half, d2])])
    d2 = Conv2D(64, (1, 1), padding="same", activation="relu", name="dec2_static_reduce")(d2)
    d1 = up_block_attention(d2, skip_1, 32, "dec1")
    d1 = Concatenate(name="dec1_static_inject")([d1, Lambda(resize_static, name="static_resize_full")([sf_full, d1])])
    d1 = Conv2D(32, (1, 1), padding="same", activation="relu", name="dec1_static_reduce")(d1)
    out = Conv2D(1, (1, 1), padding="same", activation="sigmoid", bias_initializer=bias_init, name="fire_risk_map")(d1)
    return tf.keras.Model(inputs=[inp_din, inp_sc, inp_sn], outputs=out, name="ConvLSTM_UNet_V19")

model = build_model((SEQ_LEN, *TARGET_SHAPE_HW, N_CANALES_DIN), len(STATIC_CONT_IDX), N_NALCMS_CLASSES)
_wp = os.path.join(RUN_DIR, "best_model.keras")
model.load_weights(_wp)
print("Pesos cargados desde best_model.keras")

# ---------------- Predicción determinista, en orden ----------------
def predecir(seqs, nombre):
    yt, yp, filas = [], [], []
    for k, it in enumerate(seqs):
        frames = []
        for fi in hls_files[it["seq_start"]: it["seq_start"] + SEQ_LEN]:
            dt = parse_fecha(fi)
            frames.append(sanitize_array(np.concatenate([leer_hls(fi), leer_tc(dt)], axis=-1)))
        X = np.stack(frames, axis=0)[None].astype(np.float32)
        y = leer_y(it["y_file"])
        p = model.predict({"input_dynamic": X, "input_static_cont": STATIC_CONT_B,
                           "input_static_nalcms": STATIC_OH_B}, verbose=0)
        yt.append(y.reshape(-1)); yp.append(np.clip(p.reshape(-1), 1e-7, 1 - 1e-7).astype(np.float32))
        filas.append({"orden": k, "fecha_ultima_imagen": it["fecha"].strftime("%Y-%m-%d"),
                      "y_file": it["y_file"], "positivos": int(y.sum())})
        print(f"  {nombre} {k+1}/{len(seqs)} {it['fecha']:%Y-%m-%d} positivos={int(y.sum())}", flush=True)
    if not args.quick:
        pd.DataFrame(filas).to_csv(os.path.join(OUT_DIR, f"escenas_{nombre}.csv"), index=False)
    return np.concatenate(yt), np.concatenate(yp)

if args.quick:
    paso = max(1, len(val_seqs) // args.quick)
    sel = val_seqs[::paso][:args.quick]
    _, pq = predecir(sel, "val")
    PX = TARGET_SHAPE_HW[0] * TARGET_SHAPE_HW[1]
    N = pq.reshape(-1, PX)
    orig = np.load(os.path.join(RUN_DIR, "y_pred_val.npy"))
    O = orig[: (len(orig) // PX) * PX].reshape(-1, PX)
    vistos, unicas = set(), []
    for i in range(len(O)):
        k = O[i].tobytes()
        if k not in vistos:
            vistos.add(k); unicas.append(i)
    print(f"\n=== CONFIG {args.cfg}: comparacion con las {len(unicas)} escenas originales distintas ===")
    for n in range(len(N)):
        d = np.array([np.abs(N[n] - O[i]).mean() for i in unicas])
        print(f"  escena {sel[n]['fecha']:%Y-%m-%d}: dif. minima = {d.min():.2e}  (coincide si < 1e-3)")
    sys.exit(0)

yv, pv_raw = predecir(val_seqs, "val")
yt, pt_raw = predecir(test_seqs, "test")
np.save(os.path.join(OUT_DIR, "y_pred_val_raw.npy"), pv_raw)
np.save(os.path.join(OUT_DIR, "y_pred_test_raw.npy"), pt_raw)

# ---------------- Calibración Platt sobre la validación COMPLETA ----------------
print("Calibrando (Platt) con la validación completa...")
lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
lr.fit(pv_raw.reshape(-1, 1), yv.astype(int))
pv = lr.predict_proba(pv_raw.reshape(-1, 1))[:, 1].astype(np.float32)
pt = lr.predict_proba(pt_raw.reshape(-1, 1))[:, 1].astype(np.float32)

np.save(os.path.join(OUT_DIR, "y_true_val.npy"), yv)
np.save(os.path.join(OUT_DIR, "y_true_test.npy"), yt)
np.save(os.path.join(OUT_DIR, "y_pred_val_calibrated.npy"), pv)
np.save(os.path.join(OUT_DIR, "y_pred_test_calibrated.npy"), pt)
print(f"val : n={len(yv):,} prevalencia={yv.mean()*100:.3f}%")
print(f"test: n={len(yt):,} prevalencia={yt.mean()*100:.3f}%")
print(f"Guardado en {OUT_DIR}")
print("Siguiente paso: python umbral_validacion.py", OUT_DIR)