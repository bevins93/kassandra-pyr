#!/usr/bin/env python3
# prevalencia_total.py
# Cuenta, por partición (train 2020-2022, val 2023, test 2024), cuántas secuencias hay,
# cuántos píxeles-etiqueta y cuántos positivos, con la MISMA lógica de secuencias que el
# entrenamiento (seq_len=3; etiqueta = máscara mensual de la siguiente imagen).
# Solo lee las máscaras de incendio (rápido). Uso: python prevalencia_total.py
import os, re
import numpy as np, rasterio
from datetime import datetime

BASE = "/lustre/home/kgalvan/cnn"
HLS_DIR = f"{BASE}/x/GEE_Export_HLS_Mosaicos"
Y_DIR = f"{BASE}/y_binaria/GEE_Incendios_Mensual"
SEQ_LEN = 3

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

def map_dir(d):
    m = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".tif"):
            dt = parse_fecha(f)
            if dt:
                m[dt.strftime("%Y-%m-%d")] = f; m[dt.strftime("%Y-%m")] = f
    return m

hls = sorted(f for f in os.listdir(HLS_DIR) if f.endswith(".tif"))
y_map = map_dir(Y_DIR)
y_files = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))

def buscar_y(dt, ventana=3):
    for k in [dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")]:
        if k in y_map: return y_map[k]
    mejor, md = None, ventana + 1
    for yf in y_files:
        ydt = parse_fecha(yf)
        if ydt:
            d = abs((ydt - dt).days)
            if d < md: mejor, md = yf, d
    return mejor if md <= ventana else None

seqs = []
for i in range(len(hls) - SEQ_LEN):
    last, nxt = parse_fecha(hls[i + SEQ_LEN - 1]), parse_fecha(hls[i + SEQ_LEN])
    if last is None or nxt is None: continue
    y = buscar_y(nxt)
    if y: seqs.append({"y": y, "anio": last.year, "fecha": last})

cache = {}
def positivos(yf):
    if yf not in cache:
        with rasterio.open(os.path.join(Y_DIR, yf)) as s:
            a = s.read(1)
        cache[yf] = (int((a > 0).sum()), int(a.size))
    return cache[yf]

print(f"{'particion':22s} {'secuencias':>10s} {'pixeles':>14s} {'positivos':>12s} {'prevalencia':>12s} {'meses_distintos_con_fuego':>26s}")
tot_n = tot_p = tot_s = 0
for nombre, filtro in [("train 2020-2022", lambda a: a <= 2022), ("val 2023", lambda a: a == 2023),
                       ("test 2024", lambda a: a == 2024), ("train 2020-2023", lambda a: a <= 2023)]:
    sel = [s for s in seqs if filtro(s["anio"])]
    p = sum(positivos(s["y"])[0] for s in sel); n = sum(positivos(s["y"])[1] for s in sel)
    meses = len({s["y"] for s in sel if positivos(s["y"])[0] > 0})
    print(f"{nombre:22s} {len(sel):10d} {n:14,d} {p:12,d} {100*p/max(n,1):11.3f}% {meses:26d}")
sel = [s for s in seqs if 2020 <= s["anio"] <= 2024]
p = sum(positivos(s["y"])[0] for s in sel); n = sum(positivos(s["y"])[1] for s in sel)
print(f"{'TODO 2020-2024':22s} {len(sel):10d} {n:14,d} {p:12,d} {100*p/max(n,1):11.3f}%")
print("\nAnios con secuencias:", sorted({s['anio'] for s in seqs}))
