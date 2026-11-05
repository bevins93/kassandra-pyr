#!/usr/bin/env python3
# comparar_targets.py v2: ¿la unión mensual de los incendios individuales (Binario) reproduce las máscaras mensuales (Mensual)?
# Binario = un .tif por evento de incendio (fecha en el nombre); Mensual = un .tif por mes.
import os, re, numpy as np, rasterio
from collections import defaultdict
B="/lustre/home/kgalvan/cnn/y_binaria"
A_,B_=f"{B}/GEE_Incendios_Mensual",f"{B}/GEE_Incendios_Binario"
def ym(fn):
    m=re.search(r"(\d{4})[-_](\d{2})",fn); return (m.group(1)+"-"+m.group(2)) if m else None
mens={ym(f):f for f in sorted(os.listdir(A_)) if f.endswith(".tif")}
ev=defaultdict(list)
for f in sorted(os.listdir(B_)):
    if f.endswith(".tif") and ym(f): ev[ym(f)].append(f)
print(f"Mensual: {len(mens)} meses | Binario: {sum(len(v) for v in ev.values())} eventos en {len(ev)} meses")
print("meses solo en Mensual:",sorted(set(mens)-set(ev))); print("meses solo en Binario:",sorted(set(ev)-set(mens)))
ok=0; mal=0
for k in sorted(set(mens)&set(ev)):
    with rasterio.open(f"{A_}/{mens[k]}") as s: a=s.read(1)>0
    u=np.zeros_like(a)
    for f in ev[k]:
        with rasterio.open(f"{B_}/{f}") as s:
            b=s.read(1)>0
            if b.shape!=a.shape: print("  forma distinta",f,b.shape,a.shape); continue
            u|=b
    d=int((a!=u).sum())
    if d: mal+=1; print(f"  {k}: Mensual={int(a.sum())} px | unión Binario={int(u.sum())} px | distintos={d} ({len(ev[k])} eventos)")
    else: ok+=1
print(f"\nMeses idénticos: {ok} | con diferencias: {mal}")