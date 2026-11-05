#!/usr/bin/env python3
# verificar_escenas.py
# Cuenta cuántas escenas distintas hay en las predicciones de validación y prueba,
# y cuántas tienen incendio. Uso: python verificar_escenas.py [RUN_DIR]
import sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else \
    "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
PX = 580314   # píxeles por escena (40,621,980 / 70)

for nombre in ("val", "test"):
    yt = np.load(f"{D}/y_true_{nombre}.npy")
    pr = np.load(f"{D}/y_pred_{nombre}_calibrated.npy")
    n = len(yt) // PX
    resto = len(yt) - n * PX
    P = pr[: n * PX].reshape(n, PX)
    Y = yt[: n * PX].reshape(n, PX)
    distintas = len({P[i].tobytes() for i in range(n)})
    con_fuego = int((Y.sum(axis=1) > 0).sum())
    print(f"{nombre}: {n} escenas | distintas={distintas} | "
          f"con incendio={con_fuego} | sin incendio={n - con_fuego} | píxeles sobrantes={resto}")
