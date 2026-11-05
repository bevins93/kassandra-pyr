#!/usr/bin/env python3
import os
import numpy as np

RUN_DIR = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"

shap_path = os.path.join(RUN_DIR, "shap_n30", "shap_values.npy")
labels_path = os.path.join(RUN_DIR, "shap_n30", "shap_labels.npy")

shap = np.load(shap_path)
labels = np.load(labels_path)

print(f"\n=== ESTADÍSTICAS DE MUESTREO ===")
print(f"Total muestras SHAP: {len(labels)}")
print(f"Píxeles positivos (fuego):     {np.sum(labels == 1)}")
print(f"Píxeles negativos (no fuego):  {np.sum(labels == 0)}")
print(f"Secuencias analizadas: 2 (selección manual en código)")