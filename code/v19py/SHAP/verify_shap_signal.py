#!/usr/bin/env python3
# ============================================================
# verify_shap_signal.py — Verificar señal real de SHAP
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RUN_DIR = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"

def verify_shap_signal():
    print("\n" + "="*60)
    print("  VERIFICACIÓN DE SEÑAL SHAP")
    print("="*60)
    
    # Cargar valores SHAP
    shap_path = os.path.join(RUN_DIR, "shap_n30", "shap_values.npy")
    labels_path = os.path.join(RUN_DIR, "shap_n30", "shap_labels.npy")
    
    if not os.path.exists(shap_path):
        print(f"[ERROR] No se encontró: {shap_path}")
        return
    
    shap_values = np.load(shap_path)  # [N, T, 1, 1, 12]
    labels = np.load(labels_path)      # [N]
    
    print(f"\n  Shape SHAP: {shap_values.shape}")
    print(f"  Shape labels: {labels.shape}")
    
    # Colapsar dimensiones temporales y espaciales
    shap_flat = shap_values.reshape(shap_values.shape[0], -1)  # [N, T*12]
    shap_mean_per_sample = shap_flat.mean(axis=1)  # [N]
    
    # Estadísticas generales
    print("\n  [ESTADÍSTICAS GENERALES]")
    print(f"    Mean SHAP: {shap_mean_per_sample.mean():.6f}")
    print(f"    Std SHAP:  {shap_mean_per_sample.std():.6f}")
    print(f"    Min SHAP:  {shap_mean_per_sample.min():.6f}")
    print(f"    Max SHAP:  {shap_mean_per_sample.max():.6f}")
    print(f"    Median:    {np.median(shap_mean_per_sample):.6f}")
    
    # Test de señal vs cero
    print("\n  [TEST DE HIPÓTESIS]")
    t_stat, p_value = stats.ttest_1samp(shap_mean_per_sample, 0.0)
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value:     {p_value:.2e}")
    
    if p_value < 0.05:
        print("    ✓ Los valores SHAP son SIGNIFICATIVAMENTE diferentes de cero (p < 0.05)")
    else:
        print("    ✗ Los valores SHAP NO son significativamente diferentes de cero")
    
    # Por variable
    print("\n  [ESTADÍSTICAS POR VARIABLE]")
    NOMBRES_DIN = ["NDVI", "NBR", "NDWI", "mask_NDVI", "mask_NBR", "mask_NDWI",
                   "TempMin", "TempMax", "Viento", "Precip", "PDSI", "Evapo"]
    
    shap_per_var = np.abs(shap_values).mean(axis=(0, 2, 3))  # [12]
    
    for i, name in enumerate(NOMBRES_DIN):
        mean_val = shap_per_var[i]
        std_val = np.abs(shap_values[:, :, 0, 0, i]).std()
        print(f"    {name:12s}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    # Verificar proporción de valores positivos/negativos
    print("\n  [DISTRIBUCIÓN DE SIGNOS]")
    pos_count = np.sum(shap_values > 0)
    neg_count = np.sum(shap_values < 0)
    total = pos_count + neg_count
    print(f"    Valores positivos: {pos_count} ({pos_count/total*100:.1f}%)")
    print(f"    Valores negativos: {neg_count} ({neg_count/total*100:.1f}%)")
    
    # Histograma
    print("\n  [GENERANDO HISTOGRAMA]")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(shap_mean_per_sample, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Cero')
    axes[0].set_xlabel('Mean SHAP value per sample')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribución de valores SHAP')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Boxplot por variable
    shap_for_boxplot = np.abs(shap_values[:, :, 0, 0, :])
    axes[1].boxplot(shap_for_boxplot, labels=NOMBRES_DIN, rot=45)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_ylabel('|SHAP value|')
    axes[1].set_title('Distribución por variable')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "shap_signal_verification.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Histograma guardado en: {RUN_DIR}/shap_signal_verification.png")
    
    # Conclusión
    print("\n  [CONCLUSIÓN]")
    signal_strength = np.abs(shap_mean_per_sample).mean()
    noise_level = shap_mean_per_sample.std()
    snr = signal_strength / (noise_level + 1e-8)
    
    print(f"    Signal strength (mean |SHAP|): {signal_strength:.6f}")
    print(f"    Noise level (std):             {noise_level:.6f}")
    print(f"    Signal-to-Noise Ratio:         {snr:.2f}")
    
    if snr > 1.0:
        print("    ✓ Señal FUERTE: Los valores SHAP tienen magnitud significativa")
    elif snr > 0.5:
        print("    ✓ Señal MODERADA: Los valores SHAP son razonables")
    else:
        print("    ⚠ Señal DÉBIL: Los valores SHAP son cercanos al ruido")

if __name__ == "__main__":
    verify_shap_signal()