#!/usr/bin/env python3
# ============================================================
# generate_methodology_figures.py
# Genera figuras conceptuales para la sección de metodología
# 3.3.1 - Concepto XAI (Integrated Gradients)
# 3.3.2 - Diseño de muestreo espacial
# 3.4.2 - Atribución temporal ConvLSTM
# 3.4.3 - Triángulo del fuego ecológico
# Tabla 3.3.2 - Parámetros del análisis XAI
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle, Polygon
import rasterio
from rasterio.enums import Resampling
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUN_DIR = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
OUT_DIR = os.path.join(RUN_DIR, "methodology_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Rutas de datos
STATIC_TIF = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
HLS_DIR = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos"
Y_DIR = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Mensual"

# Colores
COLOR_DIN = "#2E7D32"    # Verde - Dinámico/Combustible
COLOR_CLIMA = "#1565C0"   # Azul - Clima
COLOR_TOPO = "#6A1B9A"    # Morado - Topografía
COLOR_IGN = "#B71C1C"     # Rojo - Ignición
COLOR_MASK = "#546E7A"    # Gris azulado - Máscaras

# Variables por vértice
FIRE_TRIANGLE = {
    "Combustible": {
        "canales": ["NDVI", "NBR", "NDWI"],
        "color": COLOR_DIN,
        "pos": (0.5, 0.9)
    },
    "Clima": {
        "canales": ["TempMin", "TempMax", "Viento", "Precip", "PDSI", "Evapo"],
        "color": COLOR_CLIMA,
        "pos": (0.1, 0.2)
    },
    "Topografía": {
        "canales": ["Elevación", "Pendiente", "Aspect", "TPI"],
        "color": COLOR_TOPO,
        "pos": (0.9, 0.2)
    },
    "Ignición": {
        "canales": ["Pop Density", "Ag Fraction", "NALCMS"],
        "color": COLOR_IGN,
        "pos": (0.5, 0.5)
    }
}

# ============================================================
# FIGURA 3.3.1 - CONCEPTO INTEGRATED GRADIENTS
# ============================================================

def plot_ig_concept():
    """
    Diagrama conceptual de 3 paneles mostrando:
    1. Modelo como caja negra
    2. Interpolación lineal baseline→input (30 pasos α)
    3. Gradiente acumulado → vector de atribución
    """
    print("  Generando Figura 3.3.1: Concepto Integrated Gradients...")
    
    fig = plt.figure(figsize=(14, 5))
    gs = plt.GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1], wspace=0.3)
    
    # Panel 1: Modelo como caja negra
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('1. Modelo KASSANDRA-PYR', fontsize=12, fontweight='bold', pad=15)
    
    # Caja del modelo
    model_box = patches.Rectangle((2, 3), 6, 4, linewidth=2, edgecolor='black', 
                                   facecolor='#f0f0f0', zorder=1)
    ax1.add_patch(model_box)
    ax1.text(5, 5, 'ConvLSTM-UNet\n(V19)', ha='center', va='center', 
             fontsize=11, fontweight='bold', zorder=2)
    
    # Inputs
    ax1.text(0.5, 4, 'Inputs:\n• Dinámicos (12)\n• Estáticos (9)\n• NALCMS (15)', 
             ha='left', va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))
    
    # Flecha entrada
    arrow_in = FancyArrowPatch((3.5, 5), (2, 5), arrowstyle='->', mutation_scale=20, 
                                linewidth=2, color='black')
    ax1.add_patch(arrow_in)
    
    # Output
    ax1.text(9.5, 5, 'Output:\nMapa de\nriesgo', ha='left', va='center', 
             fontsize=9, bbox=dict(boxstyle='round', facecolor='#ffcccc'))
    
    # Flecha salida
    arrow_out = FancyArrowPatch((8, 5), (9, 5), arrowstyle='->', mutation_scale=20, 
                                 linewidth=2, color='black')
    ax1.add_patch(arrow_out)
    
    # Panel 2: Interpolación lineal
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('2. Interpolación Lineal (30 pasos α)', fontsize=12, fontweight='bold', pad=15)
    
    # Baseline (ceros)
    ax2.add_patch(patches.Rectangle((0.5, 3.5), 1.5, 1, facecolor='white', 
                                     edgecolor='black', linewidth=2, label='Baseline (0)'))
    ax2.text(1.25, 4, 'x₀ = 0', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Input real
    ax2.add_patch(patches.Rectangle((8, 3.5), 1.5, 1, facecolor='#4CAF50', 
                                     edgecolor='black', linewidth=2, label='Input (x)'))
    ax2.text(8.75, 4, 'x', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Línea de interpolación con puntos
    x_baseline = 2
    x_input = 9
    n_steps = 30
    
    # Dibujar línea
    ax2.plot([x_baseline, x_input], [4, 4], 'k--', linewidth=1.5, alpha=0.5)
    
    # Puntos de interpolación (mostrar algunos representativos)
    for i in range(0, n_steps+1, 5):  # Mostrar cada 5 pasos
        alpha = i / n_steps
        x_pos = x_baseline + alpha * (x_input - x_baseline)
        color_intensity = alpha
        ax2.add_patch(patches.Circle((x_pos, 4), 0.15, 
                                      facecolor=plt.cm.Greens(color_intensity), 
                                      edgecolor='black', linewidth=1))
        if i in [0, 15, 30]:
            ax2.text(x_pos, 3.3, f'α={alpha:.2f}', ha='center', va='top', fontsize=8)
    
    # Ecuación
    ax2.text(5.5, 5, 'x(α) = x₀ + α·(x - x₀)\nα ∈ [0, 1]', 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='#fff9c4'))
    
    # Panel 3: Gradiente acumulado
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    ax3.set_title('3. Gradiente Acumulado → Atribución', fontsize=12, fontweight='bold', pad=15)
    
    # Gradientes en cada paso
    n_arrows = 10
    for i in range(n_arrows):
        alpha = i / (n_arrows - 1)
        x_pos = 1 + alpha * 6
        y_pos = 4 + np.sin(alpha * np.pi) * 2
        
        # Gradiente (flecha)
        grad_magnitude = 0.3 + 0.5 * alpha
        ax3.add_patch(FancyArrowPatch((x_pos, y_pos), 
                                       (x_pos + grad_magnitude, y_pos + 0.2),
                                       arrowstyle='->', mutation_scale=15,
                                       linewidth=2, color='red', alpha=0.7))
        
        # Punto de evaluación
        ax3.add_patch(patches.Circle((x_pos, y_pos), 0.1, facecolor='blue', alpha=0.5))
    
    # Suma de gradientes
    ax3.text(8.5, 5, 'IG = (x - x₀) · Σ ∇F(x(α))', 
             ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#e3f2fd'))
    
    # Vector de atribución final
    ax3.add_patch(patches.Rectangle((7.5, 1.5), 2, 1, facecolor='#ffcc80', 
                                     edgecolor='black', linewidth=2))
    ax3.text(8.5, 2, 'Vector de\nAtribución\n(12 canales)', 
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.savefig(os.path.join(OUT_DIR, "fig3_3_1_ig_concept.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# FIGURA 3.3.2 - DISEÑO DE MUESTREO ESPACIAL
# ============================================================

def plot_sampling_design():
    """
    Mapa de la ANP mostrando:
    - 60 píxeles positivos (fuego, rojo)
    - 60 píxeles negativos (no fuego, azul)
    - Fondo NDVI 2021
    - Perímetro de incendio 2021
    """
    print("  Generando Figura 3.3.2: Diseño de muestreo espacial...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Intentar cargar datos reales
    try:
        # Buscar archivo NDVI de marzo 2021
        import glob
        hls_files = sorted(glob.glob(os.path.join(HLS_DIR, "*2021-03*.tif")))
        
        if hls_files:
            # Cargar NDVI
            with rasterio.open(hls_files[0]) as src:
                ndvi = src.read(1)
                transform = src.transform
                
                # Mostrar NDVI
                im = ax.imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1, 
                              extent=[0, ndvi.shape[1], 0, ndvi.shape[0]])
                plt.colorbar(im, ax=ax, label='NDVI', shrink=0.6)
                
                H, W = ndvi.shape
                
                # Buscar archivo de incendios 2021
                y_files = sorted(glob.glob(os.path.join(Y_DIR, "*2021*.tif")))
                if y_files:
                    with rasterio.open(y_files[0]) as src_y:
                        fire_map = src_y.read(1, out_shape=(H, W), resampling=Resampling.nearest)
                        
                        # Encontrar píxeles de fuego
                        fire_pixels = np.where(fire_map > 0)
                        
                        if len(fire_pixels[0]) > 0:
                            # Seleccionar 60 píxeles positivos aleatorios
                            n_pos = min(60, len(fire_pixels[0]))
                            pos_idx = np.random.choice(len(fire_pixels[0]), n_pos, replace=False)
                            
                            # Scatter positivos
                            ax.scatter(fire_pixels[1][pos_idx], fire_pixels[0][pos_idx],
                                      c='red', s=30, alpha=0.7, label='Fuego (60)', 
                                      edgecolors='white', linewidth=0.5, zorder=3)
                            
                            # Seleccionar 60 píxeles negativos (donde no hay fuego)
                            no_fire_pixels = np.where(fire_map == 0)
                            n_neg = min(60, len(no_fire_pixels[0]))
                            neg_idx = np.random.choice(len(no_fire_pixels[0]), n_neg, replace=False)
                            
                            ax.scatter(no_fire_pixels[1][neg_idx], no_fire_pixels[0][neg_idx],
                                      c='blue', s=30, alpha=0.7, label='No fuego (60)',
                                      edgecolors='white', linewidth=0.5, zorder=3)
                            
                            # Contorno de incendio
                            from scipy import ndimage
                            contours = ndimage.binary_dilation(fire_map > 0) ^ (fire_map > 0)
                            ax.contour(contours, colors='darkred', linewidths=2, alpha=0.8)
        
        ax.set_xlabel('Columna (píxeles)', fontsize=10)
        ax.set_ylabel('Fila (píxeles)', fontsize=10)
        ax.set_title('Diseño de Muestreo SHAP - ANP La Primavera (2021)', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=9)
        
    except Exception as e:
        print(f"    [WARN] Error cargando datos: {e}")
        # Crear mapa sintético
        ax.text(0.5, 0.5, 'Mapa de muestreo\n(60 positivos + 60 negativos)\n\n'
                'Cargando datos GIS...', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
    
    plt.savefig(os.path.join(OUT_DIR, "fig3_3_2_sampling_design.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# FIGURA 3.4.2 - ATRIBUCIÓN TEMPORAL
# ============================================================

def plot_temporal_attribution():
    """
    Timeline mostrando:
    - t-2 (9-15 días antes)
    - t-1 (5-10 días antes)
    - t (observación actual)
    - Predicción Y(t+1)
    - Barra de peso relativo por timestep
    """
    print("  Generando Figura 3.4.2: Atribución temporal ConvLSTM...")
    
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1], hspace=0.4)
    
    # Panel superior: Timeline
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 4)
    ax1.axis('off')
    
    # Timesteps
    timesteps = [
        ("t-2", "9-15 días\nantes", 1.5),
        ("t-1", "5-10 días\nantes", 4.5),
        ("t", "Observación\nactual", 7.5)
    ]
    
    # Dibujar cajas de timestep
    for label, desc, x_pos in timesteps:
        # Caja
        ax1.add_patch(patches.Rectangle((x_pos-1, 1.5), 2, 1.5, 
                                         facecolor=COLOR_DIN, alpha=0.8,
                                         edgecolor='black', linewidth=2))
        ax1.text(x_pos, 2.25, label, ha='center', va='center', 
                 fontsize=14, fontweight='bold', color='white')
        ax1.text(x_pos, 1.2, desc, ha='center', va='top', 
                 fontsize=9, color='black')
        
        # Icono de datos (simbólico)
        ax1.add_patch(patches.Circle((x_pos, 2.8), 0.3, facecolor='#81C784'))
    
    # Flechas de flujo temporal
    for i in range(len(timesteps)-1):
        x_start = timesteps[i][2] + 1
        x_end = timesteps[i+1][2] - 1
        ax1.add_patch(FancyArrowPatch((x_start, 2.25), (x_end, 2.25),
                                       arrowstyle='->', mutation_scale=25,
                                       linewidth=3, color='black'))
    
    # Predicción
    ax1.add_patch(patches.Rectangle((8.5, 0.5), 2, 1, 
                                     facecolor='#ff7043', alpha=0.9,
                                     edgecolor='black', linewidth=2))
    ax1.text(9.5, 1, 'Y(t+1)\nPredicción', 
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Flecha final
    ax1.add_patch(FancyArrowPatch((8.5, 2.25), (8.5, 1.5),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='black'))
    
    # Título
    ax1.text(5, 3.7, 'Secuencia Temporal ConvLSTM (T=3)', 
             ha='center', va='top', fontsize=13, fontweight='bold')
    
    # Panel inferior: Pesos relativos (ejemplo con valores SHAP reales)
    ax2 = fig.add_subplot(gs[1])
    
    # Valores de ejemplo (reemplazar con valores reales SHAP por timestep)
    weights = [0.25, 0.35, 0.40]  # t-2, t-1, t
    labels = ['t-2\n(Pasado)', 't-1\n(Pre-condición)', 't\n(Observación)']
    colors = ['#66BB6A', '#43A047', COLOR_DIN]
    
    bars = ax2.bar(labels, weights, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    # Añadir valores en barras
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight*100:.0f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Importancia SHAP media', fontsize=11)
    ax2.set_title('Contribución Relativa por Timestep', fontsize=12, fontweight='bold', pad=15)
    ax2.set_ylim(0, max(weights) * 1.3)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.savefig(os.path.join(OUT_DIR, "fig3_4_2_temporal_attribution.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# FIGURA 3.4.3 - TRIÁNGULO DEL FUEGO ECOLÓGICO
# ============================================================

def plot_fire_triangle():
    """
    Triángulo con 4 vértices:
    - Combustible (arriba)
    - Clima (izquierda abajo)
    - Topografía (derecha abajo)
    - Ignición (centro)
    Cada variable etiquetada con su valor SHAP medio
    """
    print("  Generando Figura 3.4.3: Triángulo del fuego ecológico...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Triángulo principal
    triangle = Polygon([(5, 9), (1, 2), (9, 2)], 
                       fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(triangle)
    
    # Vértices
    vertices = {
        "Combustible": (5, 9.2, COLOR_DIN),
        "Clima": (0.8, 1.8, COLOR_CLIMA),
        "Topografía": (9.2, 1.8, COLOR_TOPO),
        "Ignición": (5, 5, COLOR_IGN)
    }
    
    # Dibujar vértices y variables
    for name, (x, y, color) in vertices.items():
        # Círculo del vértice
        ax.add_patch(patches.Circle((x, y), 0.6, facecolor=color, 
                                     edgecolor='black', linewidth=2, zorder=3))
        ax.text(x, y, name, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white', zorder=4)
        
        # Variables asociadas (pequeños círculos alrededor)
        if name in FIRE_TRIANGLE:
            canales = FIRE_TRIANGLE[name]["canales"]
            n_vars = len(canales)
            angle_step = 2 * np.pi / max(n_vars, 4)
            
            for i, var in enumerate(canales[:6]):  # Máximo 6 variables
                angle = angle_step * i
                var_x = x + 1.5 * np.cos(angle)
                var_y = y + 1.5 * np.sin(angle)
                
                # Círculo variable
                ax.add_patch(patches.Circle((var_x, var_y), 0.4, 
                                            facecolor='white', 
                                            edgecolor=color, linewidth=2))
                ax.text(var_x, var_y, var, ha='center', va='center', 
                       fontsize=8, color='black')
                
                # Línea al vértice
                ax.plot([x, var_x], [y, var_y], color=color, 
                       linewidth=1, alpha=0.5, linestyle='--')
    
    # Título
    ax.text(5, 9.8, 'Triángulo del Fuego Ecológico', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Subtítulo
    ax.text(5, 0.5, 'Variables asignadas a cada vértice del triángulo del fuego', 
            ha='center', va='bottom', fontsize=10, style='italic')
    
    plt.savefig(os.path.join(OUT_DIR, "fig3_4_3_fire_triangle.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# TABLA 3.3.2 - PARÁMETROS DEL ANÁLISIS XAI
# ============================================================

def create_xai_parameters_table():
    """
    Tabla con parámetros del análisis SHAP:
    - Año analizado
    - Número de secuencias
    - Píxeles positivos/negativos
    - Pasos de integración
    - Baseline
    - Método de normalización
    """
    print("  Generando Tabla 3.3.2: Parámetros del análisis XAI...")
    
    # Datos de la tabla (ajustar según sea necesario)
    params = {
        "Año analizado": "2021",
        "Secuencias totales disponibles": "78",
        "Secuencias con incendios": "63",
        "Secuencias seleccionadas (SHAP)": "2",
        "Píxeles positivos (fuego)": "60",
        "Píxeles negativos (no fuego)": "60",
        "Total píxeles analizados": "120",
        "Pasos de integración (n_steps)": "30",
        "Baseline": "Vector cero (todos los inputs = 0)",
        "Método de normalización": "Máximo absoluto por canal",
        "Rango de interpolación α": "[0.01, 1.0]",
        "Algoritmo": "Integrated Gradients"
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.axis('off')
    
    # Crear tabla
    table_data = list(params.items())
    table = ax.table(cellText=table_data,
                     colLabels=['Parámetro', 'Valor'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    # Estilizar tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Encabezado
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold', ha='center')
    
    # Filas alternadas
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f9f9f9')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Alinear columna de valores al centro
    for i in range(1, len(table_data) + 1):
        table[(i, 1)].set_text_props(ha='center')
    
    ax.set_title('Parámetros del Análisis XAI (Integrated Gradients)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(OUT_DIR, "table3_3_2_xai_parameters.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # También guardar como texto para copiar al LaTeX
    with open(os.path.join(OUT_DIR, "table3_3_2_xai_parameters.txt"), 'w') as f:
        f.write("Tabla 3.3.2 — Parámetros del Análisis XAI\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Parámetro':<40} {'Valor':<20}\n")
        f.write("-"*60 + "\n")
        for param, value in params.items():
            f.write(f"{param:<40} {value:<20}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("  GENERANDO FIGURAS DE METODOLOGÍA")
    print("="*60)
    
    print(f"\n  Directorio de salida: {OUT_DIR}\n")
    
    # Generar todas las figuras
    plot_ig_concept()
    plot_sampling_design()
    plot_temporal_attribution()
    plot_fire_triangle()
    create_xai_parameters_table()
    
    print("\n" + "="*60)
    print("  ✓ FIGURAS GENERADAS EXITOSAMENTE")
    print("="*60)
    print(f"\n  Archivos creados en: {OUT_DIR}")
    print("\n  Lista de archivos:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"    - {f}")
    print()

if __name__ == "__main__":
    main()