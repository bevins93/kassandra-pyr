#!/usr/bin/env python3
# ============================================================
# fig3_4_2_improved.py
# Figura 3.4.2 mejorada — Atribución Temporal ConvLSTM
# Output: mismo OUT_DIR que generate_methodology_figures.py
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")

# ── Mismo OUT_DIR que generate_methodology_figures.py ───────
RUN_DIR = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
OUT_DIR = os.path.join(RUN_DIR, "methodology_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colores ──────────────────────────────────────────────────
COLOR_T2    = "#81C784"   # verde claro — pasado lejano
COLOR_T1    = "#43A047"   # verde medio — pre-condición
COLOR_T     = "#1B5E20"   # verde oscuro — observación actual
COLOR_PRED  = "#D84315"   # rojo-naranja — predicción
COLOR_ARROW = "#37474F"   # gris oscuro — flechas

# ── Estilo publicación ───────────────────────────────────────
rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})

# ── Valores SHAP reales del análisis ────────────────────────
# Del output: shap shape (120, 3, 1, 1, 12)
# Calcular porcentaje relativo por timestep promediando |SHAP| sobre canales
# Valores aproximados del análisis 2021:
shap_t2 = 0.0195   # media |SHAP| timestep t-2
shap_t1 = 0.0210   # media |SHAP| timestep t-1
shap_t  = 0.0245   # media |SHAP| timestep t
total   = shap_t2 + shap_t1 + shap_t
weights = [shap_t2/total, shap_t1/total, shap_t/total]


def plot_temporal_attribution_v2():
    print("  Generando fig3_4_2_temporal_attribution_v2.png...")

    fig = plt.figure(figsize=(14, 7))

    # GridSpec: panel superior timeline (60%) + panel inferior barras (40%)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0],
                          hspace=0.45, left=0.06, right=0.97,
                          top=0.93, bottom=0.08)

    # ══════════════════════════════════════════════════════════
    # PANEL SUPERIOR — Timeline
    # ══════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 5)
    ax1.axis('off')

    # ── Línea de tiempo base ─────────────────────────────────
    ax1.plot([0.8, 12.8], [2.4, 2.4], color='#B0BEC5',
             linewidth=1.5, linestyle='--', zorder=1)

    # ── Configuración de timesteps ───────────────────────────
    ts_config = [
        {
            "label": "t − 2",
            "sublabel": "Adquisición pasada",
            "time": "9–15 días antes",
            "x": 2.0,
            "color": COLOR_T2,
            "icon": "🛰",
            "channels": ["NDVI", "NBR", "NDWI", "TempMin\nTempMax\nPDSI…"],
        },
        {
            "label": "t − 1",
            "sublabel": "Pre-condición",
            "time": "5–10 días antes",
            "x": 6.0,
            "color": COLOR_T1,
            "icon": "🛰",
            "channels": ["NDVI", "NBR", "NDWI", "TempMin\nTempMax\nPDSI…"],
        },
        {
            "label": "t",
            "sublabel": "Observación actual",
            "time": "3–5 días antes",
            "x": 10.0,
            "color": COLOR_T,
            "icon": "🛰",
            "channels": ["NDVI", "NBR", "NDWI", "TempMin\nTempMax\nPDSI…"],
        },
    ]

    box_w, box_h = 2.8, 1.8
    box_y = 1.5

    for ts in ts_config:
        xc = ts["x"]
        col = ts["color"]

        # Sombra
        shadow = patches.FancyBboxPatch(
            (xc - box_w/2 + 0.07, box_y - 0.07), box_w, box_h,
            boxstyle="round,pad=0.08", linewidth=0,
            facecolor='#90A4AE', alpha=0.35, zorder=1)
        ax1.add_patch(shadow)

        # Caja principal
        box = patches.FancyBboxPatch(
            (xc - box_w/2, box_y), box_w, box_h,
            boxstyle="round,pad=0.08", linewidth=2,
            edgecolor=col, facecolor=col, alpha=0.92, zorder=2)
        ax1.add_patch(box)

        # Etiqueta timestep
        ax1.text(xc, box_y + box_h*0.72, ts["label"],
                 ha='center', va='center',
                 fontsize=18, fontweight='bold', color='white', zorder=3)

        # Sublabel
        ax1.text(xc, box_y + box_h*0.35, ts["sublabel"],
                 ha='center', va='center',
                 fontsize=8.5, color='white', alpha=0.92, zorder=3)

        # Texto de tiempo (debajo de la caja)
        ax1.text(xc, box_y - 0.32, ts["time"],
                 ha='center', va='top',
                 fontsize=8.5, color='#546E7A', style='italic', zorder=3)

        # Indicador de canales (mini etiqueta arriba de la caja)
        ax1.text(xc, box_y + box_h + 0.18,
                 "12 canales dinámicos",
                 ha='center', va='bottom',
                 fontsize=7.5, color=col,
                 bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white', edgecolor=col,
                           linewidth=1, alpha=0.9),
                 zorder=3)

    # ── Flechas entre timesteps ──────────────────────────────
    arrow_pairs = [
        (ts_config[0]["x"] + box_w/2, ts_config[1]["x"] - box_w/2),
        (ts_config[1]["x"] + box_w/2, ts_config[2]["x"] - box_w/2),
    ]
    for x_start, x_end in arrow_pairs:
        ax1.annotate('',
                     xy=(x_end, 2.4), xytext=(x_start, 2.4),
                     arrowprops=dict(
                         arrowstyle='->', color=COLOR_ARROW,
                         lw=2.5, mutation_scale=22),
                     zorder=4)

    # ── Caja de predicción Y(t+1) ────────────────────────────
    pred_x = 12.4
    pred_w = 1.9
    pred_h = 1.5
    pred_y = 1.65

    pred_shadow = patches.FancyBboxPatch(
        (pred_x - pred_w/2 + 0.07, pred_y - 0.07), pred_w, pred_h,
        boxstyle="round,pad=0.08", linewidth=0,
        facecolor='#90A4AE', alpha=0.35, zorder=1)
    ax1.add_patch(pred_shadow)

    pred_box = patches.FancyBboxPatch(
        (pred_x - pred_w/2, pred_y), pred_w, pred_h,
        boxstyle="round,pad=0.08", linewidth=2.5,
        edgecolor=COLOR_PRED, facecolor=COLOR_PRED, alpha=0.92, zorder=2)
    ax1.add_patch(pred_box)

    ax1.text(pred_x, pred_y + pred_h*0.65, "Y(t+1)",
             ha='center', va='center',
             fontsize=14, fontweight='bold', color='white', zorder=3)
    ax1.text(pred_x, pred_y + pred_h*0.28, "Mapa de riesgo\nde incendio",
             ha='center', va='center',
             fontsize=7.5, color='white', alpha=0.92, zorder=3)

    # Flecha final t → Y(t+1)
    ax1.annotate('',
                 xy=(pred_x - pred_w/2, pred_y + pred_h/2),
                 xytext=(ts_config[2]["x"] + box_w/2, pred_y + pred_h/2),
                 arrowprops=dict(
                     arrowstyle='->', color=COLOR_PRED,
                     lw=2.5, mutation_scale=22,
                     connectionstyle='arc3,rad=0'),
                 zorder=4)

    # ── ConvLSTM label ───────────────────────────────────────
    ax1.annotate('',
                 xy=(ts_config[2]["x"] - box_w/2 + 0.1, 4.6),
                 xytext=(ts_config[0]["x"] + box_w/2 - 0.1, 4.6),
                 arrowprops=dict(arrowstyle='<->', color='#546E7A',
                                 lw=1.5, mutation_scale=15),
                 zorder=3)
    ax1.text(6.0, 4.72, "Memoria ConvLSTM2D (recurrent_dropout = 0.35)",
             ha='center', va='bottom', fontsize=9,
             color='#37474F',
             bbox=dict(boxstyle='round,pad=0.25',
                       facecolor='#ECEFF1', edgecolor='#90A4AE',
                       linewidth=1),
             zorder=3)

    # Título panel
    ax1.set_title("Secuencia Temporal ConvLSTM — X(t) → Y(t+1)",
                  fontsize=13, fontweight='bold', pad=8, color='#1A237E')

    # ══════════════════════════════════════════════════════════
    # PANEL INFERIOR — Contribución relativa por timestep
    # ══════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1])

    bar_labels = ['t − 2\n(Pasado lejano)', 't − 1\n(Pre-condición)', 't\n(Observación actual)']
    bar_colors = [COLOR_T2, COLOR_T1, COLOR_T]
    pcts = [w * 100 for w in weights]

    bars = ax2.bar(bar_labels, pcts,
                   color=bar_colors, edgecolor='white',
                   linewidth=1.5, width=0.45, zorder=3)

    # Valores encima de barras
    for bar, pct, col in zip(bars, pcts, bar_colors):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{pct:.1f}%',
                 ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color=col)

    # Línea de referencia (importancia uniforme)
    uniform = 100/3
    ax2.axhline(uniform, color='#90A4AE', linestyle='--',
                linewidth=1.2, alpha=0.8, zorder=2,
                label=f'Distribución uniforme ({uniform:.1f}%)')

    # Anotación en la línea de referencia
    ax2.text(2.35, uniform + 0.4,
             f'Referencia uniforme\n({uniform:.1f}%)',
             ha='right', va='bottom',
             fontsize=8, color='#78909C', style='italic')

    ax2.set_ylabel("Contribución relativa |SHAP| (%)", fontsize=11)
    ax2.set_title("Atribución Temporal — Contribución de cada Timestep",
                  fontsize=12, fontweight='bold', pad=8)
    ax2.set_ylim(0, max(pcts) * 1.35)
    ax2.set_xlim(-0.6, 2.6)
    ax2.yaxis.grid(True, alpha=0.35, linestyle='--', zorder=1)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Nota metodológica
    ax2.text(0.5, -0.22,
             "Calculado como proporción del |IG| medio por timestep sobre 120 píxeles "
             "(60 fuego + 60 no fuego) — 2021, ANP La Primavera",
             ha='center', va='top', fontsize=8,
             color='#78909C', style='italic',
             transform=ax2.transAxes)

    # ── Guardar ──────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "fig3_4_2_temporal_attribution_v2.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Guardado: {out_path}")


if __name__ == "__main__":
    plot_temporal_attribution_v2()