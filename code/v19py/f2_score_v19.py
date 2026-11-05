#!/usr/bin/env python3
# f2_score_v19.py
# Calcula F2-score para V18 Run2 en distintos thresholds
# + genera plots publicables

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score

RUN_DIR = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
OUT_DIR = RUN_DIR

print("Cargando predicciones V19 Run2...")
y_true = np.load(f"{RUN_DIR}/y_true_test.npy").astype(np.int32)
y_pred = np.load(f"{RUN_DIR}/y_pred_test_calibrated.npy")

print(f"  Positivos: {y_true.sum():,} ({y_true.mean()*100:.3f}%)")
print(f"  Total    : {len(y_true):,}")

# ============================================================
# Calcular curvas
# ============================================================
print("\nCalculando curvas PR y ROC...")
prec_curve, rec_curve, thr_pr = precision_recall_curve(y_true, y_pred)
fpr_curve, tpr_curve, thr_roc = roc_curve(y_true, y_pred)
auc_roc = roc_auc_score(y_true, y_pred)

# Curva F1 y F2 vs threshold
f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-8)
f2_curve = 5 * prec_curve * rec_curve / (4 * prec_curve + rec_curve + 1e-8)

# Thresholds óptimos
best_f1_idx = np.argmax(f1_curve[:-1])
best_f2_idx = np.argmax(f2_curve[:-1])
thr_f1 = float(thr_pr[best_f1_idx])
thr_f2 = float(thr_pr[best_f2_idx])

# Thresholds fijos
thresholds = {
    "F1-óptimo":    thr_f1,
    "F2-óptimo":    thr_f2,
    "recall≥0.60":  0.0036,
    "precautorio":  0.0050,
}
colors_thr = {
    "F1-óptimo":   "#1565c0",
    "F2-óptimo":   "#b71c1c",
    "recall≥0.60": "#558b2f",
    "precautorio": "#e65100",
}

# Calcular métricas para cada threshold
metrics = {}
for nombre, thr in thresholds.items():
    y_bin = (y_pred >= thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0,1]).ravel()
    r  = tp / (tp + fn + 1e-8)
    p  = tp / (tp + fp + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    f2 = 5 * p * r / (4 * p + r + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    metrics[nombre] = {"thr": thr, "recall": r, "precision": p,
                       "f1": f1, "f2": f2, "fpr": fpr,
                       "tp": int(tp), "fp": int(fp),
                       "fn": int(fn), "tn": int(tn)}

# Imprimir tabla
print("\n" + "="*72)
print(f"  {'Threshold':<18} {'Valor':>8} {'Recall':>8} {'Precis':>8} "
      f"{'F1':>8} {'F2':>8} {'FPR':>8}")
print("  " + "─"*68)
for nombre, m in metrics.items():
    print(f"  {nombre:<18} {m['thr']:>8.5f} {m['recall']:>8.4f} "
          f"{m['precision']:>8.4f} {m['f1']:>8.4f} {m['f2']:>8.4f} "
          f"{m['fpr']:>8.4f}")
print("="*72)
print(f"\nAUC-ROC: {auc_roc:.4f}")
print(f"F2 = (5 × Precision × Recall) / (4 × Precision + Recall)")

# ============================================================
# PLOT 1: Panel completo F2
# ============================================================
print("\nGenerando plots...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Análisis F2-Score — ConvLSTM_UNet_V18_E50 (Run 2)\n"
    "Test 2024 | Calibrado Platt scaling | ANP Jalisco",
    fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32,
                       left=0.07, right=0.97, top=0.92, bottom=0.08)

# ── 1. Curva F1 y F2 vs threshold ──
ax1 = fig.add_subplot(gs[0, 0])
thr_range = thr_pr
ax1.plot(thr_range, f1_curve[:-1], color="#1565c0", lw=2, label="F1-score")
ax1.plot(thr_range, f2_curve[:-1], color="#b71c1c", lw=2.5, label="F2-score")
ax1.axvline(thr_f1, color="#1565c0", ls="--", lw=1.5,
            label=f"F1-opt={thr_f1:.4f}")
ax1.axvline(thr_f2, color="#b71c1c", ls="--", lw=1.5,
            label=f"F2-opt={thr_f2:.4f}")
ax1.scatter([thr_f1], [f1_curve[best_f1_idx]], color="#1565c0", s=80, zorder=5)
ax1.scatter([thr_f2], [f2_curve[best_f2_idx]], color="#b71c1c", s=80, zorder=5)
ax1.set_xlabel("Threshold de probabilidad calibrada")
ax1.set_ylabel("Score")
ax1.set_title("F1 y F2 vs Threshold", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax1.set_xlim(0, np.percentile(y_pred, 99.5))

# ── 2. Curva Recall y Precision vs threshold ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(thr_range, rec_curve[:-1],  color="#2e7d32", lw=2, label="Recall (TPR)")
ax2.plot(thr_range, prec_curve[:-1], color="#6a1b9a", lw=2, label="Precision")
for nombre, m in metrics.items():
    ax2.axvline(m["thr"], color=colors_thr[nombre], ls=":", lw=1.2, alpha=0.8)
    ax2.annotate(nombre.split("-")[0],
                 xy=(m["thr"], m["recall"]),
                 xytext=(m["thr"] + 0.0002, m["recall"] - 0.05),
                 fontsize=7, color=colors_thr[nombre], fontweight="bold")
ax2.set_xlabel("Threshold de probabilidad calibrada")
ax2.set_ylabel("Valor")
ax2.set_title("Recall y Precision vs Threshold", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
ax2.set_xlim(0, np.percentile(y_pred, 99.5))

# ── 3. Curva PR con puntos de operación ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(rec_curve, prec_curve, color="tomato", lw=2,
         label=f"AUC-PR={np.trapz(prec_curve, rec_curve):.4f}")
ax3.axhline(y_true.mean(), color="gray", ls="--", alpha=0.5,
            label=f"Baseline={y_true.mean():.4f}")
for nombre, m in metrics.items():
    ax3.scatter(m["recall"], m["precision"], s=120,
                color=colors_thr[nombre], zorder=5,
                label=f"{nombre} (F2={m['f2']:.4f})")
    ax3.annotate(nombre, xy=(m["recall"], m["precision"]),
                 xytext=(m["recall"] - 0.05, m["precision"] + 0.0005),
                 fontsize=7, color=colors_thr[nombre])
ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
ax3.set_title("Curva Precision-Recall\ncon puntos de operación", fontsize=10, fontweight="bold")
ax3.legend(fontsize=7, loc="upper right"); ax3.grid(alpha=0.3)

# ── 4. Barras comparativas de métricas ──
ax4 = fig.add_subplot(gs[1, 0])
nombres = list(metrics.keys())
x = np.arange(len(nombres))
w = 0.2
metricas_plot = ["recall", "precision", "f1", "f2"]
colores_m = ["#2e7d32", "#6a1b9a", "#1565c0", "#b71c1c"]
etiquetas_m = ["Recall", "Precision", "F1", "F2"]
for i, (met, col, etq) in enumerate(zip(metricas_plot, colores_m, etiquetas_m)):
    vals = [metrics[n][met] for n in nombres]
    bars = ax4.bar(x + i*w, vals, w, color=col, alpha=0.85, label=etq)
    for bar, val in zip(bars, vals):
        if val > 0.01:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.3f}", ha="center", fontsize=6.5, rotation=90)
ax4.set_xticks(x + w*1.5)
ax4.set_xticklabels([n.replace("-", "-\n") for n in nombres], fontsize=8)
ax4.set_ylabel("Valor"); ax4.set_ylim(0, 1.15)
ax4.set_title("Comparación de métricas\npor threshold", fontsize=10, fontweight="bold")
ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.3)

# ── 5. FPR vs Recall (trade-off precautorio) ──
ax5 = fig.add_subplot(gs[1, 1])
# Curva completa
ax5.plot(fpr_curve, tpr_curve, color="steelblue", lw=1.5, alpha=0.6,
         label=f"ROC (AUC={auc_roc:.4f})")
ax5.plot([0,1],[0,1], "k--", alpha=0.3)
for nombre, m in metrics.items():
    ax5.scatter(m["fpr"], m["recall"], s=120,
                color=colors_thr[nombre], zorder=5,
                label=f"{nombre}\nFPR={m['fpr']:.3f} R={m['recall']:.3f}")
ax5.set_xlabel("FPR (Tasa de Falsas Alarmas)")
ax5.set_ylabel("Recall (TPR)")
ax5.set_title("Curva ROC\ncon puntos de operación", fontsize=10, fontweight="bold")
ax5.legend(fontsize=7, loc="lower right"); ax5.grid(alpha=0.3)

# ── 6. Tabla resumen ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

col_labels = ["Threshold", "Valor", "Recall", "Prec.", "F1", "F2", "FPR"]
table_data = []
for nombre, m in metrics.items():
    table_data.append([
        nombre,
        f"{m['thr']:.5f}",
        f"{m['recall']:.4f}",
        f"{m['precision']:.4f}",
        f"{m['f1']:.4f}",
        f"{m['f2']:.4f}",
        f"{m['fpr']:.4f}",
    ])

tbl = ax6.table(cellText=table_data, colLabels=col_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 2.2)

# Colorear fila F2-óptimo
for j in range(len(col_labels)):
    tbl[2, j].set_facecolor("#ffebee")  # fila F2-óptimo (índice 2)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#37474f")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

resumen = (f"Modelo: ConvLSTM-UNet V18 Run2\n"
           f"AUC-ROC test: {auc_roc:.4f}\n"
           f"Prevalencia : {y_true.mean()*100:.3f}%\n"
           f"β=2 → penaliza FN el doble que FP\n"
           f"F2-óptimo   : threshold={thr_f2:.5f}")
ax6.text(0.05, 0.05, resumen, transform=ax6.transAxes,
         fontsize=8, va="bottom", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
ax6.set_title("Tabla resumen F2-score", fontsize=10, fontweight="bold", pad=10)

out = f"{OUT_DIR}/f2_score_analysis.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  ✅ {out}")

# ============================================================
# PLOT 2: Matriz de confusión con threshold F2-óptimo
# ============================================================
fig2 = plt.figure(figsize=(18, 10))
fig2.suptitle(
    f"Confusion Matrix — F2-óptimo (threshold={thr_f2:.5f})\n"
    f"ConvLSTM_UNet_V18_E50 | Test 2024 | ANP Jalisco",
    fontsize=12, fontweight="bold")

gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35)
m = metrics["F2-óptimo"]
tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
total = tp + fp + fn + tn
rec = m["recall"]; prec = m["precision"]
f1 = m["f1"]; f2 = m["f2"]; fpr_v = m["fpr"]
fnr = fn / (fn + tp + 1e-8)

ax = fig2.add_subplot(gs2[0, 0])
cm_abs = np.array([[tn, fp],[fn, tp]])
im = ax.imshow(cm_abs, cmap="Blues", aspect="auto")
for i, row_v in enumerate(cm_abs):
    for j, val in enumerate(row_v):
        color = "white" if val > cm_abs.max()*0.5 else "black"
        ax.text(j, i, f"{val:,}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Pred:No","Pred:Si"]); ax.set_yticklabels(["Real:No","Real:Si"])
ax.set_title("Valores absolutos", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig2.add_subplot(gs2[0, 1])
cm_n = np.array([[tn/(tn+fp+1e-8), fp/(tn+fp+1e-8)],
                  [fn/(fn+tp+1e-8), tp/(fn+tp+1e-8)]])
etiq = [[f"{cm_n[0,0]*100:.1f}%\nTNR", f"{cm_n[0,1]*100:.1f}%\nFPR"],
        [f"{cm_n[1,0]*100:.1f}%\nFNR", f"{cm_n[1,1]*100:.1f}%\nTPR"]]
im2 = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(2):
    for j in range(2):
        color = "white" if cm_n[i,j]>0.5 else "black"
        ax.text(j, i, etiq[i][j], ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Pred:No","Pred:Si"]); ax.set_yticklabels(["Real:No","Real:Si"])
ax.set_title("Normalizada por fila", fontsize=10)
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

ax = fig2.add_subplot(gs2[0, 2])
mbar = {"Recall\n(TPR)": rec, "Precision": prec, "F1": f1,
        "F2": f2, "FPR": fpr_v, "FNR\n(Miss)": fnr}
cols_b = ["#2196F3","#4CAF50","#FF9800","#b71c1c","#F44336","#9C27B0"]
bars = ax.barh(list(mbar.keys()), list(mbar.values()), color=cols_b, alpha=0.85)
for bar, val in zip(bars, mbar.values()):
    ax.text(min(val+0.01, 0.92), bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlim(0, 1.1)
ax.set_title(f"Métricas derivadas\n(threshold F2-óptimo={thr_f2:.5f})", fontsize=10)
ax.axvline(0.5, color="gray", ls="--", alpha=0.4); ax.grid(axis="x", alpha=0.3)

ax = fig2.add_subplot(gs2[1, 0:2])
pred_pos = y_pred[y_true==1]; pred_neg = y_pred[y_true==0]
idx_neg = np.random.choice(len(pred_neg), size=min(50_000, len(pred_neg)), replace=False)
ax.hist(pred_neg[idx_neg], bins=80, alpha=0.5, color="steelblue",
        density=True, label=f"Negativos (n={len(pred_neg):,})")
ax.hist(pred_pos, bins=80, alpha=0.7, color="tomato",
        density=True, label=f"Positivos (n={len(pred_pos):,})")
ax.axvline(thr_f2, color="#b71c1c", ls="--", lw=2,
           label=f"F2-opt={thr_f2:.5f}")
ax.axvline(thr_f1, color="#1565c0", ls=":", lw=1.5,
           label=f"F1-opt={thr_f1:.5f}")
ax.set_title("Distribución de predicciones calibradas", fontsize=9)
ax.set_xlabel("Probabilidad calibrada"); ax.set_ylabel("Densidad")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig2.add_subplot(gs2[1, 2]); ax.axis("off")
resumen = (
    f"RESUMEN — F2-óptimo\n{'─'*30}\n"
    f"Threshold   : {thr_f2:.5f}\n"
    f"TP  : {tp:>12,}\nFP  : {fp:>12,}\n"
    f"FN  : {fn:>12,}\nTN  : {tn:>12,}\n"
    f"Recall      : {rec:.4f}\nPrecision   : {prec:.4f}\n"
    f"F1          : {f1:.4f}\nF2 (β=2)    : {f2:.4f}\n"
    f"FNR (miss)  : {fnr:.4f}\nFPR         : {fpr_v:.4f}\n"
    f"AUC-ROC     : {auc_roc:.4f}\n"
    f"Detecta {rec*100:.1f}/100 incendios\n"
    f"{fpr_v*100:.1f}% falsas alarmas")
ax.text(0.05, 0.97, resumen, transform=ax.transAxes, fontsize=8.5,
        va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

out2 = f"{OUT_DIR}/confusion_f2_optimo.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  ✅ {out2}")

print(f"\n{'='*55}")
print(f"  Archivos generados:")
print(f"  - f2_score_analysis.png   (panel completo)")
print(f"  - confusion_f2_optimo.png (matriz de confusión)")
print(f"\n  F2-óptimo: threshold={thr_f2:.5f}")
print(f"  Recall={metrics['F2-óptimo']['recall']:.4f}  "
      f"FPR={metrics['F2-óptimo']['fpr']:.4f}  "
      f"F2={metrics['F2-óptimo']['f2']:.4f}")
print(f"{'='*55}")