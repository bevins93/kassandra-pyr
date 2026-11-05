#!/usr/bin/env python3
# umbral_validacion.py
# Elige el umbral F2 con la validación (2023), lo aplica a la prueba (2024)
# y genera la figura de desempeño del artículo (4 paneles, inglés, 300 dpi).
# Uso: python umbral_validacion.py [RUN_DIR]
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve, confusion_matrix,
                             roc_auc_score, average_precision_score)

D = sys.argv[1] if len(sys.argv) > 1 else \
    "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"

yv = np.load(f"{D}/y_true_val.npy").astype(np.int8)
pv = np.load(f"{D}/y_pred_val_calibrated.npy")
yt = np.load(f"{D}/y_true_test.npy").astype(np.int8)
pt = np.load(f"{D}/y_pred_test_calibrated.npy")
print(f"val : n={len(yv):,}  prevalencia={yv.mean()*100:.3f}%")
print(f"test: n={len(yt):,}  prevalencia={yt.mean()*100:.3f}%")

def f2_opt(y, p):
    P, R, T = precision_recall_curve(y, p)
    f = 5*P*R/(4*P+R+1e-12)
    return float(T[np.argmax(f[:-1])])

def metrics_at(thr):
    tn, fp, fn, tp = confusion_matrix(yt, (pt >= thr).astype(np.int8), labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    r = tp/(tp+fn); p = tp/(tp+fp); fpr = fp/(fp+tn)
    return dict(thr=thr, recall=r, precision=p,
                F1=2*p*r/(p+r), F2=5*p*r/(4*p+r), FPR=fpr, FNR=fn/(fn+tp),
                TP=tp, FP=fp, FN=fn, TN=tn)

def show(name, m):
    print(f"{name:<28} thr={m['thr']:.5f} recall={m['recall']:.4f} prec={m['precision']:.4f} "
          f"F1={m['F1']:.4f} F2={m['F2']:.4f} FPR={m['FPR']:.4f} "
          f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")

thr_val = f2_opt(yv, pv)          # <- UMBRAL QUE VA EN EL ARTÍCULO
thr_test = f2_opt(yt, pt)         # <- referencia (usa etiquetas de 2024)
m_val, m_test = metrics_at(thr_val), metrics_at(thr_test)
show("F2, umbral de VAL 2023", m_val)
show("F2, umbral de TEST (ref.)", m_test)

auc_roc = roc_auc_score(yt, pt)
auc_pr = average_precision_score(yt, pt)
print(f"AUC-ROC={auc_roc:.4f}  AUC-PR (average precision)={auc_pr:.4f}  "
      f"baseline={yt.mean():.4f}  razón={auc_pr/yt.mean():.2f}x")

json.dump({"umbral_val_2023": m_val, "umbral_test_referencia": m_test,
           "AUC_ROC": auc_roc, "AUC_PR": auc_pr, "n_test": int(len(yt)),
           "prevalencia_test": float(yt.mean())},
          open(f"{D}/metricas_umbral_validacion.json", "w"), indent=2)

# ---------------- Figura de desempeño (4 paneles) ----------------
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.8})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.4))
BL, RD, GR, GY = "#1f4e79", "#b71c1c", "#2e7d32", "#808080"

def thin(n, k=2000):
    return np.unique(np.linspace(0, n-1, min(k, n)).astype(int))

# A. ROC
fpr_c, tpr_c, _ = roc_curve(yt, pt)
i = thin(len(fpr_c))
ax[0, 0].plot(fpr_c[i], tpr_c[i], color=BL, lw=1.6, label=f"AUC–ROC = {auc_roc:.3f}")
ax[0, 0].plot([0, 1], [0, 1], ls="--", color=GY, lw=0.8)
ax[0, 0].scatter(m_val["FPR"], m_val["recall"], color=RD, zorder=5, s=25, label="Operating point")
ax[0, 0].set(xlabel="False positive rate", ylabel="Recall (true positive rate)")
ax[0, 0].legend(frameon=False, loc="lower right")

# B. PR
P_c, R_c, T_c = precision_recall_curve(yt, pt)
j = thin(len(R_c))
ax[0, 1].plot(R_c[j], P_c[j], color=BL, lw=1.6, label=f"AUC–PR = {auc_pr:.4f}")
ax[0, 1].axhline(yt.mean(), ls="--", color=GY, lw=0.8, label=f"No-skill = {yt.mean():.4f}")
ax[0, 1].scatter(m_val["recall"], m_val["precision"], color=RD, zorder=5, s=25, label="Operating point")
ax[0, 1].set(xlabel="Recall", ylabel="Precision")
# Zoom en el eje y: la precisión es ~0.01, así que con 0-1 la curva queda aplastada.
# (El primer punto de la curva, con recall ~0 y precisión alta, es un artefacto de pocos píxeles.)
ymax = float(np.nanmax(P_c[R_c > 0.02])) * 1.35
ax[0, 1].set_ylim(0, ymax)
ax[0, 1].legend(frameon=False, loc="upper right")

# C. Matriz de confusión (porcentaje por fila + conteo)
cm = np.array([[m_val["TN"], m_val["FP"]], [m_val["FN"], m_val["TP"]]])
cmn = cm / cm.sum(axis=1, keepdims=True)
ax[1, 0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for r in range(2):
    for c in range(2):
        ax[1, 0].text(c, r, f"{cmn[r, c]*100:.1f}%\n({cm[r, c]:,})", ha="center", va="center",
                      color="white" if cmn[r, c] > 0.5 else "black", fontsize=8)
ax[1, 0].set_xticks([0, 1]); ax[1, 0].set_xticklabels(["Not flagged", "Flagged"])
ax[1, 0].set_yticks([0, 1]); ax[1, 0].set_yticklabels(["Not burned", "Burned"])
ax[1, 0].set(xlabel="Predicted", ylabel="Observed")

# D. Métricas vs umbral (recall en el eje izquierdo; precisión y F2 en el derecho, por su escala pequeña)
f2_c = 5*P_c*R_c/(4*P_c+R_c+1e-12)
k = thin(len(T_c))
axd = ax[1, 1]
l1, = axd.plot(T_c[k], R_c[:-1][k], color=GR, lw=1.4, label="Recall (left axis)")
axd.set_xlim(0, np.percentile(pt, 99.5))
axd.set(xlabel="Calibrated probability threshold", ylabel="Recall")
axd.set_ylim(0, 1.02)
axr = axd.twinx()
axr.spines["right"].set_visible(True)
l2, = axr.plot(T_c[k], P_c[:-1][k], color="#6a1b9a", lw=1.4, label="Precision (right axis)")
l3, = axr.plot(T_c[k], f2_c[:-1][k], color=RD, lw=1.4, label="F$_2$ (right axis)")
axr.set_ylabel("Precision / F$_2$")
ymr = float(max(np.nanmax(P_c[:-1][k][T_c[k] > 0]), np.nanmax(f2_c[:-1][k][T_c[k] > 0]))) * 1.25
axr.set_ylim(0, ymr)
l4 = axd.axvline(thr_val, ls=":", color="k", lw=1, label=f"Selected threshold = {thr_val:.5f}")
axd.legend(handles=[l1, l2, l3, l4], frameon=False, fontsize=7, loc="upper right")

for a, l in zip(ax.flat, "ABCD"):
    a.text(-0.16, 1.05, l, transform=a.transAxes, fontsize=11, fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{D}/fig_performance.{ext}", dpi=300, bbox_inches="tight")
print(f"\nGuardado en {D}: fig_performance.png/.pdf y metricas_umbral_validacion.json")