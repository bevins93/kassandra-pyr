#!/usr/bin/env python3
# generar_todas_las_figuras.py -- UN SOLO ARCHIVO. Uso: python generar_todas_las_figuras.py
# Genera Fig. 3, Fig. 4, Fig. 5 y Figs. S1-S3 (png+pdf) + AUC de validación, en <base>/figuras_articulo_final/
# (no depende de los otros scripts: los lleva incrustados)
SRC_UMBRAL = '#!/usr/bin/env python3\n# umbral_validacion.py\n# Elige el umbral F2 con la validación (2023), lo aplica a la prueba (2024)\n# y genera la figura de desempeño del artículo (4 paneles, inglés, 300 dpi).\n# Uso: python umbral_validacion.py [RUN_DIR]\nimport sys, json\nimport numpy as np\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import (precision_recall_curve, roc_curve, confusion_matrix,\n                             roc_auc_score, average_precision_score)\n\nD = sys.argv[1] if len(sys.argv) > 1 else \\\n    "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"\n\nyv = np.load(f"{D}/y_true_val.npy").astype(np.int8)\npv = np.load(f"{D}/y_pred_val_calibrated.npy")\nyt = np.load(f"{D}/y_true_test.npy").astype(np.int8)\npt = np.load(f"{D}/y_pred_test_calibrated.npy")\nprint(f"val : n={len(yv):,}  prevalencia={yv.mean()*100:.3f}%")\nprint(f"test: n={len(yt):,}  prevalencia={yt.mean()*100:.3f}%")\n\ndef f2_opt(y, p):\n    P, R, T = precision_recall_curve(y, p)\n    f = 5*P*R/(4*P+R+1e-12)\n    return float(T[np.argmax(f[:-1])])\n\ndef metrics_at(thr):\n    tn, fp, fn, tp = confusion_matrix(yt, (pt >= thr).astype(np.int8), labels=[0, 1]).ravel()\n    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)\n    r = tp/(tp+fn); p = tp/(tp+fp); fpr = fp/(fp+tn)\n    return dict(thr=thr, recall=r, precision=p,\n                F1=2*p*r/(p+r), F2=5*p*r/(4*p+r), FPR=fpr, FNR=fn/(fn+tp),\n                TP=tp, FP=fp, FN=fn, TN=tn)\n\ndef show(name, m):\n    print(f"{name:<28} thr={m[\'thr\']:.5f} recall={m[\'recall\']:.4f} prec={m[\'precision\']:.4f} "\n          f"F1={m[\'F1\']:.4f} F2={m[\'F2\']:.4f} FPR={m[\'FPR\']:.4f} "\n          f"TP={m[\'TP\']} FP={m[\'FP\']} FN={m[\'FN\']} TN={m[\'TN\']}")\n\nthr_val = f2_opt(yv, pv)          # <- UMBRAL QUE VA EN EL ARTÍCULO\nthr_test = f2_opt(yt, pt)         # <- referencia (usa etiquetas de 2024)\nm_val, m_test = metrics_at(thr_val), metrics_at(thr_test)\nshow("F2, umbral de VAL 2023", m_val)\nshow("F2, umbral de TEST (ref.)", m_test)\n\nauc_roc = roc_auc_score(yt, pt)\nauc_pr = average_precision_score(yt, pt)\nprint(f"AUC-ROC={auc_roc:.4f}  AUC-PR (average precision)={auc_pr:.4f}  "\n      f"baseline={yt.mean():.4f}  razón={auc_pr/yt.mean():.2f}x")\n\njson.dump({"umbral_val_2023": m_val, "umbral_test_referencia": m_test,\n           "AUC_ROC": auc_roc, "AUC_PR": auc_pr, "n_test": int(len(yt)),\n           "prevalencia_test": float(yt.mean())},\n          open(f"{D}/metricas_umbral_validacion.json", "w"), indent=2)\n\n# ---------------- Figura de desempeño (4 paneles) ----------------\nplt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.8})\nfig, ax = plt.subplots(2, 2, figsize=(7.2, 6.4))\nBL, RD, GR, GY = "#1f4e79", "#b71c1c", "#2e7d32", "#808080"\n\ndef thin(n, k=2000):\n    return np.unique(np.linspace(0, n-1, min(k, n)).astype(int))\n\n# A. ROC\nfpr_c, tpr_c, _ = roc_curve(yt, pt)\ni = thin(len(fpr_c))\nax[0, 0].plot(fpr_c[i], tpr_c[i], color=BL, lw=1.6, label=f"AUC–ROC = {auc_roc:.3f}")\nax[0, 0].plot([0, 1], [0, 1], ls="--", color=GY, lw=0.8)\nax[0, 0].scatter(m_val["FPR"], m_val["recall"], color=RD, zorder=5, s=25, label="Operating point")\nax[0, 0].set(xlabel="False positive rate", ylabel="Recall (true positive rate)")\nax[0, 0].legend(frameon=False, loc="lower right")\n\n# B. PR\nP_c, R_c, T_c = precision_recall_curve(yt, pt)\nj = thin(len(R_c))\nax[0, 1].plot(R_c[j], P_c[j], color=BL, lw=1.6, label=f"AUC–PR = {auc_pr:.4f}")\nax[0, 1].axhline(yt.mean(), ls="--", color=GY, lw=0.8, label=f"No-skill = {yt.mean():.4f}")\nax[0, 1].scatter(m_val["recall"], m_val["precision"], color=RD, zorder=5, s=25, label="Operating point")\nax[0, 1].set(xlabel="Recall", ylabel="Precision")\n# Zoom en el eje y: la precisión es ~0.01, así que con 0-1 la curva queda aplastada.\n# (El primer punto de la curva, con recall ~0 y precisión alta, es un artefacto de pocos píxeles.)\nymax = float(np.nanmax(P_c[R_c > 0.02])) * 1.35\nax[0, 1].set_ylim(0, ymax)\nax[0, 1].legend(frameon=False, loc="upper right")\n\n# C. Matriz de confusión (porcentaje por fila + conteo)\ncm = np.array([[m_val["TN"], m_val["FP"]], [m_val["FN"], m_val["TP"]]])\ncmn = cm / cm.sum(axis=1, keepdims=True)\nax[1, 0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)\nfor r in range(2):\n    for c in range(2):\n        ax[1, 0].text(c, r, f"{cmn[r, c]*100:.1f}%\\n({cm[r, c]:,})", ha="center", va="center",\n                      color="white" if cmn[r, c] > 0.5 else "black", fontsize=8)\nax[1, 0].set_xticks([0, 1]); ax[1, 0].set_xticklabels(["Not flagged", "Flagged"])\nax[1, 0].set_yticks([0, 1]); ax[1, 0].set_yticklabels(["Not burned", "Burned"])\nax[1, 0].set(xlabel="Predicted", ylabel="Observed")\n\n# D. Métricas vs umbral (recall en el eje izquierdo; precisión y F2 en el derecho, por su escala pequeña)\nf2_c = 5*P_c*R_c/(4*P_c+R_c+1e-12)\nk = thin(len(T_c))\naxd = ax[1, 1]\nl1, = axd.plot(T_c[k], R_c[:-1][k], color=GR, lw=1.4, label="Recall (left axis)")\naxd.set_xlim(0, np.percentile(pt, 99.5))\naxd.set(xlabel="Calibrated probability threshold", ylabel="Recall")\naxd.set_ylim(0, 1.02)\naxr = axd.twinx()\naxr.spines["right"].set_visible(True)\nl2, = axr.plot(T_c[k], P_c[:-1][k], color="#6a1b9a", lw=1.4, label="Precision (right axis)")\nl3, = axr.plot(T_c[k], f2_c[:-1][k], color=RD, lw=1.4, label="F$_2$ (right axis)")\naxr.set_ylabel("Precision / F$_2$")\nymr = float(max(np.nanmax(P_c[:-1][k][T_c[k] > 0]), np.nanmax(f2_c[:-1][k][T_c[k] > 0]))) * 1.25\naxr.set_ylim(0, ymr)\nl4 = axd.axvline(thr_val, ls=":", color="k", lw=1, label=f"Selected threshold = {thr_val:.5f}")\naxd.legend(handles=[l1, l2, l3, l4], frameon=False, fontsize=7, loc="upper right")\n\nfor a, l in zip(ax.flat, "ABCD"):\n    a.text(-0.16, 1.05, l, transform=a.transAxes, fontsize=11, fontweight="bold")\nfig.tight_layout()\nfor ext in ("png", "pdf"):\n    fig.savefig(f"{D}/fig_performance.{ext}", dpi=300, bbox_inches="tight")\nprint(f"\\nGuardado en {D}: fig_performance.png/.pdf y metricas_umbral_validacion.json")\n'

SRC_SUPL = '#!/usr/bin/env python3\n# figuras_suplementarias.py\n# Genera las figuras suplementarias (inglés, 300 dpi, PNG + PDF):\n#   figS1_training_history        (antes Fig. 4: historial de entrenamiento)\n#   figS2_monthly_burned_area     (antes Fig. 3: área quemada mensual 2020-2024)\n#   figS3_slope_vs_probability    (antes Fig. 10: pendiente vs. mapa de probabilidad)\n# Uso:  python figuras_suplementarias.py [--fecha 2024-04-27] [--sub evaluacion_abril]\n# Salida: <RUN_DIR>/figuras_suplementarias/\nimport os, re, sys, argparse, warnings\nimport numpy as np, pandas as pd, rasterio\nfrom rasterio.enums import Resampling\nfrom datetime import datetime\nwarnings.filterwarnings("ignore")\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nfrom matplotlib.colors import LinearSegmentedColormap\n\nap = argparse.ArgumentParser()\nap.add_argument("--run-dir", default="/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2")\nap.add_argument("--sub", default="evaluacion_abril", help="subcarpeta con las predicciones de prueba")\nap.add_argument("--fecha", default="2024-04-27", help="fecha (YYYY-MM-DD) de la escena de la Fig. S3; se usa la más cercana")\nap.add_argument("--base", default="/lustre/home/kgalvan/cnn")\nap.add_argument("--static-tif", default=None, help="por defecto: version1 (stack de abril)")\nap.add_argument("--out", default=None)\nargs = ap.parse_args()\n\nRUN = args.run_dir\nOUT = args.out or os.path.join(RUN, "figuras_suplementarias")\nos.makedirs(OUT, exist_ok=True)\nY_DIR = f"{args.base}/y_binaria/GEE_Incendios_Mensual"\nSTATIC = args.static_tif or f"{args.base}/x/Topographic_DL_Dataset/version1/static_features_LaPrimavera_30m_V20.tif"\n\nplt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,\n                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,\n                     "axes.spines.right": False, "figure.dpi": 100})\nC_TRAIN, C_VAL, C_MARK, C_SPRING = "#4C78A8", "#E45756", "#333333", "#F2D8A7"\n\ndef guardar(fig, nombre):\n    for ext in ("png", "pdf"):\n        fig.savefig(os.path.join(OUT, f"{nombre}.{ext}"), dpi=300, bbox_inches="tight")\n    plt.close(fig)\n    print("  guardada", os.path.join(OUT, nombre + ".png"))\n\ndef parse_fecha(fn):\n    b = os.path.basename(fn)\n    for pat in [r"(\\d{4})-(\\d{2})-(\\d{2})", r"(\\d{4})_(\\d{2})_(\\d{2})"]:\n        m = re.search(pat, b)\n        if m:\n            try: return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))\n            except ValueError: pass\n    for pat in [r"(\\d{4})-(\\d{2})(?!-\\d{2})", r"(\\d{4})(\\d{2})(?!\\d)"]:\n        m = re.search(pat, b)\n        if m:\n            try: return datetime(int(m.group(1)), int(m.group(2)), 1)\n            except ValueError: pass\n    return None\n\n# ======================= S1: historial de entrenamiento =======================\nprint("S1: historial de entrenamiento")\nh = pd.read_csv(os.path.join(RUN, "history.csv"))\nep = np.arange(1, len(h) + 1)\ncol_auc = "val_roc_auc" if "val_roc_auc" in h else [c for c in h if c.startswith("val_") and "auc" in c][0]\nbest = int(np.argmax(h[col_auc].values)) + 1\nprint(f"  epocas={len(h)}  mejor epoca (max {col_auc})={best}  valor={h[col_auc].max():.4f}")\npaneles = [("loss", "val_loss", "Focal Tversky loss", "(a)"),\n           ("roc_auc", "val_roc_auc", "ROC-AUC", "(b)"),\n           ("pr_auc", "val_pr_auc", "PR-AUC", "(c)"),\n           ("recall", "val_recall", "Recall", "(d)")]\nfig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)\nfor ax, (tr, va, titulo, letra) in zip(axs.ravel(), paneles):\n    if tr in h: ax.plot(ep, h[tr], color=C_TRAIN, lw=1.4, label="Training")\n    if va in h: ax.plot(ep, h[va], color=C_VAL, lw=1.4, label="Validation")\n    ax.axvline(best, color=C_MARK, ls="--", lw=1)\n    ax.set_title(f"{letra} {titulo}", loc="left"); ax.set_ylabel(titulo)\n    ax.grid(alpha=0.25, lw=0.5)\nfor ax in axs[1]: ax.set_xlabel("Epoch")\naxs[0, 0].legend(frameon=False, loc="best")\naxs[0, 1].annotate(f"epoch {best}", xy=(best, axs[0, 1].get_ylim()[0]), xytext=(4, 6),\n                   textcoords="offset points", fontsize=8, color=C_MARK)\nfig.tight_layout()\nguardar(fig, "figS1_training_history")\n\n# ======================= S2: área quemada mensual =======================\nprint("S2: área quemada mensual (a partir de las máscaras mensuales)")\nfilas = []\nfor f in sorted(os.listdir(Y_DIR)):\n    if not f.endswith(".tif"): continue\n    dt = parse_fecha(f)\n    if dt is None: continue\n    with rasterio.open(os.path.join(Y_DIR, f)) as s:\n        a = s.read(1)\n        # Si la máscara está en grados (CRS geográfico) la resolución no está en metros: se usa 30 m (0.09 ha/píxel)\n        px_ha = 0.09 if (s.crs is None or s.crs.is_geographic) else abs(s.res[0] * s.res[1]) / 1e4\n    filas.append({"anio": dt.year, "mes": dt.month, "ha": float((a > 0).sum()) * px_ha})\nm = pd.DataFrame(filas).groupby(["anio", "mes"], as_index=False)["ha"].sum()\nm = m[(m.anio >= 2020) & (m.anio <= 2024)].sort_values(["anio", "mes"]).reset_index(drop=True)\ntot = m.ha.sum(); prim = m[m.mes.isin([3, 4, 5])].ha.sum()\nprint(f"  total 2020-2024 = {tot:,.0f} ha | marzo-mayo = {prim:,.0f} ha ({100*prim/tot:.2f} %)")\nprint(f"  pico mensual = {m.ha.max():,.0f} ha en {int(m.loc[m.ha.idxmax(),\'anio\'])}-{int(m.loc[m.ha.idxmax(),\'mes\']):02d}")\nprint("  por año (ha):", {int(a): round(v) for a, v in m.groupby(\'anio\').ha.sum().items()})\nx = np.arange(len(m))\nfig, ax = plt.subplots(figsize=(7.2, 3.2))\nfor i, r in m.iterrows():\n    if r.mes in (3, 4, 5): ax.axvspan(i - 0.5, i + 0.5, color=C_SPRING, lw=0, zorder=0)\nax.bar(x, m.ha, color=C_VAL, width=0.8, zorder=2)\nticks = [i for i, r in m.iterrows() if r.mes == 1]\nax.set_xticks(ticks); ax.set_xticklabels([str(int(m.loc[i, "anio"])) for i in ticks])\nax.set_xlim(-0.5, len(m) - 0.5); ax.set_ylabel("Burned area (ha)")\nax.grid(axis="y", alpha=0.25, lw=0.5)\nfrom matplotlib.patches import Patch\nax.legend(handles=[Patch(facecolor=C_SPRING, edgecolor="none", label="March–May (spring fire season)")], frameon=False, loc="upper right")\nfig.tight_layout()\nguardar(fig, "figS2_monthly_burned_area")\n\n# ======================= S3: pendiente vs. probabilidad =======================\nprint("S3: pendiente vs. mapa de probabilidad calibrada")\nPXF = None\ny_ref = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))[0]\nwith rasterio.open(os.path.join(Y_DIR, y_ref)) as s:\n    H, W = s.height, s.width\nPX = H * W\nesc = pd.read_csv(os.path.join(RUN, args.sub, "escenas_test.csv"))\nesc["f"] = pd.to_datetime(esc["fecha_ultima_imagen"])\nobjetivo = pd.Timestamp(args.fecha)\nk = int((esc["f"] - objetivo).abs().values.argmin())\nprint(f"  escena elegida: orden {k}  fecha de la última imagen = {esc.loc[k,\'fecha_ultima_imagen\']} (pedida {args.fecha})")\nP = np.load(os.path.join(RUN, args.sub, "y_pred_test_calibrated.npy"), mmap_mode="r")\nprob = np.asarray(P[k * PX:(k + 1) * PX]).reshape(H, W).astype(np.float32)\nwith rasterio.open(STATIC) as s:\n    slope = s.read(2, out_shape=(H, W), resampling=Resampling.bilinear).astype(np.float32)   # banda 2 = slope\n    valid = np.isfinite(slope)\n    if s.nodata is not None: valid &= slope != s.nodata\nslope = np.where(valid, slope, np.nan)\nprob = np.where(valid, prob, np.nan)\nvmax = float(np.nanpercentile(prob, 99.5))\ncmap_p = plt.get_cmap("magma_r")\nfig, axs = plt.subplots(1, 3, figsize=(7.4, 3.3))\nim0 = axs[0].imshow(slope, cmap="Greys", vmin=0, vmax=np.nanpercentile(slope, 99))\naxs[0].set_title("(A) Slope (°)", loc="left")\ncb0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02); cb0.ax.tick_params(labelsize=7)\nim1 = axs[1].imshow(prob, cmap=cmap_p, vmin=0, vmax=vmax)\naxs[1].set_title("(B) Calibrated probability", loc="left")\ncb1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02); cb1.ax.tick_params(labelsize=7)\naxs[2].imshow(slope, cmap="Greys", vmin=0, vmax=np.nanpercentile(slope, 99), alpha=0.9)\nov = np.ma.masked_invalid(prob)\naxs[2].imshow(ov, cmap=cmap_p, vmin=0, vmax=vmax, alpha=0.55)\naxs[2].set_title("(C) Composite", loc="left")\nfor a in axs:\n    a.set_xticks([]); a.set_yticks([])\n    for sp in a.spines.values(): sp.set_visible(False)\nfig.suptitle(f"Acquisition of {esc.loc[k,\'fecha_ultima_imagen\']}", fontsize=8, y=0.02, color="#555555")\nfig.tight_layout()\nguardar(fig, "figS3_slope_vs_probability")\nprint("\\nListo. Archivos en", OUT)\n'

# ============================================================================
# Figs. 4 y 5 del artículo (nuevas): mapas estacionales y serie temporal 2024
# Clases de peligro (danger classes) por percentiles de la probabilidad calibrada
# de todos los píxeles válidos de la primavera de 2024 (marzo-mayo):
#   Low < p50 <= Medium < p85 <= High < p95 <= Very High
# Por construcción, High+Very High = 15 % y Very High = 5 % del área en promedio de primavera.
# ============================================================================
def figuras_4_5(EVAL, STATIC, Y_DIR, OUT, fechas):
    import os, numpy as np, pandas as pd, rasterio
    from rasterio.enums import Resampling
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    import matplotlib.dates as mdates

    y_ref = sorted(f for f in os.listdir(Y_DIR) if f.endswith(".tif"))[0]
    with rasterio.open(os.path.join(Y_DIR, y_ref)) as s:
        H, W = s.height, s.width
    PX = H * W
    esc = pd.read_csv(os.path.join(EVAL, "escenas_test.csv"))
    esc["f"] = pd.to_datetime(esc["fecha_ultima_imagen"])
    P = np.load(os.path.join(EVAL, "y_pred_test_calibrated.npy"), mmap_mode="r")
    Y = np.load(os.path.join(EVAL, "y_true_test.npy"), mmap_mode="r")
    n = len(esc)
    with rasterio.open(STATIC) as s:
        slope = s.read(2, out_shape=(H, W), resampling=Resampling.bilinear).astype(np.float32)
        valid = np.isfinite(slope)
        if s.nodata is not None: valid &= slope != s.nodata
    PIXHA = 0.09
    print(f"  píxeles válidos = {valid.sum():,}  -> área del ANP = {valid.sum()*PIXHA:,.0f} ha (referencia: 52,228 ha)")

    spring = esc.f.dt.month.isin([3, 4, 5]).values
    pool = np.concatenate([np.asarray(P[k*PX:(k+1)*PX]).reshape(H, W)[valid] for k in range(n) if spring[k]])
    t50, t85, t95 = [float(np.percentile(pool, q)) for q in (50, 85, 95)]
    print(f"  umbrales de clase (primavera 2024): p50={t50:.5f}  p85={t85:.5f}  p95={t95:.5f}")
    cls = lambda p: np.digitize(p, [t50, t85, t95])          # 0 Low, 1 Medium, 2 High, 3 Very High

    area = np.zeros((n, 4)); obs = np.zeros(n)
    for k in range(n):
        p = np.asarray(P[k*PX:(k+1)*PX]).reshape(H, W)
        c = cls(p)[valid]
        area[k] = np.bincount(c, minlength=4) * PIXHA
        obs[k] = (np.asarray(Y[k*PX:(k+1)*PX]).reshape(H, W)[valid] > 0).sum() * PIXHA
    tot = valid.sum() * PIXHA
    hv = (area[spring, 2] + area[spring, 3]).mean(); vh = area[spring, 3].mean()
    print(f"  primavera 2024 (promedio por escena): High+Very High = {hv:,.1f} ha ({100*hv/tot:.1f} %) | Very High = {vh:,.1f} ha ({100*vh/tot:.1f} %)")
    print(f"  Very High: máximo = {area[:,3].max():,.0f} ha ({esc.f[area[:,3].argmax()]:%Y-%m-%d}), mínimo = {area[:,3].min():,.0f} ha")
    pd.DataFrame({"fecha": esc.fecha_ultima_imagen, "low_ha": area[:,0], "medium_ha": area[:,1],
                  "high_ha": area[:,2], "very_high_ha": area[:,3], "observed_burned_ha": obs}
                 ).to_csv(os.path.join(OUT, "area_por_clase_2024.csv"), index=False)

    COL = ["#ffffcc", "#fecc5c", "#fd8d3c", "#bd0026"]
    NAMES = ["Low", "Medium", "High", "Very High"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # ---------------- Fig. 4: mapas estacionales ----------------
    ks = [int((esc.f - pd.Timestamp(f)).abs().values.argmin()) for f in fechas]
    fig, axs = plt.subplots(3, len(ks), figsize=(1.9*len(ks)+0.8, 6.0))
    cmap_c = ListedColormap(COL); norm_c = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    for j, k in enumerate(ks):
        p = np.where(valid, np.asarray(P[k*PX:(k+1)*PX]).reshape(H, W), np.nan)
        y = np.where(valid, np.asarray(Y[k*PX:(k+1)*PX]).reshape(H, W), np.nan)
        c = np.where(valid, cls(np.nan_to_num(p)), np.nan)
        im = axs[0, j].imshow(p, cmap="magma_r", vmin=0, vmax=float(np.nanpercentile(pool, 99.5)))
        axs[1, j].imshow(c, cmap=cmap_c, norm=norm_c)
        axs[2, j].imshow(np.where(valid, 0.0, np.nan), cmap=ListedColormap(["#e3e3e3"]), vmin=0, vmax=1)
        axs[2, j].imshow(np.where(y > 0, 1.0, np.nan), cmap=ListedColormap(["#1f77b4"]), vmin=0, vmax=1)
        axs[0, j].set_title(f"{esc.fecha_ultima_imagen[k]}", fontsize=8)
        for r in range(3):
            axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
            for sp in axs[r, j].spines.values(): sp.set_visible(False)
    for r, lab in enumerate(["(a) Calibrated\nprobability", "(b) Danger class", "(c) Observed\nburned area"]):
        axs[r, 0].set_ylabel(lab, fontsize=8)
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COL, NAMES)] +
               [Patch(facecolor="#1f77b4", label="Burned (reference)")],
               loc="lower center", ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"Fig4_seasonal_maps.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("  -> Fig4_seasonal_maps")

    # ---------------- Fig. 5: serie temporal ----------------
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"height_ratios": [1, 2.4]})
    d = esc.f.values
    a0.bar(mdates.date2num(d), obs, width=6, color="#1f77b4", zorder=3); a0.xaxis_date(); a0.set_ylabel("Observed burned\narea (ha)")
    a1.stackplot(d, area.T, colors=COL, labels=NAMES, edgecolor="none")
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}")); a1.set_ylabel("Area per danger class (ha)"); a1.set_xlim(d[0], d[-1]); a1.set_ylim(0, tot)
    for a in (a0, a1):
        for y in sorted(set(esc.f.dt.year)):
            a.axvspan(pd.Timestamp(y, 3, 1), pd.Timestamp(y, 6, 1), color="#f2d8a7", alpha=0.35, lw=0, zorder=0)
    a1.xaxis.set_major_locator(mdates.MonthLocator()); a1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(a1.get_xticklabels(), rotation=45, ha="right", fontsize=7.5)
    a1.legend(title="Danger class", ncol=4, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"Fig5_temporal_danger.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("  -> Fig5_temporal_danger")


# ======================== PROGRAMA PRINCIPAL ========================
import os, sys, shutil, argparse, json
import numpy as np
ap = argparse.ArgumentParser(description="Genera TODAS las figuras del artículo y las suplementarias en una carpeta nueva")
RUN0 = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2"
ap.add_argument("--run-dir", default=RUN0)
ap.add_argument("--sub", default="evaluacion_abril")
ap.add_argument("--base", default="/lustre/home/kgalvan/cnn")
ap.add_argument("--fecha-s3", default="2024-04-26", help="escena de la Fig. S3")
ap.add_argument("--fechas-fig4", default="2024-01-31,2024-04-26,2024-07-31,2024-10-31", help="4 fechas (una por estación) de la Fig. 4")
ap.add_argument("--out", default=None, help="carpeta de salida (por defecto <base>/figuras_articulo_final)")
a = ap.parse_args()
EVAL = os.path.join(a.run_dir, a.sub)
OUT = a.out or os.path.join(a.base, "figuras_articulo_final")
os.makedirs(OUT, exist_ok=True)
TMP = os.path.join(OUT, "_intermedios"); os.makedirs(TMP, exist_ok=True)
Y_DIR = f"{a.base}/y_binaria/GEE_Incendios_Mensual"
STATIC = f"{a.base}/x/Topographic_DL_Dataset/version1/static_features_LaPrimavera_30m_V20.tif"

def ejecutar(src, argv, titulo):
    print(f"\n=== {titulo} ===", flush=True)
    old = sys.argv; sys.argv = argv
    try:
        exec(compile(src, titulo, "exec"), {"__name__": "__main__"})
    finally:
        sys.argv = old

# Si existe valid_mask.npy (salida de reevaluar_solo_ANP.py), las métricas y la Fig. 3 usan SOLO píxeles del ANP
EVAL_M = EVAL
_mk = os.path.join(EVAL, "valid_mask.npy")
if os.path.exists(_mk):
    EVAL_M = os.path.join(EVAL, "solo_ANP"); os.makedirs(EVAL_M, exist_ok=True)
    _m = np.load(_mk)
    for _n in ("y_true_val", "y_true_test", "y_pred_val_calibrated", "y_pred_test_calibrated"):
        _a = np.load(os.path.join(EVAL, _n + ".npy"), mmap_mode="r")
        np.save(os.path.join(EVAL_M, _n + ".npy"), np.asarray(_a)[np.tile(_m, len(_a) // len(_m))])
    print("Métricas restringidas a píxeles del ANP:", EVAL_M)
ejecutar(SRC_UMBRAL, ["umbral_validacion.py", EVAL_M], "Umbral (2023) y Fig. 3 de desempeño")
ejecutar(SRC_SUPL, ["figuras_suplementarias.py", "--run-dir", a.run_dir, "--sub", a.sub, "--fecha", a.fecha_s3,
                    "--base", a.base, "--out", os.path.join(TMP, "supl")], "Figuras suplementarias S1-S3")
print("\n=== Figs. 4 y 5 (mapas estacionales y serie temporal 2024) ===", flush=True)
figuras_4_5(EVAL, STATIC, Y_DIR, OUT, [s.strip() for s in a.fechas_fig4.split(",")])

from sklearn.metrics import roc_auc_score, average_precision_score
yv = np.load(os.path.join(EVAL_M, "y_true_val.npy")).astype(np.int8); sv = np.load(os.path.join(EVAL_M, "y_pred_val_calibrated.npy"))
yt = np.load(os.path.join(EVAL_M, "y_true_test.npy")).astype(np.int8); st = np.load(os.path.join(EVAL_M, "y_pred_test_calibrated.npy"))
resumen = (f"VALIDACIÓN 2023: AUC-ROC={roc_auc_score(yv,sv):.4f} AUC-PR={average_precision_score(yv,sv):.4f} prev={yv.mean()*100:.3f}%\n"
           f"PRUEBA 2024:     AUC-ROC={roc_auc_score(yt,st):.4f} AUC-PR={average_precision_score(yt,st):.4f} prev={yt.mean()*100:.3f}%\n")
print("\n" + resumen)

def cp(src, dst):
    if os.path.exists(src): shutil.copy2(src, os.path.join(OUT, dst)); print("  ->", dst)
    else: print("  (FALTA)", src)
for ext in ("png", "pdf"):
    cp(os.path.join(EVAL_M, f"fig_performance.{ext}"), f"Fig3_performance.{ext}")
    cp(os.path.join(TMP, "supl", f"figS1_training_history.{ext}"), f"FigS1_training_history.{ext}")
    cp(os.path.join(TMP, "supl", f"figS2_monthly_burned_area.{ext}"), f"FigS2_monthly_burned_area.{ext}")
    cp(os.path.join(TMP, "supl", f"figS3_slope_vs_probability.{ext}"), f"FigS3_slope_vs_probability.{ext}")
cp(os.path.join(EVAL_M, "metricas_umbral_validacion.json"), "metricas_umbral_validacion.json")
open(os.path.join(OUT, "resumen_metricas.txt"), "w").write(resumen)
print(f"\nLISTO. Todo en: {OUT}\nNombres finales: Fig3_performance, Fig4_seasonal_maps, Fig5_temporal_danger, FigS1_..., FigS2_..., FigS3_...")
print("Las Figs. 1 (mapa de ubicación) y 2 (arquitectura) no se generan aquí: se conservan del manuscrito.")