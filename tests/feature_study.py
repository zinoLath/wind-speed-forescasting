"""Estudo determinístico de features para a TCN.

Parte A — poder preditivo univariado (Pearson, Spearman, informação mútua)
          de cada feature candidata em t para ws100 em t+h (h = 1, 12, 36).
Parte B — redundância: matriz de correlação + clustering hierárquico.
Parte C — importância multivariada: RandomForest determinístico (seed fixa)
          treinado só no split de treino, permutation importance na validação.

Saídas: data/results/features_study/{study_tables.json, img/*.png}
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from src import common
from src.utils import wavelet_denoising

OUT = PROJECT_ROOT / "data/results/features_study"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

WAV_SRC = ["ws40", "ws100", "ws260", "disp40", "vdisp40", "v100"]
HORIZONS = (1, 12, 36)
LAG = 144  # 24 h

ds = common.load_dataset(PROJECT_ROOT / "data/dataset.csv", keep_raw=tuple(WAV_SRC))

# --- monta candidatas (série completa; análise heurística) -------------------
for c in WAV_SRC:
    ds[f"{c}_wavelet"] = wavelet_denoising(ds[c].values, level=1)
ds["shear_40_260"] = ds["ws260_wavelet"] - ds["ws40_wavelet"]
ds["turb"] = (ds["ws100"] - ds["ws100_wavelet"]).rolling(6, min_periods=1).std()
ds["ws100_wavelet_lag24h"] = ds["ws100_wavelet"].shift(LAG).bfill()

CANDIDATES = [
    "ws100_wavelet", "ws40_wavelet", "ws260_wavelet", "disp40_wavelet",
    "vdisp40_wavelet", "v100_wavelet", "v100", "dir100_sin", "dir100_cos",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "shear_40_260", "turb", "ws100_wavelet_lag24h",
]

tables = {}
for h in HORIZONS:
    y = ds["ws100"].shift(-h)
    rows = []
    for f in CANDIDATES:
        m = y.notna() & ds[f].notna()
        yy, ff = y[m].values, ds[f].loc[m].values
        from sklearn.feature_selection import mutual_info_regression
        mi = float(mutual_info_regression(ff.reshape(-1, 1), yy, random_state=0)[0])
        rows.append({"feature": f,
                     "pearson": float(pearsonr(ff, yy)[0]),
                     "spearman": float(spearmanr(ff, yy)[0]),
                     "mi": mi})
    tables[f"h{h}"] = pd.DataFrame(rows).sort_values("mi", ascending=False)

# --- Parte B: redundância ----------------------------------------------------
corr = ds[CANDIDATES].corr(method="spearman").abs()
link = hierarchy.linkage(hierarchy.distance.pdist(corr), method="average")
clusters = hierarchy.fcluster(link, t=0.05, criterion="distance")  # dist ~ 1-|r|

# --- Parte C: RF determinístico ---------------------------------------------
split = common.split_dataset(ds, 0.75, 0.20)
tr, va = split[0], split[1]
H = 36


def xy(sl):
    y = sl["ws100"].shift(-H)
    X = sl[CANDIDATES]
    m = y.notna() & X.notna().all(axis=1)
    return X[m], y[m]


Xtr, ytr = xy(tr)
Xva, yva = xy(va)
rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
rf.fit(Xtr, ytr)
r2_val = float(rf.score(Xva, yva))
pi = permutation_importance(rf, Xva, yva, n_repeats=10, random_state=0, n_jobs=-1)
imp = pd.DataFrame({"feature": CANDIDATES,
                    "perm_importance": pi.importances_mean,
                    "perm_std": pi.importances_std}).sort_values(
                        "perm_importance", ascending=False)

# --- persiste tabelas --------------------------------------------------------
out = {"r2_rf_val_h36": r2_val,
       "univariate": {k: v.to_dict("records") for k, v in tables.items()},
       "redundancy_clusters": {
           str(c): [CANDIDATES[i] for i in range(len(CANDIDATES))
                    if clusters[i] == c] for c in sorted(set(clusters))},
       "perm_importance": imp.to_dict("records")}
json.dump(out, open(OUT / "study_tables.json", "w"), indent=1)

# --- gráficos ----------------------------------------------------------------
t36 = tables["h36"].set_index("feature")
order = t36.index.tolist()
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ypos = np.arange(len(order))
ax.barh(ypos + 0.22, t36.loc[order, "mi"], height=0.42, label="Info. mútua", color="#1f77b4")
mi01 = t36.loc[order, "mi"] / t36["mi"].max()
r01 = (t36.loc[order, "pearson"].abs() ** 2)
ax.barh(ypos - 0.22, r01, height=0.42, label="r² Pearson", color="#ff7f0e")
ax.set_yticks(ypos, order)
ax.invert_yaxis()
ax.set_xlabel("Info. mútua (nats) / r² com ws100 em t+36")
ax.set_title("Poder preditivo univariado em h=36 (sem treino profundo)")
ax.legend()
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(IMG / "univariado_h36.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 7.2))
im = ax.imshow(corr.loc[order, order], cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
ax.set_yticks(range(len(order)), order, fontsize=7)
for i in range(len(order)):
    for j in range(len(order)):
        ax.text(j, i, f"{corr.loc[order, order].iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=5.5,
                color="white" if corr.loc[order, order].iloc[i, j] > 0.6 else "black")
ax.set_title("|correlação de Spearman| entre candidatas (redundância)")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(IMG / "redundancia.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
top = imp.head(12).iloc[::-1]
ax.barh(top["feature"], top["perm_importance"], xerr=top["perm_std"], color="#2ca02c")
ax.set_xlabel("Queda de R² ao permutar a feature (validação, h=36)")
ax.set_title(f"Permutation importance — RandomForest (R² val = {r2_val:.3f})")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(IMG / "perm_importance.png", dpi=140)
plt.close(fig)

# decaimento do poder preditivo por horizonte
fig, ax = plt.subplots(figsize=(8.5, 4.6))
sel = ["ws100_wavelet", "ws40_wavelet", "ws260_wavelet", "ws100_wavelet_lag24h",
       "hour_sin", "turb"]
for f in sel:
    vals = [tables[f"h{h}"].set_index("feature").loc[f, "pearson"] for h in HORIZONS]
    ax.plot(HORIZONS, vals, marker="o", label=f)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(HORIZONS)
ax.set_xlabel("Horizonte h (passos de 10 min)")
ax.set_ylabel("Pearson r com ws100 em t+h")
ax.set_title("Decaimento do r por horizonte")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG / "decaimento_horizonte.png", dpi=140)
plt.close(fig)

print("R2 RF val h36:", round(r2_val, 4))
print("\nTop-8 MI h36:")
print(tables["h36"].head(8).to_string(index=False))
print("\nClusters de redundância (|rho|~0.95+):")
for c, fs in out["redundancy_clusters"].items():
    if len(fs) > 1:
        print(" ", c, fs)
print("\nPermutation importance (top-8):")
print(imp.head(8).to_string(index=False))
