"""Figura da matriz de testes de hipótese da rodada 8020-sbsr (para o SBSR).

Condensa em uma única matriz 7×7:
  * triângulo SUPERIOR: Wilcoxon pareado por origem com correção de Holm
    (cor = −log10 p_Holm; borda verde = rejeita H0 a 5%);
  * triângulo INFERIOR: Diebold–Mariano (HLN + HAC lag 35) com p bruto;
  * diagonal: MSE agregado + rank médio de Friedman (origens como blocos);
  * modelos ordenados pelo rank de Friedman.

Lê data/results/report_8020_sbsr/testes_hipotese.json (DM + Wilcoxon +
Friedman) e mcs.json (MSE por modelo); grava PNG 300 dpi + PDF em
data/results/report_8020_sbsr/images/fig_testes_hipotese.{png,pdf}.

    python scripts/fig_testes_hipotese_sbsr.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data/results/report_8020_sbsr"
IMG_DIR = REPORT_DIR / "images"

ORDER = ["TCN-Bi", "ARIMA", "LSTM", "TCN", "P", "RF", "LSTM-Bi"]
LABEL = {"TCN-Bi": "TCN-Bi", "TCN": "TCN", "LSTM-Bi": "LSTM-Bi",
         "LSTM": "LSTM", "ARIMA": "ARIMA", "RF": "Rand. Forest",
         "P": "Persist."}
MSE_KEY = {"TCN-Bi": "TCN Bidirectional", "TCN": "TCN",
           "LSTM-Bi": "LSTM Bidirectional", "LSTM": "LSTM",
           "ARIMA": "ARIMA (baseline)", "RF": "Random Forest (baseline)",
           "P": "Persistência"}


def fmt_p(p):
    if p < 1e-6:
        return "<10⁻⁶"
    if p < 1e-3:
        return f"{p:.0e}".replace("e-0", "e−")
    return f"{p:.2f}".replace(".", ",")


def pair_lookup(df, a, b):
    m = (df["par"] == f"{a} vs {b}") | (df["par"] == f"{b} vs {a}")
    row = df[m]
    return row.iloc[0] if len(row) else None


def main():
    tests = json.load(open(REPORT_DIR / "testes_hipotese.json"))
    mcs = json.load(open(REPORT_DIR / "mcs.json"))
    dm_df = pd.DataFrame(tests["dm_pares"])
    w_df = pd.DataFrame(tests["wilcoxon_pares"])
    ranks = tests["friedman"]["ranks"]

    n = len(ORDER)
    p_w = np.full((n, n), np.nan)
    p_dm = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = ORDER[i], ORDER[j]
            if i < j:
                row = pair_lookup(w_df, a, b)
                if row is not None:
                    p_w[i, j] = row["p_holm"]
            else:
                row = pair_lookup(dm_df, a, b)
                if row is not None:
                    p_dm[i, j] = row["p"]

    score = -np.log10(np.where(np.isnan(np.where(np.triu(np.ones((n, n)), 1) > 0, p_w, p_dm)), 1.0,
                               np.where(np.triu(np.ones((n, n)), 1) > 0, p_w, p_dm)))
    score = np.clip(score, 0, 6)

    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    im = ax.imshow(score, cmap="RdYlGn", vmin=0, vmax=6)

    for i in range(n):
        for j in range(n):
            if i == j:
                nm = ORDER[i]
                mse = mcs["mse"].get(MSE_KEY[nm], float("nan"))
                ax.text(j, i - 0.14, LABEL[nm], ha="center", va="center",
                        fontsize=13, fontweight="bold")
                ax.text(j, i + 0.22, f"MSE {mse:.2f}\nrank {ranks[nm]:.2f}",
                        ha="center", va="center", fontsize=9.5, color="#333")
                continue
            upper = i < j
            p = p_w[i, j] if upper else p_dm[i, j]
            if np.isnan(p):
                continue
            sig = p < 0.05
            if not sig and upper:
                ax.text(j, i, "NS", ha="center", va="center", fontsize=10,
                        color="#7a1010", fontweight="bold")
                continue
            txt = fmt_p(p)
            ax.text(j, i, txt, ha="center", va="center", fontsize=10.5,
                    color="#111" if sig else "#666",
                    fontweight="bold" if sig else "normal")
            if sig:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#1a7d3c", lw=2.4))

    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                   edgecolor="#ffffff", lw=1.4))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=11.5, rotation=35,
                       ha="right")
    ax.set_yticklabels([LABEL[m] for m in ORDER], fontsize=11.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.30, 0.965, "▲ triângulo superior: Wilcoxon pareado + Holm",
            transform=fig.transFigure, fontsize=11.5, fontweight="bold",
            color="#1a5c2e", va="top", ha="left")
    ax.text(0.78, 0.006, "▼ triângulo inferior: Diebold–Mariano (HLN + HAC)",
            transform=fig.transFigure, fontsize=11.5, fontweight="bold",
            color="#7a3c14", va="bottom", ha="right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("−log₁₀(p)  (verde = rejeita H₀;  NS = p ≥ 0,05)",
                   fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = IMG_DIR / f"fig_testes_hipotese.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"ok: {out}")


if __name__ == "__main__":
    if "pandas" not in sys.modules:
        import pandas as pd  # noqa: F401
    main()
