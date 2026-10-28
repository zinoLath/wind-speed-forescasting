"""Diagrama de Diferença Crítica (Wilcoxon-Holm) POR HORIZONTE — reval SBSR.

Réplica do estilo do diagrama de referência (Wilcoxon-Holm, pontos no eixo
de ranks com grupos ligados por barras), calculado para cada horizonte
h = 1..36 da revalidação 72,5/12,5/15: blocos = origens, perda = MSE no
par (origem, h); pares não-significativos (Wilcoxon signed-rank + Holm
dentro do horizonte) formam cliques ligados por barras.

Saída: figures_static/cd_horizonte.png (grade 6×6) +
data/results/report_reval_sbsr/cd_horizonte.json

    python reports/lidar/reval_sbsr/cd_horizonte.py
"""

import itertools
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from friedman_horizonte import load_models  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures_static"
OUT_JSON = (PROJECT_ROOT / "data" / "results" / "report_reval_sbsr"
            / "cd_horizonte.json")
CACHE = PROJECT_ROOT / "data" / "results" / "reval_sbsr" / "arima_test_cache.csv"
RUN_ID = 7
HORIZONS = list(range(1, 37))
ALPHA = 0.05

NAMES = ["Seq2Seq_LSTM", "Seq2Seq_TCN_Bidirectional", "Seq2Seq_TCN",
         "Seq2Seq_LSTM_Bidirectional", "arima", "persistence"]
LABELS = {"Seq2Seq_LSTM": "LSTM", "Seq2Seq_TCN_Bidirectional": "TCN-Bi",
          "Seq2Seq_TCN": "TCN", "Seq2Seq_LSTM_Bidirectional": "LSTM-Bi",
          "arima": "ARIMA", "persistence": "Persistência"}
SHORT = {"LSTM": "LSTM", "LSTM-Bi": "L-Bi", "TCN": "TCN", "TCN-Bi": "T-Bi",
         "ARIMA": "ARI", "Persistência": "P"}
COLORS = {"LSTM": "#1f77b4", "LSTM-Bi": "#9467bd", "TCN": "#2ca02c",
          "TCN-Bi": "#d62728", "ARIMA": "#7f7f7f",
          "Persistência": "#000000"}


def maximal_cliques(nodes, edges):
    """Cliques maximais (grafo pequeno: força bruta sobre subconjuntos)."""
    edges = set(edges)
    cliques = []
    for size in range(len(nodes), 1, -1):
        for sub in itertools.combinations(nodes, size):
            pairs = set(itertools.combinations(sub, 2))
            if pairs <= edges and not any(set(c) > set(sub) for c in cliques):
                cliques.append(sub)
    return cliques


def main():
    con = sqlite3.connect(PROJECT_ROOT / "data" / "predictions.db")
    dl = load_models(con)
    cache = pd.read_csv(CACHE, parse_dates=["origin", "timestamp"])
    cache["timestamp"] = cache["timestamp"].dt.round("min")
    cache["origin"] = cache["origin"].dt.round("min")
    cache_long = cache.melt(
        id_vars=["origin", "timestamp", "horizon", "actual"],
        value_vars=["arima", "persistence"],
        var_name="name", value_name="predicted")
    cache_long["loss"] = (cache_long["actual"]
                          - cache_long["predicted"]) ** 2
    allm = pd.concat([dl[["name", "origin", "horizon", "loss"]],
                      cache_long[["name", "origin", "horizon", "loss"]]])

    keys = None
    for _, grp in allm.groupby("name"):
        k = set(map(tuple, grp[["origin", "horizon"]].drop_duplicates()
                    .to_numpy()))
        keys = k if keys is None else (keys & k)
    common = allm[[(o, h) in keys for o, h in
                   zip(allm["origin"], allm["horizon"])]]

    def compute(h):
        wide = (common[common["horizon"] == h]
                .pivot_table(index="origin", columns="name", values="loss")
                .dropna())
        ranks = wide[NAMES].rank(axis=1)
        mean_r = {LABELS[n]: float(ranks[n].mean()) for n in NAMES}
        pairs, pvals = [], []
        for a, b in itertools.combinations(NAMES, 2):
            diff = wide[a] - wide[b]
            p = 1.0 if (diff == 0).all() else float(wilcoxon(diff).pvalue)
            pairs.append((LABELS[a], LABELS[b]))
            pvals.append(p)
        order = np.argsort(pvals)
        m = len(pvals)
        holm = np.empty(m)
        run = 0.0
        for i, idx in enumerate(order):
            run = max(run, min(1.0, (m - i) * pvals[idx]))
            holm[idx] = run
        edges = [(a, b) for (a, b), ph in zip(pairs, holm) if ph >= ALPHA]
        return {"wide": wide, "mean_r": mean_r,
                "groups": maximal_cliques(list(mean_r), edges),
                "n_origins": int(len(wide))}

    def draw_panel(ax, h, info, scale=1.0, compact=False, tight=False):
        mean_r, groups = info["mean_r"], info["groups"]
        vals = sorted(mean_r.items(), key=lambda kv: kv[1])
        lo, hi = vals[0][1], vals[-1][1]
        span = max(hi - lo, 0.5)
        # ticks ANTES do xlim: set_xticks depois re-expande o eixo
        ax.set_yticks([])
        ax.xaxis.tick_top()
        ax.set_xticks(range(1, 7))
        ax.tick_params(axis="x", length=2, pad=1, labelsize=5.5 * scale)
        for s in ("left", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["top"].set_bounds(lo - span * 0.08, hi + span * 0.08)
        if tight:
            # todos entre 3 e 4: eixo colado nos dados + rebarba única
            ax.set_xlim(lo - 0.15, hi + 0.15)
        else:
            ax.set_xlim(lo - span * 0.55, hi + span * 0.55)
        if compact:
            if tight:
                ax.set_ylim(-2.35, 0.32)
            else:
                ax.set_ylim(-1.78, 0.28)
            bar_start, bar_step = (-0.16, 0.13) if tight else (-0.13, 0.10)
            left_levels = ([-0.58, -1.12, -1.66] if tight
                           else [-0.48, -0.86, -1.24])
            right_levels = ([-0.85, -1.39, -1.93] if tight
                            else [-0.67, -1.05, -1.43])
        else:
            ax.set_ylim(-2.15, 0.35)
            bar_start, bar_step = -0.16, 0.14
            left_levels = [-0.62, -1.12, -1.62]
            right_levels = [-0.87, -1.37, -1.87]
        ax.set_title(f"h = {h}", fontsize=7.5 * scale,
                     pad=10 if compact else 12)

        level = bar_start
        for g in sorted(groups, key=len):
            xs = [mean_r[x] for x in g]
            ax.plot([min(xs), max(xs)], [level, level], lw=2.0,
                    color="#222222", solid_capstyle="butt", zorder=2)
            level -= bar_step
        li = ri = 0
        for name, r in vals:
            if li <= ri:
                y, side = left_levels[min(li, 2)], "left"
                li += 1
            else:
                y, side = right_levels[min(ri, 2)], "right"
                ri += 1
            ax.plot([r, r], [0, y], lw=1.2, color=COLORS[name], zorder=1)
            if tight:
                ax.annotate(f"{SHORT[name]} ({r:.2f})", (r, y),
                            xytext=(-6 if side == "left" else 6, 0),
                            textcoords="offset points",
                            ha="right" if side == "left" else "left",
                            va="center", fontsize=12,
                            color=COLORS[name], clip_on=False, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="none", alpha=0.8))
            else:
                x_end = (lo - span * 0.06 if side == "left"
                         else hi + span * 0.06)
                ax.plot([r, x_end], [y, y], lw=0.9, color=COLORS[name],
                        zorder=1)
                ax.annotate(f"{SHORT[name]} ({r:.2f})",
                            (x_end + (-span * 0.03 if side == "left"
                                      else span * 0.03), y),
                            ha="right" if side == "left" else "left",
                            va="center", fontsize=5.8 * scale,
                            color=COLORS[name])
            ax.plot(r, 0, "o", ms=6 if tight else 4, color=COLORS[name],
                    markeredgecolor="k", markeredgewidth=0.5, zorder=5)

    by_h = {h: compute(h) for h in HORIZONS}
    summaries = [{"horizon": h, "n_origins": by_h[h]["n_origins"],
                  "mean_ranks": by_h[h]["mean_r"],
                  "grupos_wilcoxon_holm": [list(g) for g in by_h[h]["groups"]]}
                 for h in HORIZONS]

    # grade completa 6×6 (todos os horizontes)
    fig, axes = plt.subplots(6, 6, figsize=(13, 17))
    for ax, h in zip(axes.flat, HORIZONS):
        draw_panel(ax, h, by_h[h], scale=1.0)

    fig.suptitle("Diagrama de Diferença Crítica (Wilcoxon-Holm) por "
                 "horizonte — revalidação SBSR 72,5/12,5/15 "
                 "(blocos = origens; barras ligam métodos não-separáveis)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "cd_horizonte.png", dpi=140)
    plt.close(fig)

    # seleção 3×2 (h = 6, 12, 18, 24, 30, 36) — largura do heatmap MSE (9"),
    # fontes no mesmo nível (rótulos ~12pt, ticks ~11pt, título ~15pt)
    sel = (6, 12, 18, 24, 30, 36)
    fig, axes = plt.subplots(2, 3, figsize=(9, 7.6))
    for ax, h in zip(axes.flat, sel):
        draw_panel(ax, h, by_h[h], scale=2.0, compact=True, tight=True)
    fig.suptitle("Diagrama de Diferença Crítica (Wilcoxon-Holm)",
                 fontsize=16)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.04,
                        wspace=0.95)
    fig.savefig(FIG_DIR / "cd_horizonte_selecionados.png", dpi=140)
    plt.close(fig)

    OUT_JSON.write_text(json.dumps({
        "protocolo": "CD por horizonte com agrupamento Wilcoxon-Holm: "
                     "pares com p_holm >= 0.05 (Wilcoxon signed-rank por "
                     "origem, Holm dentro do horizonte) formam cliques "
                     "ligados por barras",
        "alpha": ALPHA, "por_horizonte": summaries,
    }, indent=1), encoding="utf-8")
    print(f"-> {FIG_DIR / 'cd_horizonte.png'}")
    print(f"-> {OUT_JSON}")
    for s in (summaries[0], summaries[17], summaries[35]):
        gs = [" < ".join(g) for g in s["grupos_wilcoxon_holm"]]
        print(f"h={s['horizon']:>2}: {' | '.join(gs) or 'sem grupos'}")


if __name__ == "__main__":
    main()
