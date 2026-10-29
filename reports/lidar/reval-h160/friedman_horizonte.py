"""Friedman POR HORIZONTE + CD (Wilcoxon-Holm) — rodada h160 (reval).

Mesmo analysis do `reports/lidar/reval_sbsr/friedman_horizonte.py`
apontado para a rodada do paper (perfil 40–160 m): Friedman por horizonte
(blocos = origens, perda = RMSE no par origem×h, em m/s) com Holm entre
os 36 testes + diagrama de diferença crítica agregado com agrupamento
Wilcoxon-Holm (pares não-separáveis por Wilcoxon signed-rank pareado por
origem com Holm entre os pares formam cliques ligados por barras).
O RMSE é raiz quadrada da perda quadrática — monotônica por bloco, os
ranks coincidem com os do critério MSE, mas os valores ficam em m/s,
comparáveis às tabelas do artigo.

Fontes: DL do banco (run 12 = lidar/reval-h160, grade já trimada) +
baselines de data/results/baselines/arima_rf_reval160/predictions_long.csv
(trimadas aqui).

Saída: figures_static/{friedman_horizonte,critical_difference}.png +
data/results/report_reval_h160/friedman_horizonte.json

    python reports/lidar/reval-h160/friedman_horizonte.py
"""

import itertools
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures_static"
OUT_JSON = (PROJECT_ROOT / "data" / "results" / "report_reval_h160"
            / "friedman_horizonte.json")
BASELINE_CSV = (PROJECT_ROOT / "data" / "results" / "baselines"
                / "arima_rf_reval160" / "predictions_long.csv")
RUN_ID = 12
TRIM = 72
HORIZONS = range(1, 37)
ALPHA = 0.05

plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25,
})


def load_models(con):
    rows = pd.read_sql_query(
        """SELECT m.name, p.origin, p.timestamp, p.horizon,
                  p.actual, p.predicted
           FROM run_model_prediction p JOIN run_model m ON m.id = p.model_id
           WHERE m.run_id = ?""", con, params=(RUN_ID,))
    rows["timestamp"] = pd.to_datetime(rows["timestamp"]).dt.round("min")
    rows["origin"] = pd.to_datetime(rows["origin"]).dt.round("min")
    rows["loss"] = (rows["actual"] - rows["predicted"]) ** 2
    return rows


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


def critical_difference(common, names, labels, colors):
    """Diferença crítica com agrupamento Wilcoxon-Holm (blocos = origens,
    perda = RMSE nos 36 passos)."""
    wide = common.pivot_table(index="origin", columns="name", values="loss")
    wide = np.sqrt(wide.dropna())  # MSE médio por origem -> RMSE (m/s)
    k = len(names)
    n = len(wide)
    ranks = wide[names].rank(axis=1).mean()
    fr = friedmanchisquare(*[wide[n].to_numpy() for n in names])
    order = ranks.sort_values()
    order.index = [labels[x] for x in order.index]

    # post-hoc: Wilcoxon signed-rank pareado por origem + Holm entre os
    # k(k-1)/2 pares; p_holm >= alpha => par estatisticamente
    # indistinguível => aresta no grafo de não-separabilidade
    pairs = []
    for a, b in itertools.combinations(names, 2):
        diff = wide[a] - wide[b]
        p = 1.0 if (diff == 0).all() else float(wilcoxon(diff).pvalue)
        pairs.append((a, b, p))
    holm_idx = np.argsort([p for _, _, p in pairs])
    m = len(pairs)
    holm = np.empty(m)
    running = 0.0
    for i, idx in enumerate(holm_idx):
        running = max(running, min(1.0, (m - i) * pairs[idx][2]))
        holm[idx] = running
    edges = [(labels[a], labels[b]) for (a, b, _), ph in zip(pairs, holm)
             if ph >= ALPHA]
    groups = maximal_cliques(list(order.index), edges)

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    lo, hi = order.iloc[0], order.iloc[-1]
    pad = 0.4
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0, 1)
    ax.axvspan(lo - pad, hi + pad, color="white")
    y = 0.78
    for name in order.index:
        ax.plot(order[name], y, "o", ms=7, color=colors[name],
                markeredgecolor="k", markeredgewidth=0.6, zorder=3)
        ax.annotate(f"{name} ({order[name]:.2f})", (order[name], y),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=colors[name])
    level = 0.45
    for g in sorted(groups, key=len):
        xs = [order[x] for x in g]
        ax.plot([min(xs), max(xs)], [level, level], lw=2.2, color="#222222",
                solid_capstyle="butt", zorder=2)
        level -= 0.13
    ax.axis("off")
    ax.set_title("Diferença crítica — agrupamento Wilcoxon-Holm "
                 f"(α = {str(ALPHA).replace('.', ',')}; blocos = {n} "
                 "origens; perda = RMSE nos 36 passos, m/s)",
                 fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "critical_difference.png", dpi=140)
    plt.close(fig)
    return {"alpha": ALPHA, "k": k, "n_origens": int(n),
            "friedman": {"chi2": float(fr.statistic), "p": float(fr.pvalue)},
            "mean_ranks": {labels[n]: float(ranks[n]) for n in names},
            "pares": [{"par": f"{labels[a]} vs {labels[b]}", "p": float(p),
                       "p_holm": float(ph)}
                      for (a, b, p), ph in zip(pairs, holm)],
            "grupos_wilcoxon_holm": [list(g) for g in groups]}


def main():
    con = sqlite3.connect(PROJECT_ROOT / "data" / "predictions.db")
    dl = load_models(con)
    cache = pd.read_csv(BASELINE_CSV, parse_dates=["origin", "timestamp"])
    cache["timestamp"] = cache["timestamp"].dt.round("min")
    cache["origin"] = cache["origin"].dt.round("min")
    origins = np.sort(cache["origin"].unique())
    cache = cache[cache["origin"] > origins[TRIM - 1]] if len(origins) > TRIM else cache
    cache_long = cache.melt(
        id_vars=["origin", "timestamp", "horizon", "actual"],
        value_vars=["arima", "persistence"],
        var_name="name", value_name="predicted")
    cache_long["loss"] = (cache_long["actual"]
                          - cache_long["predicted"]) ** 2
    allm = pd.concat([dl[["name", "origin", "timestamp", "horizon", "loss"]],
                      cache_long[["name", "origin", "timestamp", "horizon",
                                  "loss"]]])

    # grade comum: pares (origin, horizon) presentes em TODOS os modelos
    keys = None
    for name, grp in allm.groupby("name"):
        k = set(map(tuple, grp[["origin", "horizon"]].drop_duplicates()
                    .to_numpy()))
        keys = k if keys is None else (keys & k)
    common = allm[[(o, h) in keys for o, h in
                   zip(allm["origin"], allm["horizon"])]]
    n_origins = common["origin"].nunique()

    names = ["Seq2Seq_LSTM", "Seq2Seq_TCN_Bidirectional", "Seq2Seq_TCN",
             "Seq2Seq_LSTM_Bidirectional", "arima", "persistence"]
    labels = {"Seq2Seq_LSTM": "LSTM", "Seq2Seq_TCN_Bidirectional": "BiTCN",
              "Seq2Seq_TCN": "TCN", "Seq2Seq_LSTM_Bidirectional": "BiLSTM",
              "arima": "ARIMA", "persistence": "Pers."}
    colors = {"LSTM": "#1f77b4", "BiLSTM": "#9467bd", "TCN": "#2ca02c",
              "BiTCN": "#d62728", "ARIMA": "#7f7f7f",
              "Pers.": "#000000"}

    per_h = []
    for h in HORIZONS:
        sub = common[common["horizon"] == h]
        wide = sub.pivot_table(index="origin", columns="name", values="loss")
        wide = np.sqrt(wide.dropna())  # RMSE (m/s) no par (origem, h)
        stat, p = friedmanchisquare(*[wide[n].to_numpy() for n in names])
        ranks = wide[names].rank(axis=1).mean()
        row = {"horizon": h, "chi2": float(stat), "p": float(p),
               "n_origins": int(len(wide)),
               **{f"rank_{labels[n]}": float(ranks[n]) for n in names},
               **{f"rmse_{labels[n]}": float(wide[n].mean()) for n in names}}
        per_h.append(row)

    # Holm across horizons (36 testes)
    ps = np.array([r["p"] for r in per_h])
    order = np.argsort(ps)
    m = len(ps)
    holm = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = min(1.0, (m - i) * ps[idx])
        running = max(running, val)
        holm[idx] = running
    for r, ph in zip(per_h, holm):
        r["p_holm"] = float(ph)

    OUT_JSON.write_text(json.dumps({
        "protocolo": "Friedman por horizonte; blocos = origens; perda = "
                     "RMSE (m/s) no par (origem, h); grade comum (trim 72) "
                     "da rodada reval 40–160 m",
        "run_id": RUN_ID, "n_origens": n_origins,
        "modelos": [labels[n] for n in names],
        "por_horizonte": per_h,
        "wilcoxon_holm": critical_difference(common, names, labels, colors),
    }, indent=1), encoding="utf-8")

    df = pd.DataFrame(per_h)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)
    for n in names:
        lab = labels[n]
        ax1.plot(df["horizon"], df[f"rmse_{lab}"], lw=1.7,
                 color=colors[lab], ls="--" if lab == "Pers." else "-",
                 label=lab)
    ax1.set_ylabel("RMSE médio no horizonte (m/s)")
    ax1.set_yscale("log")
    ax1.set_title("Rodada 40–160 m (reval 72,5/12,5/15) — RMSE por "
                  "horizonte e rank Friedman")
    ax1.legend(ncol=3, loc="upper left")
    for n in names:
        lab = labels[n]
        ax2.plot(df["horizon"], df[f"rank_{lab}"], lw=1.7,
                 color=colors[lab], ls="--" if lab == "Pers." else "-",
                 label=lab)
    ax2.axhline(1, color="#bbbbbb", lw=0.8)
    ax2.axhline(len(names), color="#bbbbbb", lw=0.8)
    ax2.set_ylabel("Rank médio de Friedman")
    ax2.set_xlabel("Horizonte (passos de 10 min)")
    ax2.set_ylim(0.5, len(names) + 0.5)
    ax2.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "friedman_horizonte.png", dpi=140)
    plt.close(fig)

    sig = [r["horizon"] for r in per_h if r["p_holm"] < 0.05]
    cd = json.loads(OUT_JSON.read_text())["wilcoxon_holm"]
    print(f"-> {FIG_DIR / 'friedman_horizonte.png'}")
    print(f"-> {FIG_DIR / 'critical_difference.png'}")
    print(f"-> {OUT_JSON}")
    print(f"Friedman por horizonte: {len(sig)}/36 significativos após Holm")
    print("Friedman agregado (6 métodos, RMSE por origem): "
          f"chi2 = {cd['friedman']['chi2']:.1f}, "
          f"p = {cd['friedman']['p']:.2g}")
    print("Grupos Wilcoxon-Holm:", json.dumps(cd["grupos_wilcoxon_holm"]))


if __name__ == "__main__":
    main()
