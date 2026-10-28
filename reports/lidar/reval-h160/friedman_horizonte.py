"""Friedman POR HORIZONTE + Nemenyi — rodada h160 (reval).

Mesmo analysis do `reports/lidar/reval_sbsr/friedman_horizonte.py`
apontado para a rodada do paper (perfil 40–160 m): Friedman por horizonte
(blocos = origens, perda = MSE no par origem×h) com Holm entre os 36
testes + diagrama de diferença crítica de Nemenyi agregado.

Fontes: DL do banco (run 12 = lidar/reval-h160, grade já trimada) +
baselines de data/results/baselines/arima_rf_reval160/predictions_long.csv
(trimadas aqui).

Saída: figures_static/{friedman_horizonte,critical_difference}.png +
data/results/report_reval_h160/friedman_horizonte.json

    python reports/lidar/reval-h160/friedman_horizonte.py
"""

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
from scipy.stats import friedmanchisquare  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures_static"
OUT_JSON = (PROJECT_ROOT / "data" / "results" / "report_reval_h160"
            / "friedman_horizonte.json")
BASELINE_CSV = (PROJECT_ROOT / "data" / "results" / "baselines"
                / "arima_rf_reval160" / "predictions_long.csv")
RUN_ID = 12
TRIM = 72
HORIZONS = range(1, 37)

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


# q de Studentized range / sqrt(2) para alpha=0.05 (tabela Nemenyi)
NEMENYI_Q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
             8: 3.031, 9: 3.102, 10: 3.164}


def critical_difference(common, names, labels, colors):
    """CD de Nemenyi sobre blocos = origens (perda = MSE médio nos 36 passos)."""
    wide = common.pivot_table(index="origin", columns="name", values="loss")
    wide = wide.dropna()
    k = len(names)
    n = len(wide)
    ranks = wide[names].rank(axis=1).mean()
    cd = NEMENYI_Q[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    order = ranks.sort_values()
    order.index = [labels[x] for x in order.index]
    groups = []
    current = [order.index[0]]
    for i in range(1, k):
        if order.iloc[i] - order[current[0]] <= cd:
            current.append(order.index[i])
        else:
            groups.append(current)
            current = [order.index[i]]
    groups.append(current)

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    lo, hi = order.iloc[0], order.iloc[-1]
    pad = 0.4
    ax.set_xlim(min(lo - pad, lo - cd), max(hi + pad, hi + cd))
    ax.set_ylim(0, 1)
    ax.axvspan(min(lo - pad, lo - cd), max(hi + pad, hi + cd), color="white")
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
    ax.plot([lo - cd, lo - cd], [0.15, 0.9], ls="--", lw=1.2, color="#888888")
    ax.annotate(f"CD = {cd:.2f}", (lo - cd, 0.08), ha="center", fontsize=8.5,
                color="#555555")
    ax.axis("off")
    ax.set_title("Diferença crítica de Nemenyi (α = 0,05; blocos = "
                 f"{n} origens; perda = MSE médio nos 36 passos)",
                 fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "critical_difference.png", dpi=140)
    plt.close(fig)
    return {"cd": float(cd), "k": k, "n_origens": int(n),
            "mean_ranks": {labels[n]: float(ranks[n]) for n in names},
            "grupos_nao_separaveis": [list(g) for g in groups]}


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
    labels = {"Seq2Seq_LSTM": "LSTM", "Seq2Seq_TCN_Bidirectional": "TCN-Bi",
              "Seq2Seq_TCN": "TCN", "Seq2Seq_LSTM_Bidirectional": "LSTM-Bi",
              "arima": "ARIMA", "persistence": "Persistência"}
    colors = {"LSTM": "#1f77b4", "LSTM-Bi": "#9467bd", "TCN": "#2ca02c",
              "TCN-Bi": "#d62728", "ARIMA": "#7f7f7f",
              "Persistência": "#000000"}

    per_h = []
    for h in HORIZONS:
        sub = common[common["horizon"] == h]
        wide = sub.pivot_table(index="origin", columns="name", values="loss")
        wide = wide.dropna()
        stat, p = friedmanchisquare(*[wide[n].to_numpy() for n in names])
        ranks = wide[names].rank(axis=1).mean()
        row = {"horizon": h, "chi2": float(stat), "p": float(p),
               "n_origins": int(len(wide)),
               **{f"rank_{labels[n]}": float(ranks[n]) for n in names},
               **{f"mse_{labels[n]}": float(wide[n].mean()) for n in names}}
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
                     "MSE no par (origem, h); grade comum (trim 72) da "
                     "rodada reval 40–160 m",
        "run_id": RUN_ID, "n_origens": n_origins,
        "modelos": [labels[n] for n in names],
        "por_horizonte": per_h,
        "nemenyi": critical_difference(common, names, labels, colors),
    }, indent=1), encoding="utf-8")

    df = pd.DataFrame(per_h)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)
    for n in names:
        lab = labels[n]
        ax1.plot(df["horizon"], df[f"mse_{lab}"], lw=1.7,
                 color=colors[lab], ls="--" if lab == "Persistência" else "-",
                 label=lab)
    ax1.set_ylabel("MSE médio no horizonte")
    ax1.set_yscale("log")
    ax1.set_title("Rodada 40–160 m (reval 72,5/12,5/15) — MSE por "
                  "horizonte e rank Friedman")
    ax1.legend(ncol=3, loc="upper left")
    for n in names:
        lab = labels[n]
        ax2.plot(df["horizon"], df[f"rank_{lab}"], lw=1.7,
                 color=colors[lab], ls="--" if lab == "Persistência" else "-",
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
    print(f"-> {FIG_DIR / 'friedman_horizonte.png'}")
    print(f"-> {FIG_DIR / 'critical_difference.png'}")
    print(f"-> {OUT_JSON}")
    print(f"Friedman por horizonte: {len(sig)}/36 significativos após Holm")
    print("Nemenyi CD:", json.dumps(
        json.loads(OUT_JSON.read_text())["nemenyi"]["grupos_nao_separaveis"]))


if __name__ == "__main__":
    main()
