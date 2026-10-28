"""Friedman pareado por horizonte — revalidação SBSR (72,5/12,5/15).

Para cada horizonte h=1..36, testa H₀ "ranks médios iguais" entre os 4
modelos DL + ARIMA + persistência, com blocos = origens e perda = MSE no
par (origem, h). Predições DL do banco (run 7, data/predictions.db);
ARIMA/persistência do cache data/results/reval_sbsr/arima_test_cache.csv.
Saída: JSON agregado + figura (MSE por horizonte e -log10 p com Holm).

    python reports/lidar/reval_sbsr/friedman_horizonte.py
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
# NB: figures/ é WIPADA a cada build (resultados.py recopia de images/);
# figuras fora do tema padrão vão em figures_static/.
OUT_JSON = (PROJECT_ROOT / "data" / "results" / "report_reval_sbsr"
            / "friedman_horizonte.json")
CACHE = PROJECT_ROOT / "data" / "results" / "reval_sbsr" / "arima_test_cache.csv"
RUN_ID = 7
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
    wide = common.pivot_table(index="origin", columns="name",
                              values="loss", aggfunc="mean").dropna()
    n = len(wide)
    k = len(names)
    ranks = wide[names].rank(axis=1)
    mean_ranks = {labels[nm]: float(ranks[nm].mean()) for nm in names}
    cd = NEMENYI_Q[k] * np.sqrt(k * (k + 1) / (6 * n))

    order = sorted(mean_ranks, key=mean_ranks.get)
    vals = [mean_ranks[m] for m in order]
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i:
            g = tuple(order[i:j + 1])
            if g not in groups:
                groups.append(g)

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ys = np.arange(k)[::-1]
    ax.set_xlim(min(vals) - 0.35, max(vals) + 0.35)
    ax.set_ylim(-1.6, k - 0.2)
    ax.set_yticks([])
    for spine in ("left", "right"):
        ax.spines[spine].set_visible(False)
    # eixo de ranks no topo (melhor à esquerda)
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(sorted(range(1, k + 1)))
    ax_top.set_xlabel("rank médio (1 = melhor)")
    for y, m in zip(ys, order):
        ax.plot(mean_ranks[m], y, "o", ms=7, color=colors[m],
                markeredgecolor="k", markeredgewidth=0.6, zorder=5)
        side_left = y % 2 == 0
        x_txt = mean_ranks[m] - (0.06 if side_left else -0.06)
        ax.annotate(m, (x_txt, y), ha="right" if side_left else "left",
                    va="center", fontsize=9)
    # barras de grupos não-separáveis (Nemenyi)
    level = -0.55
    for g in groups:
        xs = [mean_ranks[m] for m in g]
        ax.plot([min(xs), max(xs)], [level, level], lw=3.5,
                color="#555555", solid_capstyle="butt", zorder=3)
        for m in g:
            ax.plot([mean_ranks[m], mean_ranks[m]], [level, level + 0.12],
                    lw=1.2, color="#555555", zorder=3)
        level -= 0.42
    ax.set_title(f"Diagrama de diferença crítica (Nemenyi, α = 0,05) — "
                 f"CD = {cd:.3f}, {n} origens")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "critical_difference.png")
    plt.close(fig)
    return {"cd": float(cd), "n_origins": int(n), "k": k,
            "mean_ranks": mean_ranks,
            "grupos_nao_separaveis": [list(g) for g in groups]}


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
                     "MSE no par (origem, h); grade comum (trim) da "
                     "revalidação 72,5/12,5/15",
        "n_origens": n_origins, "modelos": [labels[n] for n in names],
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
    ax1.set_title("Revalidação SBSR 72,5/12,5/15 — MSE por horizonte "
                  "(grade comum, trim)")
    ax1.legend(ncol=3, loc="upper left")
    ax2.semilogy(df["horizon"], np.maximum(-np.log10(np.maximum(df["p"], 1e-300)), 0),
                 color="#1f77b4", lw=1.8, label="−log₁₀(p bruto)")
    ax2.semilogy(df["horizon"], np.maximum(-np.log10(np.maximum(df["p_holm"], 1e-300)), 0),
                 color="#d62728", lw=1.4, ls="--", label="−log₁₀(p Holm)")
    ax2.axhline(-np.log10(0.05), color="k", lw=1, ls=":",
                label="α = 0,05")
    sig = df[df["p_holm"] >= 0.05]["horizon"].tolist()
    ax2.set_ylabel("−log₁₀(p)")
    ax2.set_xlabel("horizonte (passos de 10 min)")
    ax2.set_title("Friedman pareado por horizonte "
                  f"({n_origins} origens × 6 métodos)")
    ax2.legend(ncol=3, loc="lower right")
    if sig:
        ax2.annotate(f"H₀ não rejeitada (Holm) nos horizontes {sig}",
                     xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "friedman_horizonte.png")
    plt.close(fig)

    nsig = int((df["p_holm"] < 0.05).sum())
    cd_info = json.loads(OUT_JSON.read_text(encoding="utf-8"))["nemenyi"]
    print(f"origens na grade comum: {n_origins} | horizontes com H₀ "
          f"rejeitada (Holm 5%): {nsig}/36")
    print("não significantes:", sig or "nenhum")
    print(f"Nemenyi: CD={cd_info['cd']:.3f} | grupos não-separáveis: "
          f"{[' < '.join(g) for g in cd_info['grupos_nao_separaveis']]}")
    print(f"-> {OUT_JSON}")
    print(f"-> {FIG_DIR / 'friedman_horizonte.png'}")
    print(f"-> {FIG_DIR / 'critical_difference.png'}")


if __name__ == "__main__":
    main()
