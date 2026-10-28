"""Heatmap de desvio padrão do erro por modelo e horizonte — h160.

Complemento do heatmap de MSE: para cada (modelo, horizonte), o desvio
padrão (ddof=1) dos erros assinados (previsto − real) **entre origens**
da grade rolling oficial (trim 72). Enquanto o MSE mistura viés² +
variância, este mapa isola a dispersão: célula verde = erros consistentes
entre origens; célula vermelha = erro muito variável com a origem.

Fontes: DL do banco (run 12 = lidar/reval-h160, grade já trimada) +
baselines de data/results/baselines/arima_rf_reval160/predictions_long.csv
(trimadas aqui). Mesma interseção de origens do diagrama CD.

Saída: figures_static/std_erro_horizonte.png +
data/results/report_reval_h160/std_erro_horizonte.json

    python reports/lidar/reval-h160/std_erro_horizonte.py
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

FIG_DIR = Path(__file__).resolve().parent / "figures_static"
OUT_JSON = (PROJECT_ROOT / "data" / "results" / "report_reval_h160"
            / "std_erro_horizonte.json")
BASELINE_CSV = (PROJECT_ROOT / "data" / "results" / "baselines"
                / "arima_rf_reval160" / "predictions_long.csv")
RUN_ID = 12  # lidar/reval-h160
TRIM = 72
TICKS = [5, 11, 17, 23, 29, 35]

# ordem/labels espelham o heatmap de MSE (DLs por métrica agregada +
# baselines no mesmo estilo)
DL_ORDER = [("Seq2Seq_LSTM", "LSTM"), ("Seq2Seq_LSTM_Bidirectional", "LSTM-Bi"),
            ("Seq2Seq_TCN", "TCN"), ("Seq2Seq_TCN_Bidirectional", "TCN-Bi")]
BASE_ORDER = [("arima", "ARIMA (baseline)"),
              ("rf", "Random Forest (baseline)"),
              ("persistence", "Persistência (baseline)")]


def trim_first_origins(df, n):
    origins = np.sort(df["origin"].unique())
    if len(origins) <= n:
        return df
    return df[df["origin"] > origins[n - 1]]


def load_dl(con):
    rows = pd.read_sql_query(
        """SELECT m.name, p.origin, p.timestamp, p.horizon,
                  p.actual, p.predicted
           FROM run_model_prediction p JOIN run_model m ON m.id = p.model_id
           WHERE m.run_id = ?""", con, params=(RUN_ID,))
    for col in ("timestamp", "origin"):
        rows[col] = pd.to_datetime(rows[col]).dt.round("min")
    return rows


def load_baselines():
    cache = pd.read_csv(BASELINE_CSV, parse_dates=["origin", "timestamp"])
    cache["timestamp"] = cache["timestamp"].dt.round("min")
    cache["origin"] = cache["origin"].dt.round("min")
    cache = trim_first_origins(cache, TRIM)
    long = cache.melt(id_vars=["origin", "timestamp", "horizon", "actual"],
                      value_vars=[k for k, _ in BASE_ORDER],
                      var_name="name", value_name="predicted")
    return long


def main():
    con = sqlite3.connect(PROJECT_ROOT / "data" / "predictions.db")
    dl = load_dl(con)
    base = load_baselines()
    allm = pd.concat([dl, base], ignore_index=True)
    allm["error"] = allm["predicted"] - allm["actual"]

    keys = None
    for _, grp in allm.groupby("name"):
        k = set(map(tuple, grp[["origin", "horizon"]].drop_duplicates()
                    .to_numpy()))
        keys = k if keys is None else (keys & k)
    common = allm[[(o, h) in keys for o, h in
                   zip(allm["origin"], allm["horizon"])]]

    names = [n for n, _ in DL_ORDER] + [n for n, _ in BASE_ORDER]
    labels = [l for _, l in DL_ORDER] + [l for _, l in BASE_ORDER]
    stds, means = [], []
    for n in names:
        g = common[common["name"] == n]
        piv = g.pivot_table(index="horizon", columns="origin",
                            values="error")
        stds.append(piv.std(axis=1, ddof=1).reindex(range(1, 37)).to_numpy())
        means.append(piv.mean(axis=1).reindex(range(1, 37)).to_numpy())
    mat = np.vstack(stds)

    fig, ax = plt.subplots(figsize=(9, 1.7 + 0.34 * len(labels)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=float(mat.min()), vmax=float(mat.max()),
                   interpolation="nearest")
    ax.set_xlim(-0.5, 36.6)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(labelsize=14)
    ax.set_xticks(TICKS, [str(6 * (k + 1)) for k in range(6)])
    ax.set_xlabel("Horizonte (passos de 10 min)", fontsize=15)
    ax.set_title("Desvio padrão do erro (m/s) por modelo e horizonte "
                 "(verde = erros mais consistentes entre origens)",
                 fontsize=16)
    for j in TICKS:
        for i in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=13, color="black",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              alpha=0.75, lw=0))
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Desvio padrão do erro (m/s)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "std_erro_horizonte.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    OUT_JSON.write_text(json.dumps({
        "protocolo": "Desvio padrão (ddof=1) dos erros assinados "
                     "(previsto − real) entre origens, por modelo e "
                     "horizonte; grade rolling oficial trim 72, "
                     "interseção de origens entre métodos",
        "run_id": RUN_ID, "trim_boundary_origins": TRIM,
        "n_origens": int(common[["origin"]].drop_duplicates().shape[0]),
        "modelos": labels,
        "std_por_horizonte": {l: [round(float(v), 4) for v in s]
                              for l, s in zip(labels, stds)},
        "vies_por_horizonte": {l: [round(float(v), 4) for v in s]
                               for l, s in zip(labels, means)},
    }, indent=1), encoding="utf-8")
    print(f"-> {FIG_DIR / 'std_erro_horizonte.png'}")
    print(f"-> {OUT_JSON}")
    for l, s in zip(labels, stds):
        print(f"{l:28s} std h1 {s[0]:.2f} | h12 {s[11]:.2f} | "
              f"h24 {s[23]:.2f} | h36 {s[35]:.2f}")


if __name__ == "__main__":
    main()
