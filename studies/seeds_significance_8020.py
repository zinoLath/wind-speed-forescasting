"""Bateria de significância estatística sem retreino (modelos canônicos 80/20).

Responde, com os dados já em disco:

- **Wilcoxon pareado por seed** (n = 10, seeds 42–51): LSTM × LSTM-Bi
  (ITILDE-Q) e cada um vs persistência, em R²/MSE all e rolling
  (`data/results/itildeq_seeds_8020/predictions_{all,rolling}.csv.gz`);
  tamanho de efeito rank-biserial pareado.
- **Friedman sobre blocos diários** (papel do Kruskal–Wallis com blocos
  compartilhados) entre as 4 canônicas (seed 42, dense, predictions.db),
  com pós-hoc Wilcoxon pareado por dia + correção de Holm.
- **KSPA** (Kolmogorov–Smirnov Predictive Accuracy) para os 6 pares de
  canônicas sobre |erro|, com p-valor por bootstrap em blocos de origens
  (blocos pareados entre modelos).

Não precisa de GPU:

    python tests/seeds_significance_8020.py \
        --seeds-dir data/results/itildeq_seeds_8020 \
        --db data/predictions.db --run-id 1 \
        --out-dir data/results/itildeq_seeds_8020/significance
"""

import argparse
import json
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
BOOT_BLOCK = 144
N_BOOT = 2000
RNG_SEED = 20261022
MODEL_ORDER = ["lstm", "lstm_bi", "tcn", "tcn_bi"]
LABELS = {"lstm": "LSTM", "lstm_bi": "LSTM-Bi", "tcn": "TCN", "tcn_bi": "TCN-Bi"}


def compute_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_mse(actual, predicted):
    return float(np.mean((actual - predicted) ** 2))


def rank_biserial_paired(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(diffs))
    wp = ranks[diffs > 0].sum()
    wm = ranks[diffs < 0].sum()
    return float((wp - wm) / (wp + wm))


def wilcoxon_paired(diffs, label):
    diffs = np.asarray(diffs, dtype=float)
    res = stats.wilcoxon(diffs, alternative="two-sided", method="exact")
    return {
        "comparison": label,
        "n_pairs": int(len(diffs)),
        "median_diff": float(np.median(diffs)),
        "mean_diff": float(np.mean(diffs)),
        "wins": int(np.sum(diffs > 0)),
        "losses": int(np.sum(diffs < 0)),
        "ties": int(np.sum(diffs == 0)),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "rank_biserial": rank_biserial_paired(diffs),
        "significant_05": bool(res.pvalue < ALPHA),
    }


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = pvals[idx] * (m - rank)
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


def ks_statistic(x, y):
    x = np.sort(x)
    y = np.sort(y)
    all_vals = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, all_vals, side="right") / len(x)
    cdf_y = np.searchsorted(y, all_vals, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def block_bootstrap_indices(n, block, n_boot, rng):
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)[None, None, :]
    idx = (starts[:, :, None] + offsets) % n
    return idx.reshape(n_boot, -1)[:, :n]


def kspa_matrix(err_by_origin):
    n, width = err_by_origin.shape
    rng = np.random.default_rng(RNG_SEED)
    idx = block_bootstrap_indices(n, BOOT_BLOCK, N_BOOT, rng)
    return err_by_origin[idx, :].reshape(N_BOOT, -1)


def kspa_pair(err_a, err_b):
    obs = ks_statistic(err_a.ravel(), err_b.ravel())
    boots_a = kspa_matrix(err_a)
    boots_b = kspa_matrix(err_b)
    boot_stats = np.array([
        ks_statistic(boots_a[i], boots_b[i]) for i in range(N_BOOT)
    ])
    p_value = float(np.mean(boot_stats >= obs))
    return {
        "ks": obs,
        "p_value": p_value,
        "significant_05": bool(p_value < ALPHA),
    }


def load_seed_predictions(seeds_dir):
    all_df = pd.read_csv(seeds_dir / "predictions_all.csv.gz",
                         parse_dates=["origin", "timestamp"])
    roll_df = pd.read_csv(seeds_dir / "predictions_rolling.csv.gz",
                          parse_dates=["origin", "timestamp"])
    return all_df, roll_df


def seed_family_metrics(all_df, roll_df, wrapper):
    rows = []
    a = all_df[all_df["wrapper"] == wrapper]
    r = roll_df[roll_df["wrapper"] == wrapper]
    for seed in sorted(a["seed"].unique()):
        a_s = a[a["seed"] == seed]
        r_s = r[r["seed"] == seed]
        pers = a_s[a_s["horizon"] == a_s["horizon"].max()].drop_duplicates(
            "origin").set_index("origin")["persistence"]
        r_s = r_s.copy()
        r_s["persistence"] = r_s["origin"].map(pers)
        r_valid = r_s[r_s["persistence"].notna()]
        rows.append({
            "seed": int(seed),
            "r2": compute_r2(a_s["actual"].to_numpy(),
                             a_s["predicted"].to_numpy()),
            "mse": compute_mse(a_s["actual"].to_numpy(),
                               a_s["predicted"].to_numpy()),
            "persistence_r2": compute_r2(a_s["actual"].to_numpy(),
                                         a_s["persistence"].to_numpy()),
            "persistence_mse": compute_mse(a_s["actual"].to_numpy(),
                                           a_s["persistence"].to_numpy()),
            "rolling_r2": compute_r2(r_s["actual_raw"].to_numpy(),
                                     r_s["predicted"].to_numpy()),
            "rolling_mse": compute_mse(r_s["actual_raw"].to_numpy(),
                                       r_s["predicted"].to_numpy()),
            "rolling_persistence_r2": compute_r2(
                r_valid["actual_raw"].to_numpy(),
                r_valid["persistence"].to_numpy()),
            "rolling_persistence_mse": compute_mse(
                r_valid["actual_raw"].to_numpy(),
                r_valid["persistence"].to_numpy()),
        })
    return pd.DataFrame(rows).set_index("seed")


def part_wilcoxon_seeds(all_df, roll_df):
    fams = {w: seed_family_metrics(all_df, roll_df, w)
            for w in ("lstm", "lstm_bi")}
    results, table = [], []
    pairs = [("lstm", "lstm_bi")]
    for w in ("lstm", "lstm_bi"):
        pairs.append((w, "persistence"))
    PERS_COL = {"r2": "persistence_r2", "mse": "persistence_mse",
                "rolling_r2": "rolling_persistence_r2",
                "rolling_mse": "rolling_persistence_mse"}
    for a, b in pairs:
        for metric in ("r2", "mse", "rolling_r2", "rolling_mse"):
            if b == "persistence":
                diffs = fams[a][metric] - fams[a][PERS_COL[metric]]
                label = f"{LABELS[a]} vs persistência ({metric})"
            else:
                diffs = fams[a][metric] - fams[b][metric]
                label = f"{LABELS[a]} vs {LABELS[b]} ({metric})"
            res = wilcoxon_paired(diffs.to_numpy(), label)
            results.append(res)
            table.append({
                "comparison": label,
                "median_diff": res["median_diff"],
                "wins": res["wins"],
                "losses": res["losses"],
                "p_value": res["p_value"],
                "rank_biserial": res["rank_biserial"],
                "significant_05": res["significant_05"],
            })
    return {"n_seeds": int(len(fams["lstm"])), "alpha": ALPHA,
            "tests": results}, pd.DataFrame(table)


def load_canonical_dense(db_path, run_id):
    con = sqlite3.connect(db_path)
    q = """
        SELECT m.wrapper_key AS wrapper, p.origin, p.horizon,
               p.actual, p.predicted
        FROM run_model_prediction p
        JOIN run_model m ON m.id = p.model_id
        WHERE m.run_id = ? AND p.mode = 'dense'
    """
    df = pd.read_sql_query(q, con, params=(run_id,),
                           parse_dates=["origin"])
    con.close()
    df["abs_error"] = (df["actual"] - df["predicted"]).abs()
    return df


def part_friedman_days(dense_df):
    dense_df = dense_df.copy()
    dense_df["day"] = dense_df["origin"].dt.date
    per_day = (dense_df.groupby(["wrapper", "day"])
               .apply(lambda g: compute_mse(g["actual"], g["predicted"]),
                      include_groups=False)
               .unstack("wrapper")[MODEL_ORDER])
    stat, p = stats.friedmanchisquare(*[per_day[w] for w in MODEL_ORDER])
    ranks = per_day.rank(axis=1, ascending=True)
    mean_ranks = ranks.mean()

    pair_rows = []
    for a, b in combinations(MODEL_ORDER, 2):
        diffs = (per_day[a] - per_day[b]).to_numpy()
        res = wilcoxon_paired(diffs, f"{LABELS[a]} vs {LABELS[b]}")
        pair_rows.append({
            "comparison": res["comparison"],
            "median_diff_mse": res["median_diff"],
            "mean_diff_mse": res["mean_diff"],
            "wins": res["wins"],
            "losses": res["losses"],
            "p_value": res["p_value"],
            "rank_biserial": res["rank_biserial"],
        })
    pvals = [r["p_value"] for r in pair_rows]
    adjusted = holm(pvals)
    for row, adj in zip(pair_rows, adjusted):
        row["p_holm"] = float(adj)
        row["significant_05"] = bool(adj < ALPHA)

    return {
        "n_days": int(len(per_day)),
        "units": "MSE por dia-calendário das origens (dense, seed 42)",
        "friedman_chi2": float(stat),
        "friedman_p": float(p),
        "friedman_significant_05": bool(p < ALPHA),
        "mean_ranks": {LABELS[w]: float(mean_ranks[w]) for w in MODEL_ORDER},
        "posthoc_wilcoxon_holm": pair_rows,
    }, per_day


def part_kspa(dense_df):
    pivots = {}
    for w in MODEL_ORDER:
        sub = dense_df[dense_df["wrapper"] == w]
        pivots[w] = (sub.sort_values(["origin", "horizon"])
                     .pivot(index="origin", columns="horizon",
                            values="abs_error")
                     .to_numpy())
    rows = []
    for a, b in combinations(MODEL_ORDER, 2):
        res = kspa_pair(pivots[a], pivots[b])
        rows.append({
            "comparison": f"{LABELS[a]} vs {LABELS[b]}",
            "ks_statistic": res["ks"],
            "p_value": res["p_value"],
            "significant_05": res["significant_05"],
        })
    return {"n_boot": N_BOOT, "block_origins": BOOT_BLOCK,
            "alpha": ALPHA, "pairs": rows}, pd.DataFrame(rows)


def main():
    global N_BOOT, BOOT_BLOCK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds-dir", type=Path,
                        default=Path("data/results/itildeq_seeds_8020"))
    parser.add_argument("--db", type=Path, default=Path("data/predictions.db"))
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/results/itildeq_seeds_8020/significance"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--block", type=int, default=BOOT_BLOCK)
    args = parser.parse_args()

    N_BOOT, BOOT_BLOCK = args.n_boot, args.block

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_df, roll_df = load_seed_predictions(args.seeds_dir)
    wilcoxon_res, wilcoxon_tab = part_wilcoxon_seeds(all_df, roll_df)
    wilcoxon_tab.to_csv(args.out_dir / "wilcoxon_seeds.csv", index=False)

    dense_df = load_canonical_dense(args.db, args.run_id)
    friedman_res, per_day = part_friedman_days(dense_df)
    per_day.to_csv(args.out_dir / "friedman_day_blocks.csv")
    pd.DataFrame(friedman_res["posthoc_wilcoxon_holm"]).to_csv(
        args.out_dir / "friedman_posthoc_holm.csv", index=False)

    kspa_res, kspa_tab = part_kspa(dense_df)
    kspa_tab.to_csv(args.out_dir / "kspa_pairs.csv", index=False)

    summary = {
        "protocol": "seeds 42-51 (LSTM/LSTM-Bi ITILDE-Q) + canônicas seed 42 "
                    "(dense, predictions.db); blocos diários p/ Friedman; "
                    "bootstrap em blocos de origens p/ KSPA",
        "alpha": ALPHA,
        "wilcoxon_seeds": wilcoxon_res,
        "friedman_days": friedman_res,
        "kspa": kspa_res,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("=" * 72)
    print("WILCOXON PAREADO POR SEED (n=%d, seeds 42–51)" % wilcoxon_res["n_seeds"])
    print(wilcoxon_tab.to_string(index=False,
                                 formatters={"p_value": "{:.4f}".format,
                                             "median_diff": "{:+.4f}".format,
                                             "rank_biserial": "{:+.2f}".format}))
    print()
    print("FRIEDMAN SOBRE BLOCOS DIÁRIOS (4 canônicas, seed 42)")
    print("  chi2 = %.2f · p = %.5f · n_dias = %d" % (
        friedman_res["friedman_chi2"], friedman_res["friedman_p"],
        friedman_res["n_days"]))
    print("  ranks médios:",
          {k: round(v, 2) for k, v in friedman_res["mean_ranks"].items()})
    posthoc = pd.DataFrame(friedman_res["posthoc_wilcoxon_holm"])
    print(posthoc.to_string(index=False,
                            formatters={"p_value": "{:.4f}".format,
                                        "p_holm": "{:.4f}".format,
                                        "median_diff_mse": "{:+.4f}".format,
                                        "mean_diff_mse": "{:+.4f}".format,
                                        "rank_biserial": "{:+.2f}".format}))
    print()
    print("KSPA (|erro|, bootstrap em blocos de %d origens, %d boots)"
          % (BOOT_BLOCK, N_BOOT))
    print(kspa_tab.to_string(index=False,
                             formatters={"ks_statistic": "{:.4f}".format,
                                         "p_value": "{:.4f}".format}))
    print()
    print("Artefatos em:", args.out_dir)


if __name__ == "__main__":
    main()
