"""Análise do estudo de seeds — LSTM / LSTM-Bi com ITILDE-Q (80/20).

Lê as predições gravadas por ``scripts/lidar_8020_itildeq_seeds_train.py`` e
responde, para cada arquitetura:

- qual seed teve o **melhor** e o **pior** desempenho (MSE/MAE/R²/skill);
- como é a **distribuição do erro** (viés, dispersão, caudas, assimetria,
  curtose, quantis, % dentro de 1 e 2 m/s);
- quanto o resultado **varia entre seeds** (média ± desvio, amplitude, CV);
- se a melhor seed é **estatisticamente** melhor que a pior (Diebold–Mariano
  pareado por origem) e quão estável é o ranking (bootstrap em blocos).

Não precisa de GPU (pandas/numpy/scipy/matplotlib):

    python tests/itildeq_seeds_8020.py \\
        --models-dir pipeline/tmp/trained_8020_itildeq_seeds \\
        --out-dir data/results/itildeq_seeds_8020
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import common  # noqa: E402
from src.utils import diebold_mariano_loss_diff  # noqa: E402

ALPHA = 0.05
N_BOOT = 2000
BOOT_BLOCK = 144          # origens (24 h de passos de 10 min)
DEFAULT_SEED = 20261017

WRAPPERS = ("lstm", "lstm_bi")
LABELS = {"lstm": "LSTM", "lstm_bi": "LSTM-Bi"}
COLORS = {"lstm": "#1f77b4", "lstm_bi": "#9467bd"}
BEST_COLOR = "#2ca02c"
WORST_COLOR = "#d62728"
ALL_COLOR = "#7f7f7f"

plt.rcParams.update({
    "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
    "axes.titlesize": 11, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
})


# ---------------------------------------------------------------------------
# Métricas e estatística descritiva
# ---------------------------------------------------------------------------

def _std(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def error_stats(err):
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0}
    ae = np.abs(e)
    return {
        "n": int(e.size),
        "bias": float(e.mean()),
        "std": float(e.std(ddof=1)) if e.size > 1 else 0.0,
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "p95_ae": float(np.quantile(ae, 0.95)),
        "p99_ae": float(np.quantile(ae, 0.99)),
        "max_ae": float(ae.max()),
        "min": float(e.min()),
        "max": float(e.max()),
        "q01": float(np.quantile(e, 0.01)),
        "q05": float(np.quantile(e, 0.05)),
        "q25": float(np.quantile(e, 0.25)),
        "q50": float(np.quantile(e, 0.50)),
        "q75": float(np.quantile(e, 0.75)),
        "q95": float(np.quantile(e, 0.95)),
        "q99": float(np.quantile(e, 0.99)),
        "skew": float(stats.skew(e, bias=False)) if e.size > 2 else np.nan,
        "kurtosis": (float(stats.kurtosis(e, fisher=True, bias=False))
                     if e.size > 3 else np.nan),
        "within_1": float(np.mean(ae <= 1.0)),
        "within_2": float(np.mean(ae <= 2.0)),
    }


def per_origin_mse(df):
    """MSE agregado nos 36 horizontes, por origem (série pareável)."""
    se = (df["actual"].to_numpy(float) - df["predicted"].to_numpy(float)) ** 2
    return (pd.Series(se, index=df["origin"].to_numpy())
            .groupby(level=0).mean().sort_index())


def moving_block_indices(n, block, rng):
    if n <= block:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def bootstrap_variants(per_origin, n_boot=N_BOOT, block=BOOT_BLOCK,
                       seed=DEFAULT_SEED):
    """Bootstrap em blocos pareado entre seeds (mesmos índices p/ todas).

    Retorna CI 95% do MSE por seed e a probabilidade de cada seed ser a
    melhor/pior em uma reamostragem.
    """
    seeds = list(per_origin)
    arrays = np.vstack([per_origin[s].to_numpy(float) for s in seeds])
    n = arrays.shape[1]
    rng = np.random.default_rng(seed)
    replicates = np.empty((n_boot, len(seeds)), dtype=float)
    for b in range(n_boot):
        idx = moving_block_indices(n, block, rng)
        replicates[b] = arrays[:, idx].mean(axis=1)
    out = {}
    for j, seed_name in enumerate(seeds):
        values = replicates[:, j]
        ranks = replicates.argsort(axis=1).argsort(axis=1)
        out[seed_name] = {
            "mse": float(arrays[j].mean()),
            "ci95": [float(np.quantile(values, 0.025)),
                     float(np.quantile(values, 0.975))],
            "boot_std": float(values.std(ddof=1)),
            "p_best": float(np.mean(ranks[:, j] == 0)),
            "p_worst": float(np.mean(ranks[:, j] == len(seeds) - 1)),
            "mean_rank": float(ranks[:, j].mean() + 1.0),
        }
    return out


def dm_vs_reference(per_origin, reference):
    """DM (HAC h−1 + HLN) de cada seed contra a seed de referência."""
    ref = per_origin[reference]
    rows = {}
    for seed_name, series in per_origin.items():
        if seed_name == reference:
            continue
        idx = ref.index.intersection(series.index).sort_values()
        d = series.loc[idx].to_numpy(float) - ref.loc[idx].to_numpy(float)
        res = diebold_mariano_loss_diff(d, h=36)
        rows[seed_name] = {
            "delta_mse": float(d.mean()),
            "dm": res["dm"],
            "p": res["p_value"],
            "melhor": reference if d.mean() > 0 else seed_name,
            "sig": bool(res["p_value"] < ALPHA),
        }
    return rows


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_variant(models_dir, wrapper):
    base = models_dir / wrapper
    summary_path = base / "summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed_dirs = sorted(
        (p for p in base.glob("seed_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]))
    if not seed_dirs:
        return None

    runs, all_frames, roll_frames = {}, {}, {}
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        meta = {}
        if (seed_dir / "model.json").is_file():
            meta = json.loads((seed_dir / "model.json").read_text(
                encoding="utf-8"))
        all_df = pd.read_csv(seed_dir / "predictions_all.csv",
                             parse_dates=["origin", "timestamp"])
        roll_df = pd.read_csv(seed_dir / "predictions_rolling.csv",
                              parse_dates=["origin", "timestamp"])
        all_df["error"] = all_df["actual"] - all_df["predicted"]
        roll_df["error"] = roll_df["actual_raw"] - roll_df["predicted"]

        m_all = common.compute_metrics(all_df["actual"], all_df["predicted"])
        m_pers = common.compute_metrics(all_df["actual"],
                                        all_df["persistence"])
        m_roll = common.compute_metrics(roll_df["actual_raw"],
                                        roll_df["predicted"])
        pers36 = (all_df[all_df["horizon"] == all_df["horizon"].max()]
                  .drop_duplicates("origin").set_index("origin")["persistence"])
        roll_df["persistence"] = roll_df["origin"].map(pers36)
        valid = roll_df["persistence"].notna()
        m_pers_roll = common.compute_metrics(
            roll_df.loc[valid, "actual_raw"],
            roll_df.loc[valid, "persistence"])

        per_horizon = common.horizon_metrics(all_df).to_dict("records")
        runs[seed] = {
            "seed": seed,
            "hyperparameters": meta.get("hyperparameters"),
            "params": meta.get("params"),
            "training_time_sec": meta.get("training", {}).get(
                "training_time_sec"),
            "metrics_all": m_all,
            "metrics_rolling": m_roll,
            "persistence_all": m_pers,
            "persistence_rolling": m_pers_roll,
            "skill": float(1.0 - m_all["mse"] / m_pers["mse"]),
            "skill_rolling": float(1.0 - m_roll["mse"] / m_pers_roll["mse"]),
            "error_stats_all": error_stats(all_df["error"]),
            "error_stats_rolling": error_stats(roll_df["error"]),
            "per_horizon": per_horizon,
        }
        all_frames[seed] = all_df
        roll_frames[seed] = roll_df

    return {
        "summary": summary,
        "runs": runs,
        "all_frames": all_frames,
        "roll_frames": roll_frames,
        "seeds": sorted(runs),
    }


def analyse_variant(variant, wrapper):
    runs = variant["runs"]
    seeds = variant["seeds"]
    per_origin = {s: per_origin_mse(variant["all_frames"][s]) for s in seeds}
    # grade comum de origens (todas as seeds devem partilhar o mesmo teste)
    common_idx = per_origin[seeds[0]].index
    for s in seeds[1:]:
        common_idx = common_idx.intersection(per_origin[s].index)
    common_idx = common_idx.sort_values()
    per_origin = {s: per_origin[s].loc[common_idx] for s in seeds}

    mse = {s: float(runs[s]["metrics_all"]["mse"]) for s in seeds}
    order = sorted(seeds, key=lambda s: mse[s])
    best, worst = order[0], order[-1]
    boot = bootstrap_variants(per_origin)
    dm_best = dm_vs_reference(per_origin, best) if len(seeds) > 1 else {}
    reference_seed = 42 if 42 in per_origin else seeds[0]
    dm_reference = (dm_vs_reference(per_origin, reference_seed)
                    if len(seeds) > 1 else {})

    mse_values = np.array([mse[s] for s in seeds])
    mae_values = np.array([runs[s]["metrics_all"]["mae"] for s in seeds])
    r2_values = np.array([runs[s]["metrics_all"]["r2"] for s in seeds])
    roll_values = np.array([runs[s]["metrics_rolling"]["mse"] for s in seeds])
    pooled_all = pd.concat([variant["all_frames"][s]["error"]
                            for s in seeds], ignore_index=True)
    pooled_roll = pd.concat([variant["roll_frames"][s]["error"]
                             for s in seeds], ignore_index=True)

    return {
        "wrapper": wrapper,
        "label": LABELS[wrapper],
        "seeds": seeds,
        "best_seed": best,
        "worst_seed": worst,
        "ranking": order,
        "metrics": {str(s): runs[s] for s in seeds},
        "seed_mse": {str(s): mse[s] for s in seeds},
        "seed_mse_mean": float(mse_values.mean()),
        "seed_mse_std": _std(mse_values),
        "seed_mse_cv": (_std(mse_values) / mse_values.mean()
                        if mse_values.mean() else 0.0),
        "seed_mse_range": float(mse_values.max() - mse_values.min()),
        "seed_mae_mean": float(mae_values.mean()),
        "seed_mae_std": _std(mae_values),
        "seed_r2_mean": float(r2_values.mean()),
        "seed_r2_std": _std(r2_values),
        "seed_rolling_mse_mean": float(roll_values.mean()),
        "seed_rolling_mse_std": _std(roll_values),
        "bootstrap": {str(k): v for k, v in boot.items()},
        "dm_vs_best": {str(k): v for k, v in dm_best.items()},
        "dm_best_vs_worst": dm_best.get(worst),
        "dm_reference_seed": reference_seed,
        "dm_vs_reference": {str(k): v for k, v in dm_reference.items()},
        "error_stats_pooled": error_stats(pooled_all),
        "error_stats_rolling_pooled": error_stats(pooled_roll),
        "common_origins": int(len(common_idx)),
    }


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _present(analysis):
    return [w for w in WRAPPERS if w in analysis]


def fig_mse_seeds(analysis, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(1, len(wrappers), figsize=(6.5 * len(wrappers), 4.6),
                             sharey=True, squeeze=False)
    for ax, wrapper in zip(axes[0], wrappers):
        a = analysis[wrapper]
        seeds = a["seeds"]
        x = np.arange(len(seeds))
        vals = np.array([a["seed_mse"][str(s)] for s in seeds])
        lo = np.array([a["bootstrap"][str(s)]["ci95"][0] for s in seeds])
        hi = np.array([a["bootstrap"][str(s)]["ci95"][1] for s in seeds])
        colors = [BEST_COLOR if s == a["best_seed"] else
                  WORST_COLOR if s == a["worst_seed"] else COLORS[wrapper]
                  for s in seeds]
        ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none",
                    ecolor="#999999", elinewidth=1.0, capsize=2)
        ax.scatter(x, vals, c=colors, s=46, zorder=3)
        ax.axhline(a["seed_mse_mean"], ls="--", lw=1.0, color="#555555",
                   label=f"média = {a['seed_mse_mean']:.3f}")
        ax.set_xticks(x, [str(s) for s in seeds], rotation=45)
        ax.set_title(f"{a['label']} — MSE por seed")
        ax.set_xlabel("seed")
        ax.set_ylabel("MSE ((m/s)²)")
        ax.legend(loc="best")
    savefig(fig, img_dir / "fig_mse_seeds.png")


def fig_ranking(analysis, img_dir):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = 0
    yticks, labels = [], []
    for wrapper in _present(analysis):
        a = analysis[wrapper]
        order = a["ranking"]
        for seed in order:
            val = a["seed_mse"][str(seed)]
            color = (BEST_COLOR if seed == a["best_seed"] else
                     WORST_COLOR if seed == a["worst_seed"] else
                     COLORS[wrapper])
            ax.barh(y, val, color=color, height=0.72)
            yticks.append(y)
            labels.append(f"{a['label']} · {seed}")
            y += 1
        y += 1
    ax.set_yticks(yticks, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MSE ((m/s)²)")
    ax.set_title("Ranking das seeds (verde = melhor, vermelho = pior)")
    savefig(fig, img_dir / "fig_seed_ranking.png")


def fig_error_hist(analysis, variant, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(1, len(wrappers),
                             figsize=(6.5 * len(wrappers), 4.4),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for ax, wrapper in zip(axes[0], wrappers):
        a = analysis[wrapper]
        best, worst = a["best_seed"], a["worst_seed"]
        frames = variant[wrapper]["all_frames"]
        series = {
            f"melhor (seed {best})": frames[best]["error"].to_numpy(float),
            f"pior (seed {worst})": frames[worst]["error"].to_numpy(float),
        }
        pooled = np.concatenate([frames[s]["error"].to_numpy(float)
                                 for s in a["seeds"]])
        series["todas as seeds"] = pooled
        lo, hi = np.quantile(pooled, [0.005, 0.995])
        bins = np.linspace(lo, hi, 90)
        colors = [BEST_COLOR, WORST_COLOR, ALL_COLOR]
        styles = ["-", "-", "--"]
        for (label, values), color, ls in zip(series.items(), colors, styles):
            vals = values[np.isfinite(values)]
            if vals.size > 40000:
                vals = rng.choice(vals, 40000, replace=False)
            ax.hist(vals, bins=bins, density=True, histtype="step",
                    lw=1.5, color=color, ls=ls, label=label)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{a['label']} — distribuição do erro (previsto − real)")
        ax.set_xlabel("erro (m/s)")
        ax.set_ylabel("densidade")
        ax.legend()
    savefig(fig, img_dir / "fig_error_hist.png")


def fig_error_box(analysis, variant, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(1, len(wrappers),
                             figsize=(6.5 * len(wrappers), 4.6),
                             squeeze=False)
    for ax, wrapper in zip(axes[0], wrappers):
        a = analysis[wrapper]
        frames = variant[wrapper]["all_frames"]
        data, colors, labels = [], [], []
        for seed in a["ranking"]:
            err = frames[seed]["error"].to_numpy(float)
            if err.size > 20000:
                err = err[np.random.default_rng(seed).choice(
                    err.size, 20000, replace=False)]
            data.append(err)
            colors.append(BEST_COLOR if seed == a["best_seed"] else
                          WORST_COLOR if seed == a["worst_seed"] else
                          COLORS[wrapper])
            labels.append(str(seed))
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        widths=0.62, medianprops={"color": "black"})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=45)
        ax.set_title(f"{a['label']} — erro por seed (ordem de MSE)")
        ax.set_xlabel("seed")
        ax.set_ylabel("erro (m/s)")
    savefig(fig, img_dir / "fig_error_box.png")


def fig_ecdf(analysis, variant, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(1, len(wrappers),
                             figsize=(6.5 * len(wrappers), 4.4),
                             squeeze=False)
    for ax, wrapper in zip(axes[0], wrappers):
        a = analysis[wrapper]
        frames = variant[wrapper]["all_frames"]
        pooled = np.concatenate([
            np.abs(frames[s]["error"].to_numpy(float)) for s in a["seeds"]])
        curves = {
            f"melhor (seed {a['best_seed']})": np.abs(
                frames[a["best_seed"]]["error"].to_numpy(float)),
            f"pior (seed {a['worst_seed']})": np.abs(
                frames[a["worst_seed"]]["error"].to_numpy(float)),
            "todas as seeds": pooled,
        }
        colors = [BEST_COLOR, WORST_COLOR, ALL_COLOR]
        styles = ["-", "-", "--"]
        for (label, values), color, ls in zip(curves.items(), colors, styles):
            values = np.sort(values[np.isfinite(values)])
            if values.size == 0:
                continue
            y = np.arange(1, values.size + 1) / values.size
            step = max(values.size // 4000, 1)
            ax.plot(values[::step], y[::step], color=color, ls=ls, lw=1.5,
                    label=label)
        ax.set_xlim(0, 6)
        ax.set_title(f"{a['label']} — ECDF do erro absoluto")
        ax.set_xlabel("|erro| (m/s)")
        ax.set_ylabel("fração acumulada")
        ax.legend(loc="lower right")
    savefig(fig, img_dir / "fig_ecdf.png")


def fig_qq(analysis, variant, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(2, len(wrappers),
                             figsize=(5.5 * len(wrappers), 8.4),
                             squeeze=False)
    for j, wrapper in enumerate(wrappers):
        a = analysis[wrapper]
        frames = variant[wrapper]["all_frames"]
        for i, seed in enumerate((a["best_seed"], a["worst_seed"])):
            ax = axes[i, j]
            err = frames[seed]["error"].to_numpy(float)
            err = err[np.isfinite(err)]
            if err.size > 30000:
                err = np.random.default_rng(0).choice(
                    err, 30000, replace=False)
            stats.probplot(err, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor(
                BEST_COLOR if seed == a["best_seed"] else WORST_COLOR)
            ax.get_lines()[0].set_markeredgecolor("none")
            ax.get_lines()[0].set_markersize(2.2)
            ax.get_lines()[1].set_color("black")
            ax.get_lines()[1].set_linewidth(1.0)
            kind = "melhor" if seed == a["best_seed"] else "pior"
            ax.set_title(f"{a['label']} — {kind} (seed {seed}) — QQ")
            ax.set_xlabel("quantis teóricos (normal)")
            ax.set_ylabel("quantis do erro (m/s)")
    savefig(fig, img_dir / "fig_qq.png")


def fig_mse_horizon(analysis, variant, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(1, len(wrappers),
                             figsize=(6.5 * len(wrappers), 4.6),
                             sharey=True, squeeze=False)
    for ax, wrapper in zip(axes[0], wrappers):
        a = analysis[wrapper]
        frames = variant[wrapper]["all_frames"]
        for seed in a["seeds"]:
            ph = (frames[seed].groupby("horizon")["error"]
                  .apply(lambda e: float(np.mean(e ** 2))))
            color = (BEST_COLOR if seed == a["best_seed"] else
                     WORST_COLOR if seed == a["worst_seed"] else "#cccccc")
            lw = 2.0 if seed in (a["best_seed"], a["worst_seed"]) else 0.8
            ax.plot(ph.index, ph.values, color=color, lw=lw,
                    label=f"seed {seed}" if lw > 1 else None)
        ax.set_title(f"{a['label']} — MSE por horizonte")
        ax.set_xlabel("horizonte (passos de 10 min)")
        ax.set_ylabel("MSE ((m/s)²)")
        ax.set_xticks(range(1, 37, 2))
        ax.legend()
    savefig(fig, img_dir / "fig_mse_horizon.png")


def fig_bias_heatmap(analysis, variant, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(1, len(wrappers),
                             figsize=(6.75 * len(wrappers), 4.8),
                             squeeze=False)
    for ax, wrapper in zip(axes[0], wrappers):
        a = analysis[wrapper]
        frames = variant[wrapper]["all_frames"]
        rows, labels = [], []
        for seed in a["ranking"]:
            g = frames[seed].groupby("horizon")["error"].mean()
            rows.append(g.to_numpy())
            labels.append(str(seed))
        mat = np.vstack(rows)
        lim = np.nanmax(np.abs(mat))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_xticks(np.arange(0, 36, 4), np.arange(1, 37, 4))
        ax.set_title(f"{a['label']} — viés médio por horizonte")
        ax.set_xlabel("horizonte (passos de 10 min)")
        ax.set_ylabel("seed (ordem de MSE)")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                     label="erro médio (m/s)")
    savefig(fig, img_dir / "fig_bias_heatmap.png")


def fig_bootstrap_rank(analysis, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(1, len(wrappers),
                             figsize=(6.5 * len(wrappers), 4.4),
                             sharey=True, squeeze=False)
    for ax, wrapper in zip(axes[0], wrappers):
        a = analysis[wrapper]
        seeds = a["seeds"]
        x = np.arange(len(seeds))
        p_best = [a["bootstrap"][str(s)]["p_best"] for s in seeds]
        p_worst = [a["bootstrap"][str(s)]["p_worst"] for s in seeds]
        ax.bar(x - 0.19, p_best, width=0.38, color=BEST_COLOR,
               label="P(melhor)")
        ax.bar(x + 0.19, p_worst, width=0.38, color=WORST_COLOR,
               label="P(pior)")
        ax.set_xticks(x, [str(s) for s in seeds], rotation=45)
        ax.set_title(f"{a['label']} — estabilidade do ranking")
        ax.set_xlabel("seed")
        ax.set_ylabel("probabilidade (bootstrap)")
        ax.legend()
    savefig(fig, img_dir / "fig_bootstrap_rank.png")


def fig_rolling(analysis, variant, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(len(wrappers), 1,
                             figsize=(13, 3.5 * len(wrappers)),
                             sharex=True, squeeze=False)
    for ax, wrapper in zip(axes[:, 0], wrappers):
        a = analysis[wrapper]
        frames = variant[wrapper]["roll_frames"]
        ref = frames[a["best_seed"]]
        ax.plot(ref["timestamp"], ref["actual_raw"], color="black",
                lw=1.1, label="observado")
        ax.plot(ref["timestamp"], ref["predicted"], color=BEST_COLOR,
                lw=1.0, alpha=0.9, label=f"melhor (seed {a['best_seed']})")
        ax.plot(frames[a["worst_seed"]]["timestamp"],
                frames[a["worst_seed"]]["predicted"], color=WORST_COLOR,
                lw=0.9, alpha=0.85, label=f"pior (seed {a['worst_seed']})")
        ax.set_title(f"{a['label']} — rolling autônomo h=36")
        ax.set_ylabel("ws100 (m/s)")
        ax.legend(loc="upper left", ncol=3)
    axes[-1, 0].set_xlabel("timestamp")
    savefig(fig, img_dir / "fig_rolling.png")


def fig_arch_compare(analysis, img_dir):
    wrappers = _present(analysis)
    n = len(wrappers)
    fig, axes = plt.subplots(1, max(n, 2), figsize=(13, 4.4),
                             squeeze=False)
    ax = axes[0, 0]
    data = [np.array([analysis[w]["seed_mse"][str(s)]
                      for s in analysis[w]["seeds"]]) for w in wrappers]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black"})
    for patch, wrapper in zip(bp["boxes"], wrappers):
        patch.set_facecolor(COLORS[wrapper])
        patch.set_alpha(0.7)
    for j, (wrapper, values) in enumerate(zip(wrappers, data), start=1):
        ax.scatter(np.full(len(values), j), values, color=COLORS[wrapper],
                   s=32, zorder=3)
    ax.set_xticks(range(1, n + 1), [LABELS[w] for w in wrappers])
    ax.set_ylabel("MSE ((m/s)²)")
    ax.set_title("MSE por seed — arquiteturas")

    ax = axes[0, 1]
    for j, wrapper in enumerate(wrappers):
        a = analysis[wrapper]
        vals = np.array([a["seed_mse"][str(s)] for s in a["seeds"]])
        ax.scatter(np.full(len(vals), j), vals,
                   color=COLORS[wrapper], s=36)
    ax.set_xticks(range(n), [LABELS[w] for w in wrappers])
    ax.set_ylabel("MSE ((m/s)²)")
    ax.set_title("Dispersão entre seeds (cada ponto = 1 seed)")
    savefig(fig, img_dir / "fig_arch_compare.png")


def fig_pred_scatter(analysis, variant, img_dir):
    wrappers = _present(analysis)
    fig, axes = plt.subplots(2, len(wrappers),
                             figsize=(5.5 * len(wrappers), 9),
                             squeeze=False)
    for j, wrapper in enumerate(wrappers):
        a = analysis[wrapper]
        frames = variant[wrapper]["all_frames"]
        for i, seed in enumerate((a["best_seed"], a["worst_seed"])):
            ax = axes[i, j]
            df = frames[seed]
            hb = ax.hexbin(df["actual"], df["predicted"], gridsize=45,
                           bins="log", cmap="viridis", mincnt=1)
            lo = min(df["actual"].min(), df["predicted"].min())
            hi = max(df["actual"].max(), df["predicted"].max())
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
            kind = "melhor" if seed == a["best_seed"] else "pior"
            ax.set_title(f"{a['label']} — {kind} (seed {seed})")
            ax.set_xlabel("observado (m/s)")
            ax.set_ylabel("previsto (m/s)")
            fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    savefig(fig, img_dir / "fig_pred_scatter.png")


# ---------------------------------------------------------------------------
# Exportação e sumário
# ---------------------------------------------------------------------------

def export_predictions(variant, out_dir):
    """Concatena TODAS as predições (denso + rolling) em CSV gzip."""
    all_parts, roll_parts = [], []
    for wrapper in WRAPPERS:
        if wrapper not in variant:
            continue
        v = variant[wrapper]
        for seed in v["seeds"]:
            a = v["all_frames"][seed].copy()
            a.insert(0, "seed", seed)
            a.insert(0, "wrapper", wrapper)
            all_parts.append(a)
            r = v["roll_frames"][seed].copy()
            r.insert(0, "seed", seed)
            r.insert(0, "wrapper", wrapper)
            roll_parts.append(r)
    if all_parts:
        with gzip.open(out_dir / "predictions_all.csv.gz", "wt",
                       encoding="utf-8", newline="") as handle:
            pd.concat(all_parts, ignore_index=True).to_csv(handle, index=False)
    if roll_parts:
        with gzip.open(out_dir / "predictions_rolling.csv.gz", "wt",
                       encoding="utf-8", newline="") as handle:
            pd.concat(roll_parts, ignore_index=True).to_csv(handle,
                                                            index=False)


def seed_metrics_table(analysis):
    rows = []
    for wrapper in _present(analysis):
        a = analysis[wrapper]
        for rank, seed in enumerate(a["ranking"], start=1):
            m = a["metrics"][str(seed)]
            es = m["error_stats_all"]
            rows.append({
                "wrapper": wrapper,
                "arch": a["label"],
                "seed": seed,
                "rank": rank,
                "mse": m["metrics_all"]["mse"],
                "rmse": m["metrics_all"]["rmse"],
                "mae": m["metrics_all"]["mae"],
                "r2": m["metrics_all"]["r2"],
                "bias": es["bias"],
                "std_error": es["std"],
                "p95_ae": es["p95_ae"],
                "max_ae": es["max_ae"],
                "skill": m["skill"],
                "rolling_mse": m["metrics_rolling"]["mse"],
                "rolling_mae": m["metrics_rolling"]["mae"],
                "rolling_r2": m["metrics_rolling"]["r2"],
                "p_best": a["bootstrap"][str(seed)]["p_best"],
                "p_worst": a["bootstrap"][str(seed)]["p_worst"],
                "is_best": seed == a["best_seed"],
                "is_worst": seed == a["worst_seed"],
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir",
                        default="pipeline/tmp/trained_8020_itildeq_seeds")
    parser.add_argument("--out-dir",
                        default="data/results/itildeq_seeds_8020")
    args = parser.parse_args()

    models_dir = common.resolve(args.models_dir)
    out_dir = common.resolve(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    variants, analysis = {}, {}
    for wrapper in WRAPPERS:
        loaded = load_variant(models_dir, wrapper)
        if loaded is None:
            print(f"aviso: sem predições para {wrapper} em {models_dir}")
            continue
        variants[wrapper] = loaded
        analysis[wrapper] = analyse_variant(loaded, wrapper)
        a = analysis[wrapper]
        print(f"[{wrapper}] {len(a['seeds'])} seeds | melhor={a['best_seed']} "
              f"(MSE {a['seed_mse'][str(a['best_seed'])]:.4f}) | "
              f"pior={a['worst_seed']} "
              f"(MSE {a['seed_mse'][str(a['worst_seed'])]:.4f})")

    if not analysis:
        raise SystemExit(f"nenhuma predição encontrada em {models_dir}")

    export_predictions(variants, out_dir)
    table = seed_metrics_table(analysis)
    table.to_csv(out_dir / "seed_metrics.csv", index=False)

    fig_mse_seeds(analysis, img_dir)
    fig_ranking(analysis, img_dir)
    fig_error_hist(analysis, variants, img_dir)
    fig_error_box(analysis, variants, img_dir)
    fig_ecdf(analysis, variants, img_dir)
    fig_qq(analysis, variants, img_dir)
    fig_mse_horizon(analysis, variants, img_dir)
    fig_bias_heatmap(analysis, variants, img_dir)
    fig_bootstrap_rank(analysis, img_dir)
    fig_rolling(analysis, variants, img_dir)
    fig_arch_compare(analysis, img_dir)
    fig_pred_scatter(analysis, variants, img_dir)

    payload = {
        "study": "itildeq_seeds_8020",
        "loss": "itilde_q",
        "protocol": {
            "train_ratio": 0.80, "val_ratio": 0.0, "epochs": 60,
            "input_steps": 72, "output_steps": 36,
            "target_col": "ws100_wavelet", "denoise": ["ws100"],
            "denoise_level": 2, "loss_coeffs": {"alpha": 0.3, "beta": 0.5,
                                                "gamma": 0.1},
            "bootstrap": {"n": N_BOOT, "block": BOOT_BLOCK,
                          "seed": DEFAULT_SEED},
            "alpha_dm": ALPHA,
        },
        "variants": analysis,
        "arch_compare": {
            "best": {w: analysis[w]["best_seed"] for w in analysis},
            "worst": {w: analysis[w]["worst_seed"] for w in analysis},
            "mse_mean": {w: analysis[w]["seed_mse_mean"] for w in analysis},
            "mse_std": {w: analysis[w]["seed_mse_std"] for w in analysis},
        },
    }
    common.write_json(out_dir / "itildeq_seeds_8020.json", payload)
    print(f"OK -> {out_dir}")
    print(f"     {out_dir / 'seed_metrics.csv'}")
    print(f"     {out_dir / 'itildeq_seeds_8020.json'}")
    print(f"     {out_dir / 'predictions_all.csv.gz'}")


if __name__ == "__main__":
    main()
