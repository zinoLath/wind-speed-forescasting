"""Análise do A/B controlado de loss (LSTM/LSTM-Bi, 80/20).

Lê as predições gravadas por ``scripts/experiments/lidar_8020_lstm_lossab_train.py``
(denso + rolling) e compara, com TUDO igual exceto a função de perda:

- ``horizon_weighted_mse`` (canônica) × ``itilde_q`` (shape-aware);
- ``lstm`` e ``lstm_bi``; métricas densas e rolling, skill vs persistência,
  e Diebold–Mariano pareado (agregado + por horizonte).

Não precisa de GPU (só pandas/numpy/matplotlib):

    python studies/lstm_lossab_8020.py \
        --models-dir pipeline/tmp/trained_8020_lstm_lossab \
        --out-dir data/results/lstm_lossab_8020
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import common
from src.utils import diebold_mariano_loss_diff

ALPHA = 0.05
HORIZONS_DM = (6, 12, 24, 36)
VARIANTS = {
    "lstm_hw": ("LSTM", "hweighted"),
    "lstm_itq": ("LSTM", "itildeq"),
    "lstm_bi_hw": ("LSTM-Bi", "hweighted"),
    "lstm_bi_itq": ("LSTM-Bi", "itildeq"),
}
LABELS = {"hweighted": "horizon-weighted", "itildeq": "ITILDE-Q"}
COLORS = {"LSTM": "#1f77b4", "LSTM-Bi": "#9467bd"}
STYLE = {"hweighted": "-", "itildeq": "--"}

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.titlesize": 11,
                     "axes.labelsize": 9.5, "legend.fontsize": 8.5})


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def error_matrix(df, pred_col="predicted"):
    """origin×horizon de erros; descarta origens incompletas (grade observada)."""
    e = df["actual"] - df[pred_col]
    wide = (df.assign(e=e).pivot(index="origin", columns="horizon", values="e")
            .sort_index().sort_index(axis=1))
    return wide.dropna(axis=0)


def variant_metrics(base):
    out, frames = {}, {}
    for variant, (arch, loss) in VARIANTS.items():
        d = base / variant
        all_df = pd.read_csv(d / "predictions_all.csv",
                             parse_dates=["origin", "timestamp"])
        roll_df = pd.read_csv(d / "predictions_rolling.csv",
                              parse_dates=["origin", "timestamp"])
        # persistência no rolling = último valor observado na origem; vem da
        # coluna persistence do denso no horizonte 36 (mesma origem)
        pers36 = (all_df[all_df["horizon"] == all_df["horizon"].max()]
                  .drop_duplicates("origin").set_index("origin")["persistence"])
        roll_df["persistence"] = roll_df["origin"].map(pers36)
        frames[variant] = {"all": all_df, "roll": roll_df}
        m_all = common.compute_metrics(all_df["actual"], all_df["predicted"])
        m_roll = common.compute_metrics(roll_df["actual_raw"],
                                        roll_df["predicted"])
        m_pers = common.compute_metrics(all_df["actual"],
                                        all_df["persistence"])
        valid = roll_df["persistence"].notna()
        pers_roll = common.compute_metrics(roll_df.loc[valid, "actual_raw"],
                                           roll_df.loc[valid, "persistence"])
        out[variant] = {
            "arch": arch, "loss": loss,
            "label": f"{arch} — {LABELS[loss]}",
            "all": m_all, "rolling": m_roll,
            "skill": 1.0 - m_all["mse"] / m_pers["mse"],
            "skill_roll": 1.0 - m_roll["mse"] / pers_roll["mse"],
            "per_horizon": common.horizon_metrics(
                all_df.assign(actual=all_df["actual"])).to_dict("records"),
        }
    return out, frames


def dm_ab(frames):
    rows = []
    for arch, hw, itq in (("LSTM", "lstm_hw", "lstm_itq"),
                          ("LSTM-Bi", "lstm_bi_hw", "lstm_bi_itq")):
        eh = error_matrix(frames[hw]["all"])
        ei = error_matrix(frames[itq]["all"])
        common_idx = eh.index.intersection(ei.index).sort_values()
        H = eh.loc[common_idx].to_numpy(float)
        I = ei.loc[common_idx].to_numpy(float)
        d = (I ** 2).mean(axis=1) - (H ** 2).mean(axis=1)  # itq − hw
        agg = diebold_mariano_loss_diff(d, h=H.shape[1])
        row = {
            "arq": arch, "n": int(len(common_idx)),
            "mse_hw": float((H ** 2).mean()),
            "mse_itq": float((I ** 2).mean()),
            "delta": float((I ** 2).mean() - (H ** 2).mean()),
            "dm": agg["dm"], "p": agg["p_value"],
            "melhor": "ITILDE-Q" if agg["mean_loss_diff"] < 0
            else "horizon-weighted",
            "sig": bool(agg["p_value"] < ALPHA),
        }
        for h in HORIZONS_DM:
            res = diebold_mariano_loss_diff(I[:, h - 1] ** 2
                                            - H[:, h - 1] ** 2, h=h)
            row[f"dm_h{h}"] = res["dm"]
            row[f"p_h{h}"] = res["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)


def fig_mse_horizon(metrics, img_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, arch in zip(axes, ("LSTM", "LSTM-Bi")):
        for loss in ("hweighted", "itildeq"):
            variant = next(k for k, (a, l) in VARIANTS.items()
                           if a == arch and l == loss)
            ph = pd.DataFrame(metrics[variant]["per_horizon"])
            ax.plot(ph["horizon"], ph["mse"], color=COLORS[arch],
                    ls=STYLE[loss], lw=1.7,
                    label=LABELS[loss])
        ax.set_title(f"{arch} — MSE por horizonte")
        ax.set_xlabel("Horizonte (passos de 10 min)")
        ax.set_ylabel("MSE ((m/s)²)")
        ax.set_xticks(range(1, 37, 2))
        ax.legend()
    savefig(fig, img_dir / "fig_mse_horizon.png")


def fig_overall(metrics, img_dir):
    variants = list(VARIANTS)
    labels = [metrics[v]["label"] for v in variants]
    colors = [COLORS[metrics[v]["arch"]] for v in variants]
    hatches = ["//" if metrics[v]["loss"] == "hweighted" else ""
               for v in variants]
    y = np.arange(len(variants))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (key, title) in zip(
            axes, (("mse", "MSE ((m/s)²)"), ("rmse", "RMSE (m/s)"),
                   ("r2", "R²"))):
        vals = [metrics[v]["all"][key] for v in variants]
        bars = ax.barh(y, vals, color=colors)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(f"{title} — todos os horizontes")
    savefig(fig, img_dir / "fig_overall.png")


def fig_rolling(frames, img_dir, days=21):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True)
    for ax, (arch, hw, itq) in zip(
            axes, (("LSTM", "lstm_hw", "lstm_itq"),
                   ("LSTM-Bi", "lstm_bi_hw", "lstm_bi_itq"))):
        ref = frames[hw]["roll"]
        t1 = ref["timestamp"].max() - pd.Timedelta(days=days)
        sub = ref[ref["timestamp"] >= t1]
        ax.plot(sub["timestamp"], sub["actual_raw"], color="black", lw=1.1,
                alpha=0.6, label="observado (ws100 bruto)")
        ax.plot(sub["timestamp"], sub["predicted"], color=COLORS[arch],
                lw=1.3, ls="-", label=f"{arch} — horizon-weighted")
        sub2 = frames[itq]["roll"]
        sub2 = sub2[sub2["timestamp"] >= t1]
        ax.plot(sub2["timestamp"], sub2["predicted"], color=COLORS[arch],
                lw=1.3, ls="--", label=f"{arch} — ITILDE-Q")
        ax.set_title(f"Rolling autônomo (h=36) — {arch}")
        ax.set_ylabel("m/s")
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Timestamp")
    savefig(fig, img_dir / "fig_rolling.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path,
                        default=PROJECT_ROOT / "pipeline" / "tmp" / "trained_8020_lstm_lossab")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "results" / "lstm_lossab_8020")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    metrics, frames = variant_metrics(args.models_dir)
    dm = dm_ab(frames)

    fig_mse_horizon(metrics, img_dir)
    fig_overall(metrics, img_dir)
    fig_rolling(frames, img_dir)

    payload = {
        "alpha": ALPHA,
        "variants": metrics,
        "dm": dm.to_dict("records"),
    }
    common.write_json(args.out_dir / "lossab_8020.json", payload)

    print("Resumo (denso):")
    for v, m in metrics.items():
        print(f"  {m['label']:30} MSE={m['all']['mse']:.4f} "
              f"RMSE={m['all']['rmse']:.4f} R2={m['all']['r2']:.3f} "
              f"rolling={m['rolling']['mse']:.4f} skill={m['skill']:+.3f}")
    print("\nDM (ITILDE-Q − horizon-weighted; negativo = ITILDE-Q melhor):")
    print(dm[["arq", "mse_hw", "mse_itq", "delta", "dm", "p",
              "melhor", "sig"]].to_string(index=False))
    print(f"\nOK -> {args.out_dir}/lossab_8020.json")


if __name__ == "__main__":
    main()
