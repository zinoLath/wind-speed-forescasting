"""Análise do A/B de loss na config bihp (80/20) — hw × ITILDE-Q.

Lê as predições long-format gravadas por
``scripts/experiments/lidar_lstm_bihp_ab_train.py`` (sem GPU), calcula métricas
denso/rolling/skill e o DM pareado por origem (agregado e por horizonte),
e grava figuras + ``bihp_ab_8020.json``.

    python studies/lstm_bihp_ab_8020.py \
        --models-dir pipeline/tmp/trained_8020_lstm_bihp_ab \
        --out-dir data/results/lstm_bihp_ab_8020
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src import common  # noqa: E402
from src.utils import diebold_mariano_loss_diff  # noqa: E402

ALPHA = 0.05
HORIZONS_DM = (6, 12, 24, 36)
VARIANTS = {
    "bihp_hw": ("hweighted",),
    "bihp_itq": ("itildeq",),
}
LABELS = {"hweighted": "horizon-weighted", "itildeq": "ITILDE-Q"}
COLORS = {"hweighted": "#7f7f7f", "itildeq": "#1f77b4"}
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
    for variant, (loss,) in VARIANTS.items():
        d = base / variant
        all_df = pd.read_csv(d / "predictions_all.csv",
                             parse_dates=["origin", "timestamp"])
        roll_df = pd.read_csv(d / "predictions_rolling.csv",
                              parse_dates=["origin", "timestamp"])
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
            "loss": loss,
            "label": f"bihp — {LABELS[loss]}",
            "all": m_all, "rolling": m_roll,
            "skill": 1.0 - m_all["mse"] / m_pers["mse"],
            "skill_roll": 1.0 - m_roll["mse"] / pers_roll["mse"],
            "per_horizon": common.horizon_metrics(
                all_df.assign(actual=all_df["actual"])).to_dict("records"),
        }
    return out, frames


def dm_ab(frames):
    eh = error_matrix(frames["bihp_hw"]["all"])
    ei = error_matrix(frames["bihp_itq"]["all"])
    common_idx = eh.index.intersection(ei.index).sort_values()
    H = eh.loc[common_idx].to_numpy(float)
    I = ei.loc[common_idx].to_numpy(float)
    d = (I ** 2).mean(axis=1) - (H ** 2).mean(axis=1)  # itq − hw
    agg = diebold_mariano_loss_diff(d, h=H.shape[1])
    row = {
        "arq": "bihp (LSTM uni + hp da Bi)", "n": int(len(common_idx)),
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
    return row


def fig_mse_horizon(metrics, img_dir):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for variant, (_,) in VARIANTS.items():
        loss = metrics[variant]["loss"]
        ph = pd.DataFrame(metrics[variant]["per_horizon"])
        ax.plot(ph["horizon"], ph["mse"], STYLE[loss], color=COLORS[loss],
                label=LABELS[loss], lw=1.8)
    ax.set(xlabel="horizonte (passos de 10 min)", ylabel="MSE (ws100 bruto)",
           title="bihp 80/20 — MSE por horizonte")
    ax.legend()
    savefig(fig, img_dir / "bihp_ab_mse_horizonte.png")


def fig_overall(metrics, img_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    variants = list(VARIANTS)
    for ax, (key, title) in zip(axes, (("mae", "MAE"), ("mse", "MSE"),
                                       ("r2", "R²"))):
        vals = [metrics[v]["all"][key] for v in variants]
        bars = ax.bar(range(len(variants)), vals,
                      color=[COLORS[metrics[v]["loss"]] for v in variants])
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([LABELS[metrics[v]["loss"]] for v in variants],
                           fontsize=8.5)
        ax.set_title(f"denso — {title}")
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    savefig(fig, img_dir / "bihp_ab_geral.png")


def fig_rolling(frames, img_dir, days=21):
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    for variant, (_,) in VARIANTS.items():
        loss = metrics_loss(frames, variant)
        df = frames[variant]["roll"].tail(days * 144)
        ax.plot(df["timestamp"], df["actual_raw"], color="black", lw=0.9,
                alpha=0.65, label="observado")
        ax.plot(df["timestamp"], df["predicted"], STYLE[loss],
                color=COLORS[loss], lw=1.1, label=LABELS[loss])
    ax.set_title("bihp 80/20 — rolling autônomo (últimos 21 dias de teste)")
    ax.legend(loc="upper left", ncol=3)
    savefig(fig, img_dir / "bihp_ab_rolling.png")


def metrics_loss(frames, variant):
    return "itildeq" if variant.endswith("itq") else "hweighted"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path,
                        default=common.resolve(
                            "pipeline/tmp/trained_8020_lstm_bihp_ab"))
    parser.add_argument("--out-dir", type=Path,
                        default=common.resolve("data/results/lstm_bihp_ab_8020"))
    args = parser.parse_args()

    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    metrics, frames = variant_metrics(args.models_dir)
    dm = dm_ab(frames)
    fig_mse_horizon(metrics, img_dir)
    fig_overall(metrics, img_dir)
    fig_rolling(frames, img_dir)

    common.write_json(args.out_dir / "bihp_ab_8020.json", {
        "alpha": ALPHA,
        "variants": metrics,
        "dm": dm,
    })

    print("Métricas (80/20, bihp, mesma sessão):")
    for v, m in metrics.items():
        print(f"  {m['label']:34} denso MSE={m['all']['mse']:.4f} "
              f"MAE={m['all']['mae']:.4f} R2={m['all']['r2']:.3f} | "
              f"rolling MSE={m['rolling']['mse']:.4f} "
              f"MAE={m['rolling']['mae']:.4f}")
    print(f"\nDM pareado (n={dm['n']} origens): ΔMSE(itq−hw)={dm['delta']:+.4f} "
          f"| DM={dm['dm']:.2f} p={dm['p']:.4f} → melhor={dm['melhor']}"
          f"{' (sig)' if dm['sig'] else ' (não sig)'}")
    for h in HORIZONS_DM:
        print(f"  h{h:>2}: DM={dm[f'dm_h{h}']:+.2f} p={dm[f'p_h{h}']:.4f}")
    print(f"\nSaída: {args.out_dir / 'bihp_ab_8020.json'}")


if __name__ == "__main__":
    main()
