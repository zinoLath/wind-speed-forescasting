"""Plot, per model, the original series and the forecast for each horizon,
aligned by TARGET timestamp (same samples).

For a target timestamp T, the forecast of horizon m initiated at origin
``T - m`` predicts sample T. So every horizon curve is plotted on the shared
x-axis of T (its predicted timestamp = origin + m), and the raw series is
drawn on the same axis. Only target timestamps where all 36 horizons have a
prediction are kept, so every curve compares the same samples.

Saves one PNG per model under ``data/rolling_comparison/series/``.
"""

import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

COMPARE_DIR = Path("pipeline/tmp/trained_compare")
OUT_DIR = Path("data/rolling_comparison/series")
HORIZONS = range(1, 37)


def load_actuals():
    """Load the raw ws100 series (the ground truth) with its timestamps."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "dataset.csv")
    df["timestamp"] = pd.to_datetime(df["id"], format="mixed")
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actuals = load_actuals()

    model_dirs = sorted(
        glob.glob(str(COMPARE_DIR / "Seq2Seq_*/evaluate/predictions_all_horizons.csv"))
    )
    if not model_dirs:
        raise SystemExit(
            f"No predictions found under {COMPARE_DIR}. Run train_compare_tcn.py first."
        )

    cmap = matplotlib.colormaps["plasma"]
    horizon_colors = {h: cmap(h / len(HORIZONS)) for h in HORIZONS}

    for path in model_dirs:
        csv_path = Path(path)
        model_name = csv_path.parents[1].name
        pred_df = pd.read_csv(csv_path)
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

        # Pivot: index = target timestamp T, columns = horizon m, values =
        # predicted at T by the forecast initiated at origin T - m.
        piv = pred_df.pivot_table(
            index="timestamp", columns="horizon", values="predicted"
        )
        # Keep only target timestamps where every horizon has a prediction, so
        # all curves compare the exact same samples.
        piv = piv.dropna(how="any").sort_index()

        # Real series aligned on the same target timestamps.
        real = actuals["ws100"].reindex(piv.index)

        fig, ax = plt.subplots(figsize=(18, 8))

        # Forecast per horizon, one line each, on the shared target axis.
        for h in HORIZONS:
            ax.plot(
                piv.index,
                piv[h],
                color=horizon_colors[h],
                linewidth=0.9,
                alpha=0.85,
                label=f"t+{h}",
            )

        # Raw series on top, thickest.
        ax.plot(
            real.index,
            real.values,
            color="black",
            linewidth=2.8,
            alpha=0.95,
            label="Real (ws100)",
            zorder=10,
        )

        ax.set_title(
            f"{model_name} — Rolling forecast por horizonte (mesmas amostras)\n"
            f"Cada curva t+h = previsão de horizonte h na origem T-h; real em preto",
            fontsize=11,
        )
        ax.set_xlabel("Tempo (timestamp-alvo T)")
        ax.set_ylabel("Velocidade do vento (m/s)")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

        legend_handles = [
            plt.Line2D([0], [0], color="black", linewidth=2.8, label="Real (ws100)"),
        ]
        for h in (1, 6, 12, 18, 24, 30, 36):
            legend_handles.append(
                plt.Line2D(
                    [0], [0], color=horizon_colors[h], linewidth=0.9, label=f"t+{h}"
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", ncol=2, fontsize=9)

        fig.tight_layout()
        out_path = OUT_DIR / f"{model_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"salvo: {out_path} (amostras comuns: {len(piv)})")


if __name__ == "__main__":
    main()