"""Train + evaluate every wrapper with its Optuna best params (or defaults),
then compare under the honest protocol (step_evaluate, no future leak).

Produces, for each wrapper, artifacts under ``pipeline/tmp/trained_compare/``
and a comparison report ``pipeline/tmp/trained_compare/summary.json`` plus a
readable markdown summary.
"""

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common
from pipeline import config as config_module
from pipeline import step_evaluate, step_train

CONFIG_PATH = "pipeline/pipeline.train_tcn_compare.json"
WRAPPERS = [
    "lstm_original",
    "lstm",
    "lstm_bi",
    "gru",
    "gru_bi",
    "lstm_cnn",
    "tcn",
    "tcn_bi",
    "tcn_lstm",
    "transformer",
]
OUT_BASE = Path("pipeline/tmp/trained_compare")


def main():
    global CONFIG_PATH, OUT_BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrappers", nargs="*", default=None,
                        help="Subset of wrappers to run (default: all).")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Pipeline config file (default: train_tcn_compare).")
    parser.add_argument("--out-base", type=Path, default=OUT_BASE,
                        help="Output base dir (default: pipeline/tmp/trained_compare).")
    args = parser.parse_args()
    CONFIG_PATH = str(args.config)
    OUT_BASE = args.out_base
    wrappers = args.wrappers or WRAPPERS
    unknown = set(wrappers) - set(WRAPPERS)
    if unknown:
        raise SystemExit(f"Unknown wrappers: {sorted(unknown)}")

    base_config = config_module.load_config(CONFIG_PATH)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    results = {}
    for wrapper_key in wrappers:
        cfg = deepcopy(base_config)
        cfg["train"]["wrapper"] = wrapper_key
        if wrapper_key in ("lstm_original", "gru", "gru_bi", "lstm_cnn", "transformer"):
            # No Optuna study / no Optuna result for these wrappers: use the
            # built-in defaults so all wrappers train under the same recipe.
            cfg["train"]["params_source"] = "default"
        wrapper_name = common.wrapper_factory(wrapper_key)().name

        step_train.run(cfg)

        # The train stage overwrites pipeline/tmp/model.keras|json, so copy the
        # artifacts per wrapper before the next train clobbers them.
        out_dir = OUT_BASE / wrapper_name
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = common.resolve(cfg["paths"]["tmp_dir"])
        shutil.copy(tmp / "model.keras", out_dir / "model.keras")
        shutil.copy(tmp / "model.json", out_dir / "model.json")

        eval_cfg = deepcopy(cfg)
        eval_cfg["evaluate"]["model_path"] = str(out_dir / "model.keras")
        eval_cfg["evaluate"]["model_json_path"] = str(out_dir / "model.json")
        eval_cfg["evaluate"]["output_dir"] = str(out_dir / "evaluate")
        step_evaluate.run(eval_cfg)

        with open(out_dir / "evaluate" / "metrics.json", encoding="utf-8") as handle:
            results[wrapper_key] = json.load(handle)

    summary = build_summary(results)
    with open(OUT_BASE / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print("\n" + render_markdown(summary))
    with open(OUT_BASE / "summary.md", "w", encoding="utf-8") as handle:
        handle.write(render_markdown(summary))


def build_summary(results):
    rows = []
    for wrapper_key, metrics in results.items():
        name = metrics["model"]["name"]
        model = metrics["model"]
        row = {
            "wrapper": wrapper_key,
            "name": name,
            "loss": model["loss"],
            "epochs_run": model["training"]["epochs_run"],
            "training_time_sec": model["training"]["training_time_sec"],
            "params": model["hyperparameters"],
        }
        for agg in ("all_horizons", "rolling", "persistence"):
            for m in ("mae", "rmse", "r2"):
                row[f"{agg}_{m}"] = metrics[agg].get(m)
            for m in ("mae_denoised", "rmse_denoised", "r2_denoised"):
                if m in metrics[agg]:
                    row[f"{agg}_{m}"] = metrics[agg][m]
        row["horizon_mae"] = {
            h["horizon"]: round(h["mae"], 4) for h in metrics["by_horizon"]
        }
        rows.append(row)
    return {"models": rows}


def render_markdown(summary):
    lines = ["# Comparativo honesto de todos os wrappers (protocolo step_evaluate)", ""]
    header = (
        f"{'modelo':<38} {'loss':<7} {'epochs':<7} {'train(s)':<9} "
        f"{'all MAE':<9} {'all RMSE':<9} {'roll MAE':<9} {'roll RMSE':<9} {'r2':<7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in summary["models"]:
        lines.append(
            f"{row['name']:<38} {row['loss']:<7} {row['epochs_run']:<7} "
            f"{row['training_time_sec']:<9.0f} {row['all_horizons_mae']:<9.4f} "
            f"{row['all_horizons_rmse']:<9.4f} {row['rolling_mae']:<9.4f} "
            f"{row['rolling_rmse']:<9.4f} {row['all_horizons_r2']:<7.4f}"
        )
    lines.append("")
    lines.append("## Persistence baseline")
    for row in summary["models"]:
        lines.append(
            f"- {row['name']}: persistence MAE={row['persistence_mae']:.4f} "
            f"RMSE={row['persistence_rmse']:.4f}"
        )

    lines.append("")
    lines.append("## LSTM original vs LSTM atual")
    rows = {r["wrapper"]: r for r in summary["models"]}
    if "lstm_original" in rows and "lstm" in rows:
        lo, la = rows["lstm_original"], rows["lstm"]
        lines.append(
            f"- **all-horizons:** LSTM original MAE={lo['all_horizons_mae']:.4f} "
            f"vs LSTM atual MAE={la['all_horizons_mae']:.4f} "
            f"({(lo['all_horizons_mae']/la['all_horizons_mae']-1)*100:+.1f}%)"
        )
        lines.append(
            f"- **rolling (h=36):** LSTM original MAE={lo['rolling_mae']:.4f} "
            f"vs LSTM atual MAE={la['rolling_mae']:.4f} "
            f"({(lo['rolling_mae']/la['rolling_mae']-1)*100:+.1f}%)"
        )
        lines.append(
            f"- **r²:** LSTM original {lo['all_horizons_r2']:.4f} "
            f"vs LSTM atual {la['all_horizons_r2']:.4f}"
        )
        lines.append("")
        lines.append("### MAE por horizonte (LSTM original vs atual)")
        lines.append("| horizon | LSTM original | LSTM atual | delta |")
        lines.append("|---|---|---|---|")
        for h in sorted(lo["horizon_mae"]):
            if h in la["horizon_mae"]:
                lines.append(
                    f"| {h} | {lo['horizon_mae'][h]:.4f} | "
                    f"{la['horizon_mae'][h]:.4f} | "
                    f"{(lo['horizon_mae'][h]/la['horizon_mae'][h]-1)*100:+.1f}% |"
                )

    lines.append("")
    lines.append("## Melhores hiperparâmetros")
    for row in summary["models"]:
        lines.append(f"- **{row['name']}** (`{row['wrapper']}`)")
        lines.append(f"  - loss={row['loss']}, epochs={row['epochs_run']}, "
                     f"tempo={row['training_time_sec']:.0f}s")
        for k, v in sorted(row["params"].items()):
            lines.append(f"  - {k}: {v}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()