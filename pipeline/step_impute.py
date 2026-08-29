"""Stage: model-based data imputation.

Runs every configured imputer (random_forest, lightgbm, ...) on
``data/wind_data.csv``, saves the imputed dataset plus the validation report
(error metrics and plots) under ``data/impute/<method>/`` and mirrors them to
``pipeline/tmp/imputed/<method>/`` so the walk-forward stage can consume them.
"""

import argparse
import shutil
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import common, config as config_module
from src.impute import base

STAGE = "impute"

METHODS = {
    "random_forest": "src.impute.rf",
    "lightgbm": "src.impute.lightgbm",
}


def run(config):
    cfg = config[STAGE]
    wind_data = base.load_wind_data(
        common.resolve(cfg["input"] or config["paths"]["wind_data_csv"])
    )
    print(f"Wind data: {wind_data.shape} | missing values: {int(wind_data.isna().sum().sum())}")

    results = {}
    for method in cfg["methods"]:
        if method not in METHODS:
            raise ValueError(f"Unknown imputation method '{method}'. Choose from {list(METHODS)}.")
        module = __import__(METHODS[method], fromlist=["impute_dataframe", "validate"])

        out_dir = common.resolve(Path(cfg["output_dir"]) / method)
        tmp_dir = common.resolve(Path(config["paths"]["tmp_dir"]) / "imputed" / method)
        for directory in (out_dir, tmp_dir):
            directory.mkdir(parents=True, exist_ok=True)

        print(f"\nImputing with {method}...")
        imputer_kwargs = cfg["method_params"].get(method, {})
        imputed = module.impute_dataframe(wind_data, **imputer_kwargs)
        imputed = imputed.assign(imputed=wind_data.isna().any(axis=1))
        print(
            f"Missing after imputation: {int(imputed.isna().sum().sum())} | "
            f"rows with imputed values: {int(imputed['imputed'].sum())}"
        )

        csv_path = out_dir / f"{method}_imputed.csv"
        imputed.to_csv(csv_path)
        shutil.copy(csv_path, tmp_dir / "imputed.csv")

        print(f"Validating {method} on artificial gaps...")
        validation = cfg["validation"]
        detailed, summary, overall, settings = base.validate(
            module.impute_dataframe,
            method,
            wind_data,
            n_samples=validation["n_samples"],
            gap_lengths=validation["gap_lengths"],
            seed=validation["seed"],
            imputer_kwargs={**imputer_kwargs, "train_sample": validation["train_sample"]},
        )
        print(
            f"[{method}] validation MAE={overall['mae']:.4f} "
            f"RMSE={overall['rmse']:.4f} (n={overall['samples']})"
        )

        for directory in (out_dir, tmp_dir):
            base.save_report(directory, settings, detailed, summary, overall)
            base.plot_metrics(summary, directory)

        results[method] = {
            "imputed_csv": str(csv_path),
            "tmp_imputed_csv": str(tmp_dir / "imputed.csv"),
            "detailed_csv": str(out_dir / "detailed.csv"),
            "summary_csv": str(out_dir / "summary.csv"),
            "metrics_json": str(out_dir / "metrics.json"),
            "plots": [
                str(out_dir / p)
                for p in ("metrics_by_gap_length.png", "mae_by_column.png", "mae_heatmap.png")
            ],
            "overall_metrics": overall,
        }

    return {"artifacts": results}


def main():
    parser = argparse.ArgumentParser(description="Imputation stage.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a pipeline config file.")
    args = parser.parse_args()

    common.ensure_project_root_on_path()
    config = config_module.load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()