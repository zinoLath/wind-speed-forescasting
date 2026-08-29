"""Orchestrator for the full training/evaluation pipeline.

Run everything in order::

    python pipeline/pipeline.py

Run only a subset of stages::

    python pipeline/pipeline.py --stage train evaluate

Configuration comes from ``pipeline/pipeline.json`` (see
``pipeline/pipeline.default.json`` for the defaults). A per-run report with
the outputs of every stage is saved to ``pipeline/tmp/pipeline_run.json``.
"""

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common
from pipeline import config as config_module
from pipeline import step_evaluate, step_impute, step_optuna, step_train, step_walkforward

STAGES = {
    "optuna": step_optuna,
    "train": step_train,
    "evaluate": step_evaluate,
    "impute": step_impute,
    "walkforward": step_walkforward,
}


def main():
    parser = argparse.ArgumentParser(description="Wind speed forecasting pipeline.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a pipeline config file (default: pipeline/pipeline.json).")
    parser.add_argument(
        "--stage",
        choices=list(STAGES),
        nargs="*",
        default=None,
        help="Run only these stages (default: every enabled stage in order).",
    )
    args = parser.parse_args()

    common.ensure_project_root_on_path()
    config = config_module.load_config(args.config)

    if args.stage:
        stages = args.stage
    else:
        stages = [name for name, module in STAGES.items() if config[name].get("enabled", True)]

    tmp_dir = common.resolve(config["paths"]["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_file": str(args.config or config_module.USER_CONFIG_PATH),
        "stages": {},
    }

    for name in stages:
        print(f"\n{'=' * 60}\nSTAGE: {name}\n{'=' * 60}")
        started_at = time.perf_counter()
        try:
            result = STAGES[name].run(config)
            stage_entry = {
                "status": "ok",
                "elapsed_sec": round(time.perf_counter() - started_at, 3),
                "outputs": result,
            }
            warnings = result.get("warnings") if isinstance(result, dict) else None
            if warnings:
                stage_entry["warnings"] = warnings
            run_report["stages"][name] = stage_entry
        except Exception as exc:
            run_report["stages"][name] = {
                "status": "error",
                "elapsed_sec": round(time.perf_counter() - started_at, 3),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"\nStage {name} failed: {exc}")
            break

    run_report["finished_at"] = datetime.now(timezone.utc).isoformat()
    common.write_json(tmp_dir / "pipeline_run.json", run_report)

    failed = [name for name, info in run_report["stages"].items() if info["status"] == "error"]
    if failed:
        print(f"\nPipeline finished with errors in: {failed}")
    else:
        print(f"\nPipeline finished successfully ({len(stages)} stages).")
    print(f"Run report saved to {tmp_dir / 'pipeline_run.json'}")


if __name__ == "__main__":
    main()