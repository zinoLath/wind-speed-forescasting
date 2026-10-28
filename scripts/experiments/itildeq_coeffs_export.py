"""Exporta o progresso dos estudos de coeficientes ITILDE-Q (Optuna) para
JSON — consumido pelo report ``protocolo/itildeq-coeffs`` e rodado pelo
timer a cada 10 min.
"""

import json
import sys
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STORE_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "optuna_itildeq_coeffs"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "itildeq_coeffs_optuna"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for wrapper in ("lstm", "lstm_bi"):
        db = STORE_DIR / f"{wrapper}.db"
        if not db.is_file():
            continue
        try:
            study = optuna.load_study(
                study_name=f"itildeq_coeffs_{wrapper}",
                storage=f"sqlite:///{db}")
        except Exception as error:  # noqa: BLE001
            print(f"aviso: {wrapper}: {error}")
            continue
        done = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
        best = min(done, key=lambda t: t.value) if done else None
        payload = {
            "wrapper": wrapper,
            "n_complete": len(done),
            "n_trials": len(study.trials),
            "best": None if best is None else {
                **best.params, "val_block_mse": best.value,
                "trial": best.number},
            "trials": sorted(
                [{"trial": t.number,
                  **t.params,
                  "val_block_mse": t.value,
                  "elapsed_sec": t.user_attrs.get("elapsed_sec")}
                 for t in done], key=lambda r: r["val_block_mse"]),
        }
        (OUT_DIR / f"live_{wrapper}.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8")
        print(f"{wrapper}: {len(done)}/{len(study.trials)} trials "
              f"completos -> live_{wrapper}.json")


if __name__ == "__main__":
    main()
