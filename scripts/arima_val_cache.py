"""Cache ARIMA (e persistência) na grade de validação do round 4.

O objetivo do Optuna é o skill contra o ARIMA na fatia de validação do HPO
(split 70/10/20: treino 70%, val 10%, teste 20% intocado). Em vez de reajustar
o ARIMA a cada trial nas GPUs Kaggle, este script pré-computa as predições
uma vez, com rigor anti-vazamento:

  - ordem do ARIMA escolhida por AIC **só no treino**;
  - fit no treino e `append(refit=False)` + `forecast(36)` por origem da
    validação (mesma maquinaria de ``tests/baseline_arima_rf.py``);
  - grade de origens idêntica à usada pela avaliação dos modelos
    (primeira origem = primeira linha da validação; o encoder consome o fim
    do treino);
  - timestamps arredondados ao minuto (fix do drift de microssegundos).

Saída: CSV long (origin, timestamp, horizon, actual, arima, persistence)
consumido por ``pipeline/step_optuna.py`` (objetivo skill_vs_arima).

    python scripts/arima_val_cache.py --train-ratio 0.70 --val-ratio 0.10 \
        --out data/results/optuna_round4/arima_val_cache.csv
"""

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import common


def _load_baseline_module():
    """Importa tests/baseline_arima_rf.py (reuso da maquinaria rolling)."""
    spec = importlib.util.spec_from_file_location(
        "baseline_arima_rf", PROJECT_ROOT / "tests" / "baseline_arima_rf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data" / "dataset.csv")
    parser.add_argument("--target", default="ws100")
    parser.add_argument("--input-steps", type=int, default=72)
    parser.add_argument("--output-steps", type=int, default=36)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" /
                        "results" / "optuna_round4" / "arima_val_cache.csv")
    args = parser.parse_args()

    baseline = _load_baseline_module()

    dataset = common.load_dataset(args.dataset, keep_raw=(args.target,))
    target = dataset[args.target].to_numpy(dtype=float)
    n = len(dataset)
    train_end = int(round(n * args.train_ratio))
    val_stop = int(round(n * (args.train_ratio + args.val_ratio)))
    print(f"n={n} | treino [0,{train_end}) | val [{train_end},{val_stop}) | "
          f"teste [{val_stop},{n})")

    train_target = target[:train_end]
    order, table = baseline.select_arima_order(train_target)
    print(f"ARIMA ordem escolhida (AIC no treino): {order}")

    origins = np.arange(train_end, val_stop - args.output_steps + 1)
    print(f"origens da validação: {len(origins)}")
    preds = baseline.run_arima(target, train_target, train_end, origins,
                               args.output_steps, order)

    horizon = np.arange(1, args.output_steps + 1)
    target_positions = origins[:, None] + horizon[None, :]
    frame = pd.DataFrame({
        "origin": np.repeat(dataset.index.to_numpy()[origins],
                            args.output_steps),
        "timestamp": dataset.index.to_numpy()[target_positions].ravel(),
        "horizon": np.tile(horizon, len(origins)),
        "actual": dataset[args.target].to_numpy()[target_positions].ravel(),
        "arima": preds.ravel(),
        "persistence": np.repeat(target[origins - 1], args.output_steps),
    })
    # fix do drift de microssegundos (grade do dataset é limpa)
    frame["origin"] = frame["origin"].dt.round("min")
    frame["timestamp"] = frame["timestamp"].dt.round("min")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    for col in ("arima", "persistence"):
        rmse = float(np.sqrt(np.mean((frame["actual"] - frame[col]) ** 2)))
        mae = float(np.mean(np.abs(frame["actual"] - frame[col])))
        print(f"  {col:11s} RMSE={rmse:.4f}  MAE={mae:.4f}")
    print(f"OK -> {args.out}")


if __name__ == "__main__":
    main()
