"""Teste multi-resolução: LSTM prevendo a MÉDIA de cada hora futura.

Wrapper ``lstm_mean`` (Seq2SeqLSTMMean): entrada 72 passos de 10 min,
saída 6 valores — cada um é a média das 6 amostras de 10 min de uma hora
futura (mesmo horizonte total de 6 h, resolução de saída horária).
Mesmo protocolo canônico 80/20 (60 épocas, seed 42, hp da LSTM canônica,
loss MSE simples, decoder direto).

Avaliação em três vistas (headline vs ws100 BRUTO, protocolo honesto):
  1. resolução horária: pred vs média horária do alvo (visão do treino);
  2. broadcast 10 min: cada média repetida 6x vs ws100 bruto — comparável
     com a LSTM canônica (R^2 0.338) e a MSE-winner (0.414);
  3. teto da representação: broadcast da média horária REAL vs bruto
     (quanto do R^2 máximo se perde ao representar por médias).

    python scripts/experiments/lidar_8020_hmean_train.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

SEED = 42
WRAPPER_KEY = "lstm_mean"
INPUT_STEPS = 72
OUTPUT_STEPS = 6
POOL = 6
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_hmean")
CANONICAL_DIR = common.resolve("pipeline/tmp/trained_8020")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | lstm_mean (media horaria, pool={POOL})")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, _, test_df = common.split_dataset(dataset, 0.80, 0.0)
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | teste {len(test_df)}")

    meta = json.load(open(CANONICAL_DIR / "lstm" / "model.json"))
    hyperparameters = meta["hyperparameters"]

    tf.keras.utils.set_random_seed(args.seed)
    wrapper = common.wrapper_factory(WRAPPER_KEY)()
    wrapper.loss_mode_override = "mse"
    wrapper.prepare(
        train_df,
        train_df.tail(400),
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        target_col="ws100_wavelet",
        denoise=("ws100",),
        denoise_level=2,
    )
    batch_size = int(hyperparameters.get("batch_size", 32))
    if hasattr(wrapper, "schedule_total_steps"):
        steps = int((len(train_df) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps * args.epochs
    wrapper.build(common.FixedHyperParameters(hyperparameters))
    assert wrapper.loss == "mse", wrapper.loss

    print(f"[{WRAPPER_KEY}] treino {args.epochs} epocas (loss=mse)...")
    started = time.perf_counter()
    history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                          verbose=0, callbacks=None, use_validation=False)
    elapsed = time.perf_counter() - started
    losses = history.history.get("loss", [])
    print(f"[{WRAPPER_KEY}] {len(losses)} epocas em {elapsed:.0f}s | "
          f"loss final={losses[-1]:.5f}")

    model_dir = OUTPUT_DIR / WRAPPER_KEY
    model_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save(model_dir / "model.keras")
    common.write_json(model_dir / "model.json", {
        "wrapper_key": WRAPPER_KEY,
        "name": wrapper.name,
        "persistence_gate": False,
        "gate_mode": "static",
        "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
        "target_mode": getattr(wrapper, "target_mode", "absolute"),
        "normalize": bool(getattr(wrapper, "normalize", True)),
        "hyperparameters": hyperparameters,
        "params_source": "canonica 80/20 (LSTM)",
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "pool": POOL,
        "pool_semantics": "cada saida = media das 6 amostras de 10 min da hora futura",
        "target_col": "ws100_wavelet",
        "denoise": ["ws100"],
        "denoise_level": 2,
        "loss": "mse",
        "training": {"epochs_run": len(losses),
                     "epochs_requested": args.epochs,
                     "seed": args.seed, "patience": 0,
                     "final_loss": float(losses[-1]) if losses else None,
                     "training_time_sec": round(elapsed, 1)},
        "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                    "split": "80/20 sem validacao (media horaria)"},
    })

    all_df = common.predict_all_horizons(wrapper, test_df)
    n_seq = len(all_df) // OUTPUT_STEPS
    assert len(all_df) == n_seq * OUTPUT_STEPS

    hourly = common.compute_metrics(all_df["actual"], all_df["predicted"])

    block_starts = (
        np.arange(n_seq)[:, None]
        + INPUT_STEPS
        + POOL * np.arange(OUTPUT_STEPS)[None, :]
    )
    raw_pos = (block_starts[:, :, None] + np.arange(POOL)[None, None, :]).ravel()
    timestamps = test_df.index[raw_pos]
    actual_raw = dataset["ws100"].reindex(timestamps).to_numpy()

    pred_broadcast = (
        all_df["predicted"].to_numpy().reshape(n_seq, OUTPUT_STEPS)
        .repeat(POOL, axis=1).ravel()
    )
    bc = common.compute_metrics(actual_raw, pred_broadcast)

    actual_h_repeat = (
        all_df["actual"].to_numpy().reshape(n_seq, OUTPUT_STEPS)
        .repeat(POOL, axis=1).ravel()
    )
    ceiling = common.compute_metrics(actual_raw, actual_h_repeat)

    rolling_hour = dataset["ws100"].rolling(POOL).mean()
    last_row_pos = np.arange(n_seq) + INPUT_STEPS - 1
    pers_hourly = rolling_hour.iloc[last_row_pos].to_numpy()
    pers_bc = common.compute_metrics(
        actual_raw, np.repeat(pers_hourly, OUTPUT_STEPS * POOL))
    pers_h = common.compute_metrics(all_df["actual"].to_numpy(),
                                    np.repeat(pers_hourly, OUTPUT_STEPS))
    pers10_bc = common.compute_metrics(
        actual_raw, np.repeat(all_df["persistence"].to_numpy(), POOL))

    per_horizon = []
    for h in range(1, OUTPUT_STEPS + 1):
        sel = all_df["horizon"] == h
        h_metrics = common.compute_metrics(all_df.loc[sel, "actual"],
                                           all_df.loc[sel, "predicted"])
        h_raw_pos = (block_starts[:, h - 1][:, None]
                     + np.arange(POOL)[None, :]).ravel()
        h_actual_raw = dataset["ws100"].reindex(test_df.index[h_raw_pos]).to_numpy()
        h_pred_raw = all_df.loc[sel, "predicted"].to_numpy().repeat(POOL)
        h_bc = common.compute_metrics(h_actual_raw, h_pred_raw)
        per_horizon.append({
            "horizon_hour": h,
            "hourly": {k: h_metrics[k] for k in ("mse", "r2", "mae")},
            "broadcast_raw": {k: h_bc[k] for k in ("mse", "r2", "mae")},
        })

    evaluation = {
        "wrapper": WRAPPER_KEY,
        "protocol": "80/20 seed 42 loss=mse hp LSTM canonica",
        "resolucao": "saida horaria (pool=6) | horizonte total 6h | entrada 72x10min",
        "n_origens_teste": int(n_seq),
        "hourly_resolution": {k: hourly[k] for k in
                              ("samples", "mse", "rmse", "r2", "mae")},
        "broadcast_vs_raw": {k: bc[k] for k in
                             ("samples", "mse", "rmse", "r2", "mae")},
        "ceiling_broadcast_vs_raw": {k: ceiling[k] for k in ("mse", "r2")},
        "baselines": {
            "persistencia_horaria_broadcast_vs_raw":
                {k: pers_bc[k] for k in ("mse", "r2")},
            "persistencia_horaria_hourly":
                {k: pers_h[k] for k in ("mse", "r2")},
            "persistencia_10min_broadcast_vs_raw":
                {k: pers10_bc[k] for k in ("mse", "r2")},
        },
        "per_horizon": per_horizon,
    }
    common.write_json(model_dir / "evaluation.json", evaluation)

    print("\n=== Resultados (teste, {} origens) ===".format(n_seq))
    print(f"Horaria (alvo = media horaria)  MSE={hourly['mse']:.4f} "
          f"R2={hourly['r2']:.3f}")
    print(f"Broadcast vs ws100 BRUTO        MSE={bc['mse']:.4f} "
          f"R2={bc['r2']:.3f}   [LSTM canonica 36 passos: R2=0.338]")
    print(f"Teto representacao (media real) R2={ceiling['r2']:.3f}")
    print(f"Baseline pers. horaria (bruto)  MSE={pers_bc['mse']:.4f} "
          f"R2={pers_bc['r2']:.3f}")
    print(f"Baseline pers. 10 min (bruto)   MSE={pers10_bc['mse']:.4f} "
          f"R2={pers10_bc['r2']:.3f}")
    print("\nPor horizonte (hora 1..6):")
    print("hora | R2 horario | R2 broadcast vs bruto | MSE broadcast")
    for row in per_horizon:
        print(f"  {row['horizon_hour']:>2} | "
              f"{row['hourly']['r2']:>9.3f} | "
              f"{row['broadcast_raw']['r2']:>20.3f} | "
              f"{row['broadcast_raw']['mse']:>12.4f}")
    print(f"\nArtefatos: {model_dir}")


if __name__ == "__main__":
    main()
