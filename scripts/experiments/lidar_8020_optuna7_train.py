"""Treino 80/20 com os hiperparâmetros da Optuna round 7 (seleção 1-SE).

Lê os best_trial.json (regra 1-SE) DIRETAMENTE do storage da rodada
(pipeline/tmp/optuna_round7/<Wrapper>/) — não usa os resultados
promovidos em pipeline/tmp/optuna/ e não promove nada. Loss = MSE pura
em todos os wrappers (decisão da rodada 7). Protocolo idêntico ao
lidar_7525_train.py: split 80/20 sem validação, 60 épocas fixas,
seed 42, janela 72 → horizonte 36 (10 min), alvo ws100_wavelet.

Wrappers default: todos com best_trial.json no storage da rodada
(enquanto os kernels Kaggle não voltam, apenas lstm e lstm_bi).

    python scripts/experiments/lidar_8020_optuna7_train.py
    python reports/src/relatorio_resultados.py \
        --models-dir pipeline/tmp/trained_8020_optuna7 \
        --out-dir data/results/report_8020_optuna7 --no-baselines
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import common

WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
ROUND7_DIR = common.resolve("pipeline/tmp/optuna_round7")
TRAIN_RATIO = 0.80
EPOCHS = 60
BATCH_SIZE = 32
SEED = 42
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_optuna7")
FEATURES_EVAL_DIR = common.resolve("data/results/lidar_8020_optuna7_features")

# Dilatações estáticas C' (round 4) — só afetam os TCN.
STATIC_DILATIONS = {
    "encoder": [1, 2, 4, 8, 16, 32, 36],
    "decoder": [1, 2, 4, 8, 16],
    "encoder_depth": (4, 7),
    "decoder_depth": (3, 5),
}


def available_wrappers():
    """Wrappers com best_trial.json (1-SE) no storage da rodada 7."""
    found = []
    for key in WRAPPERS:
        name = common.wrapper_factory(key)().name
        if (ROUND7_DIR / name / "best_trial.json").is_file():
            found.append(key)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrappers", nargs="+", default=None,
                        choices=WRAPPERS,
                        help="subset (default: os disponíveis no storage round7)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--output-dir", default=None,
                        help="dir de saída (default: pipeline/tmp/trained_8020_optuna7)")
    parser.add_argument("--static-dilations", action="store_true",
                        help="dilatações estáticas C' nos TCN (round 4)")
    args = parser.parse_args()

    wrappers = args.wrappers or available_wrappers()
    if not wrappers:
        raise SystemExit("Nenhum wrapper com best_trial.json em "
                         f"{ROUND7_DIR} — rode scripts/experiments/optuna_select_1se.py "
                         "quando a rodada encerrar.")

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Wrappers (round 7, 1-SE): {', '.join(wrappers)}")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=("ws100",))
    train_df, val_df, test_df = common.split_dataset(dataset, args.train_ratio, 0.0)
    assert len(val_df) == 0 and len(test_df) > 0
    print(f"Dataset: {len(dataset)} linhas | treino {len(train_df)} "
          f"({args.train_ratio:.0%}) | teste {len(test_df)} "
          f"({1 - args.train_ratio:.0%}) | validacao: REMOVIDA")

    output_dir = common.resolve(args.output_dir) if args.output_dir else OUTPUT_DIR
    evaluations = {}
    eval_cache = FEATURES_EVAL_DIR / "all.json"

    for wrapper_key in wrappers:
        wrapper_name = common.wrapper_factory(wrapper_key)().name
        hp_path = ROUND7_DIR / wrapper_name / "best_trial.json"
        hyperparameters = json.loads(hp_path.read_text(encoding="utf-8"))["best_params"]
        print(f"[{wrapper_key}] HPs 1-SE de {hp_path.relative_to(PROJECT_ROOT)}: "
              f"{hyperparameters}")

        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        wrapper = common.wrapper_factory(wrapper_key)()
        # prepare() exige um val com sequencias; usamos o tail do PROPRIO treino
        # apenas para satisfazer a API — fit roda com use_validation=False.
        val_for_prepare = train_df.tail(400)
        wrapper.prepare(
            train_df,
            val_for_prepare,
            input_steps=72,
            output_steps=36,
            target_col="ws100_wavelet",
            denoise=("ws100",),
            denoise_level=2,
        )
        batch_size = int(hyperparameters.get("batch_size", BATCH_SIZE))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * args.epochs
        # Rodada 7: MSE pura em todos os wrappers (decisão da rodada).
        wrapper.loss = "mse"
        wrapper.loss_mode_override = "mse"
        if args.static_dilations and wrapper_key.startswith("tcn"):
            wrapper.static_dilations = STATIC_DILATIONS
        wrapper.build(common.FixedHyperParameters(hyperparameters))

        print(f"[{wrapper_key}] treino {args.epochs} epocas (sem early stopping)...")
        started = time.perf_counter()
        history = wrapper.fit(
            epochs=args.epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=None,
            use_validation=False,
        )
        elapsed = time.perf_counter() - started
        losses = history.history.get("loss", [])

        model_dir = output_dir / wrapper_key
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        common.write_json(model_dir / "model.json", {
            "model_class": common.wrapper_factory(wrapper_key).__name__,
            "wrapper_key": wrapper_key,
            "name": wrapper.name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hyperparameters": hyperparameters,
            "params_source": str(hp_path),
            "input_steps": 72,
            "output_steps": 36,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "features": None,
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "persistence_gate": False,
            "gate_mode": "static",
            "context_mode": getattr(wrapper, "context_mode", "repeat"),
            "loss": wrapper.loss,
            "training": {
                "epochs_requested": args.epochs,
                "epochs_run": len(losses),
                "batch_size": batch_size,
                "patience": 0,
                "training_time_sec": round(elapsed, 3),
                "final_loss": float(losses[-1]) if losses else None,
                "final_val_loss": None,
                "best_val_loss": None,
                "seed": SEED,
            },
            "dataset": {
                "source": "data/dataset.csv",
                "train_rows": len(train_df),
                "val_rows": 0,
                "test_rows": len(test_df),
                "train_ratio": args.train_ratio,
                "val_ratio": 0.0,
                "split": f"{round(args.train_ratio * 100)}/"
                         f"{round((1 - args.train_ratio) * 100)} sem validacao",
            },
            "features_mode": "all",
        })
        print(f"[{wrapper_key}] {len(losses)} epocas em {elapsed:.0f}s | "
              f"loss final={losses[-1]:.4f} -> {model_dir}")

        # Avaliacao inline (mesma convencao do relatorio_resultados):
        # predict_all_horizons sobre o teste, alvo = ws100 bruto.
        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]
        ).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"], all_df["predicted"])
        persistence = common.compute_metrics(
            all_df["actual_raw"], all_df["persistence"]
        )
        skill = (1 - metrics["mse"] / persistence["mse"]
                 if persistence["mse"] else None)
        per_horizon = {}
        for h in (1, 6, 12, 24, 36):
            group = all_df[all_df["horizon"] == h]
            if len(group):
                per_horizon[h] = float(np.mean(
                    (group["predicted"] - group["actual_raw"]) ** 2
                ))
        evaluations[wrapper_key] = {
            "name": wrapper_name,
            "features_mode": "all",
            "features": int(dataset.shape[1]),
            **metrics,
            "persistence_mse": persistence["mse"],
            "skill": skill,
            "per_horizon_mse": per_horizon,
            "epochs": len(losses),
        }
        print(f"[{wrapper_key}] MSE={metrics['mse']:.4f} "
              f"RMSE={metrics['rmse']:.4f} skill={skill:.3f} "
              f"(persistencia MSE={persistence['mse']:.4f})")

    eval_cache.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(eval_cache, evaluations)
    print(f"\nAvaliacao: {eval_cache}")
    print(f"Modelos: {output_dir}")


if __name__ == "__main__":
    main()
