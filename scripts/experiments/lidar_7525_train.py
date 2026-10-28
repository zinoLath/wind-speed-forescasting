"""Treino na serie LiDAR com split 80/20 (parametrizavel via --train-ratio,
sem particao de validacao).

Experimento complementar aos testes originais (75/20/5 com early stopping na
validacao): treino com orcamento fixo de EPOCHS epocas, sem early stopping e
sem restauracao de pesos; os 20% finais viram teste. Os modelos sao salvos no
mesmo formato de pipeline/step_train.py (model.keras + model.json com os
ratios no metadata) para que tests/relatorio_resultados.py os avalie e gere
o relatorio:

    python scripts/lidar_7525_train.py --train-ratio 0.80
    python tests/relatorio_resultados.py \
        --models-dir pipeline/tmp/trained_8020 \
        --out-dir data/results/report_8020 --no-baselines
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import common

WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
TRAIN_RATIO = 0.80
EPOCHS = 60
BATCH_SIZE = 32
SEED = 42
# Loss por modelo (decisão da ablação de loss, out/2026):
# LSTM/ITILDE-Q (melhor no report alinhado) e TCN/MSE puro (ITILDE-Q não
# ajudou as TCN; MSE empatou com a hwmse). Use --legacy-hwmse para voltar ao
# default horizon-weighted em todos.
LOSS_BY_WRAPPER = {"lstm": "itilde_q", "lstm_bi": "itilde_q",
                   "tcn": "mse", "tcn_bi": "mse"}
# Dilatações estáticas C' do round 4 (encoder com lag 36 direto; decoder
# truncado ao máximo útil). Só aplicadas com --static-dilations.
STATIC_DILATIONS = {
    "encoder": [1, 2, 4, 8, 16, 32, 36],
    "decoder": [1, 2, 4, 8, 16],
    "encoder_depth": (4, 7),
    "decoder_depth": (3, 5),
}
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_7525")
FEATURES_EVAL_DIR = common.resolve("data/results/lidar_7525_features")

CALENDAR_COLS = ("hour_sin", "hour_cos", "doy_sin", "doy_cos", "day_sin", "day_cos")
DIR_CYCLIC_COLS = tuple(
    f"dir{h}{suffix}" for h in common.CANONICAL_HEIGHTS for suffix in ("_sin", "_cos")
)


def apply_features_mode(dataset, mode):
    """Modos de ablação das features senoidais (comum a todos os wrappers)."""
    if mode == "all":
        return dataset
    drop = list(CALENDAR_COLS)
    if mode == "no_cyclic":
        drop += [c for c in DIR_CYCLIC_COLS if c in dataset.columns]
    dropped = [c for c in drop if c in dataset.columns]
    print(f"[features-mode {mode}] descartando {len(dropped)} colunas: "
          f"{dropped[:4]}... restam {dataset.shape[1] - len(dropped)} features")
    return dataset.drop(columns=dropped)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        choices=WRAPPERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--denoise", nargs="+", default=["ws100"],
                        help="colunas suavizadas por wavelet (default ws100)")
    parser.add_argument("--denoise-level", type=int, default=2)
    parser.add_argument("--output-dir", default=None,
                        help="dir de saída (default: deriva trained_<tag>)")
    parser.add_argument("--static-dilations", action="store_true",
                        help="dilatações estáticas C' nos TCN (round 4)")
    parser.add_argument("--legacy-hwmse", action="store_true",
                        help="ignora o mapa LOSS_BY_WRAPPER e usa "
                             "horizon-weighted em todos (protocolo antigo)")
    parser.add_argument("--features-mode", default="all",
                        choices=("all", "no_calendar", "no_cyclic"),
                        help="ablação: all = com todas as senoidais; no_calendar = "
                             "sem hour/doy/day sin-cos; no_cyclic = sem nenhuma "
                             "(apenas ws/v)")
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=tuple(args.denoise))
    dataset = apply_features_mode(dataset, args.features_mode)
    # val_ratio = 0 -> val vazio e teste = 25% finais (split_dataset)
    train_df, val_df, test_df = common.split_dataset(dataset, args.train_ratio, 0.0)
    assert len(val_df) == 0 and len(test_df) > 0
    print(f"Dataset: {len(dataset)} linhas | treino {len(train_df)} "
          f"({args.train_ratio:.0%}) | teste {len(test_df)} "
          f"({1 - args.train_ratio:.0%}) | validacao: REMOVIDA")

    tag = f"{round(args.train_ratio * 100):02d}{round((1 - args.train_ratio) * 100):02d}"
    split_label = (f"{round(args.train_ratio * 100)}/"
                   f"{round((1 - args.train_ratio) * 100)} sem validacao")
    output_dir = OUTPUT_DIR.parent / (
        f"trained_{tag}" if args.features_mode == "all"
        else f"trained_{tag}_{args.features_mode}")
    if args.output_dir:
        output_dir = common.resolve(args.output_dir)
    evaluations = {}
    eval_dir = FEATURES_EVAL_DIR.parent / f"lidar_{tag}_features"
    eval_cache = eval_dir / f"{args.features_mode}.json"
    if eval_cache.is_file():
        evaluations = __import__("json").load(open(eval_cache))

    for wrapper_key in args.wrappers:
        wrapper_name = common.wrapper_factory(wrapper_key)().name
        hyperparameters, params_source = common.resolve_hyperparameters(
            wrapper_key,
            {"params_source": "optuna", "hyperparameters": {}},
            {"paths": {"tmp_dir": "pipeline/tmp", "results_dir": "data/results"}},
        )
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
            denoise=tuple(args.denoise),
            denoise_level=args.denoise_level,
        )
        batch_size = int(hyperparameters.get("batch_size", BATCH_SIZE))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * args.epochs
        wrapper.loss = hyperparameters.get("loss", "mse")
        if not args.legacy_hwmse:
            wrapper.loss_mode_override = LOSS_BY_WRAPPER.get(wrapper_key)
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
            "params_source": params_source,
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
                "split": split_label,
            },
            "features_mode": args.features_mode,
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
            "features_mode": args.features_mode,
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

    eval_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(eval_cache, evaluations)
    print(f"\nAvaliacao: {eval_cache}")
    print(f"Modelos: {output_dir}")


if __name__ == "__main__":
    main()
