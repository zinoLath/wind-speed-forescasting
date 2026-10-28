"""Teste de regime: argmin da round 7 sob o protocolo do PRÓPRIO HPO.

Treina os HPs do argmin (best_trial_raw.json) no regime do Optuna —
split 72/8/20, EarlyStopping(val_loss, patience, restore_best) +
ReduceLROnPlateau, use_validation=True — e avalia no MESMO teste de 20%
da grade do sweep. Compara com o resultado do sweep (mesmos HPs no
regime final 80/20, 60 épocas secas): isola o efeito do protocolo.

    python scripts/experiments/lidar_optuna7_regime_test.py --wrapper lstm
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common  # noqa: E402

STORAGE = common.resolve("pipeline/tmp/optuna_round7")
OUT_DIR = common.resolve("pipeline/tmp/trained_optuna7_hpo_regime")
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrapper", default="lstm",
                        choices=("lstm", "lstm_bi"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: o do HPO por wrapper (lstm 64)")
    parser.add_argument("--out-dir", default=None,
                        help=f"default: {OUT_DIR}/<wrapper>")
    args = parser.parse_args()

    dirs = {"lstm": "Seq2Seq_LSTM", "lstm_bi": "Seq2Seq_LSTM_Bidirectional"}
    raw = json.loads(
        (STORAGE / dirs[args.wrapper] / "best_trial_raw.json").read_text())
    hp = raw["best_params"]
    print(f"[{args.wrapper}] argmin (trial {raw.get('trial', '?')}) "
          f"val={raw['best_value']:+.4f} hp={hp}")

    tf = common.setup_tensorflow(True)
    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=("ws100",))
    train_df, val_df, test_df = common.split_dataset(
        dataset, args.train_ratio, args.val_ratio)
    print(f"Split HPO: treino {len(train_df)} "
          f"({args.train_ratio:.1%}) | val {len(val_df)} "
          f"({args.val_ratio:.1%}) | teste {len(test_df)} "
          f"({1 - args.train_ratio - args.val_ratio:.1%})")

    from keras import backend as K

    K.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    wrapper = common.wrapper_factory(args.wrapper)()
    wrapper.prepare(
        train_df, val_df,
        input_steps=72, output_steps=36, target_col="ws100_wavelet",
        denoise=("ws100",), denoise_level=2,
    )
    batch_size = args.batch_size or int(hp.get("batch_size", 64))
    wrapper.loss = "mse"
    wrapper.loss_mode_override = "mse"
    wrapper.build(common.FixedHyperParameters(hp))
    callbacks = common.default_callbacks(wrapper, args.patience)
    print(f"callbacks: {[type(c).__name__ for c in callbacks]} | "
          f"batch={batch_size}")

    started = time.perf_counter()
    history = wrapper.fit(
        epochs=args.epochs, batch_size=batch_size, verbose=0,
        callbacks=callbacks, use_validation=True,
    )
    losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    elapsed = time.perf_counter() - started
    print(f"{len(losses)} épocas (ES em {args.patience} sem melhora) "
          f"em {elapsed:.0f}s | val_loss final={val_losses[-1]:.6f} "
          f"melhor={min(val_losses):.6f}" if val_losses else "")

    out = {}
    for name, df in (("val", val_df), ("teste", test_df)):
        all_df = common.predict_all_horizons(wrapper, df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"],
                                         all_df["predicted"])
        persistence = common.compute_metrics(all_df["actual_raw"],
                                             all_df["persistence"])
        skill = 1 - metrics["mse"] / persistence["mse"]
        out[name] = {**metrics, "skill_persistence": skill,
                     "persistence_mse": persistence["mse"]}
        print(f"[{name}] MSE={metrics['mse']:.4f} RMSE={metrics['rmse']:.4f} "
              f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} "
              f"skill_vs_pers={skill:+.3f}")

    model_dir = (common.resolve(args.out_dir) if args.out_dir
                 else OUT_DIR / args.wrapper)
    model_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save(model_dir / "model.keras")
    common.write_json(model_dir / "regime_test.json", {
        "wrapper": args.wrapper,
        "regime": "72/8/20 + EarlyStopping(patience "
                  f"{args.patience}, restore_best) + ReduceLROnPlateau",
        "hp": hp, "batch_size": batch_size,
        "epochs_run": len(losses), "epochs_requested": args.epochs,
        "seed": SEED, "train_sec": round(elapsed, 1),
        **{f"{k}_{m}": v for k, d in out.items() for m, v in d.items()},
    })
    print(f"-> {model_dir / 'regime_test.json'}")


if __name__ == "__main__":
    main()
