"""Multi-seed do vencedor do HPO LSTM-Bi featurex (HPs fixos, 5 seeds).

Lê os HPs do model.json do HPO, re-treina a LSTM-Bi no feature set
estendido com seeds diferentes e consolida R²/MSE por seed + média±dp.
Notifica no ntfy ao final.

    python scripts/experiments/lstm_bi_hpo_featurex_seeds.py --seeds 43 44 45 46 47
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

HPO_MODEL = common.resolve("pipeline/tmp/trained_8020_lstm_bi_hpo_featurex/lstm_bi/model.json")
OUT_DIR = common.resolve("data/results/lstm_bi_hpo_featurex_seeds")
NTFY = "https://ntfy.sh/zino-imputeoptuna"


def notify(message):
    try:
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(NTFY, data=message.encode()), timeout=10)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[43, 44, 45, 46, 47])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="sobrescreve o weight_decay dos HPs (teste 1e-4/1e-5)")
    parser.add_argument("--tag", default="",
                        help="sufixo para dirs de saída (ex.: wd1e-4)")
    args = parser.parse_args()

    canon = json.load(open(HPO_MODEL, encoding="utf-8"))
    hp = dict(canon["hyperparameters"])
    if args.weight_decay is not None:
        hp["weight_decay"] = args.weight_decay
    feature_keep = set(canon.get("feature_keep", ()))

    tf = common.setup_tensorflow(True)
    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=tuple(sorted(feature_keep)))
    train_full, _, test_df = common.split_dataset(dataset, 0.80, 0.0)

    results = {}
    for seed in args.seeds:
        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(seed)
        wrapper = common.wrapper_factory("lstm_bi")()
        wrapper.prepare(train_full, train_full.tail(400),
                        input_steps=72, output_steps=36,
                        target_col="ws100_wavelet", denoise=("ws100",),
                        denoise_level=2)
        batch = int(hp.get("batch_size", 32))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_full) + batch - 1) // batch)
            wrapper.schedule_total_steps = steps * args.epochs
        wrapper.build(common.FixedHyperParameters(hp))
        started = time.perf_counter()
        history = wrapper.fit(epochs=args.epochs, batch_size=batch,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started

        model_dir = common.resolve(
            f"pipeline/tmp/trained_8020_lstm_bi_hpo_featurex{args.tag}_s{seed}/lstm_bi")
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        meta = dict(canon)
        meta["hyperparameters"] = hp
        meta["params_source"] = (f"multi-seed do vencedor do HPO featurex "
                                 f"(seed {seed})")
        common.write_json(model_dir / "model.json", meta)

        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"],
                                         all_df["predicted"])
        results[seed] = {"r2": metrics["r2"], "mse": metrics["mse"],
                         "mae": metrics["mae"], "sec": round(elapsed)}
        print(f"[seed {seed}] MSE={metrics['mse']:.4f} R2={metrics['r2']:.4f} "
              f"({elapsed:.0f}s)", flush=True)

    import statistics

    r2s = [r["r2"] for r in results.values()]
    mses = [r["mse"] for r in results.values()]
    summary = {
        "seeds": results,
        "r2_mean": statistics.mean(r2s),
        "r2_stdev": statistics.stdev(r2s) if len(r2s) > 1 else 0.0,
        "mse_mean": statistics.mean(mses),
        "mse_stdev": statistics.stdev(mses) if len(mses) > 1 else 0.0,
        "hp": hp,
    }
    OUT_DIR = common.resolve(
        f"data/results/lstm_bi_hpo_featurex_seeds{args.tag}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json(OUT_DIR / "summary.json", summary)
    notify(f"LSTM-Bi HPO featurex{args.tag} multi-seed: "
           f"R² {summary['r2_mean']:.4f} ± {summary['r2_stdev']:.4f} "
           f"({len(r2s)} seeds)")
    print(f"\nR² médio {summary['r2_mean']:.4f} ± {summary['r2_stdev']:.4f} "
          f"| MSE médio {summary['mse_mean']:.4f} — resumo: "
          f"{OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
