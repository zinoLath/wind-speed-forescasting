"""CV 5 dobras com purge de 108 amostras — LSTM, alturas 40/100/140 + extras.

Dobras contíguas no tempo (1/5 do dataset cada); purge de 108 amostras
(72 de entrada + 36 de horizonte) em cada fronteira treino/teste. LSTM
canônica 16/10 (HPs do trained_8020_itildeq) com MSE, seed 42, 60 épocas.

    python scripts/experiments/lstm_heights3fx_cv5.py
"""
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

SEED = 42
PURGE = 108          # 72 (entrada) + 36 (horizonte)
HEIGHTS = (40, 100, 140)
KEEP = ("humid", "temp") + tuple(f"cis{i}" for i in range(1, 20))
OUT = common.resolve("data/results/lstm_heights3fx_cv/summary.json")
NTFY = "https://ntfy.sh/zino-imputeoptuna"


def notify(message):
    try:
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(NTFY, data=message.encode()), timeout=10)
    except Exception:
        pass


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    n_folds = args.folds
    canon = json.load(open(
        common.resolve("pipeline/tmp/trained_8020_itildeq/lstm/model.json"),
        encoding="utf-8"))
    hp = dict(canon["hyperparameters"])

    tf = common.setup_tensorflow(True)
    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=KEEP)
    allowed = set(HEIGHTS)
    drop = [c for c in dataset.columns
            if (m := re.match(r"^(ws|v|dir)(\d+)(?:_sin|_cos)?$", c))
            and int(m.group(2)) not in allowed]
    dataset = dataset.drop(columns=drop)
    n = len(dataset)
    fold_size = n // n_folds
    print(f"dataset {n} x {dataset.shape[1]} | {n_folds} dobras de "
          f"~{fold_size} | purge {PURGE}", flush=True)

    folds = []
    for k in range(n_folds):
        t0 = k * fold_size
        t1 = n if k == n_folds - 1 else (k + 1) * fold_size
        tr = list(range(0, max(0, t0 - PURGE))) + \
             list(range(min(n, t1 + PURGE), n))
        folds.append((t0, t1, tr))

    results = {}
    for k, (t0, t1, tr) in enumerate(folds, 1):
        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        train_df = dataset.iloc[tr]
        wrapper = common.wrapper_factory("lstm")()
        wrapper.prepare(train_df, train_df.tail(400),
                        input_steps=72, output_steps=36,
                        target_col="ws100_wavelet", denoise=("ws100",),
                        denoise_level=2)
        batch = int(hp.get("batch_size", 32))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch - 1) // batch)
            wrapper.schedule_total_steps = steps * 60
        wrapper.loss_mode_override = "mse"
        wrapper.build(common.FixedHyperParameters(hp))
        started = time.perf_counter()
        wrapper.fit(epochs=60, batch_size=batch, verbose=0,
                    callbacks=None, use_validation=False)

        # contexto de entrada (72) vem da zona de purge — inferência não
        # treina nele; avaliação fica dentro da dobra
        eval_df = dataset.iloc[max(0, t0 - 72):t1]
        all_df = common.predict_all_horizons(wrapper, eval_df)
        t_start = dataset.index[t0]
        t_end = dataset.index[t1 - 1]
        all_df = all_df[(all_df["origin"] >= t_start)
                        & (all_df["timestamp"] <= t_end)]
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        observed = common.observed_timestamps(
            common.resolve("data/dataset.csv"))
        if observed is not None:
            all_df = all_df[all_df["timestamp"].isin(observed)]
        m = common.compute_metrics(all_df["actual_raw"], all_df["predicted"])
        results[f"fold{k}"] = {"r2": m["r2"], "mse": m["mse"],
                               "mae": m["mae"], "n_origins": len(all_df),
                               "sec": round(time.perf_counter() - started)}
        print(f"[dobra {k}] treino {len(train_df)} | R2={m['r2']:.4f} "
              f"MSE={m['mse']:.4f} ({results[f'fold{k}']['sec']}s)", flush=True)

    import statistics

    r2s = [r["r2"] for r in results.values()]
    summary = {
        "config": {"n_folds": n_folds, "purge": PURGE,
                   "heights": list(HEIGHTS),
                   "features": "14 canais de altura + cis1..19 + humid/temp",
                   "hp": hp, "seed": SEED, "loss": "mse"},
        "folds": results,
        "r2_mean": statistics.mean(r2s),
        "r2_stdev": statistics.stdev(r2s),
    }
    out = OUT.parent / f"summary_folds{args.folds}{args.tag}.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(out, summary)
    notify(f"CV5 LSTM heights3fx: R² {summary['r2_mean']:.4f} "
           f"± {summary['r2_stdev']:.4f}")
    print(f"\nR² médio {summary['r2_mean']:.4f} ± {summary['r2_stdev']:.4f} "
          f"— resumo: {out}", flush=True)


if __name__ == "__main__":
    main()
