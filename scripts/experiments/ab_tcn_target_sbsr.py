"""A/B de alvo na config canônica: TCN 48col treinada no cru vs wavelet.

Mesmo protocolo da rodada SBSR (48 col, HPs canônicos + C′, MSE, seed 42,
60 épocas, split 80/20 sem validação), variando apenas o target_col.
Métricas de ambos os braços são computadas contra o ws100 CRU no teste.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

SEED = 42
EPOCHS = 60
KEEP = ("humid", "temp")
CANON = common.resolve("pipeline/tmp/trained_8020_itildeq/tcn/model.json")


def run_arm(dataset, train_df, test_df, target_col, tf):
    from keras import backend as K

    K.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    meta = json.loads(open(CANON, encoding="utf-8").read())
    hp = dict(meta["hyperparameters"])
    wrapper = common.wrapper_factory("tcn")()
    if meta.get("static_dilations"):
        wrapper.static_dilations = meta["static_dilations"]
    wrapper.prepare(
        train_df, train_df.tail(400), input_steps=72, output_steps=36,
        target_col=target_col, denoise=("ws100",), denoise_level=2,
    )
    batch = int(hp.get("batch_size", 32))
    if hasattr(wrapper, "schedule_total_steps"):
        steps = (len(train_df) + batch - 1) // batch
        wrapper.schedule_total_steps = steps * EPOCHS
    wrapper.loss_mode_override = "mse"
    wrapper.build(common.FixedHyperParameters(hp))
    wrapper.fit(epochs=EPOCHS, batch_size=batch, verbose=0,
                callbacks=None, use_validation=False)
    all_df = common.predict_all_horizons(wrapper, test_df)
    return wrapper, all_df


def main():
    tf = common.setup_tensorflow(True)
    print(f"TF {tf.__version__} | A/B alvo: TCN 48col, seed {SEED}, "
          f"{EPOCHS} épocas")
    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=KEEP)
    train_df, _val, test_df = common.split_dataset(dataset, 0.80, 0.0)
    out = {}
    for label, target in (("cru (ws100)", "ws100"),
                          ("wavelet (ws100_wavelet)", "ws100_wavelet")):
        wrapper, all_df = run_arm(dataset, train_df, test_df, target, tf)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        m = common.compute_metrics(all_df["actual_raw"],
                                   all_df["predicted"])
        out[label] = m
        print(f"[alvo {label}] MSE={m['mse']:.4f} RMSE={m['rmse']:.4f} "
              f"MAE={m['mae']:.4f} R2(vs cru)={m['r2']:.4f}")

    print(f"\n=== A/B TCN 48col ({EPOCHS} épocas, seed {SEED}, vs ws100 cru) ===")
    base = out["cru (ws100)"]
    for label, m in out.items():
        delta = (base["mse"] - m["mse"]) / base["mse"] * 100
        print(f"  {label:26s} R2={m['r2']:.4f} MSE={m['mse']:.4f} "
              f"MAE={m['mae']:.4f} ({delta:+.1f}% MSE vs cru)")
    dest = common.resolve("data/results/ab_tcn_target_sbsr.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(dest, {"seed": SEED, "epochs": EPOCHS,
                             "protocol": "48col canônica, 80/20, MSE",
                             "arms": out})
    print("salvo em", dest)


if __name__ == "__main__":
    main()
