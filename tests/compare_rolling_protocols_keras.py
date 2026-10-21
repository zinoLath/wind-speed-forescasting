"""Compare genuine vs leaky rolling forecast on the Keras branch.

Keras port of the PyTorch ``compare_rolling_protocols.py``:
- genuine: the current ``wrapper.rolling_forecast`` — each encoder window is a
  plain function of known history (no future values).
- leaky: the legacy notebook protocol (``rolling_forecasting_real_time_improved``)
  in which the encoder window is updated with the *observed* future value at
  every step (the leak), reproducing the historical "good" numbers.

Headline metrics compare against the WAVELET-DENOISED target (``ws100_wavelet``,
what the model is trained to predict); metrics against the raw ``ws100`` are
kept as ``*_raw`` secondary keys so the smoothing effect stays measurable.

Unlike the PyTorch branch there are no reliable pre-trained Keras weights, so
each wrapper is trained from scratch with the hyperparameters stored in its
``model.json``, then evaluated under both protocols on the held-out test split.

Usage:
    python tests/compare_rolling_protocols_keras.py [--epochs 100]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

MODELS = {
    "tcn_bi": ("models/best_model/model.json", "data/results/rolling_protocols_keras/tcn_bi"),
    "lstm_bi": ("models/lstmbi_ref/model.json", "data/results/rolling_protocols_keras/lstm_bi"),
}


def leaky_rolling_forecast(wrapper, data, test_start, batch_size=256):
    """Legacy leaky protocol from the original notebook (Keras model.predict)."""
    data_scaled, target_idx = wrapper._to_scaled_array(data)
    n = len(data_scaled)
    count = n - wrapper.output_steps + 1 - test_start
    if count <= 0:
        return np.empty((0, 1)), np.empty((0, 1))

    window = data_scaled[test_start - wrapper.input_steps:test_start].copy()
    predictions = []
    for i in range(test_start, test_start + count):
        encoder_input = window.reshape(1, wrapper.input_steps, data_scaled.shape[1])

        decoder_input = np.zeros(
            (1, wrapper.output_steps, wrapper.num_decoder_features), dtype=np.float32
        )
        decoder_input[0, 0, 0] = encoder_input[0, -1, target_idx]
        for t in range(1, wrapper.output_steps):
            decoder_input[0, t, 0] = decoder_input[0, t - 1, 0]
        if wrapper.persistence_gate:
            decoder_input[0, :, 1] = encoder_input[0, -1, target_idx]

        pred = wrapper.model.predict([encoder_input, decoder_input], verbose=0)
        predictions.append(pred[0, -1, 0])

        # Leak: the window ingests the observed value output_steps-1 ahead.
        window = np.vstack([window, data_scaled[i + wrapper.output_steps - 1]])[1:]

    predictions = np.array(predictions)
    actuals = data_scaled[test_start + wrapper.output_steps - 1:n, target_idx]
    pred_inv = wrapper.scaler_target.inverse_transform(predictions.reshape(-1, 1))
    act_inv = wrapper.scaler_target.inverse_transform(
        actuals.reshape(-1, 1).astype(np.float64)
    )
    return pred_inv, act_inv


def evaluate(wrapper_key, model_json, out_dir, epochs, verbose=0):
    tf = common.setup_tensorflow(True)
    from keras import backend as K

    with open(common.resolve(model_json), encoding="utf-8") as handle:
        metadata = json.load(handle)

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_ratio = metadata["dataset"].get("train_ratio", 0.75)
    val_ratio = metadata["dataset"].get("val_ratio", 0.20)
    train_df, val_df, _ = common.split_dataset(dataset, train_ratio, val_ratio)
    val_end = len(train_df) + len(val_df)
    print(f"Dataset: {len(dataset)} rows | test start (val_end)={val_end}")

    hp = metadata["hyperparameters"]
    tf.keras.utils.set_random_seed(42)
    K.clear_session()
    wrapper_class = common.wrapper_factory(wrapper_key)
    wrapper = wrapper_class()
    wrapper.prepare(
        train_df, val_df,
        input_steps=metadata["input_steps"],
        output_steps=metadata["output_steps"],
        target_col=metadata["target_col"],
        denoise=metadata["denoise"],
        denoise_level=metadata["denoise_level"],
        persistence_gate=bool(metadata.get("persistence_gate", False)),
    )
    batch_size = int(hp.get("batch_size", 32))
    if hasattr(wrapper, "schedule_total_steps") and hp.get("lr_schedule") == "warmup_cosine":
        steps_per_epoch = int((len(train_df) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps_per_epoch * epochs
    wrapper.loss = metadata.get("loss", "mse")

    wrapper.build(common.FixedHyperParameters(hp))
    callbacks = common.default_callbacks(wrapper, int(metadata["training"]["patience"]))
    print(f"[{wrapper_key}] training up to {epochs} epochs (batch {batch_size})...")
    t0 = time.perf_counter()
    wrapper.fit(
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        use_validation=True,
    )
    train_sec = time.perf_counter() - t0
    print(f"[{wrapper_key}] trained in {train_sec:.1f}s")

    results = {}
    # Both protocols cover target rows [val_end + output_steps - 1, n); the
    # raw reference is that same slice of the undenoised ws100 column.
    raw_col = metadata["target_col"].removesuffix("_wavelet")
    out_steps = int(metadata["output_steps"])
    count = len(dataset) - out_steps + 1 - val_end
    raw_actuals = dataset[raw_col].to_numpy()[
        val_end + out_steps - 1 : val_end + out_steps - 1 + count
    ] if raw_col in dataset.columns else None

    def _pair_metrics(pred, act_wavelet):
        metrics = common.compute_metrics(act_wavelet.ravel(), pred.ravel())
        if raw_actuals is not None:
            raw = common.compute_metrics(raw_actuals, pred.ravel())
            metrics["mae_raw"] = raw["mae"]
            metrics["rmse_raw"] = raw["rmse"]
            metrics["r2_raw"] = raw["r2"]
        return metrics

    def _report(label, protocol_results):
        m = protocol_results["metrics"]
        line = (f"{label}: wavelet MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                f"R²={m['r2']:.4f}")
        if "mae_raw" in m:
            line += f" | raw MAE={m['mae_raw']:.4f} RMSE={m['rmse_raw']:.4f} R²={m['r2_raw']:.4f}"
        print(line)

    t0 = time.perf_counter()
    pred_g, act_g = wrapper.rolling_forecast(dataset, test_start=val_end)
    results["genuine"] = {
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "metrics": _pair_metrics(pred_g, act_g),
    }
    _report("genuine", results["genuine"])

    t0 = time.perf_counter()
    pred_l, act_l = leaky_rolling_forecast(wrapper, dataset, test_start=val_end)
    results["leaky"] = {
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "metrics": _pair_metrics(pred_l, act_l),
    }
    _report("leaky", results["leaky"])

    out_dir = common.resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(out_dir / "metrics.json", {
        "model": {"wrapper_key": wrapper_key, "name": wrapper.name},
        "protocols": results,
    })
    print(f"Saved to {out_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare genuine vs leaky rolling forecast (Keras).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--wrappers", nargs="*", default=list(MODELS))
    args = parser.parse_args()

    summary = {}
    for wk in args.wrappers:
        print(f"\n{'='*60}\nEvaluating {wk}\n{'='*60}")
        summary[wk] = evaluate(wk, *MODELS[wk], epochs=args.epochs)

    print("\n" + render(summary))
    common.write_json(common.resolve("data/results/rolling_protocols_keras/summary.json"), {
        "models": {k: {"protocols": v} for k, v in summary.items()}
    })


def render(summary):
    lines = [
        "# Comparativo genuíno vs leaky — branch Keras (baseline_keras)",
        "",
        "Métricas principais contra o alvo waveletado (`ws100_wavelet`);",
        "colunas `raw` comparam as mesmas previsões contra o `ws100` sem denoising.",
        "",
        "| wrapper | protocolo | MAE wavelet | RMSE wavelet | R² wavelet | MAE raw | RMSE raw | R² raw |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for wk, prots in summary.items():
        for proto in ("genuine", "leaky"):
            m = prots[proto]["metrics"]
            has_raw = "mae_raw" in m
            lines.append(
                f"| {wk} | {proto} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | "
                + (f"{m['mae_raw']:.4f} | {m['rmse_raw']:.4f} | {m['r2_raw']:.4f} |"
                   if has_raw else "— | — | — |")
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()