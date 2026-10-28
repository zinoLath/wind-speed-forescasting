"""Faithful reproduction of the original notebook's evaluation protocol.

Reproduces ``OLD/notebooks/wind_speed_forecasting-original.ipynb`` as closely
as possible -- including its intentional quirks (future leakage in the rolling
window and the growing misalignment), so every wrapper is compared under the
exact same protocol.

Protocol replicated:
- raw dataset.csv read directly (canonical column names), wavelet denoising
  on the notebook's columns (sym18, level 2);
- MinMax scaling: target (ws100_wavelet) with its own scaler, everything else
  with another;
- teacher-forcing sequences (decoder = last observed target);
- split: 77% train / 18% val / 5% test (computed on sequences, like the
  notebook);
- training: MSE loss, Adam lr=1e-3, batch 32, early stopping patience 10,
  reduce LR patience 5, up to 100 epochs;
- rolling forecast exactly like ``rolling_forecasting_real_time_improved``:
  decoder fed with the last observed value repeated, prediction taken from the
  last decoder step, actuals from ``test_data[i + output_steps - 1]``, and the
  REAL future sample appended to the encoder window (the notebook's leak).

Outputs predictions/metrics per model under ``pipeline/tmp/notebook_protocol/``
and a comparison report.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src import common
from src.utils import wavelet_denoising

OUT_BASE = Path("pipeline/tmp/notebook_protocol")

# Wrappers compared under the notebook protocol. lstm_original uses the
# notebook's hardcoded hyperparameters; the others use the shared defaults so
# the training recipe (MSE / Adam 1e-3 / constant LR) is identical across
# models and only the architecture differs.
WRAPPERS = [
    "lstm_original",
    "lstm",
    "lstm_bi",
    "gru",
    "gru_bi",
    "lstm_cnn",
    "tcn",
    "tcn_bi",
    "tcn_lstm",
    "transformer",
]

INPUT_STEPS = 72
OUTPUT_STEPS = 36
TARGET_COL = "ws100_wavelet"
# Columns denoised in the original notebook (canonical names in the current
# dataset: wdisp40 -> disp40, vertdisp40 -> vdisp40, wdir40 -> dir40).
DENOISE_COLS = ["ws100", "disp40", "vdisp40", "dir40", "cis1", "humid", "temp"]

# Fixed training recipe taken from the notebook.
LOSS = "mse"
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 100
EARLY_PATIENCE = 10
LR_PATIENCE = 5


def load_raw_dataset():
    """Read dataset.csv directly (like the notebook) with canonical columns."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "dataset.csv")
    df["id"] = pd.to_datetime(df["id"], format="mixed")
    df = df.sort_values("id").set_index("id")
    return common.canonicalize_columns(df)


def build_notebook_features(df):
    """Apply the notebook's wavelet denoising columns to *df* (in place copy)."""
    df = df.copy()
    for col in DENOISE_COLS:
        if col not in df.columns:
            raise ValueError(f"Column {col} required by the notebook protocol is missing.")
        if f"{col}_wavelet" not in df.columns:
            df[f"{col}_wavelet"] = wavelet_denoising(df[col].values, level=2)
    return df


def rolling_forecast_notebook(wrapper, data_scaled, target_col_index, test_start):
    """Exact port of ``rolling_forecasting_real_time_improved`` from the notebook."""
    predictions = []
    actuals = []

    test_data = data_scaled[test_start:]
    window = data_scaled[test_start - wrapper.input_steps:test_start].copy()

    for i in range(len(test_data) - wrapper.output_steps + 1):
        encoder_input = window.reshape(1, wrapper.input_steps, data_scaled.shape[1])

        decoder_input = np.zeros((1, wrapper.output_steps, 1))
        decoder_input[0, 0, 0] = encoder_input[0, -1, target_col_index]
        for t in range(1, wrapper.output_steps):
            decoder_input[0, t, 0] = decoder_input[0, t - 1, 0]

        pred = wrapper.model.predict([encoder_input, decoder_input], verbose=0)
        predictions.append(pred[0, -1, 0])

        actuals.append(test_data[i + wrapper.output_steps - 1, target_col_index])

        # The notebook's leak: the REAL future sample is appended to the window.
        window = np.vstack([window, test_data[i + wrapper.output_steps - 1]])
        window = window[1:]

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    predictions = wrapper.scaler_target.inverse_transform(predictions)
    actuals = wrapper.scaler_target.inverse_transform(actuals)
    return predictions, actuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrappers", nargs="*", default=None,
        help="Subset of wrappers to run (default: all).",
    )
    parser.add_argument(
        "--optuna", action="store_true",
        help="Use the Optuna best hyperparameters for wrappers that have an "
             "Optuna study (instead of the notebook's fixed recipe).",
    )
    args = parser.parse_args()
    wrappers = args.wrappers or WRAPPERS
    unknown = set(wrappers) - set(WRAPPERS)
    if unknown:
        raise SystemExit(f"Unknown wrappers: {sorted(unknown)}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset()
    df = build_notebook_features(df)
    print(f"Dataset: {df.shape}")

    variables_scaled = df.copy()
    ws100_cols = [TARGET_COL]
    other_cols = df.columns.drop(TARGET_COL).tolist()

    scaler_ws100 = MinMaxScaler()
    scaler_other = MinMaxScaler()
    variables_scaled[ws100_cols] = scaler_ws100.fit_transform(df[ws100_cols])
    variables_scaled[other_cols] = scaler_other.fit_transform(df[other_cols])
    data_array = variables_scaled.values.astype(np.float32)
    target_col_index = variables_scaled.columns.get_loc(TARGET_COL)
    print(f"Scaled data: {data_array.shape} | target index: {target_col_index}")

    # Sequence split exactly like the notebook.
    n_seq = len(df) - INPUT_STEPS - OUTPUT_STEPS + 1
    train_end_seq = int(n_seq * 0.77)
    val_end_seq = int(n_seq * 0.95)
    print(f"Sequences: {n_seq} | train_end={train_end_seq} val_end={val_end_seq}")

    # Split rows so the sequence boundaries match the notebook's split.
    train_end_row = train_end_seq + INPUT_STEPS - 1
    val_end_row = val_end_seq + INPUT_STEPS - 1
    train_df = df.iloc[:train_end_row].copy()
    val_df = df.iloc[train_end_row:val_end_row].copy()
    print(f"Rows: train={len(train_df)} val={len(val_df)} test={len(df) - val_end_row}")

    test_start = int(len(df) * 0.95)
    print(f"Rolling test_start (0.95*n): {test_start}")

    results = {}
    for wrapper_key in wrappers:
        print(f"\n=== Training {wrapper_key} ===")
        from keras import backend as K

        K.clear_session()
        wrapper_class = common.wrapper_factory(wrapper_key)
        wrapper = wrapper_class()
        # Notebook recipe: teacher forcing, no persistence gate, MSE, Adam 1e-3.
        wrapper.input_steps = INPUT_STEPS
        wrapper.output_steps = OUTPUT_STEPS
        wrapper.target_col = TARGET_COL
        wrapper.denoise = ()
        wrapper.denoise_level = 2
        wrapper.decoder_mode = "teacher_forcing"
        wrapper.target_mode = "absolute"
        wrapper.persistence_gate = False

        # Prepare with the GLOBAL scalers (fit on the whole scaled array, exactly
        # like the notebook) so the rolling forecast and training share one scale.
        wrapper.train, _ = wrapper.prepare_data(
            train_df, INPUT_STEPS, OUTPUT_STEPS, TARGET_COL,
            scaler_target=scaler_ws100, scaler_other=scaler_other,
            denoise=(), decoder_mode="teacher_forcing",
            target_mode="absolute", persistence_gate=False,
        )
        wrapper.val, _ = wrapper.prepare_data(
            val_df, INPUT_STEPS, OUTPUT_STEPS, TARGET_COL,
            scaler_target=scaler_ws100, scaler_other=scaler_other,
            denoise=(), decoder_mode="teacher_forcing",
            target_mode="absolute", persistence_gate=False,
        )
        wrapper.scaler_target = wrapper.train["scaler_target"]
        wrapper.scaler_other = wrapper.train["scaler_other"]
        wrapper.target_col_index = wrapper.train["target_col_index"]
        wrapper.num_encoder_features = wrapper.train["X_encoder"].shape[2]
        wrapper.num_decoder_features = wrapper.train["X_decoder"].shape[2]
        wrapper.loss = LOSS
        wrapper.schedule_total_steps = None

        if args.optuna:
            best_params_path = common.find_optuna_result(wrapper_key)
            if best_params_path is not None:
                with open(best_params_path, encoding="utf-8") as handle:
                    optuna_params = json.load(handle)["best_params"]
            else:
                optuna_params = {}
        else:
            optuna_params = {}

        hp = common.FixedHyperParameters(
            {
                # Use each wrapper's built-in architecture defaults so build()
                # has all keys, then force the notebook's training recipe.
                **common.DEFAULT_HYPERPARAMETERS[wrapper_key],
                "loss": LOSS,
                "lr_schedule": "constant",
                "learning_rate": LR,
                "weight_decay": 0.0,
                # Optuna winners win over the fixed recipe when requested.
                **optuna_params,
            }
        )

        # With a warmup_cosine schedule (from Optuna), the LR horizon must match
        # the training length, mirroring how step_train/step_optuna set it.
        schedule_mode = getattr(wrapper, "lr_schedule_mode", None)
        if schedule_mode == "warmup_cosine":
            steps_per_epoch = int(np.ceil(len(wrapper.train["X_encoder"]) / BATCH_SIZE))
            wrapper.schedule_total_steps = steps_per_epoch * EPOCHS
        wrapper.build(hp)

        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=EARLY_PATIENCE, restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=LR_PATIENCE, min_lr=1e-6
            ),
        ]
        history = wrapper.fit(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=callbacks,
            use_validation=True,
        )
        epochs_run = len(history.history.get("loss", []))

        print(f"  trained {epochs_run} epochs | best val_loss="
              f"{min(history.history['val_loss']):.6f}")

        # Notebook rolling forecast on the SCALED full array.
        predictions, actuals = rolling_forecast_notebook(
            wrapper, data_array, target_col_index, test_start
        )

        # Notebook metrics: MAE/RMSE against aligned denoised actuals, and
        # against the last len(predictions) raw ws100 values.
        mae_denoised = mean_absolute_error(actuals, predictions)
        rmse_denoised = np.sqrt(mean_squared_error(actuals, predictions))

        raw_ws100 = df["ws100"].iloc[-len(predictions):].values
        mae_raw = mean_absolute_error(raw_ws100, predictions)
        rmse_raw = np.sqrt(mean_squared_error(raw_ws100, predictions))

        results[wrapper_key] = {
            "name": wrapper.name,
            "epochs": epochs_run,
            "mae_denoised": mae_denoised,
            "rmse_denoised": rmse_denoised,
            "mae_raw_lastN": mae_raw,
            "rmse_raw_lastN": rmse_raw,
            "n_preds": len(predictions),
        }
        print(f"  denoised MAE={mae_denoised:.4f} RMSE={rmse_denoised:.4f}")
        print(f"  raw(last {len(predictions)}) MAE={mae_raw:.4f} RMSE={rmse_raw:.4f}")

        # Save per-model artifacts.
        model_dir = OUT_BASE / wrapper.name
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"actual_denoised": actuals.ravel(), "predicted": predictions.ravel(),
             "raw_ws100": raw_ws100}
        ).to_csv(model_dir / "predictions.csv", index=False)

    with open(OUT_BASE / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=float)

    # Merge with previously saved partial results, if any.
    prev = OUT_BASE / "summary.json"
    merged = {}
    if prev.is_file():
        with open(prev, encoding="utf-8") as handle:
            merged.update(json.load(handle))
    merged.update(results)
    with open(OUT_BASE / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, default=float)

    print("\n" + render_markdown(merged))
    with open(OUT_BASE / "summary.md", "w", encoding="utf-8") as handle:
        handle.write(render_markdown(merged))


def render_markdown(results):
    lines = [
        "# Comparativo sob o protocolo do notebook original (com leak)",
        "",
        "Treino: MSE, Adam 1e-3, batch 32, teacher forcing, sem persistence "
        "gate. Rolling forecast idêntico ao notebook "
        "(`rolling_forecasting_real_time_improved`): janela recebe o valor real "
        "futuro (leak) e a comparação usa `test_data[i + output_steps - 1]`.",
        "",
        f"| modelo | epochs | MAE denoised | RMSE denoised | MAE raw(últ. N) | RMSE raw(últ. N) |",
        f"|---|---|---|---|---|---|",
    ]
    for row in results.values():
        lines.append(
            f"| {row['name']} | {row['epochs']} | {row['mae_denoised']:.4f} | "
            f"{row['rmse_denoised']:.4f} | {row['mae_raw_lastN']:.4f} | "
            f"{row['rmse_raw_lastN']:.4f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()