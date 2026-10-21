"""Re-execução fiel da implementação antiga (OLD/notebooks/wind_speed_forecasting.ipynb)
com medição expandida sob o protocolo atual.

Parte 1 — código original preservado (split 77/18/5, todas as colunas, teacher
forcing, scalers ajustados no dataset inteiro, BiLSTM 256 + atenção, mse).
Parte 2 — medição expandida (rigor atual): teacher-forced com vazamento,
all-horizons honesto (decoder de inferência), rolling autônomo h=36,
persistência, por horizonte, tudo contra o ws100 bruto e contra a wavelet.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Bidirectional, Dropout, Dense,
                                     Concatenate, TimeDistributed, Attention)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

from src import common
from src.utils import wavelet_denoising

OUT = PROJECT_ROOT / "pipeline" / "tmp" / "old_impl_rerun"
OUT.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Parte 1 — implementação original (células do notebook, com paths adaptados)
# ===========================================================================
variables = pd.read_csv(PROJECT_ROOT / "data" / "dataset.csv", index_col=0,
                        parse_dates=True)
# O dataset.csv atual não traz as colunas wavelet; gera a do alvo com a
# wavelet padrão do projeto (sym18, nível 2 — default histórico da utils).
variables["ws100_wavelet"] = wavelet_denoising(variables["ws100"].values, level=2)

scaler_ws100 = MinMaxScaler()
scaler_other = MinMaxScaler()

variables_scaled = variables.copy()

ws100_columns = ["ws100_wavelet"]
other_columns = variables.columns.drop("ws100_wavelet").tolist()

variables_scaled[ws100_columns] = scaler_ws100.fit_transform(variables[ws100_columns])
variables_scaled[other_columns] = scaler_other.fit_transform(variables[other_columns])

print(f"Shape of scaled data: {variables_scaled.shape}")

input_steps = 72
output_steps = 36


def create_sequences(data, input_steps, output_steps, target_col_index):
    X_encoder, X_decoder, y_decoder = [], [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X_encoder.append(data[i:(i + input_steps)])
        decoder_input = np.zeros((output_steps, 1))
        decoder_input[0] = data[i + input_steps - 1, target_col_index]
        decoder_input[1:] = data[i + input_steps:i + input_steps + output_steps - 1,
                                 target_col_index].reshape(-1, 1)
        X_decoder.append(decoder_input)
        y_decoder.append(data[i + input_steps:i + input_steps + output_steps,
                              target_col_index].reshape(-1, 1))
    return (np.array(X_encoder), np.array(X_decoder), np.array(y_decoder))


data_array = variables_scaled.values
target_col_index = variables_scaled.columns.get_loc("ws100_wavelet")
X_encoder, X_decoder, y_decoder = create_sequences(data_array, input_steps,
                                                   output_steps, target_col_index)
print(f"X_encoder: {X_encoder.shape} | X_decoder: {X_decoder.shape} | y: {y_decoder.shape}")

train_end = int(X_encoder.shape[0] * 0.77)
val_end = int(X_encoder.shape[0] * 0.95)

X_encoder_train = X_encoder[:train_end]
X_decoder_train = X_decoder[:train_end]
y_decoder_train = y_decoder[:train_end]
X_encoder_val = X_encoder[train_end:val_end]
X_decoder_val = X_decoder[train_end:val_end]
y_decoder_val = y_decoder[train_end:val_end]
X_encoder_test = X_encoder[val_end:]
X_decoder_test = X_decoder[val_end:]
y_decoder_test = y_decoder[val_end:]
print(f"test: {X_encoder_test.shape}")

num_encoder_features = X_encoder_train.shape[2]
num_decoder_features = 1


def build_seq2seq_attention_model(input_steps, output_steps, num_encoder_features,
                                  num_decoder_features):
    optimizer = Adam(learning_rate=1e-3)
    encoder_inputs = Input(shape=(input_steps, num_encoder_features),
                           name="encoder_inputs")
    encoder_lstm = Bidirectional(LSTM(256, return_sequences=True, return_state=True),
                                 name="bidirectional_encoder_lstm")
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = \
        encoder_lstm(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_outputs = Dropout(0.1)(encoder_outputs)
    decoder_inputs = Input(shape=(output_steps, num_decoder_features),
                           name="decoder_inputs")
    decoder_lstm = LSTM(256 * 2, return_sequences=True, return_state=True,
                        name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=[state_h, state_c])
    decoder_outputs = Dropout(0.1)(decoder_outputs)
    attention_outputs = Attention(name="attention_layer")(
        [decoder_outputs, encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1)(
        [decoder_outputs, attention_outputs])
    decoder_outputs_final = TimeDistributed(Dense(1, activation="linear"),
                                            name="output_layer")(decoder_combined_context)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


model = build_seq2seq_attention_model(input_steps, output_steps,
                                      num_encoder_features, num_decoder_features)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
]

history = model.fit(
    [X_encoder_train, X_decoder_train], y_decoder_train,
    epochs=100, batch_size=32,
    validation_data=([X_encoder_val, X_decoder_val], y_decoder_val),
    callbacks=callbacks, verbose=1,
)
print(f"treino: {len(history.history['loss'])} épocas | "
      f"melhor val_loss={min(history.history['val_loss']):.5f}")

# ===========================================================================
# Parte 2 — medição expandida (protocolo atual)
# ===========================================================================
raw_ws100 = variables["ws100"].to_numpy()
n_seq = X_encoder.shape[0]


def inverse_targets(scaled):
    return scaler_ws100.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()


def metrics_block(pred_wav_scaled, first_seq, last_seq):
    """Métricas de um conjunto de previsões (espaço wavelet escalado).

    first_seq/last_seq: índices de sequência (no grid completo) que as
    previsões cobrem; alvo do passo h fica na linha first_seq + 72 + h - 1
    do dataset original. Retorna métricas vs ws100 bruto e vs wavelet.
    """
    h_idx = np.arange(output_steps)
    rows = first_seq[:, None] + input_steps + h_idx[None, :]
    pred_wav_vals = inverse_targets(pred_wav_scaled.ravel())
    wav_real = inverse_targets(y_decoder[first_seq].ravel())
    real = raw_ws100[rows.ravel()]
    persist = np.tile(
        raw_ws100[first_seq + input_steps - 1][:, None], (1, output_steps)).ravel()
    def blk(p, a):
        return {"mae": float(mean_absolute_error(a, p)),
                "mse": float(mean_squared_error(a, p)),
                "rmse": float(np.sqrt(mean_squared_error(a, p))),
                "r2": float(r2_score(a, p))}
    return {"vs_raw": blk(pred_wav_vals, real),
            "vs_wavelet": blk(pred_wav_vals, wav_real)}, {
        "pred_raw": pred_wav_vals, "real": real, "persist": persist, "rows": rows}


# --- 1. protocolo antigo: teacher forcing com vazamento (como o notebook media)
tf_scaled = model.predict([X_encoder_test, X_decoder_test], verbose=0)
first_test = np.arange(val_end, n_seq)
block_tf, data_tf = metrics_block(tf_scaled, first_test, n_seq)
print(f"[antigo/teacher-forced] vs wavelet: MAE={block_tf['vs_wavelet']['mae']:.4f} "
      f"(o '0.15' histórico é este número em escala MinMax: "
      f"{mean_absolute_error(y_decoder_test.ravel(), tf_scaled.ravel()):.4f})")
print(f"[antigo/teacher-forced] vs bruto : MAE={block_tf['vs_raw']['mae']:.4f}")

# --- 2. all-horizons honesto: decoder de inferência (último observado repetido)
dec_inf = np.zeros_like(X_decoder_test)
dec_inf[:, :, 0] = X_decoder_test[:, :1, 0]
inf_scaled = model.predict([X_encoder_test, dec_inf], verbose=0)
block_inf, data_inf = metrics_block(inf_scaled, first_test, n_seq)

# --- 3. rolling autônomo h=36 (função original do notebook)
def rolling_forecasting_real_time_improved(model, data_scaled, input_steps,
                                           output_steps, scaler_ws100, target_col_index):
    predictions, actuals = [], []
    test_start = int(data_scaled.shape[0] * 0.95)
    test_data = data_scaled[test_start:]
    window_start = test_start - input_steps
    window_end = test_start
    window = data_scaled[window_start:window_end].copy()
    for i in range(len(test_data) - output_steps + 1):
        encoder_input = window.reshape(1, input_steps, data_scaled.shape[1])
        decoder_input = np.zeros((1, output_steps, 1))
        decoder_input[0, 0, 0] = encoder_input[0, -1, target_col_index]
        for t in range(1, output_steps):
            previous_pred = decoder_input[0, t - 1, 0]
            decoder_input[0, t, 0] = previous_pred
        pred = model.predict([encoder_input, decoder_input], verbose=0)
        predictions.append(pred[0, -1, 0])
        actuals.append(test_data[i + output_steps - 1, target_col_index])
        window = np.vstack([window, test_data[i + output_steps - 1]])[1:]
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    return scaler_ws100.inverse_transform(predictions), scaler_ws100.inverse_transform(actuals)


roll_pred_w, roll_act_w = rolling_forecasting_real_time_improved(
    model, data_array, input_steps, output_steps, scaler_ws100, target_col_index)
roll_real = raw_ws100[val_end + output_steps - 1: val_end + output_steps - 1 + len(roll_act_w)]
persist_roll = raw_ws100[val_end - 1: val_end - 1 + len(roll_act_w)]

out = {
    "protocolo": "notebook original: split 77/18/5, todas as colunas, teacher forcing, scaler no dataset inteiro",
    "treino": {"epochs_run": len(history.history["loss"]),
               "best_val_loss": float(min(history.history["val_loss"]))},
    "teacher_forced_leaky": {"vs_wavelet": block_tf["vs_wavelet"],
                             "vs_raw": block_tf["vs_raw"],
                             "mae_scaled_historico": float(mean_absolute_error(
                                 y_decoder_test.ravel(), tf_scaled.ravel()))},
    "all_horizons_honesto": {"vs_raw": block_inf["vs_raw"],
                             "vs_wavelet": block_inf["vs_wavelet"]},
    "rolling_h36": {"vs_wavelet": {
        "mae": float(mean_absolute_error(roll_act_w, roll_pred_w)),
        "mse": float(mean_squared_error(roll_act_w, roll_pred_w)),
        "rmse": float(np.sqrt(mean_squared_error(roll_act_w, roll_pred_w))),
        "r2": float(r2_score(roll_act_w, roll_pred_w))},
        "vs_raw": {
        "mae": float(mean_absolute_error(roll_real, roll_pred_w)),
        "mse": float(mean_squared_error(roll_real, roll_pred_w)),
        "rmse": float(np.sqrt(mean_squared_error(roll_real, roll_pred_w))),
        "r2": float(r2_score(roll_real, roll_pred_w))}},
    "persistence_all_horizons": {"vs_raw": {
        "mae": float(mean_absolute_error(data_inf["real"], data_inf["persist"])),
        "mse": float(mean_squared_error(data_inf["real"], data_inf["persist"]))}},
    "persistence_rolling": {"mae": float(mean_absolute_error(roll_real, persist_roll))},
}
json.dump(out, open(OUT / "metrics.json", "w"), indent=1)

# por horizonte (rolling grid não tem; usa all-horizons honesto vs raw)
ph = []
for h in range(output_steps):
    rows_h = data_inf["rows"][:, h]
    pred_h = data_inf["pred_raw"][h::output_steps]
    real_h = raw_ws100[rows_h]
    ph.append({"horizon": h + 1, "mae": float(np.mean(np.abs(pred_h - real_h)))})
pd.DataFrame(ph).to_csv(OUT / "per_horizon_metrics.csv", index=False)

print("\n=== RESUMO (implementação antiga, medido com rigor atual) ===")
print(f"teacher-forced c/ vazamento vs wavelet: MAE {block_tf['vs_wavelet']['mae']:.4f} | "
      f"vs bruto: {block_tf['vs_raw']['mae']:.4f}")
print(f"all-horizons honesto vs bruto         : MAE {block_inf['vs_raw']['mae']:.4f} "
      f"MSE {block_inf['vs_raw']['mse']:.3f} R² {block_inf['vs_raw']['r2']:+.3f}")
print(f"rolling h36 autônomo vs bruto         : MAE {out['rolling_h36']['vs_raw']['mae']:.4f} "
      f"MSE {out['rolling_h36']['vs_raw']['mse']:.3f} R² {out['rolling_h36']['vs_raw']['r2']:+.3f}")
print(f"persistência (all / rolling)          : MAE {out['persistence_all_horizons']['vs_raw']['mae']:.4f} / {out['persistence_rolling']['mae']:.4f}")
