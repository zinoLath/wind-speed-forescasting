"""AB: 9 features + sem pós-dropout + lag de 24h (144 passos).

Os lags são calculados por fatia (train/val/test) sobre as colunas wavelet,
exatamente como o prepare as produz, com preenchimento bfill nas 144 linhas
iniciais de cada fatia (só afeta o começo do treino).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from keras import backend as K

import numpy as np
from src import common
import src.models.seq2seq_wrapper as sw
from src.models.s2s_tcn_wrapper import S2STCNWrapper
from src.utils import wavelet_denoising


class _IdentityScaler:
    """Substituto do MinMaxScaler: nao normaliza nada."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X, y=None):
        return X
    def inverse_transform(self, X):
        return np.asarray(X)


sw.MinMaxScaler = _IdentityScaler

CFG = json.load(open(PROJECT_ROOT / "pipeline/tmp/ab_nopostdrop_9feat.json"))["common"]
HP = CFG["hyperparameters"]
LAG = 144  # 24 h em passos de 10 min

WAV_SRC = CFG["denoise"]                      # ws40, ws100, ws260, disp40, vdisp40
FEATURES = list(CFG["features"]) + [f"{c}_wavelet_lag{LAG}" for c in WAV_SRC]

dataset = common.load_dataset(PROJECT_ROOT / "data/dataset.csv",
                              keep_raw=tuple(WAV_SRC))
n = len(dataset)
splits = common.split_dataset(dataset, CFG["train_ratio"], CFG["val_ratio"])
names = ("train", "val", "test")
prepared = {}
for name, sl in zip(names, splits):
    sl = sl.copy()
    for col in WAV_SRC:
        sl[f"{col}_wavelet"] = wavelet_denoising(sl[col].values, level=CFG["denoise_level"])
    for col in WAV_SRC:
        sl[f"{col}_wavelet_lag{LAG}"] = sl[f"{col}_wavelet"].shift(LAG).bfill()
    prepared[name] = sl
    print(f"{name}: {sl.shape[1]} colunas, {sl.shape[0]} linhas")

K.clear_session()
np.random.seed(CFG["seed"])
wrapper = S2STCNWrapper()
wrapper.prepare(prepared["train"], prepared["val"],
                input_steps=CFG["input_steps"], output_steps=CFG["output_steps"],
                target_col=CFG["target_col"], denoise=tuple(WAV_SRC),
                denoise_level=CFG["denoise_level"], features=FEATURES,
                decoder_mode=CFG.get("decoder_mode", "teacher_forcing"),
                persistence_gate=bool(CFG.get("persistence_gate", False)))
batch = int(HP.get("batch_size", 32))
if hasattr(wrapper, "schedule_total_steps"):
    wrapper.schedule_total_steps = -(-wrapper.train["X_encoder"].shape[0] // batch) * 100
wrapper.loss = HP.get("loss", CFG["loss"])
wrapper.build(common.FixedHyperParameters(HP))
callbacks = common.default_callbacks(wrapper, CFG["patience"])
history = wrapper.fit(epochs=100, batch_size=batch, verbose=0,
                      callbacks=callbacks, use_validation=True)
epochs = len(history.history["loss"])
val_loss = float(min(history.history["val_loss"]))
print(f"treino: {epochs} épocas | val_loss={val_loss:.5f} | "
      f"inputs={wrapper.num_encoder_features}")


def blk(df):
    df = common.predict_all_horizons(wrapper, df)
    pred = df["predicted"].to_numpy()
    real = dataset["ws100"].reindex(df["timestamp"]).to_numpy()
    return {"mae": float(np.mean(np.abs(pred - real))),
            "mse": float(np.mean((pred - real) ** 2)),
            "r2": float(common.compute_metrics(real, pred)["r2"]),
            "bias": float(np.mean(pred - real))}


val_m = blk(prepared["val"])
test_m = blk(prepared["test"])
import pandas as pd
full = pd.concat([prepared["train"], prepared["val"], prepared["test"]])
roll_pred, roll_act = wrapper.rolling_forecast(full, test_start=len(prepared["train"]) + len(prepared["val"]))
roll = {"mae": float(np.mean(np.abs(roll_pred - roll_act))),
        "mse": float(np.mean((roll_pred - roll_act) ** 2)),
        "r2": float(common.compute_metrics(roll_act, roll_pred)["r2"])}

out = {"condicao": "9 features + lag24h, sem pos-dropout, SEM normalizacao",
       "epochs": epochs, "val_loss_scaled": val_loss,
       "val_all_horizons": val_m, "test_all_horizons": test_m, "rolling_h36": roll}
json.dump(out, open(PROJECT_ROOT / "pipeline/tmp/ab_lag24_nonorm/metrics.json", "w"), indent=1)
# layout padrao, para o build_investigacao coletar
std_dir = PROJECT_ROOT / "pipeline/tmp/ab_lag24_nonorm/Seq2Seq_TCN/evaluate"
std_dir.mkdir(parents=True, exist_ok=True)
json.dump({"all_horizons": out["test_all_horizons"],
           "rolling": {"mae": out["rolling_h36"]["mae"], "mse": out["rolling_h36"]["mse"],
                       "rmse": out["rolling_h36"]["mse"] ** 0.5,
                       "r2": out["rolling_h36"]["r2"]},
           "model": {"training": {"epochs_run": out["epochs"],
                                  "best_val_loss": out["val_loss_scaled"]}},
           "condicao": out["condicao"]},
          open(std_dir / "metrics.json", "w"), indent=1)
print(json.dumps(out, indent=1))
