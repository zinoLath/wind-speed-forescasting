"""Split invertido com a LSTM: validação (0–20%) → treino (20–95%) → teste (95–100%).

A janela de teste é a mesma do split padrão (últimos 5%), mas a validação
agora vem de um regime distante (início da série) e o treino termina
encostado no teste. Compara com o split padrão (75/20/5).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from keras import backend as K

from src import common
from src.models.s2s_lstm_wrapper import S2SLSTMWrapper

CFG = json.load(open("/tmp/opencode/report_compare.json"))["common"]
HP = json.load(open(PROJECT_ROOT / "pipeline/tmp/hp_v2_stage/optuna/"
                    "Seq2Seq_LSTM/best_trial.json"))["best_params"]
OUT = PROJECT_ROOT / "pipeline/tmp/split_val_train_test"
OUT.mkdir(parents=True, exist_ok=True)

dataset = common.load_dataset(PROJECT_ROOT / "data/dataset.csv",
                              keep_raw=tuple(CFG["denoise"]))
n = len(dataset)
val_df = dataset.iloc[:int(n * 0.20)]
train_df = dataset.iloc[int(n * 0.20):int(n * 0.95)]
test_df = dataset.iloc[int(n * 0.95):]
print(f"val {len(val_df)} | treino {len(train_df)} | teste {len(test_df)}")

K.clear_session()
wrapper = S2SLSTMWrapper()
wrapper.prepare(train_df, val_df, input_steps=CFG["input_steps"],
                output_steps=CFG["output_steps"], target_col=CFG["target_col"],
                denoise=tuple(CFG["denoise"]), denoise_level=CFG["denoise_level"],
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
print(f"treino: {epochs} épocas | val_loss={val_loss:.5f}")


def blk(df):
    df = common.predict_all_horizons(wrapper, df)
    pred = df["predicted"].to_numpy()
    real = dataset["ws100"].reindex(df["timestamp"]).to_numpy()
    return {"mae": float(np.mean(np.abs(pred - real))),
            "mse": float(np.mean((pred - real) ** 2)),
            "r2": float(common.compute_metrics(real, pred)["r2"]),
            "bias": float(np.mean(pred - real))}


val_m = blk(val_df)
test_m = blk(test_df)
roll_pred, roll_act = wrapper.rolling_forecast(dataset, test_start=len(train_df) + len(val_df))
roll = {"mae": float(np.mean(np.abs(roll_pred - roll_act))),
        "mse": float(np.mean((roll_pred - roll_act) ** 2))}
persist_roll = float(np.mean(np.abs(dataset["ws100"].values[
    len(train_df) + len(val_df) - 1: len(train_df) + len(val_df) - 1 + len(roll_act)]
    - roll_act)))

out = {
    "split": "val 0-20% | treino 20-95% | teste 95-100%",
    "epochs": epochs, "val_loss_scaled": val_loss,
    "val_all_horizons": val_m, "test_all_horizons": test_m,
    "rolling_h36_vs_raw": roll, "persistence_rolling_mae": persist_roll,
    "vento_medio": {"val": float(val_df["ws100"].mean()),
                    "treino": float(train_df["ws100"].mean()),
                    "teste": float(test_df["ws100"].mean())},
}
json.dump(out, open(OUT / "metrics.json", "w"), indent=1)
print(json.dumps(out, indent=1))
