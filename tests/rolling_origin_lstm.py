"""Rolling-origin com a LSTM: várias divisões treino/val/teste para medir
o gap validação → teste em diferentes regimes de vento.

Em cada fold: treino expansivo, val de 10%, teste de 5% (janelas deslizantes).
Mesmos hp do staging v2, decoder teacher forcing, paciencia 12.
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
OUT = PROJECT_ROOT / "pipeline/tmp/rolling_origin_lstm"
OUT.mkdir(parents=True, exist_ok=True)

dataset = common.load_dataset(PROJECT_ROOT / "data/dataset.csv",
                              keep_raw=tuple(CFG["denoise"]))
n = len(dataset)
train_df_full, val_df_full, test_df_full = common.split_dataset(dataset)
input_steps, output_steps = CFG["input_steps"], CFG["output_steps"]

folds = []
for k in range(6):
    s = int(n * (0.45 + 0.075 * k))          # inicio da val
    v_end = s + int(n * 0.10)                # fim da val / inicio do teste
    t_end = v_end + int(n * 0.05)            # fim do teste
    folds.append((0, s, v_end, min(t_end, n)))

results = []
for k, (t0, t1, v1, t2) in enumerate(folds):
    train = dataset.iloc[t0:t1]
    val = dataset.iloc[t1:v1]
    test = dataset.iloc[v1:t2]
    val_end_abs = v1
    print(f"\n=== fold {k}: treino {t0}-{t1} | val {t1}-{v1} | teste {v1}-{t2} ===")

    K.clear_session()
    np.random.seed(CFG["seed"])
    wrapper = S2SLSTMWrapper()
    wrapper.prepare(train, val, input_steps=input_steps, output_steps=output_steps,
                    target_col=CFG["target_col"], denoise=tuple(CFG["denoise"]),
                    denoise_level=CFG["denoise_level"],
                    decoder_mode=CFG.get("decoder_mode", "teacher_forcing"),
                    persistence_gate=bool(CFG.get("persistence_gate", False)))
    if hasattr(wrapper, "schedule_total_steps"):
        batch = int(HP.get("batch_size", CFG["batch_size"]))
        wrapper.schedule_total_steps = -(-wrapper.train["X_encoder"].shape[0] // batch) * 100
    wrapper.loss = HP.get("loss", CFG["loss"])
    wrapper.build(common.FixedHyperParameters(HP))
    callbacks = common.default_callbacks(wrapper, CFG["patience"])
    wrapper.fit(epochs=100, batch_size=int(HP.get("batch_size", 32)),
                verbose=0, callbacks=callbacks, use_validation=True)
    epochs = len(wrapper.history["loss"]) if hasattr(wrapper, "history") else -1

    # val e teste sob o MESMO protocolo (all-horizons, decoder de inferencia)
    val_df = common.predict_all_horizons(wrapper, val)
    tst_df = common.predict_all_horizons(wrapper, test)

    def blk(df):
        pred, real = df["predicted"], df["actual_raw"]
        bias = float(np.mean(pred - real))
        return {"mae": float(np.mean(np.abs(pred - real))),
                "mse": float(np.mean((pred - real) ** 2)),
                "r2": float(common.compute_metrics(real, pred)["r2"]),
                "bias": bias}

    v_m, t_m = blk(val_df), blk(tst_df)
    gap = t_m["mse"] - v_m["mse"]
    wind_shift = float(test["ws100"].mean() - train["ws100"].mean())
    fold_out = {
        "fold": k,
        "treino_fim_pct": round(t1 / n, 3),
        "val_mae": v_m["mae"], "val_mse": v_m["mse"], "val_bias": v_m["bias"],
        "test_mae": t_m["mae"], "test_mse": t_m["mse"], "test_bias": t_m["bias"],
        "test_r2": t_m["r2"], "gap_mse": gap, "vento_teste_menos_treino": wind_shift,
        "epochs": len(wrapper.history_ if hasattr(wrapper, "history_") else []) or None,
    }
    results.append(fold_out)
    print(json.dumps(fold_out, indent=1))
    json.dump(results, open(OUT / "folds.json", "w"), indent=1)
    K.clear_session()

vals = [r["val_mse"] for r in results]
tests = [r["test_mse"] for r in results]
gaps = [r["gap_mse"] for r in results]
summary = {"gap_medio": float(np.mean(gaps)), "gap_std": float(np.std(gaps)),
           "val_mse_medio": float(np.mean(vals)), "test_mse_medio": float(np.mean(tests)),
           "folds": len(results)}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print("\nRESUMO:", json.dumps(summary, indent=1))
