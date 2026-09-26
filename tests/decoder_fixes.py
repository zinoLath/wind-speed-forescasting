"""Testa as três correções para o decoder achatado (trajetória reta).

Correções (todas com features evid10, HP nopostdrop, seed 42, com norma):
  fix1 — decoder recebe senoides de hora/dia dos instantes-alvo (TF)
  fix2 — decoder recebe codificação posicional senoidal (TF)
  fix3 — decoder direto (convenção de inferência) + senoides futuras
Métrica extra: amplitude média dentro da trajetória (max-min por origem).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from keras import backend as K

from src import common
from src.models.s2s_tcn_wrapper import S2STCNWrapper
from src.utils import wavelet_denoising

CFG = json.load(open(PROJECT_ROOT / "pipeline/tmp/ab_nopostdrop_9feat.json"))["common"]
HP = CFG["hyperparameters"]
CYC = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
WAV_SRC = ("ws40", "ws100", "ws260", "disp40", "vdisp40")
FEATURES = ["ws40_wavelet", "ws260_wavelet", "dir100_sin", "dir100_cos",
            "disp40_wavelet", "vdisp40_wavelet"] + list(CYC)

CONDITIONS = [
    {"slug": "fix1_tf_time", "label": "fix1: TF + senoides futuras",
     "decoder_mode": "teacher_forcing", "decoder_extra": "time"},
    {"slug": "fix2_tf_pos", "label": "fix2: TF + posicional",
     "decoder_mode": "teacher_forcing", "decoder_extra": "pos"},
    {"slug": "fix3_dir_time", "label": "fix3: direto + senoides futuras",
     "decoder_mode": "direct", "decoder_extra": "time"},
]

dataset = common.load_dataset(PROJECT_ROOT / "data/dataset.csv", keep_raw=WAV_SRC)
splits = common.split_dataset(dataset, CFG["train_ratio"], CFG["val_ratio"])

for cond in CONDITIONS:
    if (PROJECT_ROOT / "pipeline/tmp" / cond["slug"] / "metrics.json").exists():
        print(f"[{cond['slug']}] já concluída, pulando")
        continue
    prepared = {}
    for name, sl in zip(("train", "val", "test"), splits):
        sl = sl.copy()
        for col in WAV_SRC:
            sl[f"{col}_wavelet"] = wavelet_denoising(sl[col].values,
                                                     level=CFG["denoise_level"])
        prepared[name] = sl

    K.clear_session()
    np.random.seed(CFG["seed"])
    wrapper = S2STCNWrapper()
    wrapper.prepare(prepared["train"], prepared["val"],
                    input_steps=CFG["input_steps"], output_steps=CFG["output_steps"],
                    target_col=CFG["target_col"], denoise=WAV_SRC,
                    denoise_level=CFG["denoise_level"], features=FEATURES,
                    decoder_mode=cond["decoder_mode"],
                    decoder_extra=cond["decoder_extra"],
                    persistence_gate=False)
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
    print(f"[{cond['slug']}] {epochs} épocas | val_loss={val_loss:.5f} | "
          f"inputs={wrapper.num_encoder_features} | decoder ch={wrapper.num_decoder_features}")

    def blk(df):
        df = common.predict_all_horizons(wrapper, df)
        pred = df["predicted"].to_numpy()
        real = dataset["ws100"].reindex(df["timestamp"]).to_numpy()
        return {"mae": float(np.mean(np.abs(pred - real))),
                "mse": float(np.mean((pred - real) ** 2)),
                "r2": float(common.compute_metrics(real, pred)["r2"]),
                "bias": float(np.mean(pred - real))}

    val_m = blk(prepared["val"])
    all_test = common.predict_all_horizons(wrapper, prepared["test"])
    all_test["actual_raw"] = dataset["ws100"].reindex(all_test["timestamp"]).to_numpy()
    test_m = blk(prepared["test"])
    # amplitude média dentro da trajetória (por origem) vs real
    amp_pred = float(all_test.groupby("origin")["predicted"].apply(
        lambda s: s.max() - s.min()).mean())
    amp_real = float(all_test.groupby("origin")["actual_raw"].apply(
        lambda s: s.max() - s.min()).mean())
    full = pd.concat([prepared["train"], prepared["val"], prepared["test"]])
    roll_pred, roll_act = wrapper.rolling_forecast(
        full, test_start=len(prepared["train"]) + len(prepared["val"]))
    roll = {"mae": float(np.mean(np.abs(roll_pred - roll_act))),
            "mse": float(np.mean((roll_pred - roll_act) ** 2)),
            "r2": float(common.compute_metrics(roll_act, roll_pred)["r2"])}

    model_dir = PROJECT_ROOT / "pipeline/tmp" / cond["slug"] / "Seq2Seq_TCN"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "evaluate").mkdir(parents=True, exist_ok=True)
    wrapper.model.save(model_dir / "model.keras")
    json.dump({"wrapper_key": "tcn", "name": "Seq2Seq_TCN",
               "hyperparameters": HP, "denoise": list(WAV_SRC),
               "denoise_level": CFG["denoise_level"], "features": FEATURES,
               "decoder_mode": cond["decoder_mode"],
               "decoder_extra": cond["decoder_extra"],
               "persistence_gate": False, "gate_mode": "static",
               "context_mode": "horizon",
               "input_steps": CFG["input_steps"], "output_steps": CFG["output_steps"],
               "target_col": CFG["target_col"], "loss": HP.get("loss", "mse"),
               "dataset": {"source": "data/dataset.csv",
                           "train_ratio": CFG["train_ratio"],
                           "val_ratio": CFG["val_ratio"]},
               "training": {"epochs_run": epochs, "best_val_loss": val_loss}},
              open(model_dir / "model.json", "w"), indent=1)
    all_test.to_csv(model_dir / "evaluate" / "predictions_all_horizons.csv", index=False)
    val_end = len(prepared["train"]) + len(prepared["val"])
    roll_ts = dataset.index[val_end + CFG["output_steps"] - 1:
                            val_end + CFG["output_steps"] - 1 + len(roll_pred)]
    pd.DataFrame({"timestamp": roll_ts, "predicted": roll_pred.ravel(),
                  "actual_raw": roll_act.ravel()}).to_csv(
        model_dir / "evaluate" / "predictions_rolling.csv", index=False)
    out = {"condicao": cond["label"], "n_features": len(FEATURES) + 1,
           "epochs": epochs, "val_loss_scaled": val_loss,
           "val_all_horizons": val_m, "test_all_horizons": test_m,
           "rolling_h36": roll,
           "amplitude_pred": amp_pred, "amplitude_real": amp_real}
    json.dump(out, open(PROJECT_ROOT / "pipeline/tmp" / cond["slug"] / "metrics.json",
                        "w"), indent=1)
    json.dump({"all_horizons": test_m,
               "rolling": {"mae": roll["mae"], "mse": roll["mse"],
                           "rmse": roll["mse"] ** 0.5, "r2": roll["r2"]},
               "model": {"training": {"epochs_run": epochs,
                                      "best_val_loss": val_loss}},
               "condicao": cond["label"]},
              open(model_dir / "evaluate" / "metrics.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
