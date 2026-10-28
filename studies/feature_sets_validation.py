"""Validação empírica dos conjuntos de features sugeridos pelo estudo.

Todos: TCN, sem pós-dropout, COM normalização, seed 42, HP do nopostdrop.
Conjuntos (além do alvo ws100_wavelet, sempre incluído como entrada):
  ciclo4    — day/hour senoides (4 canais)
  compl6    — ws40+ws260 wavelet, day/hour senoides (6 canais)
  evid8     — compl6 + dir100 senoidal + disp40 + vdisp40 (10 canais)
  dirv100-13— 9feat padrão + dir100 senoidal + v100 wavelet (13 canais)
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
WAV_ALL = ("ws40", "ws100", "ws260", "disp40", "vdisp40", "v100")

CONDITIONS = [
    {"slug": "fs_ciclo4", "label": "ciclo4",
     "decoder_mode": "teacher_forcing",
     "wav_src": ("ws100",),
     "features": list(CYC)},
    {"slug": "fs_compl6", "label": "compl6",
     "decoder_mode": "teacher_forcing",
     "wav_src": ("ws40", "ws100", "ws260"),
     "features": ["ws40_wavelet", "ws260_wavelet"] + list(CYC)},
    {"slug": "fs_evid8", "label": "evid10",
     "decoder_mode": "teacher_forcing",
     "wav_src": ("ws40", "ws100", "ws260", "disp40", "vdisp40"),
     "features": ["ws40_wavelet", "ws260_wavelet", "dir100_sin", "dir100_cos",
                  "disp40_wavelet", "vdisp40_wavelet"] + list(CYC)},
    {"slug": "fs_dirv100_13", "label": "dirv100-13",
     "decoder_mode": "teacher_forcing",
     "wav_src": WAV_ALL,
     "features": ["ws40_wavelet", "ws100_wavelet", "ws260_wavelet",
                  "disp40_wavelet", "vdisp40_wavelet", "v100_wavelet",
                  "dir100_sin", "dir100_cos"] + list(CYC)},
    # réplicas com decoder direto (sem teacher forcing no treino)
    {"slug": "fs2_ciclo4", "label": "ciclo4-df",
     "decoder_mode": "direct",
     "wav_src": ("ws100",),
     "features": list(CYC)},
    {"slug": "fs2_compl6", "label": "compl6-df",
     "decoder_mode": "direct",
     "wav_src": ("ws40", "ws100", "ws260"),
     "features": ["ws40_wavelet", "ws260_wavelet"] + list(CYC)},
    {"slug": "fs2_evid8", "label": "evid10-df",
     "decoder_mode": "direct",
     "wav_src": ("ws40", "ws100", "ws260", "disp40", "vdisp40"),
     "features": ["ws40_wavelet", "ws260_wavelet", "dir100_sin", "dir100_cos",
                  "disp40_wavelet", "vdisp40_wavelet"] + list(CYC)},
    {"slug": "fs2_dirv100_13", "label": "dirv100-13-df",
     "decoder_mode": "direct",
     "wav_src": WAV_ALL,
     "features": ["ws40_wavelet", "ws100_wavelet", "ws260_wavelet",
                  "disp40_wavelet", "vdisp40_wavelet", "v100_wavelet",
                  "dir100_sin", "dir100_cos"] + list(CYC)},
]

dataset = common.load_dataset(PROJECT_ROOT / "data/dataset.csv", keep_raw=WAV_ALL)
splits = common.split_dataset(dataset, CFG["train_ratio"], CFG["val_ratio"])

for cond in CONDITIONS:
    if (PROJECT_ROOT / "pipeline/tmp" / cond["slug"] / "metrics.json").exists():
        print(f"[{cond['slug']}] já concluída, pulando")
        continue
    prepared = {}
    for name, sl in zip(("train", "val", "test"), splits):
        sl = sl.copy()
        for col in cond["wav_src"]:
            sl[f"{col}_wavelet"] = wavelet_denoising(sl[col].values,
                                                     level=CFG["denoise_level"])
        prepared[name] = sl

    K.clear_session()
    np.random.seed(CFG["seed"])
    wrapper = S2STCNWrapper()
    wrapper.prepare(prepared["train"], prepared["val"],
                    input_steps=CFG["input_steps"], output_steps=CFG["output_steps"],
                    target_col=CFG["target_col"], denoise=cond["wav_src"],
                    denoise_level=CFG["denoise_level"], features=cond["features"],
                    decoder_mode=cond["decoder_mode"], persistence_gate=False)
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
    all_test = common.predict_all_horizons(wrapper, prepared["test"])
    all_test["actual_raw"] = dataset["ws100"].reindex(all_test["timestamp"]).to_numpy()
    test_m = blk(prepared["test"])
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
               "hyperparameters": HP, "denoise": list(cond["wav_src"]),
               "denoise_level": CFG["denoise_level"], "features": cond["features"],
               "decoder_mode": cond["decoder_mode"], "persistence_gate": False,
               "gate_mode": "static", "context_mode": "horizon",
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
    out = {"condicao": cond["label"], "n_features": len(cond["features"]) + 1,
           "epochs": epochs, "val_loss_scaled": val_loss,
           "val_all_horizons": val_m, "test_all_horizons": test_m,
           "rolling_h36": roll}
    json.dump(out, open(PROJECT_ROOT / "pipeline/tmp" / cond["slug"] / "metrics.json",
                        "w"), indent=1)
    json.dump({"all_horizons": test_m,
               "rolling": {"mae": roll["mae"], "mse": roll["mse"],
                           "rmse": roll["mse"] ** 0.5, "r2": roll["r2"]},
               "model": {"training": {"epochs_run": epochs,
                                      "best_val_loss": val_loss}},
               "condicao": cond["label"]},
              open(model_dir / "evaluate" / "metrics.json", "w"), indent=1)
    print(json.dumps(out["test_all_horizons"], indent=1))
    print(json.dumps(out["rolling_h36"], indent=1))
