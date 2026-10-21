"""AB: 9 features e todas as features, sem pós-dropout, SEM normalização.

Ambas as condições adicionam dir100 (decomposta em senoides) e v100 com
wavelet. Reutiliza o HP do experimento sem pós-dropout (huber, post-dropouts
zerados). A normalização é desligada substituindo o MinMaxScaler do módulo
seq2seq_wrapper por um scaler identidade.
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

CONDITIONS = [
    {
        "slug": "ab_nonorm_9feat_dir_v100",
        "condicao": "9 features + dir100 senoidal + v100 wavelet, sem pos-dropout, SEM normalizacao",
        "wav_src": ("ws40", "ws100", "ws260", "disp40", "vdisp40", "v100"),
        "features": ["ws40_wavelet", "ws100_wavelet", "ws260_wavelet",
                     "disp40_wavelet", "vdisp40_wavelet", "v100_wavelet",
                     "dir100_sin", "dir100_cos",
                     "hour_sin", "hour_cos", "day_sin", "day_cos"],
        "drop_cols": (),
    },
    {
        "slug": "ab_nonorm_allfeat_dir_v100",
        "condicao": "todas as features + v100 wavelet, dir100 so senoidal, sem pos-dropout, SEM normalizacao",
        "wav_src": ("ws100", "ws40", "v40", "v100"),
        "features": None,
        "drop_cols": ("dir100",),
    },
]

dataset = common.load_dataset(PROJECT_ROOT / "data/dataset.csv",
                              keep_raw=tuple(
                                  {c for cond in CONDITIONS for c in cond["wav_src"]}))
splits = common.split_dataset(dataset, CFG["train_ratio"], CFG["val_ratio"])


def run(cond):
    prepared = {}
    for name, sl in zip(("train", "val", "test"), splits):
        sl = sl.copy()
        for col in cond["wav_src"]:
            sl[f"{col}_wavelet"] = wavelet_denoising(sl[col].values,
                                                     level=CFG["denoise_level"])
        sl = sl.drop(columns=[c for c in cond["drop_cols"] if c in sl.columns])
        prepared[name] = sl
        print(f"[{cond['slug']}] {name}: {sl.shape[1]} colunas, {sl.shape[0]} linhas")

    K.clear_session()
    np.random.seed(CFG["seed"])
    wrapper = S2STCNWrapper()
    wrapper.prepare(prepared["train"], prepared["val"],
                    input_steps=CFG["input_steps"], output_steps=CFG["output_steps"],
                    target_col=CFG["target_col"], denoise=cond["wav_src"],
                    denoise_level=CFG["denoise_level"], features=cond["features"],
                    decoder_mode=CFG.get("decoder_mode", "teacher_forcing"),
                    persistence_gate=bool(CFG.get("persistence_gate", False)))
    batch = int(HP.get("batch_size", 32))
    if hasattr(wrapper, "schedule_total_steps"):
        wrapper.schedule_total_steps = -(-wrapper.train["X_encoder"].shape[0] // batch) * 100
    wrapper.loss = HP.get("loss", CFG["loss"])

    model_dir = PROJECT_ROOT / "pipeline/tmp" / cond["slug"] / "Seq2Seq_TCN"
    if (model_dir / "model.keras").exists():
        wrapper.build(common.FixedHyperParameters(HP))
        wrapper.model.load_weights(model_dir / "model.keras")
        meta_old = json.load(open(model_dir / "model.json"))
        epochs = meta_old["training"]["epochs_run"]
        val_loss = meta_old["training"]["best_val_loss"]
        print(f"[{cond['slug']}] modelo carregado do disco ({epochs} épocas)")
    else:
        wrapper.build(common.FixedHyperParameters(HP))
        callbacks = common.default_callbacks(wrapper, CFG["patience"])
        history = wrapper.fit(epochs=100, batch_size=batch, verbose=0,
                              callbacks=callbacks, use_validation=True)
        epochs = len(history.history["loss"])
        val_loss = float(min(history.history["val_loss"]))
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        print(f"[{cond['slug']}] treino: {epochs} épocas | val_loss={val_loss:.5f} | "
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

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "evaluate").mkdir(parents=True, exist_ok=True)
    meta_model = {
        "wrapper_key": "tcn", "name": "Seq2Seq_TCN",
        "hyperparameters": HP, "denoise": list(cond["wav_src"]),
        "denoise_level": CFG["denoise_level"], "features": cond["features"],
        "normalize": False, "drop_columns": list(cond["drop_cols"]),
        "decoder_mode": CFG.get("decoder_mode", "teacher_forcing"),
        "persistence_gate": bool(CFG.get("persistence_gate", False)),
        "gate_mode": "static", "context_mode": "horizon",
        "input_steps": CFG["input_steps"], "output_steps": CFG["output_steps"],
        "target_col": CFG["target_col"], "loss": HP.get("loss", "mse"),
        "dataset": {"source": "data/dataset.csv",
                    "train_ratio": CFG["train_ratio"], "val_ratio": CFG["val_ratio"]},
        "training": {"epochs_run": epochs, "best_val_loss": val_loss},
    }
    json.dump(meta_model, open(model_dir / "model.json", "w"), indent=1)
    all_test.to_csv(model_dir / "evaluate" / "predictions_all_horizons.csv", index=False)

    val_end_abs = len(prepared["train"]) + len(prepared["val"])
    roll_ts = dataset.index[val_end_abs + CFG["output_steps"] - 1:
                            val_end_abs + CFG["output_steps"] - 1 + len(roll_pred)]
    pd.DataFrame({"timestamp": roll_ts, "predicted": roll_pred.ravel(),
                  "actual_raw": roll_act.ravel()}).to_csv(
        model_dir / "evaluate" / "predictions_rolling.csv", index=False)

    out = {"condicao": cond["condicao"], "epochs": epochs,
           "val_loss_scaled": val_loss,
           "val_all_horizons": val_m, "test_all_horizons": test_m,
           "rolling_h36": roll}
    json.dump(out, open(PROJECT_ROOT / "pipeline/tmp" / cond["slug"] / "metrics.json",
                        "w"), indent=1)
    std_dir = model_dir / "evaluate"
    json.dump({"all_horizons": out["test_all_horizons"],
               "rolling": {"mae": out["rolling_h36"]["mae"],
                           "mse": out["rolling_h36"]["mse"],
                           "rmse": out["rolling_h36"]["mse"] ** 0.5,
                           "r2": out["rolling_h36"]["r2"]},
               "model": {"training": {"epochs_run": out["epochs"],
                                      "best_val_loss": out["val_loss_scaled"]}},
               "condicao": out["condicao"]},
              open(std_dir / "metrics.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


for cond in CONDITIONS:
    run(cond)
