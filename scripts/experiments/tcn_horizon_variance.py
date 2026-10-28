"""Variância por horizonte das TCNs testadas (todas as configs de 21/10).

Para cada config: reconstrói o wrapper EXATAMENTE como o treino
(mesmo keep_raw, mesmo filtro de alturas, mesma ordem de colunas),
MSE por horizonte (h1..36) e estatísticas de dispersão.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import common
from src.models.tcn_hp import STATIC_DILATIONS

OUT = common.resolve("data/results/tcn_horizon_variance/summary.json")
H3 = (40, 100, 140)
CIS = tuple(f"cis{i}" for i in (1, 7, 11))
EXTRA_ALL = (["humid", "temp", "press"] + list(CIS)
             + [f"{p}{h}" for h in H3 for p in ("disp", "vdisp")])

CONFIGS = [
    ("canônica 46", "trained_8020_itildeq", (), False),
    ("featurex 67", "trained_8020_featurex",
     tuple(f"cis{i}" for i in range(1, 20)) + ("humid", "temp"), False),
    ("h3 (14col)", "trained_8020_heights3_s42", (), True),
    ("h3+cis+press+meteo (20col)", "trained_8020_h3cis",
     CIS + ("press", "humid", "temp"), True),
    ("h3+cis+meteo+disp (25col)", "trained_8020_h3cisvdisp",
     CIS + ("humid", "temp") + tuple(f"{p}{h}" for h in H3
                                     for p in ("disp", "vdisp")), True),
    ("h3+cis+meteo s/press (19col)", "trained_8020_h3cisnp",
     CIS + ("humid", "temp"), True),
]

tf = common.setup_tensorflow(True)
results = {}
for label, dirname, keep, filter_heights in CONFIGS:
    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=keep)
    df = dataset
    if filter_heights:
        allowed = set(H3)
        drop = [c for c in dataset.columns
                if (m := re.match(r"^(ws|v|dir)(\d+)(?:_sin|_cos)?$", c))
                and int(m.group(2)) not in allowed]
        df = dataset.drop(columns=drop)
    n = len(df)
    val_end = int(n * 0.80)
    for wrapper_key in ("tcn", "tcn_bi"):
        mdir = common.resolve(f"pipeline/tmp/{dirname}/{wrapper_key}")
        meta = json.load(open(mdir / "model.json", encoding="utf-8"))
        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(42)
        wrapper = common.wrapper_factory(wrapper_key)()
        wrapper.prepare(df, df.tail(400), input_steps=72, output_steps=36,
                        target_col="ws100_wavelet", denoise=("ws100",),
                        denoise_level=2)
        # o grafo autêntico é o salvo; o wrapper fica só com os scalers
        import keras
        import tcn  # registra a classe TCN do keras-tcn

        keras.config.enable_unsafe_deserialization()
        wrapper.model = keras.models.load_model(
            mdir / "model.keras", compile=False, safe_mode=False)
        wrapper.input_steps = 72
        wrapper.output_steps = 36

        eval_df = df.iloc[max(0, val_end - 72):]
        all_df = common.predict_all_horizons(wrapper, eval_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        observed = common.observed_timestamps(
            common.resolve("data/dataset.csv"))
        if observed is not None:
            all_df = all_df[all_df["timestamp"].isin(observed)]
        ph = common.horizon_metrics(
            all_df.assign(actual=all_df["actual_raw"]))
        mse = ph["mse"].to_numpy()
        name = f"{label} — {wrapper_key}"
        results[name] = {
            "mse_h1": float(mse[0]), "mse_h12": float(mse[11]),
            "mse_h24": float(mse[23]), "mse_h36": float(mse[35]),
            "var": float(np.var(mse)), "std": float(np.std(mse)),
            "min": float(np.min(mse)), "max": float(np.max(mse)),
            "crescimento_h36_h1": float(mse[35] / mse[0]),
            "series": [float(x) for x in mse],
        }
        print(f"{name}: var={results[name]['var']:.4f} "
              f"std={results[name]['std']:.4f} h1={mse[0]:.2f} "
              f"h36={mse[35]:.2f}", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
common.write_json(OUT, results)
print(f"escrito: {OUT}")
