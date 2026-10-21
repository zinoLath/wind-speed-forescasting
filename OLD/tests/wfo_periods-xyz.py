# %%
# Importação das bibliotecas necessárias
# Manipulação de dados
import os
import json
import glob
import ctypes
import site
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


import pywt  # Biblioteca para transformacoes wavelet
import keras_tuner as kt

# Pré-processamento
from sklearn.preprocessing import MinMaxScaler

import keras

from keras import backend as K


# Componentes do modelo Seq2Seq com Atenção
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
    Concatenate,
    TimeDistributed,
    Attention,
)
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    Activation,
    Add,
    LayerNormalization,
    Flatten,
    Reshape,
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Métricas de avaliação
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Otimizar alocacao de memoria GPU - reduz fragmentacao
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TensorFlow logging verbosity


# Garante runtime de GPU no Arch Linux (cuSOLVER e libs CUDA via wheel)
def configure_tensorflow_gpu_runtime():
    """Configura paths CUDA do venv e pre-carrega bibliotecas antes do import do TensorFlow."""
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    candidate_dirs = []

    for sp in site.getsitepackages():
        candidate_dirs.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))

    if "VIRTUAL_ENV" in os.environ:
        candidate_dirs.extend(
            glob.glob(
                os.path.join(
                    os.environ["VIRTUAL_ENV"],
                    "lib",
                    "python*",
                    "site-packages",
                    "nvidia",
                    "*",
                    "lib",
                )
            )
        )

    valid_dirs = []
    for lib_dir in candidate_dirs:
        if os.path.isdir(lib_dir) and lib_dir not in valid_dirs:
            valid_dirs.append(lib_dir)

    if valid_dirs:
        merged = ":".join(valid_dirs)
        os.environ["LD_LIBRARY_PATH"] = (
            f"{merged}:{current_ld}" if current_ld else merged
        )

        # Pre-carrega libs CUDA para evitar falha de resolucao dinamica no kernel do VS Code
        preload_libs = [
            "libcudart.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
            "libcusolver.so.11",
        ]
        for lib_name in preload_libs:
            for lib_dir in valid_dirs:
                lib_path = os.path.join(lib_dir, lib_name)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
                    break

    return valid_dirs


cuda_lib_dirs = configure_tensorflow_gpu_runtime()
print("CUDA lib dirs found:", len(cuda_lib_dirs))
from tcn import TCN

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Python:", os.environ.get("VIRTUAL_ENV", "sem VIRTUAL_ENV"))
print("GPUs:", tf.config.list_physical_devices("GPU"))

import sys
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
import keras_tuner as kt

project_root = Path.cwd().resolve().parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.s2s_lstm_bi_wrapper import S2SLSTMBidirectionalWrapper

from src.models.s2s_lstm_wrapper import S2SLSTMWrapper
from src.models.s2s_tcn_wrapper import S2STCNWrapper
from src.models.s2s_tcn_bi_wrapper import S2STCNBidirectionalWrapper
from src.utils import wavelet_denoising

wrappers = [
    S2SLSTMWrapper(),
    S2SLSTMBidirectionalWrapper(),
    S2STCNWrapper(),
    S2STCNBidirectionalWrapper(),
]

def calculate_features(df, col_list):
    heights = [i for i in range(40, 51, 10)]
    for h in heights:
        if h > 40:
            df["cis" + str(h)] = (df["ws" + str(h)] - df["ws" + str(h-10)])/10
        for i in range(0, len(df)):
            moving_average_ws = df["ws" + str(h)].iloc[max(0, i-6):min(max(0, i-6)+6, len(df))].mean()
            moving_average_v = df["v" + str(h)].iloc[max(0, i-6):min(max(0, i-6)+6, len(df))].mean()
            moving_average_dir = df["dir" + str(h)].iloc[max(0, i-6):min(max(0, i-6)+6, len(df))].mean()

            df.at[df.index[i], "wsdisp" + str(h)] = df["ws" + str(h)].iloc[i] - moving_average_ws
            df.at[df.index[i], "vdisp" + str(h)] = df["v" + str(h)].iloc[i] - moving_average_v
            df.at[df.index[i], "dirdisp" + str(h)] = df["dir" + str(h)].iloc[i] - moving_average_dir
    df_trim = df.copy()

    df_trim = df_trim[col_list]

    return df_trim

def execute_test(wrapper):
    wavelet_level = 2

    K.clear_session()  # Limpa o estado do Keras para evitar acúmulo de memória entre execuções
    #K.backend.clear_session()  # Limpa o estado do backend para liberar recursos
    results_dir = project_root / "data" / "results-xyz" / wrapper.name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Carregando e preparando os dados para {wrapper.name}...")

    dataset = pd.read_csv("../data/dataset.csv")
    dataset["timestamp"] = pd.to_datetime(dataset["id"], format="mixed")
    dataset = dataset.sort_values("timestamp").set_index("timestamp")
    # Drop unnecessary columns if present
    cols_to_drop = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "press",
        "humid",
        "temp",
        "id",
        "cis1",
        "cis2",
        "cis3",
        "cis4",
        "cis5",
        "cis6",
        "cis7",
        "cis8",
        "cis9",
        "cis10",
        "cis11",
        "cis12",
        "cis13",
        "cis14",
        "cis15",
        "cis16",
        "cis17",
        "cis18",
        "cis19",
        "wdisp40",
        "wdisp50",
        "wdisp60",
        "wdisp70",
        "wdisp80",
        "wdisp90",
        "wdisp100",
        "wdisp110",
        "wdisp120",
        "wdisp130",
        "wdisp140",
        "wdisp150",
        "wdisp160",
        "wdisp170",
        "wdisp180",
        "wdisp190",
        "wdisp200",
        "wdisp220",
        "wdisp240",
        "wdisp260",
        "vertdisp40",
        "vertdisp50",
        "vertdisp60",
        "vertdisp70",
        "vertdisp80",
        "vertdisp90",
        "vertdisp100",
        "vertdisp110",
        "vertdisp120",
        "vertdisp130",
        "vertdisp140",
        "vertdisp150",
        "vertdisp160",
        "vertdisp170",
        "vertdisp180",
        "vertdisp190",
        "vertdisp200",
        "vertdisp220",
        "vertdisp240",
        "vertdisp260",
        "wdir150",
        "wdir160",
        "wdir170",
        "wdir180",
        "wdir190",
        "wdir200",
        "wdir220",
        "wdir240",
        "wdir260",
        "verts150",
        "verts160",
        "verts170",
        "verts180",
        "verts190",
        "verts200",
        "verts220",
        "verts240",
        "verts260",
        "ws150",
        "ws160",
        "ws170",
        "ws180",
        "ws190",
        "ws200",
        "ws220",
        "ws240",
        "ws260",
    ]
    dataset = dataset.drop(
        columns=[c for c in cols_to_drop if c in dataset.columns], errors="ignore"
    )
    heights = [
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        220,
        240,
        260,
    ]
    cols_to_rename = {}
    for h in heights:
        cols_to_rename[f"wdir{h}"] = f"dir{h}"
        cols_to_rename[f"verts{h}"] = f"v{h}"

    dataset = dataset.rename(columns=cols_to_rename)
    col_list = ["ws100", "wsdisp40", "vdisp40", "dirdisp40", "dir40", "cis50"]
    dataset = calculate_features(dataset,col_list)
    

    train_ratio = 0.75
    val_ratio = 0.20

    train_end = int(len(dataset) * train_ratio)
    val_end = train_end + int(len(dataset) * val_ratio)

    train_df = dataset.iloc[:train_end].copy()
    val_df = dataset.iloc[train_end:val_end].copy()
    test_df = dataset.iloc[val_end:].copy()

    input_steps = 72
    output_steps = 36

    print(f"Treinando o modelo {wrapper.name}...")

    time_start_train = time.perf_counter()


    wrapper.prepare(
        train_df,
        val_df,
        input_steps=input_steps,
        output_steps=output_steps,
        target_col="ws100_wavelet",
        denoise=col_list,
        denoise_level=wavelet_level,
    )

    hp = kt.HyperParameters()
    model = wrapper.build(hp)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            "../models/best_model.h5.keras", monitor="val_loss", save_best_only=True
        ),
    ]
    history = wrapper.fit(
        epochs=100, batch_size=32, verbose=1, callbacks=callbacks, use_validation=True
    )
    #wrapper.model.load_weights("../models/best_model.h5.keras")
    time_end_train = time.perf_counter()

    #history_df = pd.DataFrame(history.history)
    #history_df.to_csv(results_dir / "training_history.csv", index=False)

    train_time_sec = time_end_train - time_start_train

    json.dump(
        {
            "train_time_sec": train_time_sec,
        },
        open(results_dir / "training_time.json", "w"),
        indent=4,
    )
    print(
        f"Treinamento do modelo {wrapper.name} concluído em {train_time_sec:.2f} segundos."
    )

    periods_summary = json.load(open("../data/complete_periods_summary.json", "r"))
    pred_results = []
    rolling_results = []

    predictions = []
    actuals = []

    for period_metadata in periods_summary:
        if period_metadata["length"] < 142:
            continue
        df_period = pd.read_csv(
            "../data/series_list/complete_period_"
            + str(period_metadata["period"])
            + ".csv"
        )
        df_period["DT"] = pd.to_datetime(df_period["DT"], format="mixed")
        df_period = df_period.sort_values("DT").set_index("DT")
        df_period = calculate_features(df_period,col_list)
        print(
            f"Evaluating on period {period_metadata['period']} with {len(df_period)} records..."
        )

        for col in wrapper.denoise:
            if (
                col in df_period.columns
                and f"{col}_wavelet" not in df_period.columns
            ):
                df_period[f"{col}_wavelet"] = wavelet_denoising(
                    df_period[col].values, level=wrapper.denoise_level
                )
        last_values_predicted = []
        last_values_target = []
        for i in range(0, len(df_period) - input_steps - output_steps, 1):
            input_seq = df_period.iloc[i : i + input_steps]

            target_seq = df_period.iloc[
                i + input_steps : i + input_steps + output_steps
            ]
            time_start = time.perf_counter()
            prediction = wrapper.predict(input_seq)
            time_end = time.perf_counter()
            target_values = target_seq[wrapper.target_col].values
            last_values_predicted.extend(prediction.flatten()[-1:])
            last_values_target.extend(np.array(target_values)[-1:])
            mae = mean_absolute_error(target_values, prediction.flatten())
            mse = mean_squared_error(target_values, prediction.flatten())
            r2 = r2_score(target_values, prediction.flatten())
            predictions.extend(prediction.flatten())
            actuals.extend(target_values)
            pred_results.append(
                {
                    "period": period_metadata["period"],
                    "window_start": input_seq.index[0],
                    "window_end": input_seq.index[-1],
                    "mae": mae,
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "r2": r2,
                    "prediction_time_sec": time_end - time_start,
                }
            )
        if len(last_values_target) > 0:
            mae_rolling = mean_absolute_error(last_values_target, last_values_predicted)
            mse_rolling = mean_squared_error(last_values_target, last_values_predicted)
            r2_rolling = r2_score(last_values_target, last_values_predicted)
            rolling_results.append(
                {
                    "period": period_metadata["period"],
                    "mae": mae_rolling,
                    "mse": mse_rolling,
                    "rmse": np.sqrt(mse_rolling),
                    "r2": r2_rolling,
                }
            )

    pred_results_df = pd.DataFrame(pred_results)
    pred_results_df.to_csv(results_dir / "prediction_results.csv", index=False)

    rolling_results_df = pd.DataFrame(rolling_results)
    rolling_results_df.to_csv(results_dir / "rolling_results.csv", index=False)

    # Save overall predictions and actuals
    pd.DataFrame({"prediction": predictions, "actual": actuals}).to_csv(
        results_dir / "overall_predictions.csv", index=False
    )


for wrapper in wrappers:
    print(f"Executing test for {wrapper.name}...")
    execute_test(wrapper)
