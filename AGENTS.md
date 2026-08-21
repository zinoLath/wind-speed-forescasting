# AGENTS.md

## Setup

```bash
./setup_environment.sh        # creates .venv (Python 3.10), installs deps
source .venv/bin/activate
source ./scripts/tf_gpu_env.sh  # REQUIRED on Arch Linux with NVIDIA GPU
```

The venv uses Python 3.10 specifically. TensorFlow 2.21 with CUDA is installed via pip.

## Key Commands

```bash
# Hyperparameter search (Optuna) — GPU required
python tests/optuna_all_hyperparameters.py --n-trials 100 --epochs 60

# Walk-forward evaluation using Optuna best params
python tests/wfo_optuna.py

# Walk-forward over individual periods (older scripts, run from project root)
python tests/wfo_periods.py
python tests/wfo_periods-fixed.py
python tests/wfo_periods-xyz.py

# GUI simulation (opens matplotlib window)
./run_simulation.sh
# NOTE: src/simulation.py does not exist — the script will fail until it is created.

# Wavelet-based imputation for short gaps in wind_data.csv
python src/impute_wind_data.py --input data/wind_data.csv --output data/wind_data_imputed.csv

# Random Forest (row-wise) imputation for wind_data.csv
python src/impute_wind_data_rf.py --input data/wind_data.csv --output data/wind_data_imputed_rf.csv --estimators 50 --max-iter 3 --train-sample 3000

# Validate imputation by injecting artificial gaps and measuring error
python tests/validate_impute_wind_data.py --n-samples 10 --gap-lengths 1 2 3 5 10

# Per-row KNN imputation validation
python tests/validate_knn_impute.py --n-samples 10 --gap-lengths 1 2 3 5 10

# Per-row Random Forest imputation validation
python tests/validate_rf_impute.py --n-samples 10 --gap-lengths 1 2 3 5 10

# Per-row LightGBM imputation validation
python tests/validate_lgbm_impute.py --n-samples 10 --gap-lengths 1 2 3 5 10

# Linear interpolation + Kalman filter (column-wise) imputation validation
python tests/validate_kalman_impute.py --n-samples 10 --gap-lengths 1 2 3 5 10 20 36

# Side-by-side comparison of KNN vs Random Forest vs LightGBM vs Kalman on the same injected gaps
python tests/compare_imputers.py --n-samples 10 --gap-lengths 1 2 3 5 10 20 36

# Detailed statistical evaluation of Random Forest imputation (CSV + plots)
python tests/validate_rf_impute_detailed.py --n-samples 30 --gap-lengths 1 2 3 5 10 20 36

# Walk-forward validation on imputed data, retraining at each window
python tests/wfo_imputed.py --models lstm lstm_bi tcn tcn_bi --train-window-days 30 --test-window-days 7 --step-days 7
```

There are no linter, formatter, type-checker, or test runner commands configured. No CI workflows exist.

## Architecture

Seq2Seq wind speed forecasting models (Keras/TensorFlow). All models inherit from `Seq2SeqWrapper` (`src/models/seq2seq_wrapper.py`), which provides `prepare()`, `build()`, `fit()`, `predict()`, and `rolling_forecast()`.

Model variants in `src/models/`:
- `S2SLSTMWrapper` — unidirectional LSTM encoder + attention
- `S2SLSTMBidirectionalWrapper` — bidirectional LSTM encoder + attention
- `S2STCNWrapper` — TCN encoder + attention
- `S2STCNBidirectionalWrapper` — bidirectional TCN encoder + attention
- `S2SLSTMBidirectionalWrapperAttentionExtract` — variant that also extracts attention scores

All wrappers use Teacher Forcing. The decoder always receives 1 feature (the target column).

Wavelet denoising utility: `src/utils.py:wavelet_denoising()` — uses `sym18` wavelet, default level 2.

## Data

- Raw data: `data/dataset.csv` (gitignored, not in repo)
- Period CSVs: `data/series_list/complete_period_*.csv`
- Period summary: `data/complete_periods_summary.json`
- Results written to: `data/results/optuna/<WrapperName>/` and `data/results/wfo_optuna/<WrapperName>/`

`tests/` files are **not** pytest/unittest — they are standalone scripts meant to be run directly with `python`.

## Important Gotchas

- **`tests/wfo_periods*.py` must be run from the project root.** They use relative paths like `../data/` and `Path.cwd().resolve().parent`.
- **`tests/optuna_all_hyperparameters.py`** is the primary entry point for model training. It manages `sys.path` and GPU runtime setup internally.
- **CUDA library loading** is duplicated across scripts (`configure_tensorflow_gpu_runtime`). If you modify GPU setup, update all copies in `tests/wfo_periods*.py` and `tests/optuna_all_hyperparameters.py`.
- The `build()` method returns `NotImplementedError` (as an expression, not raised) in `Seq2SeqWrapper` — subclasses must override it.
- Data files (`*.csv`, `*.h5`, `*.keras`) are gitignored. The models directory is also gitignored.
- The `OLD/` directory contains archived code; ignore it.
- **`src/simulation.py` is referenced by `run_simulation.sh` and README but does not exist.**
- `FixedHyperParameters._get()` in `tests/wfo_optuna.py:33` was fixed: it now raises `KeyError` only when the param is **missing**.
