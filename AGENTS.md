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
# Pipeline in stages (optuna -> train -> evaluate -> impute -> walkforward)
python pipeline/pipeline.py                  # full pipeline
python pipeline/pipeline.py --stage train evaluate   # only some stages
python pipeline/step_train.py                # run a single stage standalone
# All pipeline stages accept --config <file> (merged over
# pipeline/pipeline.default.json). Shared values live in the "common"
# config section; stage-level values win. See pipeline/README.md.

# Regression smoke tests (standalone scripts, no pytest)
python tests/test_wrappers_build.py              # build+predict every wrapper
python tests/test_create_sequences_equivalence.py  # vectorised vs loop oracle

# A/B experiment: raw (ws100) vs wavelet-denoised target
python tests/ab_raw_vs_wavelet.py --epochs 15

# Fair fixed-budget benchmark of wrappers (appends to data/results/benchmark_transformer.jsonl)
python tests/benchmark_transformer.py --wrapper lstm_bi --epochs 40

# Legacy Optuna driver (kept for reference; pipeline/step_optuna.py is primary)
python tests/optuna_all_hyperparameters.py --n-trials 100 --epochs 60

# Optuna-tuned focused-column imputer cache (data/imputation_optuna/best_model.json)
python tests/compare_rf_lgbm_impute_detailed.py --optuna

# Regenerate complete-period slices (data/series_list/)
python src/periods.py --min-length 108

# Distributed Optuna round 2 (local GPU + Colab notebooks + Kaggle kernels;
# see skill optuna-colab / docs below)
python scripts/colab_round2.py all --colab-gpus 1 --kaggle-gpus 1  # plan + package + notebooks
python scripts/colab_round2.py local --start        # local watchdog (light wrappers)
python scripts/colab_round2.py kaggle               # dataset + push Kaggle kernels
python scripts/colab_monitor.py --watch 60          # monitor local + Colab + Kaggle
python scripts/colab_round2.py collect --kaggle --drive-dir <mirror> --promote  # finish
```

There are no linter, formatter, type-checker, or test runner commands configured. No CI workflows exist.
For linting during development, `pyflakes` and `radon` are useful but are not in requirements.txt.

## Distributed Optuna round (local + Colab + Kaggle)

`pipeline/pipeline.optuna_round2.json` defines the `round2` studies (wrappers
`lstm, lstm_bi, tcn, tcn_bi`; one sqlite per wrapper under
`pipeline/tmp/optuna_round2/`; every worker resumes via `load_if_exists=True`).
`scripts/colab_round2.py` plans the distribution (heavy: tcn/tcn_bi -> Colab
notebooks and/or Kaggle kernels; light: lstm/lstm_bi -> local
`scripts/optuna_watchdog.sh`), balances by median trial cost, packages the repo
zip (Drive `MyDrive/wind-speed-colab/round2/package/` + Kaggle dataset) and
generates the notebooks in `notebooks/colab_round2/` — **never edit those by
hand; regenerate**. RAM limits are calibrated per environment at runtime
(local 85% of MemTotal, Colab/Kaggle 82%). Colab workers publish heartbeats +
sqlite snapshots to Drive; Kaggle workers (`scripts/kaggle_round2_worker.py`,
pushed via `kaggle_runner.py` with `--progress-subdir optuna_round2`) snapshot
progress into the kernel output every 10 min and are monitored via
`kaggle kernels status`. `scripts/colab_monitor.py` consolidates everything
(exit 3 = error/stale). The `optuna-colab` skill has the full runbook incl.
the error-fix playbook.

## Architecture

Seq2Seq wind speed forecasting models (Keras/TensorFlow). All models inherit from `Seq2SeqWrapper` (`src/models/seq2seq_wrapper.py`), which provides `prepare()`, `build()`, `fit()`, `predict()`, and a batched `rolling_forecast()` (windows are plain functions of known history; a single `model.predict` call).

Model registry lives in `src/common.py:WRAPPERS` with lazy imports; keys: `lstm`, `lstm_bi`, `gru`, `gru_bi`, `lstm_cnn`, `tcn`, `tcn_bi`, `tcn_lstm`, `transformer` (Pre-LN). `validate_wrapper_names()` fails fast on duplicate `.name` values (names double as results-directory names).

- Training loss is the **horizon-weighted MSE** (`src/models/losses.py`), preferring the longer forecast horizons; the pipeline `loss` config key is kept for compatibility but wrappers fix the loss.
- Decoder inputs use the **direct (deployment-valid) convention** — last observed target + horizon fraction, no teacher forcing and no data leakage — so val_loss/early stopping and the Optuna objective (horizon-weighted MSE over validation blocks) measure deployment behaviour.
- Headline metrics compare against the RAW `ws100`; denoised-actual metrics are kept as `*_denoised` secondary columns.
- TCN search space is shared and reduced: `src/models/tcn_hp.py:tcn_hyperparameters()` — a single `filters` value feeds encoder and decoder, no post-block dropout, small kernels (2-3) and shallow dilations. The bidirectional TCN uses a direct last-step encoder→decoder mapping (no context pooling).
- `src/common.py` is the single source of truth for GPU setup, dataset loading/splitting, hp adapters (`FixedHyperParameters`, `OptunaHyperParameters`), metrics and batched prediction (`predict_all_horizons`). Pipeline stages import it; keep new shared logic there.

Wavelet denoising utility: `src/utils.py:wavelet_denoising()` — uses `sym18` wavelet, default level 2.

Imputation package `src/impute/`: `rf.py`, `lightgbm.py`, `knn.py`, `kalman.py` plus `base.py` (loaders, direction sin/cos handling, gap-injection validation, metrics, plots). Registered in `pipeline/step_impute.py:METHODS`.

## Data

- Inputs: `data/dataset.csv` (forecast series) and `data/wind_data.csv` (long series for imputation/walk-forward) — gitignored
- Pipeline outputs: `pipeline/tmp/` (overwritten each run) and `data/impute/<method>/`
- Results: `data/results/{optuna,wfo_optuna,benchmark*}/`
- Archived/orphaned artifacts: `data/archive/` (do not consume)
- Complete-period slices: `data/series_list/` regenerated by `src/periods.py`

`tests/` files are **not** pytest/unittest — they are standalone scripts meant to be run directly with `python`.

## Important Gotchas

- **`src/common.py` must stay import-side-effect free**: TensorFlow is configured via `common.setup_tensorflow()` before model imports (lazy in stages). Do not add module-level keras/tf imports there.
- **`pipeline/` package vs `pipeline/pipeline.py` module**: every file under `pipeline/` inserts the project root at `sys.path[0]` before `from pipeline import ...` imports — otherwise `pipeline/pipeline.py` shadows the package. Keep the bootstrap block when adding stage files.
- **Optuna studies are persisted to sqlite** (`pipeline/tmp/optuna/<name>/optuna.db`) and resume with `load_if_exists=True`; delete the `.db` to restart a search from scratch.
- Data files (`/data/**/*.csv`, `*.h5`, `*.keras`) are gitignored; `/models/`, `/OLD/`, `/images/` are gitignored but some files inside are tracked.
- The `OLD/` directory contains retired code (kept, not maintained): archived test scripts in `OLD/tests/` (see its README for replacements), retired wrappers in `OLD/models/`, legacy notebooks in `OLD/notebooks/`, `run_simulation.sh` (its `src/simulation.py` never existed).
- Historical metrics in older docs/commits (e.g. MAE 0.1589) came from an older dataset and a rolling protocol that leaked future rows; current honest numbers are documented in `docs/`.
