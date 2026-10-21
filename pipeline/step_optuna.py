"""Stage: hyperparameter optimization with Optuna.

Trains every configured wrapper for ``n_trials`` trials and saves the best
trial for each wrapper under ``pipeline/tmp/optuna/<WrapperName>/``.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd

from src import common
from pipeline import config as config_module

STAGE = "optuna"


def _legacy_val_value(wrapper, wrapper_key, cfg, val_blocks, history, trial,
                      elapsed, tf):
    """Objetivo legado: hw-MSE médio em blocos da validação (wrapper.val)."""
    from src.models.losses import horizon_weights

    val_losses = history.history.get("val_loss", [])
    if not val_losses:
        raise RuntimeError("Training returned no val_loss.")
    val = wrapper.val
    try:
        predicted_scaled = wrapper.model.predict(
            [val["X_encoder"], val["X_decoder"]], batch_size=128, verbose=0
        )[:, :, 0]
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError,
            MemoryError) as error:
        trial.set_user_attr("oom", f"eval: {str(error)[:200]}")
        raise optuna.TrialPruned("OOM (evaluation)") from error
    actual_scaled = val["y_decoder"][:, :, 0]
    if wrapper.target_mode == "residual":
        persistence = val["X_encoder"][:, -1, wrapper.target_col_index]
        predicted_scaled = predicted_scaled + persistence
        actual_scaled = actual_scaled + persistence
    predictions = wrapper.scaler_target.inverse_transform(
        predicted_scaled.reshape(-1, 1)).ravel()
    actuals = wrapper.scaler_target.inverse_transform(
        actual_scaled.reshape(-1, 1)).ravel()
    errors = (actuals - predictions).reshape(-1, cfg["output_steps"])
    seq_idx = np.arange(len(errors))
    hw = horizon_weights(cfg["output_steps"])
    block_hw_mses = [float(np.mean(hw * (errors[b] ** 2)))
                     for b in np.array_split(seq_idx, val_blocks)]
    val_hw_mse = float(np.mean(hw * (errors ** 2)))
    val_mse = float(np.mean(errors ** 2))
    trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
    trial.set_user_attr("elapsed_sec", elapsed)
    trial.set_user_attr("val_loss", float(np.min(val_losses)))
    trial.set_user_attr("val_mae", float(np.mean(np.abs(errors))))
    trial.set_user_attr("val_rmse", float(np.sqrt(val_mse)))
    trial.set_user_attr("val_block_mse", block_hw_mses)
    trial.set_user_attr("val_mse", val_mse)
    trial.set_user_attr("val_hw_mse", val_hw_mse)
    value = float(np.mean(block_hw_mses))
    print(f"[{wrapper_key}] trial {trial.number} block_hw_mse={value:.6f} "
          f"val_hw_mse={val_hw_mse:.6f} val_rmse={np.sqrt(val_mse):.6f} "
          f"epochs={len(history.history.get('loss', []))} elapsed={elapsed:.1f}s")
    return value


def _skill_val_value(wrapper, wrapper_key, cfg, val_blocks, eval_df, cache,
                     history, trial, elapsed, tf):
    """Objetivo round4: skill (1 − wmse_modelo/wmse_ARIMA) médio em blocos.

    Erros do modelo na fatia de validação (mesma grade de origens do cache
    ARIMA, alinhada por timestamp×horizonte). Retorna o negativo do skill
    médio (Optuna minimiza)."""
    from src.models.losses import horizon_weights

    all_df = common.predict_all_horizons(wrapper, eval_df)
    all_df["actual_raw"] = eval_df["ws100"].reindex(
        all_df["timestamp"]).to_numpy()
    keys = all_df[["timestamp", "horizon"]].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"]).dt.round("min")
    keys["e_model"] = (all_df["actual_raw"] - all_df["predicted"]).to_numpy()
    merged = cache.merge(keys, on=["timestamp", "horizon"], how="inner")
    if merged.empty:
        raise RuntimeError("cache ARIMA não alinhou com a validação.")
    hw = horizon_weights(cfg["output_steps"])[merged["horizon"].to_numpy() - 1]
    merged["lw_model"] = hw * merged["e_model"] ** 2
    merged["lw_arima"] = hw * merged["e_arima"] ** 2
    skills = []
    for block in np.array_split(np.sort(merged["origin"].unique()), val_blocks):
        sub = merged[merged["origin"].isin(block)]
        wa = float(sub["lw_arima"].mean())
        skills.append(1.0 - float(sub["lw_model"].mean()) / wa if wa > 0 else 0.0)
    skill = float(np.mean(skills))
    trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
    trial.set_user_attr("elapsed_sec", elapsed)
    trial.set_user_attr("skill", skill)
    trial.set_user_attr("skill_blocks", [round(s, 4) for s in skills])
    trial.set_user_attr("val_wmse_model", float(merged["lw_model"].mean()))
    trial.set_user_attr("val_wmse_arima", float(merged["lw_arima"].mean()))
    trial.set_user_attr("n_origins_val", int(merged["origin"].nunique()))
    print(f"[{wrapper_key}] trial {trial.number} skill={skill:+.4f} "
          f"blocks={[round(s, 3) for s in skills]} "
          f"epochs={len(history.history.get('loss', []))} elapsed={elapsed:.1f}s")
    return -skill


def _run_search(wrapper_key, train_df, val_df, cfg, out_dir):
    from keras import backend as K
    # TF is already configured (CUDA runtime, memory growth) by the time this
    # stage runs; importing here keeps the module import side-effect free.
    import tensorflow as tf

    # Data preparation (wavelet denoising, scalers, sequences) does not depend
    # on the trial hyperparameters, so it runs once and is reused by every
    # trial through the same wrapper instance (build() replaces only the model).
    wrapper = common.wrapper_factory(wrapper_key)()
    wrapper.prepare(
        train_df,
        val_df,
        input_steps=cfg["input_steps"],
        output_steps=cfg["output_steps"],
        target_col=cfg["target_col"],
        denoise=cfg["denoise"],
        denoise_level=cfg["denoise_level"],
        features=cfg.get("features"),
        decoder_mode=cfg.get("decoder_mode", "direct"),
        persistence_gate=cfg.get("persistence_gate", False),
    )
    wrapper.loss = cfg.get("loss", "mse")
    wrapper.context_mode = cfg.get("context_mode", "repeat")

    # Round 4: loss por modelo e sequências de dilatação estáticas (opt-in).
    loss_by_wrapper = cfg.get("loss_by_wrapper") or {}
    if wrapper_key in loss_by_wrapper:
        wrapper.loss_mode_override = loss_by_wrapper[wrapper_key]
    if wrapper_key.startswith("tcn") and cfg.get("filters_max"):
        wrapper.filters_max = int(cfg["filters_max"])
    static = cfg.get("static_dilations")
    if static and wrapper_key.startswith("tcn"):
        wrapper.static_dilations = {
            "encoder": static.get("encoder"),
            "decoder": static.get("decoder"),
            "encoder_depth": (tuple(static["encoder_depth"])
                              if static.get("encoder_depth") else None),
            "decoder_depth": (tuple(static["decoder_depth"])
                              if static.get("decoder_depth") else None),
        }

    objective_cfg = cfg.get("objective") or {}
    val_blocks = max(1, int(objective_cfg.get("val_blocks", 4)))
    obj_type = objective_cfg.get("type", "hw_mse")

    cache = None
    eval_df = None
    if obj_type == "skill_vs_arima":
        cache_path = PROJECT_ROOT / objective_cfg["cache"]
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"cache do objetivo ausente: {cache_path}. Rode "
                f"scripts/arima_val_cache.py antes.")
        cache = pd.read_csv(cache_path, parse_dates=["timestamp"])
        cache["timestamp"] = cache["timestamp"].dt.round("min")
        cache["e_arima"] = cache["actual"] - cache["arima"]
        # mesma convenção do relatório: o encoder consome o fim do treino,
        # então a primeira origem de validação = primeira linha da validação.
        eval_df = pd.concat([train_df.tail(wrapper.input_steps), val_df])
        print(f"[{wrapper_key}] objetivo=skill_vs_arima | "
              f"cache={cache_path.name} | origens={cache['origin'].nunique()} "
              f"| val_blocks={val_blocks}")

    def objective(trial):
        K.clear_session()
        hp = common.BoundedOptunaHyperParameters(
            trial, (cfg.get("space_override") or {}).get(wrapper_key))
        # Fixed batch size (search space focus); LR schedules are fixed to
        # constant inside the wrappers. Round 5: batch pode diferir por
        # wrapper (batch_size_by_wrapper) p/ reproduzir a receita do
        # warm start (ex.: lstm do HPO 80/20 usava batch 64).
        batch_size = int((cfg.get("batch_size_by_wrapper") or {})
                         .get(wrapper_key, cfg.get("batch_size", 32)))
        wrapper.build(hp)
        callbacks = common.default_callbacks(wrapper, cfg["patience"])
        if cfg.get("pruning", True):
            callbacks.append(common.optuna_pruning_callback(trial))

        started_at = time.perf_counter()
        try:
            history = wrapper.fit(
                epochs=cfg["epochs"],
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
                use_validation=True,
            )
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError,
                MemoryError) as error:
            # Large sampled configs can exceed the GPU budget mid-training.
            # Discard the trial instead of killing the whole study.
            trial.set_user_attr("oom", str(error)[:200])
            wrapper.model = None
            gc.collect()
            K.clear_session()
            raise optuna.TrialPruned("OOM") from error
        elapsed = time.perf_counter() - started_at

        try:
            if obj_type == "skill_vs_arima":
                return _skill_val_value(wrapper, wrapper_key, cfg, val_blocks,
                                        eval_df, cache, history, trial,
                                        elapsed, tf)
            return _legacy_val_value(wrapper, wrapper_key, cfg, val_blocks,
                                     history, trial, elapsed, tf)
        finally:
            # The wrapper is reused across trials and is the only persistent
            # reference to the trained model, so gc_after_trial alone cannot
            # free it. Drop the reference and tear down the TF graph right
            # after scoring so CPU/GPU memory does not accumulate per trial.
            wrapper.model = None
            gc.collect()
            K.clear_session()

    sampler = optuna.samplers.TPESampler(
        seed=cfg["seed"],
        multivariate=True,
        group=True,
        n_startup_trials=min(20, cfg["n_trials"] // 2),
        n_ei_candidates=64,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(cfg.get("pruner_n_startup_trials", 10)),
        n_warmup_steps=int(cfg.get("pruner_n_warmup_steps", 8)))
    storage = f"sqlite:///{out_dir / 'optuna.db'}"
    study_name = cfg.get("study_name", wrapper_key)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner,
        storage=storage, study_name=study_name, load_if_exists=True,
    )

    # Warm start: lista unica (legado, entra em todos os wrappers) ou
    # enqueue_params_by_wrapper {wrapper: [params, ...]} (round 5).
    enqueues = cfg.get("enqueue_params_by_wrapper") or {}
    for params in enqueues.get(wrapper_key, cfg.get("enqueue_params") or []):
        study.enqueue_trial(params)
        print(f"[{wrapper_key}] enqueue (warm start): {params}")

    started_at = time.perf_counter()
    # Trials persist in the sqlite storage, so a rerun after Ctrl+C / OOM
    # resumes the study where it left off.
    from optuna.trial import TrialState
    # Conta só trials FINALIZADOS: os WAITING (enfileirados p/ warm start)
    # ainda serão executados pelo optimize, então não gastam o orçamento aqui.
    n_finished = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED,
                                 TrialState.FAIL) for t in study.trials)
    n_remaining = max(0, cfg["n_trials"] - n_finished)
    if n_remaining:
        study.optimize(
            objective,
            n_trials=n_remaining,
            timeout=cfg.get("timeout_sec"),
            gc_after_trial=True,
            show_progress_bar=True,
        )
    else:
        print(
            f"[{wrapper_key}] study already has {len(study.trials)} trials "
            f"(budget {cfg['n_trials']}); nothing left to run."
        )
    elapsed = time.perf_counter() - started_at

    pd.DataFrame(study.trials_dataframe()).to_csv(out_dir / "trials.csv", index=False)
    objective_label = (
        "skill_vs_arima (retorna -skill; maximizar)"
        if obj_type == "skill_vs_arima" else f"val_block_hw_mse(k={val_blocks})")
    common.write_json(
        out_dir / "best_trial.json",
        {
            "wrapper": wrapper_key,
            "objective": objective_label,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "elapsed_sec": elapsed,
            "seed": cfg["seed"],
            "timeout_sec": cfg.get("timeout_sec"),
        },
    )
    print(f"[{wrapper_key}] best value={study.best_value:.6f}")
    return {
        "best_trial_json": str(out_dir / "best_trial.json"),
        "trials_csv": str(out_dir / "trials.csv"),
    }


def run(config):
    tf = common.setup_tensorflow(config["gpu"]["enabled"])
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    cfg = config[STAGE]
    if cfg.get("xla"):
        tf.config.optimizer.set_jit(True)
        print("[opt] XLA (jit) ligado")
    if cfg.get("mixed_precision"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[opt] mixed precision: mixed_float16")
    tf.keras.utils.set_random_seed(cfg["seed"])
    dataset = common.load_dataset(common.resolve(config["paths"]["dataset_csv"]),
                                  keep_raw=tuple(cfg.get("denoise", ())))
    train_df, val_df, _ = common.split_dataset(
        dataset, cfg.get("train_ratio", 0.75), cfg.get("val_ratio", 0.20)
    )
    print(f"Dataset: {len(dataset)} rows | train={len(train_df)} val={len(val_df)}")

    results = {}
    for wrapper_key in cfg["wrappers"]:
        wrapper_name = common.wrapper_factory(wrapper_key)().name
        out_dir = common.resolve(
            Path(config["paths"]["tmp_dir"])
            / cfg.get("storage_subdir", "optuna")
            / wrapper_name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOptimizing {wrapper_key} ({wrapper_name})...")
        results[wrapper_key] = _run_search(wrapper_key, train_df, val_df, cfg, out_dir)

    return {"artifacts": results}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization stage.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a pipeline config file.")
    parser.add_argument("--wrappers", nargs="*", default=None,
                        help="Subset of wrappers to optimize (overrides config).")
    args = parser.parse_args()

    common.ensure_project_root_on_path()
    config = config_module.load_config(args.config)
    if args.wrappers:
        unknown = set(args.wrappers) - set(config[STAGE]["wrappers"])
        if unknown:
            raise SystemExit(f"Unknown wrappers: {sorted(unknown)}")
        config[STAGE]["wrappers"] = args.wrappers
    run(config)


if __name__ == "__main__":
    main()