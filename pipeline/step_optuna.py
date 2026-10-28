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


def _skill_metrics(wrapper, cfg, eval_df, cache, val_blocks, stability_lambda=0.0,
                   all_df=None):
    """Skill em blocos vs ARIMA + termo de estabilidade (round 7+).

    Retorna dict com skills por bloco, skill medio, ``skill_stability``
    (= skill - lambda*std(blocos)) e ``value`` (negativo; Optuna minimiza).
    ``all_df`` permite reaproveitar a avaliação já feita pelo objetivo
    round 8 (evita segunda passada de predict_all_horizons).
    """
    from src.models.losses import horizon_weights

    if all_df is None:
        all_df = common.predict_all_horizons(wrapper, eval_df)
    if "actual_raw" not in all_df:
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
    stability = skill - stability_lambda * float(np.std(skills, ddof=1))
    return {
        "skills": skills,
        "skill": skill,
        "skill_stability": stability,
        "value": -stability,
        "wmse_model": float(merged["lw_model"].mean()),
        "wmse_arima": float(merged["lw_arima"].mean()),
        "n_origins": int(merged["origin"].nunique()),
    }


def _block_mse_metrics(wrapper, eval_df, val_blocks):
    """MSE absoluto por bloco de origens na grade de inferência (round 8).

    Mesma mecânica da avaliação final (predict_all_horizons, decoder de
    inferência, alvo ws100 bruto) — sem horizon weighting e sem referência
    a ARIMA: o valor é o MSE médio em blocos da fatia de validação, na
    MESMA unidade da métrica do teste do artigo.
    """
    all_df = common.predict_all_horizons(wrapper, eval_df)
    all_df["actual_raw"] = eval_df["ws100"].reindex(
        all_df["timestamp"]).to_numpy()
    err2 = (all_df["actual_raw"] - all_df["predicted"]).to_numpy() ** 2
    origins = all_df["origin"].to_numpy()
    unique_origins = np.sort(pd.unique(origins))
    blocks = [float(err2[np.isin(origins, block)].mean())
              for block in np.array_split(unique_origins, val_blocks)]
    return {
        "all_df": all_df,
        "blocks": blocks,
        "mse": float(err2.mean()),
        "value": float(np.mean(blocks)),
        "n_origins": int(unique_origins.size),
    }


def _block_mse_val_value(wrapper, wrapper_key, cfg, val_blocks, eval_df,
                         history, trial, elapsed, cache=None):
    """Objetivo round 8: MSE absoluto por bloco na validação (inferência).

    Evidência do sweep da round 7: o objetivo skill-vs-ARIMA não transfere
    para o teste (r ≈ 0,05 na banda 1-SE; ver findings/optuna-hpo), enquanto
    o argmin por MSE absoluto teria levado o melhor trial ao promote. Se um
    cache ARIMA estiver configurado, skill/stability continuam gravados como
    user_attrs de diagnóstico (sem entrar no valor).
    """
    metrics = _block_mse_metrics(wrapper, eval_df, val_blocks)
    trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
    trial.set_user_attr("elapsed_sec", elapsed)
    trial.set_user_attr("val_mse", metrics["mse"])
    trial.set_user_attr("val_rmse", float(np.sqrt(metrics["mse"])))
    trial.set_user_attr("val_mse_blocks",
                        [round(b, 4) for b in metrics["blocks"]])
    trial.set_user_attr("n_origins_val", metrics["n_origins"])
    extra = ""
    if cache is not None:
        sk = _skill_metrics(wrapper, cfg, eval_df, cache, val_blocks,
                            all_df=metrics["all_df"])
        trial.set_user_attr("skill", sk["skill"])
        trial.set_user_attr("skill_blocks",
                            [round(s, 4) for s in sk["skills"]])
        trial.set_user_attr("val_wmse_model", sk["wmse_model"])
        trial.set_user_attr("val_wmse_arima", sk["wmse_arima"])
        extra = (f" skill(diag)={sk['skill']:+.4f} "
                 f"wmse_model={sk['wmse_model']:.4f}")
    print(f"[{wrapper_key}] trial {trial.number} "
          f"block_mse={metrics['value']:.6f} "
          f"val_mse={metrics['mse']:.6f} "
          f"rmse={np.sqrt(metrics['mse']):.6f}{extra} "
          f"blocks={[round(b, 3) for b in metrics['blocks']]} "
          f"epochs={len(history.history.get('loss', []))} "
          f"elapsed={elapsed:.1f}s")
    return metrics["value"]


def _skill_val_value(wrapper, wrapper_key, cfg, val_blocks, eval_df, cache,
                     history, trial, elapsed, stability_lambda=0.0):
    """Objetivo round4/5/7: skill vs ARIMA em blocos da validação.

    Round 7: valor = -(skill medio - lambda*std entre blocos), penalizando
    configs que desabam num bloco. Erros do modelo na fatia de validação
    (mesma grade de origens do cache ARIMA, alinhada por
    timestamp×horizonte)."""
    metrics = _skill_metrics(wrapper, cfg, eval_df, cache, val_blocks,
                             stability_lambda)
    trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
    trial.set_user_attr("elapsed_sec", elapsed)
    trial.set_user_attr("skill", metrics["skill"])
    trial.set_user_attr("skill_stability", metrics["skill_stability"])
    trial.set_user_attr("stability_lambda", stability_lambda)
    trial.set_user_attr("skill_blocks", [round(s, 4) for s in metrics["skills"]])
    trial.set_user_attr("val_wmse_model", metrics["wmse_model"])
    trial.set_user_attr("val_wmse_arima", metrics["wmse_arima"])
    trial.set_user_attr("n_origins_val", metrics["n_origins"])
    print(f"[{wrapper_key}] trial {trial.number} "
          f"skill={metrics['skill']:+.4f} "
          f"stability={metrics['skill_stability']:+.4f} "
          f"(lambda={stability_lambda}) "
          f"blocks={[round(s, 3) for s in metrics['skills']]} "
          f"epochs={len(history.history.get('loss', []))} elapsed={elapsed:.1f}s")
    return metrics["value"]


def _objective_pruning_callback(trial, compute_value, report_epochs):
    """Callback que reporta o PRÓPRIO objetivo (decoder inferência) ao pruner.

    Round 7: o pruner (SHP) precisa de valores intermediários comparáveis
    ao objetivo final; ``val_mae`` teacher-forced escalado tem correlação
    fraca com o skill final (Spearman 0,07–0,51 na round 5). O custo é uma
    avaliação na validação apenas nas épocas de rung (ex.: 12 e 36).
    """
    from keras.callbacks import Callback

    class _ObjectivePruningCallback(Callback):
        def __init__(self, trial, compute_value, report_epochs):
            super().__init__()
            self.trial = trial
            self.compute_value = compute_value
            self.report_epochs = report_epochs

        def on_epoch_end(self, epoch, logs=None):
            step = epoch + 1
            if step not in self.report_epochs:
                return
            value = self.compute_value()
            self.trial.report(float(value), step=step)
            if self.trial.should_prune():
                import optuna

                raise optuna.TrialPruned(
                    f"Trial pruned at epoch {step} (objective rung)")

    return _ObjectivePruningCallback(trial, compute_value, report_epochs)


def _make_pruner(cfg):
    """MedianPruner (legado) ou SuccessiveHalvingPruner via cfg["pruner"]."""
    pruner_cfg = cfg.get("pruner") or {}
    if str(pruner_cfg.get("type", "median")).lower() in (
            "successive_halving", "sha", "asha"):
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(pruner_cfg.get("min_resource", 12)),
            reduction_factor=int(pruner_cfg.get("reduction_factor", 3)),
        )
        print(f"[pruner] SuccessiveHalving "
              f"(min_resource={pruner_cfg.get('min_resource', 12)}, "
              f"reduction_factor={pruner_cfg.get('reduction_factor', 3)})")
        return pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(cfg.get("pruner_n_startup_trials", 10)),
        n_warmup_steps=int(cfg.get("pruner_n_warmup_steps", 8)))
    print(f"[pruner] Median (startup={cfg.get('pruner_n_startup_trials', 10)}, "
          f"warmup={cfg.get('pruner_n_warmup_steps', 8)})")
    return pruner


def _make_sampler(cfg):
    return optuna.samplers.TPESampler(
        seed=cfg["seed"],
        multivariate=True,
        group=True,
        n_startup_trials=int(cfg.get("sampler_n_startup_trials",
                                     min(20, cfg["n_trials"] // 2))),
        n_ei_candidates=64,
        constant_liar=bool(cfg.get("sampler_constant_liar", False)),
    )


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
    stability_lambda = float(objective_cfg.get("stability_lambda", 0.0))

    cache = None
    eval_df = None
    if obj_type in ("skill_vs_arima", "block_mse"):
        if obj_type == "skill_vs_arima" or objective_cfg.get("cache"):
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
        cache_note = (f"cache={cache_path.name} " if cache is not None
                      else "sem cache ARIMA ")
        print(f"[{wrapper_key}] objetivo={obj_type} | {cache_note}"
              f"| val_blocks={val_blocks}")

    def objective(trial):
        K.clear_session()
        # Round 8: re-seed por trial — trials comparáveis entre si e o
        # escolhido reproduz o treino final (seed única do protocolo).
        tf.keras.utils.set_random_seed(cfg["seed"])
        hp = common.BoundedOptunaHyperParameters(
            trial, (cfg.get("space_override") or {}).get(wrapper_key))
        # Fixed batch size (search space focus); LR schedules are fixed to
        # constant inside the wrappers. Round 5: batch pode diferir por
        # wrapper (batch_size_by_wrapper) p/ reproduzir a receita do
        # warm start (ex.: lstm do HPO 80/20 usava batch 64).
        batch_size = int((cfg.get("batch_size_by_wrapper") or {})
                         .get(wrapper_key, cfg.get("batch_size", 32)))
        wrapper.build(hp)
        # Round 8 (`final_protocol_callbacks`): o treino final 80/20 roda
        # todas as épocas SEM EarlyStopping/ReduceLROnPlateau — o HPO
        # avaliava "melhor época + lr decaído no plateau", regime que o
        # deploy nunca usa. Com a flag, os callbacks padrão somem e resta
        # apenas o pruning (rungs do SHP).
        if cfg.get("final_protocol_callbacks"):
            callbacks = []
        else:
            callbacks = common.default_callbacks(wrapper, cfg["patience"])
        if cfg.get("pruning", True):
            # Round 7/8: reportar o próprio objetivo (decoder inferência)
            # nos rungs em vez de val_mae teacher-forced, quando o config
            # pedir. block_mse dispensa cache; skill_vs_arima exige.
            if (cfg.get("pruning_metric") == "objective"
                    and obj_type in ("skill_vs_arima", "block_mse")
                    and (obj_type == "block_mse" or cache is not None)):
                report_epochs = set(int(e) for e in
                                    cfg.get("pruning_report_epochs", [12, 36]))
                if obj_type == "block_mse":
                    rung_value = lambda: _block_mse_metrics(  # noqa: E731
                        wrapper, eval_df, val_blocks)["value"]
                else:
                    rung_value = lambda: _skill_metrics(  # noqa: E731
                        wrapper, cfg, eval_df, cache, val_blocks,
                        stability_lambda)["value"]
                callbacks.append(_objective_pruning_callback(
                    trial, rung_value, report_epochs))
            else:
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
                                        elapsed, stability_lambda)
            if obj_type == "block_mse":
                return _block_mse_val_value(wrapper, wrapper_key, cfg,
                                            val_blocks, eval_df, history,
                                            trial, elapsed, cache=cache)
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

    sampler = _make_sampler(cfg)
    pruner = _make_pruner(cfg)
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
        f"skill_vs_arima stability (lambda={stability_lambda}; retorna "
        "-(skill-lambda*std); minimizar)"
        if obj_type == "skill_vs_arima"
        else (f"val_block_mse(k={val_blocks}) — inferência, MSE absoluto"
              if obj_type == "block_mse"
              else f"val_block_hw_mse(k={val_blocks})"))
    common.write_json(
        out_dir / "best_trial.json",
        {
            "wrapper": wrapper_key,
            "objective": objective_label,
            "stability_lambda": stability_lambda,
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
    # Round 8: features por wrapper alinhadas ao lineup final do artigo —
    # LSTM/Bi com featurex (46 + cis1..19 + humid/temp), TCN/Bi com meteo
    # (46 + humid/temp). Default: um dataset só (keep_raw = denoise).
    keep_by_wrapper = cfg.get("keep_raw_by_wrapper") or {}
    splits_by_keep = {}

    def _split_for(wrapper_key):
        extras = tuple(keep_by_wrapper.get(wrapper_key, ()))
        keep = tuple(dict.fromkeys(tuple(cfg.get("denoise", ())) + extras))
        if keep not in splits_by_keep:
            ds = common.load_dataset(
                common.resolve(config["paths"]["dataset_csv"]),
                keep_raw=keep)
            splits_by_keep[keep] = common.split_dataset(
                ds, cfg.get("train_ratio", 0.75), cfg.get("val_ratio", 0.20)
            ) + (ds,)
            tr, va, _te, _ds = splits_by_keep[keep]
            print(f"[{wrapper_key}] dataset keep_raw={len(keep)} extras: "
                  f"{tr.shape[0]} treino / {va.shape[0]} val / "
                  f"{_ds.shape[1]} colunas")
        return splits_by_keep[keep]

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
        train_df, val_df, _test_df, _ds = _split_for(wrapper_key)
        results[wrapper_key] = _run_search(wrapper_key, train_df, val_df,
                                           cfg, out_dir)

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