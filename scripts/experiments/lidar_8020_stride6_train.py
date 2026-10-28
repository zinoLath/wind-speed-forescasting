"""80/20 stride-6: decoder esparso que só emite os horizontes 6, 12, 18, 24,
30 e 36 (1 h em 1 h até 6 h; passos de 10 min).

Experimento de isolamento de variável sobre a rodada canônica mista
(``scripts/experiments/lidar_8020_itildeq_train.py``): mesmos HPs (Round 5), mesma seed
(42), mesmo orçamento (60 épocas, sem validação), mesmo split 80/20, mesma
loss por família (TCN=MSE puro + dilatações C′, LSTM=ITILDE-Q) — só muda a
grade de saída: ``output_steps=6`` com o alvo amostrado a passo 6 a partir da
origem. O canal "fração de horizonte" do decoder direto preserva a semântica
real (k/6 = 6k/36).

O override de ``create_sequences`` segue o precedente do
``s2s_lstm_mean_wrapper`` (multi-resolução): delega para o método base com a
janela cheia de 36 passos e amostro X_decoder/y a passo STRIDE — o decoder
recebe exatamente as linhas dos horizontes-alvo (covariadas futuras inclusive).

Avaliação: ``predict_all_horizons_stride`` espelha ``common.predict_all_horizons``
com posições/labels de horizonte em stride; comparação lado a lado com a
rodada canônica NOS MESMOS horizontes (per_horizon de
``data/results/report_8020_itildeq/evaluation.json``).

    python scripts/experiments/lidar_8020_stride6_train.py
    # saída: pipeline/tmp/trained_8020_stride6 + data/results/lidar_8020_stride6/evaluation.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

STRIDE = 6
HORIZONS = tuple(STRIDE * k for k in range(1, 7))  # 6, 12, 18, 24, 30, 36
SEED = 42
WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_stride6")
EVAL_DIR = common.resolve("data/results/lidar_8020_stride6")
CANONICAL_EVAL = common.resolve(
    "data/results/report_8020_itildeq/evaluation.json")
ROUND5_DIR = common.resolve("pipeline/tmp/optuna")

SEQ2SEQ_NAME = {
    "lstm": "Seq2Seq_LSTM",
    "lstm_bi": "Seq2Seq_LSTM_Bidirectional",
    "tcn": "Seq2Seq_TCN",
    "tcn_bi": "Seq2Seq_TCN_Bidirectional",
}
# Mesmas loss/HPs da rodada canônica mista (isola só a grade de saída).
CONFIGS = {
    "lstm": {"loss": "itilde_q", "static_dilations": False},
    "lstm_bi": {"loss": "itilde_q", "static_dilations": False},
    "tcn": {"loss": "mse", "static_dilations": True},
    "tcn_bi": {"loss": "mse", "static_dilations": True},
}
WRAPPER_MODULES = {
    "lstm": "src.models.s2s_lstm_wrapper",
    "lstm_bi": "src.models.s2s_lstm_bi_wrapper",
    "tcn": "src.models.s2s_tcn_wrapper",
    "tcn_bi": "src.models.s2s_tcn_bi_wrapper",
}


def round5_hp(wrapper_key):
    """Hiperparâmetros do melhor trial promovido da Optuna Round 5."""
    path = ROUND5_DIR / SEQ2SEQ_NAME[wrapper_key] / "best_trial.json"
    data = json.load(open(path, encoding="utf-8"))
    return dict(data["best_params"])


def _stride_create_sequences(
    self,
    data,
    input_steps,
    output_steps,
    target_col_index,
    decoder_mode="direct",
    target_mode="absolute",
    persistence_gate=False,
    future_indices=None,
):
    """Base com janela cheia (output_steps*STRIDE) + amostragem a passo STRIDE.

    Só X_decoder e y são fatiados: as linhas selecionadas (índices
    STRIDE-1, 2·STRIDE-1, …) são exatamente os horizontes 6, 12, …, 36 —
    inclusive o canal "fração de horizonte" (6k/36 = k/6) e as covariadas
    futuras nos timestamps-alvo. O encoder (janela passada) não muda.
    """
    from src.models.seq2seq_wrapper import Seq2SeqWrapper

    X_encoder, X_decoder, y = Seq2SeqWrapper.create_sequences(
        data, input_steps, output_steps * STRIDE, target_col_index,
        decoder_mode, target_mode, persistence_gate, future_indices,
    )
    return (X_encoder, X_decoder[:, STRIDE - 1::STRIDE],
            y[:, STRIDE - 1::STRIDE])


def stride_wrapper(wrapper_key):
    """Instância do wrapper canônico com create_sequences em stride."""
    base_cls = common.wrapper_factory(wrapper_key)

    class Stride6Wrapper(base_cls):
        def create_sequences(self, *args, **kwargs):
            return _stride_create_sequences(self, *args, **kwargs)

    Stride6Wrapper.__name__ = f"Stride6{base_cls.__name__}"
    return Stride6Wrapper()


def apply_itildeq_coeffs(wrapper_key, alpha, beta, gamma):
    """Fixa os coeficientes do ITILDE-Q no símbolo do módulo do wrapper."""
    import importlib

    from src.models import losses

    module = importlib.import_module(WRAPPER_MODULES[wrapper_key])
    module.itilde_q = (
        lambda output_steps: losses.itilde_q(output_steps, alpha=alpha,
                                             beta=beta, gamma=gamma))


def predict_all_horizons_stride(wrapper, eval_df):
    """Como ``common.predict_all_horizons``, com horizontes em stride.

    A origem i prevê os timestamps i+6, i+12, …, i+36 (labels 6..36); o
    persistence permanece o último valor observado (flat, válido em h≠1).
    """
    input_steps = wrapper.input_steps
    output_steps = wrapper.output_steps

    prepared, _ = wrapper.prepare_data(
        eval_df,
        input_steps,
        output_steps,
        wrapper.target_col,
        scaler_target=wrapper.scaler_target,
        scaler_other=wrapper.scaler_other,
        denoise=wrapper.denoise,
        decoder_mode=wrapper.decoder_mode,
        target_mode=wrapper.target_mode,
        features=getattr(wrapper, "features", None),
    )
    if wrapper.decoder_mode == "teacher_forcing":
        prepared["X_decoder"][:, :, 0] = prepared["X_decoder"][:, :1, 0]
    common.align_decoder_channels(wrapper, prepared)

    predicted_scaled = wrapper.model.predict(
        [prepared["X_encoder"], prepared["X_decoder"]], verbose=0
    )[:, :, 0]
    actual_scaled = prepared["y_decoder"][:, :, 0]
    persistence_scaled = prepared["X_encoder"][:, -1, wrapper.target_col_index]

    if wrapper.target_mode == "residual":
        predicted_scaled = predicted_scaled + persistence_scaled[:, None]
        actual_scaled = actual_scaled + persistence_scaled[:, None]

    def _inverse(values):
        return wrapper.scaler_target.inverse_transform(
            values.reshape(-1, 1)
        ).reshape(values.shape)

    predicted = _inverse(predicted_scaled)
    actual = _inverse(actual_scaled)
    persistence = wrapper.scaler_target.inverse_transform(
        persistence_scaled.reshape(-1, 1)
    ).ravel()

    sequence_count = len(predicted)
    step_offsets = STRIDE * np.arange(1, output_steps + 1)  # 6..36
    target_positions = (
        np.arange(sequence_count)[:, None] + input_steps
        + (step_offsets - 1)[None, :]
    )
    origins = eval_df.index.to_numpy()[np.arange(sequence_count) + input_steps - 1]

    return pd.DataFrame(
        {
            "origin": np.repeat(origins, output_steps),
            "timestamp": eval_df.index.to_numpy()[target_positions].ravel(),
            "horizon": np.tile(step_offsets, sequence_count),
            "actual": actual.ravel(),
            "predicted": predicted.ravel(),
            "persistence": np.repeat(persistence, output_steps),
        }
    )


def canonical_per_horizon(wrapper_key):
    """Métricas da rodada canônica nos horizontes do stride (evaluation.json)."""
    data = json.load(open(CANONICAL_EVAL, encoding="utf-8"))
    rows = data[SEQ2SEQ_NAME[wrapper_key]]["per_horizon"]
    return {int(r["horizon"]): r for r in rows if int(r["horizon"]) in HORIZONS}


def aggregate(rows):
    """Métricas agregadas sobre os horizontes do stride (média por horizonte)."""
    keys = ("mae", "mse", "rmse", "r2")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def remount_wrapper(wrapper_key, meta, train_df, model_path):
    """Remonta o wrapper salvo (padrão do relatorio_resultados.build_wrapper)."""
    from keras import backend as K

    from src.models.tcn_hp import apply_static_dilations

    K.clear_session()
    wrapper = stride_wrapper(wrapper_key)
    wrapper.gate_mode = meta.get("gate_mode", "static")
    wrapper.context_mode = meta.get("context_mode", "repeat")
    prepared_cols = list(train_df.columns)
    if "time_delta" not in prepared_cols:
        prepared_cols.append("time_delta")
    names = tuple(n for n in ("time_delta", "hour_sin", "hour_cos")
                  if n in prepared_cols)
    wrapper.future_indices = (tuple(prepared_cols.index(n) for n in names)
                              or None)
    wrapper.future_covariate_names = names
    wrapper.prepare(
        train_df,
        train_df.tail(400),
        input_steps=meta["input_steps"],
        output_steps=meta["output_steps"],
        target_col=meta["target_col"],
        denoise=tuple(meta["denoise"]),
        denoise_level=meta["denoise_level"],
        decoder_mode=meta.get("decoder_mode", "direct"),
        target_mode=meta.get("target_mode", "absolute"),
        normalize=bool(meta.get("normalize", True)),
        create_sequences=False,
        persistence_gate=bool(meta.get("persistence_gate", False)),
    )
    if meta.get("static_dilations"):
        wrapper.static_dilations = meta["static_dilations"]
    else:
        apply_static_dilations(wrapper, meta.get("hyperparameters"))
    wrapper.build(common.FixedHyperParameters(meta["hyperparameters"]))
    wrapper.model.load_weights(model_path)
    return wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        choices=WRAPPERS)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--eval-only", action="store_true",
                        help="pula o treino; avalia modelos já salvos")
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | stride-{STRIDE} "
          f"(horizontes {HORIZONS}, alvo {STRIDE * 10}..{STRIDE * 6 * 10} min)")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | teste {len(test_df)}")

    output_dir = common.resolve(args.output_dir)
    results = {}

    for wrapper_key in args.wrappers:
        cfg = CONFIGS[wrapper_key]
        hyperparameters = round5_hp(wrapper_key)
        model_dir = output_dir / wrapper_key
        model_path = model_dir / "model.keras"

        wrapper = None
        if not args.eval_only:
            from keras import backend as K

            K.clear_session()
            tf.keras.utils.set_random_seed(args.seed)
            wrapper = stride_wrapper(wrapper_key)
            wrapper.prepare(
                train_df,
                train_df.tail(400),
                input_steps=72,
                output_steps=6,
                target_col="ws100_wavelet",
                denoise=("ws100",),
                denoise_level=2,
            )

            batch_size = int(hyperparameters.get("batch_size", 32))
            if hasattr(wrapper, "schedule_total_steps"):
                steps = int((len(train_df) + batch_size - 1) // batch_size)
                wrapper.schedule_total_steps = steps * args.epochs

            wrapper.loss_mode_override = cfg["loss"]
            if cfg["loss"] == "itilde_q":
                apply_itildeq_coeffs(wrapper_key, args.alpha, args.beta,
                                     args.gamma)
            if cfg["static_dilations"]:
                from src.models.tcn_hp import STATIC_DILATIONS

                wrapper.static_dilations = STATIC_DILATIONS

            wrapper.build(common.FixedHyperParameters(hyperparameters))
            assert wrapper.loss == cfg["loss"], (wrapper_key, wrapper.loss)
            print(f"[{wrapper_key}] loss={wrapper.loss} output_steps=6 "
                  f"stride={STRIDE} hp=round5 static={cfg['static_dilations']}")

            print(f"[{wrapper_key}] treino {args.epochs} epocas...")
            started = time.perf_counter()
            history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                                  verbose=0, callbacks=None,
                                  use_validation=False)
            elapsed = time.perf_counter() - started
            losses = history.history.get("loss", [])
            print(f"[{wrapper_key}] {len(losses)} epocas em {elapsed:.0f}s | "
                  f"loss final={losses[-1]:.5f}")

            model_dir.mkdir(parents=True, exist_ok=True)
            wrapper.model.save(model_path)
            common.write_json(model_dir / "model.json", {
                "wrapper_key": wrapper_key,
                "name": wrapper.name,
                "experiment": "stride6 (decoder esparso h=6..36)",
                "output_stride": STRIDE,
                "horizons": list(HORIZONS),
                "real_horizon_steps": STRIDE * 6,
                "persistence_gate": False,
                "gate_mode": "static",
                "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
                "target_mode": getattr(wrapper, "target_mode", "absolute"),
                "normalize": bool(getattr(wrapper, "normalize", True)),
                "hyperparameters": hyperparameters,
                "params_source": "HPs promovidos da Optuna Round 5 "
                                 "(pipeline/tmp/optuna) — iguais à canônica",
                "input_steps": 72,
                "output_steps": 6,
                "target_col": "ws100_wavelet",
                "denoise": ["ws100"],
                "denoise_level": 2,
                "loss": cfg["loss"],
                "loss_coeffs": ({"alpha": args.alpha, "beta": args.beta,
                                 "gamma": args.gamma}
                                if cfg["loss"] == "itilde_q" else None),
                "static_dilations": (wrapper.static_dilations
                                     if cfg["static_dilations"] else None),
                "training": {"epochs_run": len(losses),
                             "epochs_requested": args.epochs,
                             "seed": args.seed, "patience": 0,
                             "final_loss": float(losses[-1]) if losses else None,
                             "training_time_sec": round(elapsed, 1)},
                "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                            "split": "80/20 sem validacao (stride-6)"},
            })

        # ---- avaliação stride: wrapper em memória ou remount --------------
        if wrapper is None:
            meta = json.load(open(model_dir / "model.json", encoding="utf-8"))
            hyperparameters = dict(meta["hyperparameters"])
            wrapper = remount_wrapper(wrapper_key, meta, train_df, model_path)

        all_df = predict_all_horizons_stride(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        # métricas contra o ws100 BRUTO (convenção do relatorio_resultados)
        long_df = all_df.assign(actual=all_df["actual_raw"])
        per_h = common.horizon_metrics(long_df)
        sparse_all = aggregate(per_h.to_dict(orient="records"))
        persistence_all = aggregate([
            common.compute_metrics(g["actual"], g["persistence"])
            for _, g in long_df.groupby("horizon")
        ])

        canon = canonical_per_horizon(wrapper_key)
        table = []
        for row in per_h.to_dict(orient="records"):
            h = int(row["horizon"])
            table.append({
                "horizon": h,
                "stride6": {k: round(float(row[k]), 4)
                            for k in ("mae", "mse", "rmse", "r2")},
                "canonical": {k: round(float(canon[h][k]), 4)
                              for k in ("mae", "mse", "rmse", "r2")},
                "persistence_mae": round(float(row["persistence_mae"]), 4),
            })
        results[wrapper_key] = {
            "name": wrapper.name,
            "loss": cfg["loss"],
            "horizons": list(HORIZONS),
            "all_sparse": {k: round(v, 4) for k, v in sparse_all.items()},
            "canonical_same_horizons": {
                k: round(v, 4)
                for k, v in aggregate(list(canon.values())).items()},
            "persistence_sparse": {k: round(v, 4)
                                   for k, v in persistence_all.items()},
            "per_horizon": table,
        }

        print(f"\n[{wrapper_key}] stride-6 vs canônica (36 passos) — "
              f"mesmos horizontes {HORIZONS}")
        print(f"{'h':>3} | {'MAE s6':>7} {'MAE c36':>7} | "
              f"{'RMSE s6':>7} {'RMSE c36':>8} | {'R² s6':>6} {'R² c36':>7} | "
              f"{'MAE pers':>8}")
        for t in table:
            print(f"{t['horizon']:>3} | {t['stride6']['mae']:>7.4f} "
                  f"{t['canonical']['mae']:>7.4f} | "
                  f"{t['stride6']['rmse']:>7.4f} "
                  f"{t['canonical']['rmse']:>8.4f} | "
                  f"{t['stride6']['r2']:>6.3f} {t['canonical']['r2']:>7.3f} | "
                  f"{t['persistence_mae']:>8.4f}")
        print(f"     | agregado: MAE {sparse_all['mae']:.4f} vs "
              f"{aggregate(list(canon.values()))['mae']:.4f} | "
              f"R² {sparse_all['r2']:.3f} vs "
              f"{aggregate(list(canon.values()))['r2']:.3f}\n")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "evaluation.json"
    common.write_json(out, {
        "experiment": "stride6 (decoder emite só h=6,12,18,24,30,36)",
        "canonical_round": str(CANONICAL_EVAL),
        "stride": STRIDE,
        "wrappers": results,
    })
    print(f"Avaliação: {out}")
    print(f"Modelos: {output_dir}")


if __name__ == "__main__":
    main()
