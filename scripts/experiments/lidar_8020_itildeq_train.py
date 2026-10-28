"""80/20 rodada mista — HPs dos vencedores promovidos da Optuna Round 5
(pipeline/tmp/optuna/, combinação canônica do artigo); LSTM com loss
ITILDE-Q e TCN com MSE puro + dilatações estáticas C′.

Protocolo comum (idêntico ao canônico 80/20): split 80/20 sem validação, 60
épocas fixas, seed 42, janela 72 → 36 passos, alvo ``ws100_wavelet`` (sym18
nível 2), métricas contra o ``ws100`` bruto.

O que muda por família:

- **HPs (todos os wrappers)**: melhor trial da Round 5 promovido
  (``pipeline/tmp/optuna/<Seq2Seq_*>/best_trial.json``) — estudo com
  objetivo skill bloco k=4 na validação (split 72/8/20) e o mesmo espaço
  loss/dilatações desta rodada.
- **LSTM / LSTM-Bi**: perda ITILDE-Q (shape-aware: MSE + termos de
  amplitude/fase/amplificação — Lee et al. 2022, arXiv:2210.15050;
  variante Eq. 15 de Dong Sheng et al. 2026).
- **TCN / TCN-Bi**: perda MSE pura com a sequência de dilatações estática
  C′ (encoder ``[1,2,4,8,16,32,36]`` — o lag de 36 passos/6 h direto no
  campo receptivo; decoder ``[1,2,4,8,16]``); a PROFUNDIDADE (prefixo das
  dilatações) vem do HP da round 5.

O patch de perda é feito pelo atributo ``loss_mode_override`` dos wrappers
(código canônico intocado).

    python scripts/experiments/lidar_8020_itildeq_train.py
    python reports/src/relatorio_resultados.py \\
        --models-dir pipeline/tmp/trained_8020_itildeq \\
        --out-dir data/results/report_8020_itildeq
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common
from src.models.tcn_hp import STATIC_DILATIONS

SEED = 42
WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_itildeq")
ROUND5_DIR = common.resolve("pipeline/tmp/optuna")

# HPs de TODOS os wrappers vêm dos vencedores promovidos da Optuna
# Round 5 (pipeline/tmp/optuna/<Seq2Seq_*>/best_trial.json), que otimizou
# exatamente esta combinação (LSTM ITILDE-Q / TCN MSE + dilatações C′).
SEQ2SEQ_NAME = {
    "lstm": "Seq2Seq_LSTM",
    "lstm_bi": "Seq2Seq_LSTM_Bidirectional",
    "tcn": "Seq2Seq_TCN",
    "tcn_bi": "Seq2Seq_TCN_Bidirectional",
}
CONFIGS = {
    "lstm": {"loss": "itilde_q", "hp_source": "round5", "static_dilations": False},
    "lstm_bi": {"loss": "itilde_q", "hp_source": "round5", "static_dilations": False},
    "tcn": {"loss": "mse", "hp_source": "round5", "static_dilations": True},
    "tcn_bi": {"loss": "mse", "hp_source": "round5", "static_dilations": True},
}


def round5_hp(wrapper_key):
    """Hiperparâmetros do melhor trial promovido da Optuna Round 5."""
    path = ROUND5_DIR / SEQ2SEQ_NAME[wrapper_key] / "best_trial.json"
    data = json.load(open(path, encoding="utf-8"))
    return dict(data["best_params"])


WRAPPER_MODULES = {
    "lstm": "src.models.s2s_lstm_wrapper",
    "lstm_bi": "src.models.s2s_lstm_bi_wrapper",
    "tcn": "src.models.s2s_tcn_wrapper",
    "tcn_bi": "src.models.s2s_tcn_bi_wrapper",
}


def apply_itildeq_coeffs(wrapper_key, alpha, beta, gamma):
    """Fixa os coeficientes do ITILDE-Q no símbolo do módulo do wrapper.

    O wrapper chama ``itilde_q(output_steps)`` com os defaults da função; para
    usar α, β e γ escolhidos na rodada, o símbolo importado no módulo do
    wrapper é substituído por uma closure com esses coeficientes.
    """
    import importlib

    from src.models import losses

    module = importlib.import_module(WRAPPER_MODULES[wrapper_key])
    module.itilde_q = (
        lambda output_steps: losses.itilde_q(output_steps, alpha=alpha,
                                             beta=beta, gamma=gamma))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="pasta de saída (default: trained_8020_itildeq")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        choices=WRAPPERS)
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | rodada mista 8020-itildeq "
          f"(LSTM=HPO+ITILDE-Q, TCN=canônica+MSE+C′)")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | teste {len(test_df)}")

    itildeq_coeffs = {"alpha": args.alpha, "beta": args.beta,
                      "gamma": args.gamma}

    for wrapper_key in args.wrappers:
        cfg = CONFIGS[wrapper_key]
        if cfg["hp_source"] == "round5":
            hyperparameters = round5_hp(wrapper_key)
        else:  # fallback: HPs canônicos de trained_8020
            meta = json.load(open(
                common.resolve("pipeline/tmp/trained_8020") / wrapper_key
                / "model.json", encoding="utf-8"))
            hyperparameters = dict(meta["hyperparameters"])

        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        wrapper = common.wrapper_factory(wrapper_key)()
        wrapper.prepare(
            train_df,
            train_df.tail(400),
            input_steps=72,
            output_steps=36,
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
            apply_itildeq_coeffs(wrapper_key, args.alpha, args.beta, args.gamma)
        if cfg["static_dilations"]:
            # grade C′ fixa; a PROFUNDIDADE (prefixo) vem do HP da round 5
            wrapper.static_dilations = STATIC_DILATIONS

        wrapper.build(common.FixedHyperParameters(hyperparameters))
        assert wrapper.loss == cfg["loss"], (wrapper_key, wrapper.loss)
        print(f"[{wrapper_key}] loss={wrapper.loss} "
              f"hp_source={cfg['hp_source']} "
              f"static={cfg['static_dilations']}")

        print(f"[{wrapper_key}] treino {args.epochs} epocas "
              f"(loss={cfg['loss']})...")
        started = time.perf_counter()
        history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started
        losses = history.history.get("loss", [])
        print(f"[{wrapper_key}] {len(losses)} epocas em {elapsed:.0f}s | "
              f"loss final={losses[-1]:.5f}")

        model_dir = Path(args.output_dir) / wrapper_key
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        meta = {
            "wrapper_key": wrapper_key,
            "name": wrapper.name,
            "persistence_gate": False,
            "gate_mode": "static",
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "hyperparameters": hyperparameters,
            "params_source": ("HPs promovidos da Optuna Round 5 "
                              "(pipeline/tmp/optuna)"
                              if cfg["hp_source"] == "round5"
                              else "canonica 80/20 (loss mse) + dilatações C′"),
            "input_steps": 72,
            "output_steps": 36,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "loss": cfg["loss"],
            "training": {"epochs_run": len(losses),
                         "epochs_requested": args.epochs, "seed": args.seed,
                         "patience": 0,
                         "final_loss": float(losses[-1]) if losses else None,
                         "best_val_loss": None, "final_val_loss": None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                        "split": "80/20 sem validacao (rodada mista)"},
        }
        if cfg["loss"] == "itilde_q":
            meta["loss_coeffs"] = itildeq_coeffs
        if cfg["static_dilations"]:
            meta["static_dilations"] = STATIC_DILATIONS
        common.write_json(model_dir / "model.json", meta)

        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"],
                                         all_df["predicted"])
        print(f"[{wrapper_key}] MSE teste={metrics['mse']:.4f}")

    print(f"\nModelos: {args.output_dir}")
    print("Relatorio: python reports/src/relatorio_resultados.py --models-dir "
          "pipeline/tmp/trained_8020_itildeq --out-dir "
          "data/results/report_8020_itildeq")


if __name__ == "__main__":
    main()
