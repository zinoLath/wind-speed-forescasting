"""80/20 rodada de CONJUNTO DE ATRIBUTOS estendido (feature set A/B).

Pergunta respondida: o que acontece se os modelos consumirem também o
cisalhamento (``cis1..cis19``) e as variáveis meteorológicas ``humid``/``temp``
que hoje são descartadas pelo ``load_dataset``? A direção permanece limitada
a 40–140 m (``dir{h}_sin/cos``), como na canônica.

Diferenças para a rodada canônica (execução 16/10,
``pipeline/tmp/trained_8020_itildeq``) — conjunto de atributos E perda:

- dataset carregado com ``keep_raw=(cis1..19, humid, temp)`` ⇒ entrada do
  encoder passa de 46 para 67 canais (+ alvo);
- **perda MSE pura em TODOS os wrappers** (incluindo LSTM/LSTM-Bi, que na
  canônica usam ITILDE-Q) — decisão 21/10 para a linha de features;
- **HPs e dilatações C′ copiados do ``model.json`` dos modelos canônicos
  restaurados** (não da Round 5, que sofreu transferência negativa e foi
  revertida).

Protocolo idêntico ao canônico: split 80/20 sem validação, 60 épocas fixas,
seed 42, janela 72 → 36, alvo ``ws100_wavelet`` (sym18 nível 2), métricas
contra o ``ws100`` bruto.

    python scripts/experiments/lidar_8020_featurex_train.py
    python reports/src/relatorio_resultados.py \\
        --models-dir pipeline/tmp/trained_8020_featurex \\
        --out-dir data/results/report_8020_featurex
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

SEED = 42
WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_featurex")
CANON_DIR = common.resolve("pipeline/tmp/trained_8020_itildeq")
EXTRA_KEEP = ("humid", "temp") + tuple(f"cis{i}" for i in range(1, 20))
FEATURE_SET = ("ws/v 40-140, dir sin/cos 40-140, hour ciclico "
               "+ cis1..19 + humid + temp (sem disp/vdisp/press)")


def canonical_meta(wrapper_key):
    """Meta do modelo canônico restaurado (fonte de HPs/loss/dilatações)."""
    path = CANON_DIR / wrapper_key / "model.json"
    return json.load(open(path, encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        choices=WRAPPERS)
    parser.add_argument("--units", type=int, default=None,
                        help="sobrescreve lstm_units (testes de capacidade "
                             "para lstm/lstm_bi)")
    parser.add_argument("--enc-dropout", type=float, default=None,
                        help="sobrescreve encoder_dropout_rate (testes de "
                             "regularização para lstm/lstm_bi)")
    parser.add_argument("--heights", type=int, nargs="+", default=None,
                        help="mantém ws/v/dir apenas nestas alturas "
                             "(base canônica 46, sem extras featurex)")
    parser.add_argument("--featurex-extras", action="store_true",
                        help="com --heights: inclui cis1..19/humid/temp "
                             "(featurex) sobre o filtro de alturas")
    parser.add_argument("--extra-cols", default=None,
                        help="com --heights: lista explícita de colunas "
                             "extras (vírgula), ex.: cis1,cis7,cis11,press,"
                             "humid,temp")
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | rodada 8020-featurex "
          f"(HPs/loss da canônica {CANON_DIR.name})")

    if not args.heights:
        keep = EXTRA_KEEP
    elif args.extra_cols:
        keep = tuple(c.strip() for c in args.extra_cols.split(",") if c.strip())
    elif args.featurex_extras:
        keep = EXTRA_KEEP
    else:
        keep = ()
    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=keep)
    if args.heights:
        import re as _re

        allowed = set(args.heights)
        drop_cols = [
            c for c in dataset.columns
            if (m := _re.match(r"^(ws|v|dir)(\d+)(?:_sin|_cos)?$", c))
            and int(m.group(2)) not in allowed
        ]
        dataset = dataset.drop(columns=drop_cols)
        print(f"filtro de alturas {sorted(allowed)}: {dataset.shape[1]} "
              f"colunas (removidas {len(drop_cols)})")
    print(f"Dataset {len(dataset)} linhas x {dataset.shape[1]} colunas "
          f"(entrada estendida: {FEATURE_SET})")
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    print(f"treino {len(train_df)} | teste {len(test_df)}")

    for wrapper_key in args.wrappers:
        canon = canonical_meta(wrapper_key)
        hyperparameters = dict(canon["hyperparameters"])
        if args.units and wrapper_key in ("lstm", "lstm_bi"):
            hyperparameters["lstm_units"] = args.units
        if args.enc_dropout is not None and wrapper_key in ("lstm", "lstm_bi"):
            hyperparameters["encoder_dropout_rate"] = args.enc_dropout
        loss = "mse"  # run de features: perda pura em todos (decisão 21/10)
        coeffs = None
        static = canon.get("static_dilations")

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

        wrapper.loss_mode_override = loss
        if coeffs:
            import importlib

            from src.models import losses

            modules = {
                "lstm": "src.models.s2s_lstm_wrapper",
                "lstm_bi": "src.models.s2s_lstm_bi_wrapper",
                "tcn": "src.models.s2s_tcn_wrapper",
                "tcn_bi": "src.models.s2s_tcn_bi_wrapper",
            }
            module = importlib.import_module(modules[wrapper_key])
            module.itilde_q = (
                lambda output_steps: losses.itilde_q(
                    output_steps, alpha=coeffs["alpha"], beta=coeffs["beta"],
                    gamma=coeffs["gamma"]))
        if static:
            wrapper.static_dilations = static

        wrapper.build(common.FixedHyperParameters(hyperparameters))
        assert wrapper.loss == loss, (wrapper_key, wrapper.loss)
        print(f"[{wrapper_key}] loss={loss} hp=canônica16/10 "
              f"static={bool(static)} coeffs={bool(coeffs)} "
              f"entradas={wrapper.num_encoder_features}")

        print(f"[{wrapper_key}] treino {args.epochs} epocas (loss={loss})...")
        started = time.perf_counter()
        history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started
        losses_hist = history.history.get("loss", [])
        print(f"[{wrapper_key}] {len(losses_hist)} epocas em {elapsed:.0f}s | "
              f"loss final={losses_hist[-1]:.5f}")

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
            "params_source": ("canônica 16/10 (trained_8020_itildeq) com "
                              "perda MSE pura — run de features: mudam "
                              "conjunto de entrada e perda"),
            "feature_keep": list(keep),
            "feature_set": (f"alturas {sorted(args.heights)} (ws/v/dir), "
                            "base canônica, MSE"
                            if args.heights else FEATURE_SET),
            "input_steps": 72,
            "output_steps": 36,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "loss": loss,
            "training": {"epochs_run": len(losses_hist),
                         "epochs_requested": args.epochs, "seed": args.seed,
                         "patience": 0,
                         "final_loss": (float(losses_hist[-1])
                                        if losses_hist else None),
                         "best_val_loss": None, "final_val_loss": None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                        "split": "80/20 sem validacao (rodada de features)"},
        }
        if coeffs:
            meta["loss_coeffs"] = coeffs
        if static:
            meta["static_dilations"] = static
        if args.units and wrapper_key in ("lstm", "lstm_bi"):
            meta["units_override"] = args.units
        if args.enc_dropout is not None and wrapper_key in ("lstm", "lstm_bi"):
            meta["enc_dropout_override"] = args.enc_dropout
        common.write_json(model_dir / "model.json", meta)

        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"],
                                         all_df["predicted"])
        print(f"[{wrapper_key}] MSE teste={metrics['mse']:.4f} "
              f"R2={metrics['r2']:.4f}")

    print(f"\nModelos: {args.output_dir}")
    print("Relatorio: python reports/src/relatorio_resultados.py --models-dir "
          "pipeline/tmp/trained_8020_featurex --out-dir "
          "data/results/report_8020_featurex")


if __name__ == "__main__":
    main()
