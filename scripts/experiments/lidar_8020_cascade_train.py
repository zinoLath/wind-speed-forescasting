"""Cascata multi-resolução 80/20: head 1 (médias horárias) + head 2 (detalhe).

Head 1: ``lstm_mean`` congelado (pipeline/tmp/trained_8020_hmean) prevê a
média de cada hora futura (6 valores). Head 2: ``lstm_detail`` prevê o
resíduo de 10 min ``ws100 − broadcast(médias)`` com a predição da head 1
como canal extra do decoder. Predição final = broadcast(médias) + detalhe
(em unidades do ws100, com a correção do offset do MinMax no painel).

Variantes do canal de dica (o vazamento in-sample da head 1 faz a head 2
aprender correções que não transferem para o teste):
  * ``noisy`` — ruído gaussiano σ=0.6 nos hints durante o TREINO (estilo
    DART), casando a distribuição do canal com a qualidade out-of-sample
    da head; na avaliação o hint real entra sem ruído;
  * o canal do decoder é LEVEL-FREE: (hint − última observação), para
    bloquear o atalho espúrio "nível do hint → nível do resíduo".
Controle: mesmo wrapper/mesmos HPs/seed SEM head anexada (LSTM direto no
ws100 bruto, sem canal de dica) — isola o efeito do condicionamento.

Protocolo canônico: split 80/20, seed 42, 60 épocas, loss MSE,
decoder direto, avaliação vs ws100 bruto.

    python scripts/experiments/lidar_8020_cascade_train.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

SEED = 42
INPUT_STEPS = 72
OUTPUT_STEPS = 36
POOL = 6
HINT_NOISE = 0.6
HEAD_DIR = common.resolve("pipeline/tmp/trained_8020_hmean/lstm_mean")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_cascade")
KEYS = ("r2", "mse", "mae")


def compact(metrics):
    return {k: float(metrics[k]) for k in ("samples", "mse", "rmse", "r2", "mae")
            if k in metrics}


def evaluate_panel(wrapper, test_df, dataset, head_broadcast=None):
    """Métricas vs ws100 bruto; ``head_broadcast`` soma o componente suave.
    A saída do detalhe é invertida no espaço do scaler da cascata, então a
    composição em unidades carrega o data_min do MinMax uma vez:
    inverse(ŷ_s + m_s) = inverse(ŷ_s) + inverse(m_s) − data_min_."""
    df = common.predict_all_horizons(wrapper, test_df)
    actual_raw = dataset["ws100"].reindex(df["timestamp"]).to_numpy()
    predicted = df["predicted"].to_numpy()
    if head_broadcast is not None:
        predicted = predicted + head_broadcast - float(
            wrapper.scaler_target.data_min_[0])
    metrics = common.compute_metrics(actual_raw, predicted)
    std_ratio = float(np.std(predicted) / np.std(actual_raw))
    per_h = []
    for h in range(1, OUTPUT_STEPS + 1):
        sel = (df["horizon"] == h).to_numpy()
        per_h.append({
            "horizon": h,
            **{k: float(common.compute_metrics(
                actual_raw[sel], predicted[sel])[k]) for k in KEYS},
        })
    residual = common.compute_metrics(df["actual"], df["predicted"])
    return {
        "metrics": compact(metrics),
        "std_ratio": std_ratio,
        "per_horizon": per_h,
        "residual_skill": compact(residual),
    }, df, actual_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | cascata lstm_mean + lstm_detail")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, _, test_df = common.split_dataset(dataset, 0.80, 0.0)
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | teste {len(test_df)}")

    hyperparameters = json.load(
        open(HEAD_DIR / "model.json"))["hyperparameters"]

    def train_variant(key, wrapper):
        tf.keras.utils.set_random_seed(args.seed)
        wrapper.loss_mode_override = "mse"
        wrapper.prepare(
            train_df,
            train_df.tail(400),
            input_steps=INPUT_STEPS,
            output_steps=OUTPUT_STEPS,
            target_col="ws100",
            denoise=("ws100",),
            denoise_level=2,
        )
        batch_size = int(hyperparameters.get("batch_size", 32))
        steps = int((len(train_df) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps * args.epochs
        wrapper.build(common.FixedHyperParameters(hyperparameters))
        assert wrapper.loss == "mse", wrapper.loss
        print(f"[{key}] treino {args.epochs} epocas (loss=mse)...")
        started = time.perf_counter()
        history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started
        losses = history.history.get("loss", [])
        print(f"[{key}] {len(losses)} epocas em {elapsed:.0f}s | "
              f"loss final={losses[-1]:.5f}")
        model_dir = OUTPUT_DIR / key
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        common.write_json(model_dir / "model.json", {
            "wrapper_key": key,
            "name": wrapper.name,
            "head": type(wrapper.head).__name__ if wrapper.head else None,
            "hint_noise": float(getattr(wrapper, "hint_noise", 0.0)),
            "decoder_mode": "direct",
            "target_mode": "absolute",
            "hyperparameters": hyperparameters,
            "params_source": "canonica 80/20 (LSTM)",
            "input_steps": INPUT_STEPS,
            "output_steps": OUTPUT_STEPS,
            "target_col": "ws100",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "loss": "mse",
            "training": {"epochs_run": len(losses),
                         "epochs_requested": args.epochs,
                         "seed": args.seed,
                         "final_loss": float(losses[-1]) if losses else None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0},
        })
        return wrapper

    head = common.wrapper_factory("lstm_mean")()
    head.prepare(
        train_df,
        train_df.tail(400),
        input_steps=INPUT_STEPS,
        output_steps=POOL,
        target_col="ws100_wavelet",
        denoise=("ws100",),
        denoise_level=2,
    )
    head.model = tf.keras.models.load_model(HEAD_DIR / "model.keras")
    print(f"Head 1 carregada: {HEAD_DIR}")

    control = train_variant("lstm_detail_ctrl",
                            common.wrapper_factory("lstm_detail")())

    m_df = common.predict_all_horizons(head, test_df)
    n_seq = len(m_df) // POOL
    head_broadcast = (
        m_df["predicted"].to_numpy().reshape(n_seq, POOL)
        .repeat(OUTPUT_STEPS // POOL, axis=1).ravel()
    )
    head_eval = None

    control_eval, ctrl_df, actual_raw = evaluate_panel(control, test_df, dataset)

    cascade = common.wrapper_factory("lstm_detail")()
    cascade.attach_head(head)
    cascade.hint_noise = HINT_NOISE
    train_variant("lstm_detail", cascade)
    cascade.hint_noise = 0.0
    cascade_eval, cas_df, _ = evaluate_panel(
        cascade, test_df, dataset, head_broadcast=head_broadcast)

    head_eval = common.compute_metrics(actual_raw, head_broadcast)
    last_obs = cas_df.groupby("origin", sort=False).first()["persistence"]
    pers_eval = common.compute_metrics(
        actual_raw, np.repeat(last_obs.to_numpy(), OUTPUT_STEPS))

    evaluation = {
        "protocolo": "80/20 seed 42 loss=mse hp LSTM canonica 60 epocas",
        "hint_noise_sigma_treino": HINT_NOISE,
        "head1_broadcast_vs_raw": compact(head_eval),
        "controle_lstm_raw_vs_raw": control_eval,
        "cascata_vs_raw": cascade_eval,
        "referencias": {"lstm_canonica_36p_itildeq_r2": 0.338,
                        "lstm_mean_broadcast_r2": 0.346},
        "persistencia_broadcast": compact(pers_eval),
    }
    common.write_json(OUTPUT_DIR / "evaluation.json", evaluation)

    print(f"\n=== Resultados (teste, {n_seq} origens x 36 passos) ===")
    print(f"{'modelo':<34} {'MSE':>8} {'R2':>7} {'std_ratio':>9}")
    rows = [
        ("head1 broadcast (médias)", head_eval, None),
        ("controle LSTM raw (sem dica)", control_eval["metrics"],
         control_eval["std_ratio"]),
        ("CASCATA (head1 + detalhe)", cascade_eval["metrics"],
         cascade_eval["std_ratio"]),
        ("persistência broadcast", pers_eval, None),
    ]
    for name, m, sr in rows:
        sr_txt = f"{sr:9.3f}" if sr is not None else "        -"
        print(f"{name:<34} {m['mse']:>8.4f} {m['r2']:>7.3f} {sr_txt:>9}")
    print(f"head2 residual: R2 = "
          f"{cascade_eval['residual_skill']['r2']:.3f}")
    print("\nPor horizonte (R2): passo | head1 | controle | cascata")
    for h_idx in range(OUTPUT_STEPS):
        mask = (cas_df["horizon"] == h_idx + 1).to_numpy()
        h1 = float(common.compute_metrics(
            actual_raw[mask], head_broadcast[mask])["r2"])
        print(f"  h{h_idx + 1:>2} | {h1:6.3f} | "
              f"{control_eval['per_horizon'][h_idx]['r2']:6.3f} | "
              f"{cascade_eval['per_horizon'][h_idx]['r2']:6.3f}")
    print(f"\nArtefatos: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
