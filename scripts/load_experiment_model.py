"""Carrega um modelo de experimento usando as features gravadas no model.json.

Funciona com qualquer diretório em models/experiments/<condição>/<Wrapper>/
(por extenso: pipeline/tmp/trained_compare*/<Wrapper>/ também).

Uso programático:
    from scripts.load_experiment_model import load_experiment_model
    wrapper, dataset, _ = load_experiment_model(
        "models/experiments/baseline_todas_features/Seq2Seq_LSTM")
    pred, real = wrapper.rolling_forecast(dataset)   # ou:
    df = common.predict_all_horizons(wrapper, test_df)

CLI:
    python scripts/load_experiment_model.py <model_dir> [--rolling] [--next36]
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_experiment_model(model_dir, dataset_path=None):
    """Reconstrói o wrapper com as features/protocolo do model.json e carrega os pesos.

    O model.json é a fonte da verdade: denoise (colunas cruas a preservar no
    loader), denoise_level, features (filtro do encoder, se houver),
    decoder_mode e gate. Funciona para qualquer variante da sessão.
    """
    from src import common

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        candidates = [PROJECT_ROOT / model_dir]
        if model_dir.parts and model_dir.parts[0] == "experiments":
            candidates.append(PROJECT_ROOT / "models/experiments"
                              / Path(*model_dir.parts[1:]))
        model_dir = next((c for c in candidates if c.is_dir()), model_dir)
    meta = json.load(open(model_dir / "model.json"))

    dataset_path = dataset_path or meta.get("dataset", {}).get("source", "data/dataset.csv")
    dataset = common.load_dataset(PROJECT_ROOT / dataset_path,
                                  keep_raw=tuple(meta["denoise"]))
    ratios = meta.get("dataset", {})
    train_df, val_df, test_df = common.split_dataset(
        dataset, ratios.get("train_ratio", 0.75), ratios.get("val_ratio", 0.20))

    wrapper = common.wrapper_factory(meta["wrapper_key"])()
    wrapper.prepare(
        train_df, val_df,
        input_steps=meta["input_steps"], output_steps=meta["output_steps"],
        target_col=meta["target_col"], denoise=meta["denoise"],
        denoise_level=meta["denoise_level"], features=meta.get("features"),
        decoder_mode=meta.get("decoder_mode", "teacher_forcing"),
        persistence_gate=bool(meta.get("persistence_gate", False)),
        create_sequences=False,
    )
    wrapper.gate_mode = meta.get("gate_mode", "static")
    wrapper.context_mode = meta.get("context_mode", "repeat")
    wrapper.build(common.FixedHyperParameters(meta["hyperparameters"]))
    wrapper.model.load_weights(model_dir / "model.keras")
    return wrapper, dataset, model_dir


def main():
    import argparse

    from src import common

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="diretório com model.keras + model.json")
    parser.add_argument("--rolling", action="store_true",
                        help="recalcula o rolling h=36 e imprime as métricas")
    parser.add_argument("--next36", action="store_true",
                        help="previsão das próximas 6 h a partir do fim da série")
    args = parser.parse_args()

    wrapper, dataset, model_dir = load_experiment_model(args.model_dir)
    meta = json.load(open(model_dir / "model.json"))
    feats = meta.get("features")
    print(f"modelo: {meta['name']} | loss={meta['loss']} | "
          f"features={len(feats) if feats else 'todas as colunas do frame'} | "
          f"decoder={meta.get('decoder_mode', 'teacher_forcing')}")

    if args.rolling:
        import numpy as np

        val_rows = int(len(dataset) * (meta["dataset"]["train_ratio"]
                                       + meta["dataset"]["val_ratio"]))
        pred, act = wrapper.rolling_forecast(dataset, test_start=val_rows)
        mae = float(np.mean(np.abs(pred - act)))
        rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
        print(f"rolling h=36 (teste): MAE={mae:.4f} RMSE={rmse:.4f} ({len(act)} origens)")

    if args.next36:
        import numpy as np

        data_scaled, target_idx = wrapper._to_scaled_array(dataset)
        encoder_input = data_scaled[-wrapper.input_steps:][np.newaxis, ...]
        decoder_input, _ = wrapper._build_decoder_input(
            encoder_input, wrapper.output_steps, wrapper.decoder_mode, target_idx,
            persistence_gate=wrapper.persistence_gate)
        pred = wrapper.model.predict([encoder_input, decoder_input], verbose=0)[:, :, 0]
        values = wrapper.scaler_target.inverse_transform(
            pred.reshape(-1, 1)).ravel()
        print("próximas 36 passos (6 h) a partir do fim da série:")
        print(np.round(values, 3))


if __name__ == "__main__":
    main()
