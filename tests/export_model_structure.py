"""Exporta os diagramas de arquitetura (sem persistence gate) para notebooks/.

Gera ``notebooks/model_structure_<ModelClass>.png`` a partir dos wrappers
treinados sem gate (pipeline/tmp/trained_compare_nogate), refletindo a
arquitetura definitiva: formas, nomes de camada, ativações e tag de
treinabilidade.

Executar a partir da raiz do projeto:
    python tests/export_model_structure.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common
from relatorio_resultados import NOGATE_DIR, build_wrapper, load_metadata

DATA_FILE = PROJECT_ROOT / "data" / "dataset.csv"
OUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    common.setup_tensorflow(True)
    dataset = common.load_dataset(DATA_FILE)
    for entry in load_metadata(NOGATE_DIR):
        meta = entry["meta"]
        ratios = meta.get("dataset", {})
        train_df, val_df, _ = common.split_dataset(
            dataset, ratios.get("train_ratio", 0.75), ratios.get("val_ratio", 0.20)
        )
        print(f"[{meta['wrapper_key']}] exportando {meta['model_class']} ...")
        wrapper = build_wrapper(meta, train_df, val_df, entry["dir"] / "model.keras")
        out = OUT_DIR / f"model_structure_{meta['model_class']}.png"
        common.setup_tensorflow(True).keras.utils.plot_model(
            wrapper.model,
            to_file=str(out),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
            rankdir="TB",
            dpi=110,
        )
        print(f"  -> {out}")
    print(f"Diagramas salvos em {OUT_DIR}")


if __name__ == "__main__":
    main()
