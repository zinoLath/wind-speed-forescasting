"""AB: wavelet por fatia (padrão) vs wavelet na série inteira, antes do split.

A wavelet pré-computada no dataset completo torna as colunas *_wavelet
idênticas para treino/val/teste — o guard de prepare_data ('já existe')
evita o re-denoising por fatia. Roda lstm e tcn com os hp do staging v2.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import common
from src.utils import wavelet_denoising

LEVEL = 1
ORIG = common.load_dataset


def patched(path, keep_raw=()):
    frame = ORIG(path, keep_raw=keep_raw)
    for col in keep_raw:
        if f"{col}_wavelet" not in frame.columns:
            frame[f"{col}_wavelet"] = wavelet_denoising(frame[col].values, level=LEVEL)
    return frame


common.load_dataset = patched

import tests.train_compare_tcn as driver  # noqa: E402  (após o patch)

sys.argv = [
    "train_compare", "--wrappers", "lstm", "tcn",
    "--config", "/tmp/opencode/report_compare.json",
    "--out-base", "pipeline/tmp/trained_compare_wavfull",
]
driver.main()
