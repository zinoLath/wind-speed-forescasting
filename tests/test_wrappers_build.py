"""Build every registered wrapper on synthetic data to catch regressions.

Run from the project root:  python tests/test_wrappers_build.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common


def main():
    common.validate_wrapper_names()
    print("wrapper names unique: OK")

    rng = np.random.default_rng(0)
    index = pd.date_range("2022-01-01", periods=400, freq="10min")
    df = pd.DataFrame(
        {
            "ws100": rng.uniform(3, 12, 400),
            "ws90": rng.uniform(3, 12, 400),
            "dir100": rng.uniform(0, 360, 400),
        },
        index=index,
    )

    for gate in (False, True):
        for key in common.WRAPPERS:
            wrapper = common.wrapper_factory(key)()
            wrapper.prepare(
                df.iloc[:320], df.iloc[320:],
                input_steps=24, output_steps=12, target_col="ws100_wavelet",
                persistence_gate=gate,
            )
            model = wrapper.build(common.FixedHyperParameters(common.DEFAULT_HYPERPARAMETERS[key]))
            params = model.count_params()
            pred = wrapper.predict(df.iloc[320:320 + 24])
            assert pred.shape == (12,), f"{key}: unexpected prediction shape {pred.shape}"
            label = f"{key} gate={'on' if gate else 'off'}"
            print(f"OK {label:<22} {wrapper.name:<28} params={params:>9,} predict={pred.round(2)}")

    print("all wrappers build: PASS")


if __name__ == "__main__":
    main()