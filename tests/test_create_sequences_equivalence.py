"""Verify that the vectorised create_sequences matches the original loop.

Run from the project root:  python tests/test_create_sequences_equivalence.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.seq2seq_wrapper import Seq2SeqWrapper


def reference_loop(data, input_steps, output_steps, target_col_index, decoder_mode, target_mode):
    """The original per-sample loop implementation, kept as the oracle."""
    data = np.asarray(data)
    X_encoder, X_decoder, y_decoder = [], [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X_encoder.append(data[i:(i + input_steps)])
        if decoder_mode == "teacher_forcing":
            decoder_input = np.zeros((output_steps, 1))
            decoder_input[0] = data[i + input_steps - 1, target_col_index]
            decoder_input[1:] = data[
                i + input_steps:i + input_steps + output_steps - 1, target_col_index
            ].reshape(-1, 1)
        elif decoder_mode == "direct":
            decoder_input = np.empty((output_steps, 2))
            decoder_input[:, 0] = data[i + input_steps - 1, target_col_index]
            decoder_input[:, 1] = np.arange(1, output_steps + 1) / output_steps
        else:
            raise ValueError(decoder_mode)
        X_decoder.append(decoder_input)
        target = data[i + input_steps:i + input_steps + output_steps, target_col_index].reshape(-1, 1)
        if target_mode == "residual":
            target = target - data[i + input_steps - 1, target_col_index]
        y_decoder.append(target)
    return np.array(X_encoder), np.array(X_decoder), np.array(y_decoder)


def main():
    rng = np.random.default_rng(0)
    data = rng.random((500, 7)).astype(np.float32)

    for decoder_mode in ("teacher_forcing", "direct"):
        for target_mode in ("absolute", "residual"):
            expected = reference_loop(data, 24, 12, 3, decoder_mode, target_mode)
            actual = Seq2SeqWrapper.create_sequences(data, 24, 12, 3, decoder_mode, target_mode)
            for name, exp, act in zip(("X_encoder", "X_decoder", "y_decoder"), expected, actual):
                if not np.allclose(exp, act, atol=1e-6):
                    raise AssertionError(f"{decoder_mode}/{target_mode}: {name} mismatch")
                if act.dtype != np.float32:
                    raise AssertionError(f"{decoder_mode}/{target_mode}: {name} is not float32")
            print(f"OK {decoder_mode}/{target_mode}")

    print("create_sequences equivalence: PASS")


if __name__ == "__main__":
    main()
