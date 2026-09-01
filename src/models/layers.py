"""Shared Keras layers/helpers for the Seq2Seq wrappers."""

import tensorflow as tf
from tensorflow.keras.layers import Lambda


class PersistenceGate(tf.keras.layers.Layer):
    """Learned per-horizon blend between the model output and persistence.

    output_h = sigmoid(g_h) * prediction_h + (1 - sigmoid(g_h)) * last_observed_h

    The gate starts at 0.5 (zero-initialized logits) and each horizon learns
    how much of the raw forecast to trust versus repeating the last observed
    value. Low horizons typically converge toward persistence, where the raw
    forecasts are weakest.
    """

    def __init__(self, output_steps, **kwargs):
        super().__init__(**kwargs)
        self.output_steps = int(output_steps)

    def build(self, input_shape):
        self.gate_logits = self.add_weight(
            name="gate_logits",
            shape=(self.output_steps, 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        prediction, last_observed = inputs
        gate = tf.sigmoid(self.gate_logits)[None, :, :]
        return gate * prediction + (1.0 - gate) * last_observed

    def get_config(self):
        config = super().get_config()
        config.update({"output_steps": self.output_steps})
        return config


def apply_persistence_gate(outputs, decoder_inputs, output_steps, name="output"):
    """Blend *outputs* with the last observed target channel of the decoder.

    ``decoder_inputs`` must carry the last observed value in its final channel
    (see ``Seq2SeqWrapper.create_sequences`` with ``persistence_gate=True``).
    """
    last_observed = Lambda(
        lambda x: x[:, :, -1:], name=f"{name}_last_observed"
    )(decoder_inputs)
    return PersistenceGate(output_steps, name=f"{name}_persistence_gate")(
        [outputs, last_observed]
    )
