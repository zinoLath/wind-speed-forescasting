"""Shared Keras layers/helpers for the Seq2Seq wrappers."""

import numpy as np
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


class DynamicPersistenceGate(tf.keras.layers.Layer):
    """Input-conditional per-horizon blend with persistence.

    g[t, h]   = sigmoid(W · f[t, h] + b_h)
    output[t, h] = g[t, h] * prediction[t, h] + (1 - g[t, h]) * last_observed[t]

    ``f[t, h]`` is the decoder feature vector feeding the output Dense layer,
    ``W`` a learned scalar projection and ``b_h`` a per-horizon bias. Both are
    zero-initialized so the gate starts at 0.5, matching the static
    ``PersistenceGate``; with W = 0 the layer degenerates to it. The gate
    tensor of the last call is kept in ``self.gate_values`` for diagnostics.
    """

    def __init__(self, output_steps, **kwargs):
        super().__init__(**kwargs)
        self.output_steps = int(output_steps)
        self.gate_values = None

    def build(self, input_shape):
        features_shape = tuple(input_shape[0])
        feature_dim = int(features_shape[-1])
        self.feature_kernel = self.add_weight(
            name="feature_kernel",
            shape=(feature_dim, 1),
            initializer="zeros",
            trainable=True,
        )
        self.horizon_bias = self.add_weight(
            name="horizon_bias",
            shape=(self.output_steps, 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        features, prediction, last_observed = inputs
        logits = tf.tensordot(features, self.feature_kernel, axes=[[-1], [0]])
        logits = logits + self.horizon_bias[None, :, :]
        gate = tf.sigmoid(logits)
        self.gate_values = gate
        return gate * prediction + (1.0 - gate) * last_observed

    def get_config(self):
        config = super().get_config()
        config.update({"output_steps": self.output_steps})
        return config


def apply_persistence_gate(outputs, decoder_inputs, output_steps, name="output",
                           features=None, mode="static"):
    """Blend *outputs* with the last observed target channel of the decoder.

    ``decoder_inputs`` must carry the last observed value in its final channel
    (see ``Seq2SeqWrapper.create_sequences`` with ``persistence_gate=True``).
    With ``mode="dynamic"`` the blend weight is conditioned on ``features``
    (the decoder feature tensor feeding the output Dense layer) through a
    learned projection plus a per-horizon bias; otherwise a static per-horizon
    parameter is used.
    """
    last_observed = Lambda(
        lambda x: x[:, :, -1:], name=f"{name}_last_observed"
    )(decoder_inputs)
    if mode == "dynamic":
        if features is None:
            raise ValueError("mode='dynamic' requires the decoder feature tensor.")
        return DynamicPersistenceGate(output_steps, name=f"{name}_dynamic_gate")(
            [features, outputs, last_observed]
        )
    return PersistenceGate(output_steps, name=f"{name}_persistence_gate")(
        [outputs, last_observed]
    )


def get_dynamic_gate_layer(model):
    """Return the DynamicPersistenceGate layer of a built model, if any."""
    return next(
        (layer for layer in model.layers
         if isinstance(layer, DynamicPersistenceGate)),
        None,
    )


def make_gate_extractor(model):
    """Return f(inputs) -> gate values (n, output_steps, 1) for diagnostics.

    Recomputes sigmoid(features @ W + b) in numpy from the dynamic gate
    layer's weights, using a sub-model for the decoder feature tensor.
    """
    gate_layer = get_dynamic_gate_layer(model)
    if gate_layer is None:
        raise ValueError("Model has no DynamicPersistenceGate layer.")
    features_model = tf.keras.Model(model.inputs, gate_layer.input[0])
    kernel = gate_layer.feature_kernel.numpy()
    bias = gate_layer.horizon_bias.numpy()

    def extract(inputs):
        features = features_model.predict(inputs, verbose=0)
        logits = features @ kernel + bias[None, :, :]
        return 1.0 / (1.0 + np.exp(-logits))

    return extract
