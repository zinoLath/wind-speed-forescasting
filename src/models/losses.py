"""Training losses shared by the Seq2Seq wrappers.

The wrappers train against the horizon-weighted MSE so that the model
prioritises the longer forecast horizons (where the error is largest and the
persistence bias is strongest). The weight grows linearly with the horizon,
so the last step counts ~2x the first while the mean weight stays 1.
"""

import numpy as np
import tensorflow as tf

LOSS_NAME = "horizon_weighted_mse"


def horizon_weights(output_steps):
    """(output_steps,) float32 weights growing gently with the horizon.

    Ramp from ~0.7 to ~1.3 (1 + h/output_steps, mean-normalised): every
    horizon stays meaningful in the loss while the far end counts ~2x the
    first step.
    """
    steps = np.arange(1, output_steps + 1, dtype=np.float32)
    ramp = 1.0 + steps / float(output_steps)
    return ramp / float(ramp.mean())


def horizon_weighted_mse(output_steps):
    """Keras loss: weighted MSE over the output horizon (weight grows with h)."""
    weights = tf.constant(
        horizon_weights(output_steps)[None, :, None], dtype=tf.float32
    )

    def loss(y_true, y_pred):
        return tf.reduce_mean(weights * tf.square(y_pred - y_true))

    return loss