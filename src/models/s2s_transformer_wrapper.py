import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Add,
    Dense,
    Dropout,
    Lambda,
    LayerNormalization,
    MultiHeadAttention,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from .seq2seq_wrapper import Seq2SeqWrapper


class WarmupCosineSchedule(LearningRateSchedule):
    """Linear warmup followed by cosine decay to a minimum fraction of base LR."""

    def __init__(self, base_lr, total_steps, warmup_steps, min_lr_ratio=0.01):
        super().__init__()
        self.base_lr = tf.cast(base_lr, dtype=tf.float32)
        self.total_steps = tf.cast(total_steps, dtype=tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        warmup = tf.minimum(step / tf.maximum(self.warmup_steps, 1.0), 1.0)
        progress = (step - self.warmup_steps) / tf.maximum(
            1.0, self.total_steps - self.warmup_steps
        )
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * progress))
        scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return self.base_lr * warmup * scale

    def get_config(self):
        return {
            "base_lr": float(self.base_lr),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "min_lr_ratio": self.min_lr_ratio,
        }


def make_learning_rate(hp, schedule_total_steps=None):
    """Builds the optimizer LR, wrapping a warmup+cosine schedule when requested."""
    lr = hp.Float(
        'learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3
    )
    if schedule_total_steps:
        warmup_steps = max(1, int(0.1 * schedule_total_steps))
        return WarmupCosineSchedule(lr, schedule_total_steps, warmup_steps)
    return lr


def _positional_encoding(length, d_model):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model)
    )
    positions = np.arange(length, dtype=np.float32)[:, None]
    pe = np.zeros((length, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


def _add_positional_encoding(length, d_model, name):
    """Returns a Lambda layer that adds a fixed sinusoidal positional encoding."""
    encoding = _positional_encoding(length, d_model)
    encoding_tensor = tf.constant(encoding, dtype=tf.float32)

    def add_encoding(x):
        return x + encoding_tensor

    return Lambda(add_encoding, name=name)


def _encoder_block(x, d_model, num_heads, key_dim, ff_dim, dropout_rate, name):
    attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name=f"{name}_attention",
    )(x, x)
    attention = Dropout(dropout_rate, name=f"{name}_attention_dropout")(attention)
    x = Add(name=f"{name}_add1")([x, attention])
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)

    ffn = Dense(ff_dim, activation="relu", name=f"{name}_ffn1")(x)
    ffn = Dense(d_model, name=f"{name}_ffn2")(ffn)
    ffn = Dropout(dropout_rate, name=f"{name}_ffn_dropout")(ffn)
    x = Add(name=f"{name}_add2")([x, ffn])
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    return x


def _decoder_block(x, encoder_outputs, d_model, num_heads, key_dim, ff_dim, dropout_rate, name):
    attention_self = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name=f"{name}_self_attention",
    )(x, x, use_causal_mask=True)
    attention_self = Dropout(
        dropout_rate, name=f"{name}_self_attention_dropout"
    )(attention_self)
    x = Add(name=f"{name}_add1")([x, attention_self])
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)

    attention_cross = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name=f"{name}_cross_attention",
    )(x, encoder_outputs)
    attention_cross = Dropout(
        dropout_rate, name=f"{name}_cross_attention_dropout"
    )(attention_cross)
    x = Add(name=f"{name}_add2")([x, attention_cross])
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)

    ffn = Dense(ff_dim, activation="relu", name=f"{name}_ffn1")(x)
    ffn = Dense(d_model, name=f"{name}_ffn2")(ffn)
    ffn = Dropout(dropout_rate, name=f"{name}_ffn_dropout")(ffn)
    x = Add(name=f"{name}_add3")([x, ffn])
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ln3")(x)
    return x


class S2STransformerWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_Transformer"

    def build(self, hp):
        self._require_prepared()

        learning_rate = make_learning_rate(
            hp, schedule_total_steps=getattr(self, 'schedule_total_steps', None)
        )
        d_model = hp.Int('d_model', min_value=32, max_value=128, step=16, default=64)
        num_heads = hp.Int('num_heads', min_value=1, max_value=8, step=1, default=4)
        num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1, default=2)
        ff_dim = hp.Int('ff_dim', min_value=64, max_value=512, step=32, default=128)
        dropout_rate = hp.Float(
            'dropout_rate', min_value=0.0, max_value=0.3, step=0.05, default=0.1
        )

        key_dim = max(1, d_model // num_heads)

        encoder_inputs = Input(
            shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs'
        )
        encoder_outputs = Dense(d_model, name='encoder_input_projection')(encoder_inputs)
        encoder_outputs = _add_positional_encoding(
            self.input_steps, d_model, 'encoder_positional_encoding'
        )(encoder_outputs)
        encoder_outputs = Dropout(dropout_rate, name='encoder_input_dropout')(encoder_outputs)
        for i in range(num_layers):
            encoder_outputs = _encoder_block(
                encoder_outputs,
                d_model,
                num_heads,
                key_dim,
                ff_dim,
                dropout_rate,
                f'encoder_block_{i}',
            )

        decoder_inputs = Input(
            shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs'
        )
        decoder_outputs = Dense(d_model, name='decoder_input_projection')(decoder_inputs)
        decoder_outputs = _add_positional_encoding(
            self.output_steps, d_model, 'decoder_positional_encoding'
        )(decoder_outputs)
        decoder_outputs = Dropout(dropout_rate, name='decoder_input_dropout')(decoder_outputs)
        for i in range(num_layers):
            decoder_outputs = _decoder_block(
                decoder_outputs,
                encoder_outputs,
                d_model,
                num_heads,
                key_dim,
                ff_dim,
                dropout_rate,
                f'decoder_block_{i}',
            )

        decoder_dense = TimeDistributed(
            Dense(1, activation='linear', name='output_dense'), name='output_layer'
        )
        decoder_outputs_final = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=getattr(self, 'loss', 'mse'),
            metrics=['mae'],
        )

        return self.model