import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Add,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW
from .seq2seq_wrapper import Seq2SeqWrapper
from .s2s_transformer_wrapper import make_learning_rate


class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """Trainable per-position embedding added to the sequence."""

    def __init__(self, max_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self.max_length, self.d_model),
            initializer="uniform",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.position_embeddings[None, :, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_length": self.max_length, "d_model": self.d_model})
        return config


def _preln_encoder_block(x, d_model, num_heads, key_dim, ff_dim, dropout_rate, name):
    attn_in = LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    attention = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_attention"
    )(attn_in, attn_in)
    attention = Dropout(dropout_rate, name=f"{name}_attention_dropout")(attention)
    x = Add(name=f"{name}_add1")([x, attention])

    ffn_in = LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    ffn = Dense(ff_dim, activation="relu", name=f"{name}_ffn1")(ffn_in)
    ffn = Dense(d_model, name=f"{name}_ffn2")(ffn)
    ffn = Dropout(dropout_rate, name=f"{name}_ffn_dropout")(ffn)
    x = Add(name=f"{name}_add2")([x, ffn])
    return x


def _preln_decoder_block(x, encoder_outputs, d_model, num_heads, key_dim, ff_dim, dropout_rate, name):
    self_ln = LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    attention_self = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_self_attention"
    )(self_ln, self_ln, use_causal_mask=True)
    attention_self = Dropout(
        dropout_rate, name=f"{name}_self_attention_dropout"
    )(attention_self)
    x = Add(name=f"{name}_add1")([x, attention_self])

    cross_ln = LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    attention_cross = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_cross_attention"
    )(cross_ln, encoder_outputs)
    attention_cross = Dropout(
        dropout_rate, name=f"{name}_cross_attention_dropout"
    )(attention_cross)
    x = Add(name=f"{name}_add2")([x, attention_cross])

    ffn_ln = LayerNormalization(epsilon=1e-6, name=f"{name}_ln3")(x)
    ffn = Dense(ff_dim, activation="relu", name=f"{name}_ffn1")(ffn_ln)
    ffn = Dense(d_model, name=f"{name}_ffn2")(ffn)
    ffn = Dropout(dropout_rate, name=f"{name}_ffn_dropout")(ffn)
    x = Add(name=f"{name}_add3")([x, ffn])
    return x


class S2STransformerPrelnWrapper(Seq2SeqWrapper):
    """Pre-LayerNorm transformer with learned positional encoding.

    Pre-LN residual blocks are generally more stable to train than the
    post-LN variant and do not require gradient clipping. Positional
    information is a trainable embedding instead of a fixed sinusoid.
    """

    def __init__(self):
        self.name = "Seq2Seq_Transformer_PreLN"

    def build(self, hp):
        self._require_prepared()

        learning_rate = make_learning_rate(
            hp, schedule_total_steps=getattr(self, 'schedule_total_steps', None)
        )
        d_model = hp.Int('d_model', min_value=32, max_value=256, step=16, default=64)
        num_heads = hp.Int('num_heads', min_value=1, max_value=8, step=1, default=4)
        num_layers = hp.Int('num_layers', min_value=1, max_value=4, step=1, default=2)
        ff_dim = hp.Int('ff_dim', min_value=64, max_value=512, step=32, default=128)
        dropout_rate = hp.Float(
            'dropout_rate', min_value=0.0, max_value=0.3, step=0.05, default=0.1
        )

        key_dim = max(1, d_model // num_heads)

        encoder_inputs = Input(
            shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs'
        )
        encoder_outputs = Dense(d_model, name='encoder_input_projection')(encoder_inputs)
        encoder_outputs = LearnedPositionalEncoding(
            self.input_steps, d_model, name='encoder_positional_encoding'
        )(encoder_outputs)
        encoder_outputs = Dropout(dropout_rate, name='encoder_input_dropout')(encoder_outputs)
        for i in range(num_layers):
            encoder_outputs = _preln_encoder_block(
                encoder_outputs, d_model, num_heads, key_dim, ff_dim, dropout_rate,
                f'encoder_block_{i}',
            )
        encoder_outputs = LayerNormalization(epsilon=1e-6, name='encoder_final_ln')(
            encoder_outputs
        )

        decoder_inputs = Input(
            shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs'
        )
        decoder_outputs = Dense(d_model, name='decoder_input_projection')(decoder_inputs)
        decoder_outputs = LearnedPositionalEncoding(
            self.output_steps, d_model, name='decoder_positional_encoding'
        )(decoder_outputs)
        decoder_outputs = Dropout(dropout_rate, name='decoder_input_dropout')(decoder_outputs)
        for i in range(num_layers):
            decoder_outputs = _preln_decoder_block(
                decoder_outputs, encoder_outputs, d_model, num_heads, key_dim, ff_dim,
                dropout_rate, f'decoder_block_{i}',
            )
        decoder_outputs = LayerNormalization(epsilon=1e-6, name='decoder_final_ln')(
            decoder_outputs
        )

        decoder_dense = TimeDistributed(
            Dense(1, activation='linear', name='output_dense'), name='output_layer'
        )
        decoder_outputs_final = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        weight_decay = getattr(self, 'weight_decay', None)
        if weight_decay:
            optimizer = AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                clipnorm=getattr(self, 'clipnorm', None),
            )
        else:
            optimizer = Adam(
                learning_rate=learning_rate,
                clipnorm=getattr(self, 'clipnorm', None),
            )
        self.model.compile(
            optimizer=optimizer,
            loss=getattr(self, 'loss', 'mse'),
            metrics=['mae'],
        )

        return self.model