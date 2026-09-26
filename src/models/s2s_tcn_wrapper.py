import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Concatenate,
    TimeDistributed,
    Attention,
    RepeatVector,
    Lambda,
)
from tensorflow.keras.optimizers import Adam, AdamW
from tcn import TCN
from .seq2seq_wrapper import Seq2SeqWrapper
from .layers import apply_persistence_gate
from .s2s_transformer_wrapper import make_learning_rate
from .tcn_hp import tcn_hyperparameters, TCN_RANGES
from .losses import horizon_weighted_mse, LOSS_NAME


class S2STCNWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_TCN"

    def build(self, hp):
        self._require_prepared()

        self.lr_schedule_mode = "constant"
        learning_rate = make_learning_rate(
            hp, lr_min=1e-4, lr_max=1e-2, lr_default=0.0026677478212305725,
        )
        weight_decay = hp.Choice('weight_decay', [0.0, 1e-4], default=0.0)
        loss = horizon_weighted_mse(self.output_steps)
        self.loss = LOSS_NAME

        if weight_decay:
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = Adam(learning_rate=learning_rate)

        # A single shared filter count feeds both encoder and decoder.
        shared_filters = hp.Int(
            "filters", 32, TCN_RANGES["filters"]["max_filters"], step=16, default=48
        )
        encoder_tcn_hp = tcn_hyperparameters(
            hp, "encoder", filters=48, kernel_size=2, nb_stacks=1,
            dropout_rate=0.4, dilation_rate=4,
            min_receptive_field=self.input_steps, filters_value=shared_filters,
        )
        decoder_tcn_hp = tcn_hyperparameters(
            hp, "decoder", filters=48, kernel_size=2, nb_stacks=1,
            dropout_rate=0.0, dilation_rate=4,
            min_receptive_field=self.output_steps, filters_value=shared_filters,
        )
        # How the encoder summary is injected into the decoder: mean pooling
        # over the whole window, the most recent step, or both concatenated.
        # "mean" keeps the historical behaviour; "last" mirrors the LSTM
        # family's final-state conditioning.
        context_pooling = hp.Choice(
            'context_pooling', ['mean', 'last', 'mean_last'], default='mean'
        )

        encoder_inputs = Input(
            shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs'
        )
        encoder_outputs = TCN(
            nb_filters=encoder_tcn_hp['filters'],
            kernel_size=encoder_tcn_hp['kernel_size'],
            nb_stacks=encoder_tcn_hp['nb_stacks'],
            dropout_rate=encoder_tcn_hp['dropout_rate'],
            dilations=encoder_tcn_hp['dilations'],
            use_layer_norm=True,
            use_skip_connections=False,
            return_sequences=True,
            name='encoder_tcn',
        )(encoder_inputs)

        encoder_dim = encoder_tcn_hp['filters']

        if context_pooling == 'mean':
            encoder_context = Lambda(
                lambda x: K.mean(x, axis=1), name='encoder_context'
            )(encoder_outputs)
        elif context_pooling == 'last':
            encoder_context = Lambda(
                lambda x: x[:, -1, :], name='encoder_context'
            )(encoder_outputs)
        else:
            encoder_context = Concatenate(name='encoder_context')([
                Lambda(lambda x: K.mean(x, axis=1), name='encoder_context_mean')(encoder_outputs),
                Lambda(lambda x: x[:, -1, :], name='encoder_context_last')(encoder_outputs),
            ])
        encoder_context_repeated = RepeatVector(
            self.output_steps, name='encoder_context_repeated'
        )(encoder_context)

        decoder_inputs = Input(
            shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs'
        )

        # How the pooled encoder context reaches the decoder TCN: broadcast to
        # every step ("repeat", the historical behaviour), not at all ("none",
        # conditioning flows only through the attention path), or broadcast
        # plus a per-step horizon fraction channel ("horizon").
        self.context_mode = getattr(self, 'context_mode', 'repeat')
        if self.context_mode == 'none':
            decoder_inputs_with_context = decoder_inputs
        else:
            decoder_features = [decoder_inputs]
            if self.context_mode == 'horizon':
                output_steps = int(self.output_steps)
                decoder_features.append(Lambda(
                    lambda x, steps=output_steps: tf.tile(tf.reshape(
                        tf.linspace(0.0, 1.0, steps), (1, -1, 1)),
                        [tf.shape(x)[0], 1, 1]),
                    output_shape=(output_steps, 1),
                    name='decoder_horizon_channel',
                )(decoder_inputs))
            decoder_features.append(encoder_context_repeated)
            decoder_inputs_with_context = Concatenate(
                axis=-1, name='decoder_inputs_with_context'
            )(decoder_features)

        decoder_outputs = TCN(
            nb_filters=decoder_tcn_hp['filters'],
            kernel_size=decoder_tcn_hp['kernel_size'],
            nb_stacks=decoder_tcn_hp['nb_stacks'],
            dropout_rate=decoder_tcn_hp['dropout_rate'],
            dilations=decoder_tcn_hp['dilations'],
            use_layer_norm=True,
            use_skip_connections=False,
            return_sequences=True,
            name='decoder_tcn',
        )(decoder_inputs_with_context)

        # Project the decoder outputs to the encoder dimension so the
        # attention dot-product is well-defined.
        decoder_query = Dense(
            encoder_dim, activation='linear', name='decoder_query_projection'
        )(decoder_outputs)

        attention_layer = Attention(name='attention_layer')
        attention_outputs = attention_layer([decoder_query, encoder_outputs])

        decoder_combined_context = Concatenate(
            axis=-1, name='decoder_combined_context'
        )([decoder_outputs, attention_outputs])

        decoder_dense = TimeDistributed(
            Dense(1, activation='linear', name='output_dense'), name='output_layer'
        )
        decoder_outputs_final = decoder_dense(decoder_combined_context)
        if self.persistence_gate:
            decoder_outputs_final = apply_persistence_gate(
                decoder_outputs_final, decoder_inputs, self.output_steps,
                features=decoder_combined_context,
                mode=getattr(self, "gate_mode", "static"),
            )

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

        return self.model