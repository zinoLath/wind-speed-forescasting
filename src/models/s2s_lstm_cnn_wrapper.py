from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
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
from .tcn_hp import tcn_hyperparameters


class S2SLSTMCNNWrapper(Seq2SeqWrapper):
    """Seq2Seq with an LSTM encoder and a CNN (TCN) decoder.

    Mirrors ``Seq2Seq_TCN_LSTM`` in reverse: the recurrent encoder summarises
    the known history, and the decoder stacks causal convolutions over the
    horizon inputs instead of an RNN.
    """

    def __init__(self):
        self.name = "Seq2Seq_LSTM_CNN"

    def build(self, hp):
        self._require_prepared()

        schedule_mode = hp.Choice('lr_schedule', ['constant', 'warmup_cosine'], default='constant')
        schedule_steps = getattr(self, 'schedule_total_steps', None) if schedule_mode == 'warmup_cosine' else None
        self.lr_schedule_mode = schedule_mode
        learning_rate = make_learning_rate(
            hp, schedule_total_steps=schedule_steps,
            lr_min=1e-4, lr_max=1e-2, lr_default=0.001,
        )
        weight_decay = hp.Choice('weight_decay', [0.0, 1e-5, 1e-4, 1e-3], default=0.0)
        loss = hp.Choice('loss', ['mse', 'mae', 'huber'], default=getattr(self, 'loss', 'mse'))
        self.loss = loss

        if weight_decay:
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = Adam(learning_rate=learning_rate)

        encoder_layers = hp.Int('encoder_layers', min_value=1, max_value=2, step=1, default=1)
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=512, step=32, default=64)
        encoder_dropout_rate = hp.Float(
            'encoder_dropout_rate', min_value=0.0, max_value=0.6, step=0.05, default=0.1
        )

        decoder_tcn_hp = tcn_hyperparameters(
            hp, "decoder", filters=48, kernel_size=2, nb_stacks=1,
            dropout_rate=0.1, dilation_rate=4,
        )

        encoder_inputs = Input(
            shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs'
        )
        encoder_x = encoder_inputs
        for i in range(encoder_layers - 1):
            encoder_x = LSTM(
                lstm_units, return_sequences=True, name=f'encoder_lstm_{i}'
            )(encoder_x)
        encoder_outputs, _, _ = LSTM(
            lstm_units, return_sequences=True, return_state=True, name='encoder_lstm'
        )(encoder_x)
        encoder_outputs = Dropout(encoder_dropout_rate, name='encoder_dropout')(encoder_outputs)

        encoder_dim = lstm_units

        # Mean-pooled encoder summary repeated across decoder steps.
        encoder_context = Lambda(
            lambda x: K.mean(x, axis=1), name='encoder_context'
        )(encoder_outputs)
        encoder_context_repeated = RepeatVector(
            self.output_steps, name='encoder_context_repeated'
        )(encoder_context)

        decoder_inputs = Input(
            shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs'
        )
        decoder_inputs_with_context = Concatenate(
            axis=-1, name='decoder_inputs_with_context'
        )([decoder_inputs, encoder_context_repeated])

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
        decoder_outputs = Dropout(0.1, name='decoder_dropout')(decoder_outputs)

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
                decoder_outputs_final, decoder_inputs, self.output_steps
            )

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

        return self.model
