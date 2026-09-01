from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense,
    Concatenate,
    TimeDistributed,
    Attention,
    Lambda,
)
from tensorflow.keras.optimizers import Adam, AdamW
from tcn import TCN
from .seq2seq_wrapper import Seq2SeqWrapper
from .layers import apply_persistence_gate
from .s2s_transformer_wrapper import make_learning_rate
from .tcn_hp import tcn_hyperparameters


class S2STCNLSTMWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_TCN_LSTM"

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

        encoder_tcn_hp = tcn_hyperparameters(
            hp, "encoder", filters=128, kernel_size=3, nb_stacks=1,
            dropout_rate=0.1, dilation_rate=4,
        )

        lstm_units = hp.Int(
            'lstm_units', min_value=64, max_value=256, step=32, default=128
        )
        decoder_dropout_rate = hp.Float(
            'decoder_dropout_rate', min_value=0.0, max_value=0.5, step=0.05, default=0.1
        )

        optimizer = Adam(learning_rate=learning_rate)

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
        encoder_outputs = Dropout(0.1, name='encoder_dropout')(encoder_outputs)

        encoder_dim = encoder_tcn_hp['filters']

        # Use the last encoder step to initialise the LSTM decoder state.
        encoder_context = Lambda(
            lambda x: x[:, -1, :], name='encoder_context'
        )(encoder_outputs)
        state_h = Dense(
            lstm_units, activation='tanh', name='state_h_projection'
        )(encoder_context)
        state_c = Dense(
            lstm_units, activation='tanh', name='state_c_projection'
        )(encoder_context)

        decoder_inputs = Input(
            shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs'
        )
        decoder_lstm = LSTM(
            lstm_units,
            return_sequences=True,
            return_state=True,
            name='decoder_lstm',
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, initial_state=[state_h, state_c]
        )
        decoder_outputs = Dropout(decoder_dropout_rate, name='decoder_dropout')(decoder_outputs)

        # Project decoder outputs to the encoder dimension before attention.
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
