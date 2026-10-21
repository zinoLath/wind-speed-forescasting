"""Faithful reimplementation of the original notebook LSTM (Bi-LSTM 256 / LSTM 512).

Reproduces ``OLD/notebooks/wind_speed_forecasting-original.ipynb``'s
architecture exactly: bidirectional encoder LSTM with 256 units per direction,
decoder LSTM with 512 units initialized from the concatenated encoder states,
attention, dropout 0.1 on encoder/decoder outputs, Adam lr=1e-3 and MSE loss.
No persistence gate (the original model had none).
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
    Concatenate,
    TimeDistributed,
    Attention,
)
from tensorflow.keras.optimizers import Adam
from .seq2seq_wrapper import Seq2SeqWrapper


class S2SLSTMOriginalWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_LSTM_Original"

    def build(self, hp):
        self._require_prepared()

        # Hyperparameters hardcoded exactly as in the original notebook.
        learning_rate = 1e-3
        lstm_units = 256
        dropout_rate = 0.1
        loss = "mse"
        self.loss = loss
        self.lr_schedule_mode = "constant"

        encoder_inputs = Input(
            shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs'
        )
        encoder_lstm = Bidirectional(
            LSTM(lstm_units, return_sequences=True, return_state=True, name='lstm_encoder'),
            name='bidirectional_encoder_lstm',
        )
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_inputs)

        state_h = Concatenate(name='state_h')([forward_h, backward_h])
        state_c = Concatenate(name='state_c')([forward_c, backward_c])

        encoder_outputs = Dropout(dropout_rate, name='encoder_dropout')(encoder_outputs)

        decoder_inputs = Input(
            shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs'
        )
        decoder_lstm = LSTM(
            lstm_units * 2, return_sequences=True, return_state=True, name='decoder_lstm'
        )
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

        decoder_outputs = Dropout(dropout_rate, name='decoder_dropout')(decoder_outputs)

        attention_layer = Attention(name='attention_layer')
        attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

        decoder_combined_context = Concatenate(
            axis=-1, name='decoder_combined_context'
        )([decoder_outputs, attention_outputs])

        decoder_dense = TimeDistributed(
            Dense(1, activation='linear', name='output_dense'), name='output_layer'
        )
        decoder_outputs_final = decoder_dense(decoder_combined_context)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=['mae']
        )

        return self.model