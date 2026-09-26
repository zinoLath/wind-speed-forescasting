"""Seq2Seq Bi-GRU, mirroring the original notebook LSTM structure with a bidirectional GRU encoder."""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Bidirectional, Dropout, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.optimizers import Adam, AdamW
from .seq2seq_wrapper import Seq2SeqWrapper
from .layers import apply_persistence_gate
from .s2s_transformer_wrapper import make_learning_rate
from .losses import horizon_weighted_mse, LOSS_NAME


class S2SGRUBidirectionalWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_GRU_Bidirectional"

    def build(self, hp):
        self._require_prepared()

        self.lr_schedule_mode = "constant"
        learning_rate = make_learning_rate(hp)
        weight_decay = hp.Choice('weight_decay', [0.0, 1e-4], default=0.0)

        gru_units = hp.Int('gru_units', min_value=32, max_value=192, step=32, default=64)
        encoder_dropout_rate = hp.Float('encoder_dropout_rate', min_value=0.0, max_value=0.3, step=0.05, default=0.05)
        decoder_dropout_rate = hp.Float('decoder_dropout_rate', min_value=0.0, max_value=0.3, step=0.05, default=0.15)
        loss = horizon_weighted_mse(self.output_steps)
        self.loss = LOSS_NAME

        if weight_decay:
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = Adam(learning_rate=learning_rate)

        encoder_inputs = Input(shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs')

        encoder_gru = Bidirectional(
            GRU(gru_units, return_sequences=True, return_state=True, name='gru_encoder'),
            name='bidirectional_encoder_gru',
        )
        encoder_outputs, forward_h, backward_h = encoder_gru(encoder_inputs)

        state_h = Concatenate(name='state_h')([forward_h, backward_h])

        encoder_outputs = Dropout(encoder_dropout_rate, name='encoder_dropout')(encoder_outputs)

        decoder_inputs = Input(shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs')

        decoder_gru = GRU(
            gru_units * 2, return_sequences=True, return_state=True, name='decoder_gru'
        )
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=[state_h])

        decoder_outputs = Dropout(decoder_dropout_rate, name='decoder_dropout')(decoder_outputs)

        attention_layer = Attention(name='attention_layer')
        attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

        decoder_combined_context = Concatenate(
            axis=-1, name='decoder_combined_context'
        )([decoder_outputs, attention_outputs])

        decoder_dense = TimeDistributed(Dense(1, activation='linear', name='output_dense'), name='output_layer')
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