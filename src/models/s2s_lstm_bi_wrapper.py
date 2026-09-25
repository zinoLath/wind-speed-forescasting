
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.optimizers import Adam, AdamW
from .seq2seq_wrapper import Seq2SeqWrapper
from .layers import apply_persistence_gate
from .s2s_transformer_wrapper import make_learning_rate

class S2SLSTMBidirectionalWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_LSTM_Bidirectional"

    def build(self, hp):
        self._require_prepared()

        schedule_mode = hp.Choice('lr_schedule', ['constant', 'warmup_cosine'], default='constant')
        schedule_steps = getattr(self, 'schedule_total_steps', None) if schedule_mode == 'warmup_cosine' else None
        self.lr_schedule_mode = schedule_mode
        learning_rate = make_learning_rate(hp, schedule_total_steps=schedule_steps)
        weight_decay = hp.Choice('weight_decay', [0.0, 1e-5, 1e-4, 1e-3], default=0.0)

        encoder_layers = hp.Int('encoder_layers', min_value=1, max_value=2, step=1, default=1)
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=512, step=32, default=64)
        encoder_dropout_rate = hp.Float('encoder_dropout_rate', min_value=0.0, max_value=0.6, step=0.05, default=0.05)
        decoder_dropout_rate = hp.Float('decoder_dropout_rate', min_value=0.0, max_value=0.6, step=0.05, default=0.15)
        loss = hp.Choice('loss', ['mse', 'mae', 'huber'], default=getattr(self, 'loss', 'mse'))
        self.loss = loss

        if weight_decay:
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = Adam(learning_rate=learning_rate, name='Adam')

        encoder_inputs = Input(shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs')

        encoder_x = encoder_inputs
        for i in range(encoder_layers - 1):
            encoder_x = Bidirectional(
                LSTM(lstm_units, return_sequences=True, name=f"lstm_encoder_{i}"),
                name=f'bidirectional_encoder_lstm_{i}',
            )(encoder_x)
        encoder_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, return_state=True, name="lstm_encoder"), name='bidirectional_encoder_lstm')
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_x)

        state_h = Concatenate(name="state_h")([forward_h, backward_h])
        state_c = Concatenate(name="state_c")([forward_c, backward_c])


        encoder_outputs = Dropout(encoder_dropout_rate, name="encoder_output_dropout")(encoder_outputs)

        decoder_inputs = Input(shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs')

        decoder_lstm = LSTM(lstm_units * 2, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

        decoder_outputs = Dropout(decoder_dropout_rate, name="decoder_output_dropout")(decoder_outputs)

        attention_layer = Attention(name='attention_layer')
        attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

        decoder_combined_context = Concatenate(axis=-1, name="decoder_combined_context")([decoder_outputs, attention_outputs])

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
