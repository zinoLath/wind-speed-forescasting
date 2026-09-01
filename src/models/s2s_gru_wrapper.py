
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dropout, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.optimizers import Adam, AdamW
from .seq2seq_wrapper import Seq2SeqWrapper
from .layers import apply_persistence_gate
from .s2s_transformer_wrapper import make_learning_rate

class S2SGRUWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_GRU"

    def build(self, hp):
        self._require_prepared()

        schedule_mode = hp.Choice('lr_schedule', ['constant', 'warmup_cosine'], default='constant')
        schedule_steps = getattr(self, 'schedule_total_steps', None) if schedule_mode == 'warmup_cosine' else None
        self.lr_schedule_mode = schedule_mode
        learning_rate = make_learning_rate(hp, schedule_total_steps=schedule_steps)
        weight_decay = hp.Choice('weight_decay', [0.0, 1e-5, 1e-4, 1e-3], default=0.0)

        encoder_layers = hp.Int('encoder_layers', min_value=1, max_value=2, step=1, default=1)
        gru_units = hp.Int('gru_units', min_value=32, max_value=512, step=32, default=64)
        encoder_dropout_rate = hp.Float('encoder_dropout_rate', min_value=0.0, max_value=0.6, step=0.05, default=0.05)
        decoder_dropout_rate = hp.Float('decoder_dropout_rate', min_value=0.0, max_value=0.6, step=0.05, default=0.0)
        loss = hp.Choice('loss', ['mse', 'mae', 'huber'], default=getattr(self, 'loss', 'mse'))
        self.loss = loss

        if weight_decay:
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = Adam(learning_rate=learning_rate)

        encoder_inputs = Input(shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs')

        encoder_x = encoder_inputs
        for i in range(encoder_layers - 1):
            encoder_x = GRU(gru_units, return_sequences=True, name=f'encoder_gru_{i}')(encoder_x)
        encoder_gru = GRU(gru_units, return_sequences=True, return_state=True, name='encoder_gru')
        encoder_outputs, state_h = encoder_gru(encoder_x)

        state_h = Concatenate()([state_h])


        encoder_outputs = Dropout(encoder_dropout_rate)(encoder_outputs)

        decoder_inputs = Input(shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs')

        decoder_gru = GRU(gru_units, return_sequences=True, return_state=True, name='decoder_gru')
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=[state_h])

        decoder_outputs = Dropout(decoder_dropout_rate)(decoder_outputs)

        attention_layer = Attention(name='attention_layer')
        attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

        decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, attention_outputs])

        decoder_dense = TimeDistributed(Dense(1, activation='linear'), name='output_layer')
        decoder_outputs_final = decoder_dense(decoder_combined_context)
        if self.persistence_gate:
            decoder_outputs_final = apply_persistence_gate(
                decoder_outputs_final, decoder_inputs, self.output_steps
            )

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

        return self.model
