
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.optimizers import Adam
from .seq2seq_wrapper import Seq2SeqWrapper

class S2SLSTMBidirectionalWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_LSTM_Bidirectional"
        
    def build(self, hp):
        if not hasattr(self, 'train') or not hasattr(self, 'val'):
            raise ValueError("Os dados de treinamento e validação devem ser preparados antes de construir o modelo. Chame o método 'prepare' primeiro.")

        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default = 0.006403023029268099)

        lstm_units = hp.Int('lstm_units', min_value=64, max_value=384, step=32, default=64)
        encoder_dropout_rate = hp.Float('encoder_dropout_rate', min_value=0.0, max_value=0.5, step=0.05, default=0.05)
        decoder_dropout_rate = hp.Float('decoder_dropout_rate', min_value=0.0, max_value=0.5, step=0.05, default=0.15)

        optimizer = Adam(learning_rate=learning_rate,name='Adam')
    
        encoder_inputs = Input(shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs')
        
        encoder_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, return_state=True, name="lstm_encoder"), name='bidirectional_encoder_lstm')
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_inputs)
        
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
        
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        self.model.compile(optimizer=optimizer, loss=getattr(self, 'loss', 'mse'), metrics=['mae'])
    
        return self.model