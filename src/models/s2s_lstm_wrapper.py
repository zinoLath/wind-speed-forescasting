from curses import window

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ..utils import wavelet_denoising
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tcn import TCN
from .seq2seq_wrapper import Seq2SeqWrapper

class S2SLSTMWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_LSTM"
    
    def build(self, hp):
        if not hasattr(self, 'train') or not hasattr(self, 'val'):
            raise ValueError("Os dados de treinamento e validação devem ser preparados antes de construir o modelo. Chame o método 'prepare' primeiro.")

        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default = 0.001249176597990083)
        
        encoder_tcn_hp = {}
        encoder_tcn_hp['filters_power'] = hp.Int('filters_power', min_value=4, max_value=9, default=9)
        encoder_tcn_hp['filters'] = 2 ** encoder_tcn_hp['filters_power']
        encoder_tcn_hp['kernel_size'] = hp.Int('encoder_kernel_size', min_value=2, max_value=5, step=1, default=2)
        encoder_tcn_hp['nb_stacks'] = hp.Int('encoder_nb_stacks', min_value=1, max_value=2, step=1, default=1)
        encoder_tcn_hp['dropout_rate'] = hp.Float('encoder_dropout_rate', min_value=0.1, max_value=0.5, step=0.1, default=0.2)
        encoder_tcn_hp['dilation_rate'] = hp.Int('encoder_dilation_rate', min_value=1, max_value=4, step=1, default=3)
        encoder_tcn_hp['dilations'] = [2 ** i for i in range(encoder_tcn_hp['dilation_rate'])]


        decoder_tcn_hp = {}
        decoder_tcn_hp['filters_power'] = encoder_tcn_hp['filters_power']
        decoder_tcn_hp['filters'] = 2 ** decoder_tcn_hp['filters_power']
        decoder_tcn_hp['kernel_size'] = hp.Int('decoder_kernel_size', min_value=2, max_value=5, step=1, default=4)
        decoder_tcn_hp['nb_stacks'] = hp.Int('decoder_nb_stacks', min_value=1, max_value=2, step=1, default=2)
        decoder_tcn_hp['dropout_rate'] = hp.Float('decoder_dropout_rate', min_value=0.1, max_value=0.5, step=0.1, default=0.1)
        decoder_tcn_hp['dilation_rate'] = hp.Int('decoder_dilation_rate', min_value=1, max_value=4, step=1, default=4)
        decoder_tcn_hp['dilations'] = [2 ** i for i in range(decoder_tcn_hp['dilation_rate'])]


        optimizer = Adam(learning_rate=1e-3)
    
        encoder_inputs = Input(shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs')
        
        encoder_lstm = LSTM(256, return_sequences=True, return_state=True, name='encoder_lstm')
        encoder_outputs, forward_h, forward_c = encoder_lstm(encoder_inputs)
        
        state_h = Concatenate()([forward_h])
        state_c = Concatenate()([forward_c])
        
        
        encoder_outputs = Dropout(0.1)(encoder_outputs)
        
        decoder_inputs = Input(shape=(self.output_steps, self.num_decoder_features), name='decoder_inputs')
        
        decoder_lstm = LSTM(256, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
        
        decoder_outputs = Dropout(0.1)(decoder_outputs)
        
        attention_layer = Attention(name='attention_layer')
        attention_outputs = attention_layer([decoder_outputs, encoder_outputs])
        
        decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, attention_outputs])
        
        decoder_dense = TimeDistributed(Dense(1, activation='linear'), name='output_layer')
        decoder_outputs_final = decoder_dense(decoder_combined_context)
        
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
        return self.model
