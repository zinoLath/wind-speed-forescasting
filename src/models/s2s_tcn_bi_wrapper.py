import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    Dropout,
    Dense,
    Concatenate,
    TimeDistributed,
    Attention,
    RepeatVector,
    Lambda,
)
from tensorflow.keras.optimizers import Adam
from tcn import TCN
from .seq2seq_wrapper import Seq2SeqWrapper


class S2STCNBidirectionalWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_TCN_Bidirectional"

    def build(self, hp):
        if not hasattr(self, 'train') or not hasattr(self, 'val'):
            raise ValueError(
                "Os dados de treinamento e validação devem ser preparados antes de construir o modelo. "
                "Chame o método 'prepare' primeiro."
            )

        learning_rate = hp.Float(
            'learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=0.003969484893321028
        )

        encoder_tcn_hp = {}
        encoder_tcn_hp['filters'] = hp.Int(
            'encoder_filters', min_value=32, max_value=256, step=16, default=64
        )
        encoder_tcn_hp['kernel_size'] = hp.Int(
            'encoder_kernel_size', min_value=2, max_value=3, step=1, default=2
        )
        encoder_tcn_hp['nb_stacks'] = hp.Int(
            'encoder_nb_stacks', min_value=1, max_value=2, step=1, default=1
        )
        encoder_tcn_hp['dropout_rate'] = hp.Float(
            'encoder_dropout_rate', min_value=0.0, max_value=0.5, step=0.05, default=0.2
        )
        encoder_tcn_hp['dilation_rate'] = hp.Int(
            'encoder_dilation_rate', min_value=1, max_value=5, step=1, default=2
        )
        encoder_tcn_hp['dilations'] = [2 ** i for i in range(encoder_tcn_hp['dilation_rate'])]

        decoder_tcn_hp = {}
        decoder_tcn_hp['filters'] = hp.Int(
            'decoder_filters', min_value=32, max_value=128, step=16, default=80
        )
        decoder_tcn_hp['kernel_size'] = hp.Int(
            'decoder_kernel_size', min_value=2, max_value=3, step=1, default=2
        )
        decoder_tcn_hp['nb_stacks'] = hp.Int(
            'decoder_nb_stacks', min_value=1, max_value=2, step=1, default=2
        )
        decoder_tcn_hp['dropout_rate'] = hp.Float(
            'decoder_dropout_rate', min_value=0.0, max_value=0.5, step=0.05, default=0.0
        )
        decoder_tcn_hp['dilation_rate'] = hp.Int(
            'decoder_dilation_rate', min_value=1, max_value=4, step=1, default=1
        )
        decoder_tcn_hp['dilations'] = [2 ** i for i in range(decoder_tcn_hp['dilation_rate'])]

        optimizer = Adam(learning_rate=learning_rate)

        encoder_inputs = Input(
            shape=(self.input_steps, self.num_encoder_features), name='encoder_inputs'
        )
        encoder_outputs = Bidirectional(
            TCN(
                nb_filters=encoder_tcn_hp['filters'],
                kernel_size=encoder_tcn_hp['kernel_size'],
                nb_stacks=encoder_tcn_hp['nb_stacks'],
                dropout_rate=encoder_tcn_hp['dropout_rate'],
                dilations=encoder_tcn_hp['dilations'],
                use_layer_norm=True,
                use_skip_connections=False,
                return_sequences=True,
                name='encoder_tcn',
            ),
            name='bidirectional_encoder_tcn',
        )(encoder_inputs)
        encoder_outputs = Dropout(0.1, name='encoder_dropout')(encoder_outputs)

        encoder_dim = 2 * encoder_tcn_hp['filters']

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

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return self.model
