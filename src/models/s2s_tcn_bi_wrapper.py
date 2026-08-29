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
from .tcn_hp import tcn_hyperparameters


class S2STCNBidirectionalWrapper(Seq2SeqWrapper):

    def __init__(self):
        self.name = "Seq2Seq_TCN_Bidirectional"

    def build(self, hp):
        self._require_prepared()

        learning_rate = hp.Float(
            'learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=0.003969484893321028
        )

        encoder_tcn_hp = tcn_hyperparameters(
            hp, "encoder", filters=64, kernel_size=2, nb_stacks=1,
            dropout_rate=0.2, dilation_rate=2,
        )
        decoder_tcn_hp = tcn_hyperparameters(
            hp, "decoder", filters=80, kernel_size=2, nb_stacks=2,
            dropout_rate=0.0, dilation_rate=1,
        )

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
        self.model.compile(optimizer=optimizer, loss=getattr(self, 'loss', 'mse'), metrics=['mae'])

        return self.model
