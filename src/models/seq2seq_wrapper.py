from curses import window

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ..utils import wavelet_denoising
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tcn import TCN

class Seq2SeqWrapper:

    def __init__(self):
        self.name = "Seq2Seq_Base"

    @staticmethod
    def create_sequences(
        data,
        input_steps,
        output_steps,
        target_col_index,
        decoder_mode="teacher_forcing",
        target_mode="absolute",
    ):
        """
        Create input and output sequences for the Seq2Seq model using Teacher forcing.
        """
        data = np.asarray(data)
        X_encoder = []
        X_decoder = []
        y_decoder = []
        
        for i in range(len(data) - input_steps - output_steps + 1):
            X_encoder.append(data[i:(i + input_steps)])
            
            if decoder_mode == "teacher_forcing":
                decoder_input = np.zeros((output_steps, 1))
                decoder_input[0] = data[i + input_steps - 1, target_col_index]
                decoder_input[1:] = data[
                    i + input_steps:i + input_steps + output_steps - 1,
                    target_col_index,
                ].reshape(-1, 1)
            elif decoder_mode == "direct":
                decoder_input = np.empty((output_steps, 2))
                decoder_input[:, 0] = data[i + input_steps - 1, target_col_index]
                decoder_input[:, 1] = np.arange(1, output_steps + 1) / output_steps
            else:
                raise ValueError(f"Unknown decoder mode: {decoder_mode}")
            
            X_decoder.append(decoder_input)
            target = data[
                i + input_steps:i + input_steps + output_steps, target_col_index
            ].reshape(-1, 1)
            if target_mode == "residual":
                target = target - data[i + input_steps - 1, target_col_index]
            elif target_mode != "absolute":
                raise ValueError(f"Unknown target mode: {target_mode}")
            y_decoder.append(target)
        
        return np.array(X_encoder), np.array(X_decoder), np.array(y_decoder)

    def prepare_data(self, data, input_steps, output_steps, target_col, scaler_target=None, scaler_other=None, create_sequences=True, denoise=["ws100"], decoder_mode=None, target_mode=None):
        data = data.copy()
        values = {}# Aplica o denoising em cada coluna e armazena os resultados
        for col in denoise:
            if col in data.columns and f'{col}_wavelet' not in data.columns:
                denoised_signal = wavelet_denoising(data[col].values, level=self.denoise_level)
                data[f'{col}_wavelet'] = denoised_signal
            elif col not in data.columns:
                raise ValueError(f"A coluna '{col}' não existe no conjunto de dados.")
        if scaler_target is None:
            scaler_target = MinMaxScaler()
            scaler_target.fit(data[[target_col]])
        if scaler_other is None:
            scaler_other = MinMaxScaler()
            scaler_other.fit(data.drop(columns=target_col))

        variables_scaled = data.copy()
        ws100_columns = [target_col]
        other_columns = data.columns.drop(target_col).tolist()
        variables_scaled[ws100_columns] = scaler_target.transform(data[ws100_columns])
        variables_scaled[other_columns] = scaler_other.transform(data[other_columns])
        target_col_index = data.columns.get_loc(target_col)

        if create_sequences:
            X_encoder, X_decoder, y_decoder = self.create_sequences(
                variables_scaled,
                input_steps,
                output_steps,
                target_col_index,
                decoder_mode=decoder_mode or getattr(self, "decoder_mode", "teacher_forcing"),
                target_mode=target_mode or getattr(self, "target_mode", "absolute"),
            )
            values['X_encoder'] = X_encoder
            values['X_decoder'] = X_decoder
            values['y_decoder'] = y_decoder
        else:
            X_encoder, X_decoder, y_decoder = None, None, None

        values['scaler_target'] = scaler_target
        values['scaler_other'] = scaler_other
        values['target_col_index'] = target_col_index
        return values, variables_scaled

    def prepare(self, train_data, val_data, input_steps=72, output_steps=36, denoise_level=2, target_col='ws100_wavelet', denoise=["ws100"], decoder_mode="teacher_forcing", target_mode="absolute"):
        
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.target_col = target_col
        self.denoise = denoise
        self.denoise_level = denoise_level
        self.decoder_mode = decoder_mode
        self.target_mode = target_mode
        
        self.train, _ = self.prepare_data(train_data, 
                                       input_steps, 
                                       output_steps, 
                                       target_col,
                                       denoise=denoise)
        self.scaler_target = self.train['scaler_target']
        self.scaler_other = self.train['scaler_other']
        self.target_col_index = self.train['target_col_index']
        self.val, _ = self.prepare_data(val_data, 
                                     input_steps, 
                                     output_steps, 
                                     target_col, 
                                     scaler_target=self.train['scaler_target'], 
                                     scaler_other=self.train['scaler_other'],
                                     denoise=denoise)
        self.num_encoder_features = self.train["X_encoder"].shape[2]
        self.num_decoder_features = self.train["X_decoder"].shape[2]
        
        return self
    def build(self, hp):
        return NotImplementedError("O método 'build' deve ser implementado nas subclasses específicas do modelo.")

    def fit(self, epochs=100, batch_size=32, callbacks=None, verbose=1, use_validation=False):
        fit_kwargs = {
            'epochs': epochs,
            'batch_size': batch_size,
            'callbacks': callbacks,
            'verbose': verbose,
        }

        if use_validation:
            fit_kwargs['validation_data'] = (
                [self.val['X_encoder'], self.val['X_decoder']],
                self.val['y_decoder'],
            )

        history = self.model.fit(
            [self.train['X_encoder'], self.train['X_decoder']],
            self.train['y_decoder'],
            **fit_kwargs,
        )
        return history
    def predict(self, input_data):
        _, input_data = self.prepare_data(input_data,
                                       self.input_steps,
                                       self.output_steps,
                                       self.target_col,
                                       scaler_target=self.scaler_target,
                                       scaler_other=self.scaler_other,
                                       create_sequences=False)
        encoder_input = input_data.to_numpy(copy=True).reshape(1, self.input_steps, input_data.shape[1])
        if self.decoder_mode == "direct":
            decoder_input = np.empty((1, self.output_steps, 2))
            decoder_input[0, :, 0] = encoder_input[0, -1, self.target_col_index]
            decoder_input[0, :, 1] = np.arange(1, self.output_steps + 1) / self.output_steps
        else:
            decoder_input = np.zeros((1, self.output_steps, 1))
            decoder_input[0, :, 0] = encoder_input[0, -1, self.target_col_index]
        
        prediction = self.model.predict([encoder_input, decoder_input], verbose=0)
        if self.target_mode == "residual":
            prediction = prediction + encoder_input[0, -1, self.target_col_index]
        prediction = self.scaler_target.inverse_transform(prediction.reshape(-1, 1)).flatten()
        return prediction
    def rolling_forecast(self, data, test_start=None):
        predictions = []
        actuals = []
        
        if hasattr(data, 'columns'):
            data = data.copy()
            raw_target_col = self.target_col.removesuffix('_wavelet')
            if self.target_col not in data.columns and raw_target_col in data.columns:
                data[self.target_col] = wavelet_denoising(data[raw_target_col].values, level=self.denoise_level)
            for col in self.denoise:
                if col in data.columns and f'{col}_wavelet' not in data.columns:
                    data[f'{col}_wavelet'] = wavelet_denoising(data[col].values, level=self.denoise_level)

            data_columns = list(data.columns)
            target_idx = data_columns.index(self.target_col)
            other_columns = [column for column in data_columns if column != self.target_col]

            data_scaled_df = data.copy()
            data_scaled_df[self.target_col] = self.scaler_target.transform(data[[self.target_col]])
            data_scaled_df[other_columns] = self.scaler_other.transform(data[other_columns])
            data_array = data_scaled_df.to_numpy(copy=True)
        else:
            data_columns = None
            data_array = np.asarray(data).copy()

        data_scaled = data_array.copy()

        if data_columns is not None and self.target_col in data_columns:
            other_indices = [idx for idx, column in enumerate(data_columns) if column != self.target_col]
        else:
            target_idx = self.target_col_index
            other_indices = [idx for idx in range(data_scaled.shape[1]) if idx != target_idx]

        if data_columns is None:
            data_scaled[:, target_idx] = self.scaler_target.transform(data_array[:, target_idx].reshape(-1, 1)).flatten()
            data_scaled[:, other_indices] = self.scaler_other.transform(data_array[:, other_indices])

        if test_start is None:
            test_start = max(self.input_steps, int(len(data_scaled) * 0.95))
        else:
            test_start = max(self.input_steps, test_start)

        test_data = data_scaled
        
        window_start = test_start - self.input_steps
        window_end = test_start
        if window_start < 0:
            raise ValueError(
                "rolling_forecast needs at least input_steps rows before test_start. "
                "Pass the full dataset and set test_start to the first test index."
            )
        if window_end > len(data_scaled):
            raise ValueError(
                "test_start must be within the provided data. Pass the full dataset instead of only the test slice."
            )
        window = data_scaled[window_start:window_end].copy()
        print(f"Starting rolling forecast from index {test_start} to {len(test_data) - self.output_steps + 1}")
        for i in range(test_start, len(test_data) - self.output_steps + 1):
            print(f"Rolling forecast step {i+1}/{len(test_data) - self.output_steps + 1}", end='\r')
            encoder_input = window.reshape(1, self.input_steps, data_scaled.shape[1])
            
            
            if self.decoder_mode == "direct":
                decoder_input = np.empty((1, self.output_steps, 2))
                decoder_input[0, :, 0] = encoder_input[0, -1, self.target_col_index]
                decoder_input[0, :, 1] = np.arange(1, self.output_steps + 1) / self.output_steps
            else:
                decoder_input = np.zeros((1, self.output_steps, 1))
                decoder_input[0, :, 0] = encoder_input[0, -1, self.target_col_index]
            
            pred = self.model.predict([encoder_input, decoder_input], verbose=0)
            if self.target_mode == "residual":
                pred = pred + encoder_input[0, -1, self.target_col_index]
            
            last_pred = pred[0, -1, 0]
            predictions.append(last_pred)
            
            last_actual = test_data[i + self.output_steps - 1, self.target_col_index]
            actuals.append(last_actual)
            
            window = np.vstack([window, test_data[i + self.output_steps - 1]])
            window = window[1:] 
        
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        
        predictions_inverse = self.scaler_target.inverse_transform(predictions)
        actuals_inverse = self.scaler_target.inverse_transform(actuals)
        
        return predictions_inverse, actuals_inverse
