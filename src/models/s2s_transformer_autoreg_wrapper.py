import numpy as np
from ..utils import wavelet_denoising
from .s2s_transformer_preln_wrapper import S2STransformerPrelnWrapper


class S2STransformerAutoregressiveWrapper(S2STransformerPrelnWrapper):
    """Transformer that decodes autoregressively at inference.

    During training the decoder is teacher-forced. At inference the base
    class broadcasts the last observed target to every decoder step, which
    mismatches the training distribution (exposure bias). This wrapper feeds
    each predicted output back as the next decoder input, matching the
    causal-mask structure the model was trained with.
    """

    def __init__(self):
        self.name = "Seq2Seq_Transformer_Autoreg"

    def _decode(self, encoder_input):
        """Runs the decoder one step at a time, feeding predictions back.

        encoder_input: scaled (1, input_steps, num_encoder_features).
        Returns unscaled-normalized predictions (1, output_steps, 1).
        """
        decoder_input = np.zeros((1, self.output_steps, 1), dtype=np.float32)
        decoder_input[0, 0, 0] = encoder_input[0, -1, self.target_col_index]
        predictions = np.zeros((1, self.output_steps, 1), dtype=np.float32)
        for step in range(self.output_steps):
            output = self.model.predict([encoder_input, decoder_input], verbose=0)
            value = output[0, step, 0]
            predictions[0, step, 0] = value
            if step + 1 < self.output_steps:
                decoder_input[0, step + 1, 0] = value
        return predictions

    def predict(self, input_data):
        _, input_data = self.prepare_data(
            input_data,
            self.input_steps,
            self.output_steps,
            self.target_col,
            scaler_target=self.scaler_target,
            scaler_other=self.scaler_other,
            create_sequences=False,
        )
        encoder_input = input_data.to_numpy(copy=True).reshape(
            1, self.input_steps, input_data.shape[1]
        )
        prediction = self._decode(encoder_input)
        if self.target_mode == "residual":
            prediction = prediction + encoder_input[0, -1, self.target_col_index]
        prediction = self.scaler_target.inverse_transform(
            prediction.reshape(-1, 1)
        ).flatten()
        return prediction

    def rolling_forecast(self, data, test_start=None):
        predictions = []
        actuals = []

        if hasattr(data, 'columns'):
            data = data.copy()
            raw_target_col = self.target_col.removesuffix('_wavelet')
            if self.target_col not in data.columns and raw_target_col in data.columns:
                data[self.target_col] = wavelet_denoising(
                    data[raw_target_col].values, level=self.denoise_level
                )
            for col in self.denoise:
                if col in data.columns and f'{col}_wavelet' not in data.columns:
                    data[f'{col}_wavelet'] = wavelet_denoising(
                        data[col].values, level=self.denoise_level
                    )

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
            data_scaled[:, target_idx] = self.scaler_target.transform(
                data_array[:, target_idx].reshape(-1, 1)
            ).flatten()
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

            pred = self._decode(encoder_input)
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