
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ..utils import wavelet_denoising


class Seq2SeqWrapper:

    # Total optimizer steps used by warmup+cosine LR schedules; stages set it
    # before build() when the chosen lr_schedule is "warmup_cosine".
    schedule_total_steps = None

    def __init__(self):
        self.name = "Seq2Seq_Base"
        # Training loss defaults to MSE so large forecast errors are punished
        # harder than with MAE. build() overrides it from the searched
        # ``loss`` hyperparameter; the pipeline ``loss`` config key feeds it.
        self.loss = "mse"
        # When True the decoder inputs gain an extra channel carrying the last
        # observed target, and the output passes through a learned per-horizon
        # persistence gate (see src/models/layers.py).
        self.persistence_gate = False

    @staticmethod
    def create_sequences(
        data,
        input_steps,
        output_steps,
        target_col_index,
        decoder_mode="direct",
        target_mode="absolute",
        persistence_gate=False,
    ):
        """
        Create encoder/decoder/target sequences for the Seq2Seq model.

        Vectorised with sliding windows and returned as float32 arrays:
        - X_encoder: (n, input_steps, n_features)
        - X_decoder: (n, output_steps, 1-3)
        - y_decoder: (n, output_steps, 1)

        With ``persistence_gate`` the decoder inputs gain a trailing channel
        holding the last observed target at every step (the blend reference
        used by the output gate).
        """
        data = np.asarray(data, dtype=np.float32)
        n_sequences = len(data) - input_steps - output_steps + 1
        if n_sequences <= 0:
            raise ValueError(
                f"Not enough rows ({len(data)}) for input_steps={input_steps} "
                f"and output_steps={output_steps}."
            )

        # (n, input_steps, n_features) sliding windows over every position.
        encoder_windows = sliding_window_view(data, (input_steps, data.shape[1]))
        X_encoder = np.ascontiguousarray(encoder_windows[:n_sequences, 0])

        # (n, input_steps + output_steps) sliding windows over the target.
        target_windows = sliding_window_view(
            data[:, target_col_index], input_steps + output_steps
        )[:n_sequences]
        last_observed = target_windows[:, input_steps - 1]

        if decoder_mode == "teacher_forcing":
            width = 2 if persistence_gate else 1
            X_decoder = np.zeros((n_sequences, output_steps, width), dtype=np.float32)
            X_decoder[:, 0, 0] = last_observed
            X_decoder[:, 1:, 0] = target_windows[:, input_steps:-1]
            if persistence_gate:
                X_decoder[:, :, 1] = last_observed[:, None]
        elif decoder_mode == "direct":
            width = 3 if persistence_gate else 2
            X_decoder = np.empty((n_sequences, output_steps, width), dtype=np.float32)
            X_decoder[:, :, 0] = last_observed[:, None]
            X_decoder[:, :, 1] = np.arange(1, output_steps + 1) / output_steps
            if persistence_gate:
                X_decoder[:, :, 2] = last_observed[:, None]
        else:
            raise ValueError(f"Unknown decoder mode: {decoder_mode}")

        y = target_windows[:, input_steps:].astype(np.float32)
        if target_mode == "residual":
            y -= last_observed[:, None]
        elif target_mode != "absolute":
            raise ValueError(f"Unknown target mode: {target_mode}")

        return X_encoder, X_decoder, y.reshape(n_sequences, output_steps, 1)

    def prepare_data(self, data, input_steps, output_steps, target_col, scaler_target=None, scaler_other=None, create_sequences=True, denoise=("ws100",), decoder_mode=None, target_mode=None, persistence_gate=None, features=None):
        if persistence_gate is None:
            persistence_gate = getattr(self, "persistence_gate", False)
        if decoder_mode is None:
            decoder_mode = getattr(self, "decoder_mode", "direct")
        data = data.copy()
        values = {}# Aplica o denoising em cada coluna e armazena os resultados
        for col in denoise:
            if col in data.columns and f'{col}_wavelet' not in data.columns:
                denoised_signal = wavelet_denoising(data[col].values, level=self.denoise_level)
                data[f'{col}_wavelet'] = denoised_signal
            elif col not in data.columns:
                raise ValueError(f"A coluna '{col}' não existe no conjunto de dados.")
        if features:
            keep = [c for c in data.columns if c in set(features) | {target_col}]
            missing = (set(features) | {target_col}) - set(keep)
            if missing:
                raise ValueError(f"Features ausentes no dataset: {sorted(missing)}")
            data = data[keep]
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
                decoder_mode=decoder_mode,
                target_mode=target_mode or getattr(self, "target_mode", "absolute"),
                persistence_gate=persistence_gate,
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

    def prepare(self, train_data, val_data, input_steps=72, output_steps=36, denoise_level=2, target_col='ws100_wavelet', denoise=("ws100",), decoder_mode="direct", target_mode="absolute", create_sequences=True, validate_with_inference_decoder=True, persistence_gate=False, features=None):

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.target_col = target_col
        self.denoise = tuple(denoise)
        self.denoise_level = denoise_level
        self.decoder_mode = decoder_mode
        self.target_mode = target_mode
        self.persistence_gate = persistence_gate
        self.features = tuple(features) if features else None

        self.train, processed_train = self.prepare_data(train_data,
                                       input_steps,
                                       output_steps,
                                       target_col,
                                       denoise=denoise,
                                       create_sequences=create_sequences,
                                       features=features)
        self.scaler_target = self.train['scaler_target']
        self.scaler_other = self.train['scaler_other']
        self.target_col_index = self.train['target_col_index']
        self.val, _ = self.prepare_data(val_data,
                                     input_steps,
                                     output_steps,
                                     target_col,
                                     scaler_target=self.train['scaler_target'],
                                     scaler_other=self.train['scaler_other'],
                                     denoise=denoise,
                                     create_sequences=create_sequences,
                                     features=features)
        if create_sequences:
            self.num_encoder_features = self.train["X_encoder"].shape[2]
            self.num_decoder_features = self.train["X_decoder"].shape[2]
        else:
            # Scaler-only preparation (used before loading saved weights).
            self.num_decoder_features = (2 if decoder_mode == "direct" else 1) + (
                1 if persistence_gate else 0
            )

        if create_sequences and decoder_mode == "teacher_forcing" and validate_with_inference_decoder:
            # At deployment the decoder never sees ground-truth future steps;
            # it repeats the last observed value. Validating (and early
            # stopping) under the same convention keeps val_loss aligned with
            # the inference behaviour instead of the teacher-forced training
            # behaviour.
            self.val["X_decoder"][:, :, 0] = self.val["X_decoder"][:, :1, 0]

        return self
    def _require_prepared(self):
        """Raise if prepare() has not run yet (subclasses call this in build())."""
        if not hasattr(self, "train") or not hasattr(self, "val"):
            raise ValueError(
                "Os dados de treinamento e validação devem ser preparados antes de "
                "construir o modelo. Chame o método 'prepare' primeiro."
            )

    def build(self, hp):
        raise NotImplementedError("O método 'build' deve ser implementado nas subclasses específicas do modelo.")

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
    def _to_scaled_array(self, data):
        """Convert a DataFrame or ndarray to the scaled float32 inference array.

        Adds the wavelet columns the wrapper was prepared with, then applies
        the train-fitted scalers. Returns (scaled_array, target_col_index).
        """
        if not hasattr(data, "columns"):
            array = np.asarray(data, dtype=np.float32)
            target_idx = self.target_col_index
        else:
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
            if getattr(self, "features", None):
                keep = [c for c in data.columns
                        if c in set(self.features) | {self.target_col}]
                data = data[keep]
            target_idx = data.columns.get_loc(self.target_col)
            array = data.to_numpy(dtype=np.float32)

        other_indices = [j for j in range(array.shape[1]) if j != target_idx]
        array[:, [target_idx]] = self.scaler_target.transform(array[:, [target_idx]])
        array[:, other_indices] = self.scaler_other.transform(array[:, other_indices])
        return array, target_idx

    @staticmethod
    def _build_decoder_input(encoder_window, output_steps, decoder_mode, target_col_index, persistence_gate=False):
        """Inference-time decoder input: the last observed target, as seen at
        deployment (the model never receives ground-truth future steps). With
        ``persistence_gate`` a trailing channel repeats the last observed
        value at every decoder step (the output gate reference)."""
        last_observed = encoder_window[..., -1, target_col_index]
        if decoder_mode == "direct":
            width = 3 if persistence_gate else 2
            decoder_input = np.empty((*encoder_window.shape[:-2], output_steps, width), dtype=np.float32)
            decoder_input[..., :, 0] = last_observed[..., None]
            decoder_input[..., :, 1] = np.arange(1, output_steps + 1) / output_steps
            if persistence_gate:
                decoder_input[..., :, 2] = last_observed[..., None]
        else:
            width = 2 if persistence_gate else 1
            decoder_input = np.zeros((*encoder_window.shape[:-2], output_steps, width), dtype=np.float32)
            decoder_input[..., 0, 0] = last_observed
            if persistence_gate:
                decoder_input[..., :, 1] = last_observed[..., None]
        return decoder_input, last_observed

    def predict(self, input_data):
        """Forecast one output_steps horizon from exactly input_steps observed rows."""
        data_scaled, target_idx = self._to_scaled_array(input_data)
        if len(data_scaled) != self.input_steps:
            raise ValueError(
                f"predict expects exactly {self.input_steps} rows, got {len(data_scaled)}."
            )
        encoder_input = data_scaled.reshape(1, self.input_steps, data_scaled.shape[1])
        decoder_input, last_observed = self._build_decoder_input(
            encoder_input, self.output_steps, self.decoder_mode, target_idx,
            persistence_gate=self.persistence_gate,
        )

        prediction = self.model.predict([encoder_input, decoder_input], verbose=0)
        if self.target_mode == "residual":
            prediction = prediction + last_observed
        return self.scaler_target.inverse_transform(prediction.reshape(-1, 1)).flatten()

    def rolling_forecast(self, data, test_start=None, batch_size=256):
        """Forecast the target at every origin from ``test_start`` onward.

        For each origin i the encoder receives the scaled window
        ``[i - input_steps, i)`` and the forecast is compared against the
        target observed at ``i + output_steps - 1``. Every window is a plain
        function of known history, so the whole forecast runs as one batched
        model call. Returns (predictions, actuals), both inverse-transformed.
        """
        data_scaled, target_idx = self._to_scaled_array(data)
        n = len(data_scaled)
        if test_start is None:
            test_start = max(self.input_steps, int(n * 0.95))
        test_start = max(self.input_steps, test_start)
        count = n - self.output_steps + 1 - test_start
        if count <= 0:
            return np.empty((0, 1)), np.empty((0, 1))

        # Encoder window for origin i covers rows [i - input_steps, i).
        windows = sliding_window_view(
            data_scaled[test_start - self.input_steps:], self.input_steps, axis=0
        )[:count]
        encoder_input = np.ascontiguousarray(np.moveaxis(windows, 2, 1))
        decoder_input, last_observed = self._build_decoder_input(
            encoder_input, self.output_steps, self.decoder_mode, target_idx,
            persistence_gate=self.persistence_gate,
        )

        predictions = self.model.predict(
            [encoder_input, decoder_input], batch_size=batch_size, verbose=0
        )[:, -1, 0]
        if self.target_mode == "residual":
            predictions = predictions + last_observed

        actuals = data_scaled[
            test_start + self.output_steps - 1 : n, target_idx
        ]
        predictions = self.scaler_target.inverse_transform(predictions.reshape(-1, 1))
        actuals = self.scaler_target.inverse_transform(actuals.reshape(-1, 1).astype(np.float64))
        return predictions, actuals
