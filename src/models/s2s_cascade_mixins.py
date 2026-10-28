"""Mixins compartilhados das cascatas multi-resolução (mean/detail).

``PooledMixin`` — saída agrupada: prevê a MÉDIA de cada bloco de ``POOL``
amostras futuras (``POOL=6`` → média horária). Semântica do alvo na
origem i (janela do encoder = linhas [i, i+input_steps)):
    y[k] = mean(target[i + input_steps + k*POOL : i + input_steps + (k+1)*POOL])
para k = 0..output_steps-1 (blocos alinhados à origem, não ao relógio).
As covariadas futuras do decoder entram como a média do bloco (para
sin/cos, média ≈ centro circular do bloco).

``DetailMixin`` — head 2 da cascata: prevê o DETALHE de 10 min
(resíduo ``ws100_bruto − broadcast(médias previstas pela head 1)``)
com a predição da head 1 como canal extra do decoder, LEVEL-FREE
(``(hint − última observação)``) para bloquear o atalho espúrio de
nível, e ``hint_noise`` (σ, em unidades) para ruído estilo DART só no
treino. A predição final da cascata = broadcast(médias) + detalhe — em
unidades do ws100, com a correção do ``data_min`` do MinMax na
composição (ver finding `lstm-mean-multiresolucao`).

Sem head anexada, ``DetailMixin`` degrada para o wrapper base treinado
no próprio ``target_col`` — é o controle do experimento. Os mixins não
herdam de wrapper nenhum: a subclasse traz a arquitetura
(LSTM/TCN/...); composição:
    class Seq2SeqTCNDetail(DetailMixin, S2STCNWrapper)
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

from .seq2seq_wrapper import Seq2SeqWrapper


class PooledMixin:

    POOL = 6

    def create_sequences(
        self,
        data,
        input_steps,
        output_steps,
        target_col_index,
        decoder_mode="direct",
        target_mode="absolute",
        persistence_gate=False,
        future_indices=None,
    ):
        """Como ``Seq2SeqWrapper.create_sequences``, com o alvo (e as
        covariadas futuras) agrupados em blocos de ``POOL`` passos brutos;
        ``output_steps`` aqui é o número de blocos (horizonte bruto =
        ``output_steps * POOL``)."""
        pool = self.POOL
        horizon = output_steps * pool
        data = np.asarray(data, dtype=np.float32)
        n_sequences = len(data) - input_steps - horizon + 1
        if n_sequences <= 0:
            raise ValueError(
                f"Not enough rows ({len(data)}) for input_steps={input_steps} "
                f"and horizon={horizon} ({output_steps} blocos de {pool})."
            )

        encoder_windows = sliding_window_view(data, (input_steps, data.shape[1]))
        X_encoder = np.ascontiguousarray(encoder_windows[:n_sequences, 0])

        target_windows = sliding_window_view(
            data[:, target_col_index], input_steps + horizon
        )[:n_sequences]
        last_observed = target_windows[:, input_steps - 1]

        future_covariates = None
        if future_indices:
            cov = np.stack(
                [
                    sliding_window_view(
                        data[:, idx], input_steps + horizon
                    )[:n_sequences, input_steps:]
                    for idx in future_indices
                ],
                axis=-1,
            ).astype(np.float32)  # (n, horizon, k)
            future_covariates = cov.reshape(
                n_sequences, output_steps, pool, cov.shape[-1]
            ).mean(axis=2)

        y = (
            target_windows[:, input_steps:]
            .reshape(n_sequences, output_steps, pool)
            .mean(axis=2)
            .astype(np.float32)
        )

        extra = future_covariates.shape[-1] if future_covariates is not None else 0
        if decoder_mode == "teacher_forcing":
            width = 1 + extra + (1 if persistence_gate else 0)
            X_decoder = np.zeros((n_sequences, output_steps, width), dtype=np.float32)
            X_decoder[:, 0, 0] = last_observed
            X_decoder[:, 1:, 0] = y[:, :-1]
            if future_covariates is not None:
                X_decoder[:, :, 1:1 + extra] = future_covariates
            if persistence_gate:
                X_decoder[:, :, -1] = last_observed[:, None]
        elif decoder_mode == "direct":
            width = 2 + extra + (1 if persistence_gate else 0)
            X_decoder = np.empty((n_sequences, output_steps, width), dtype=np.float32)
            X_decoder[:, :, 0] = last_observed[:, None]
            X_decoder[:, :, 1] = np.arange(1, output_steps + 1) / output_steps
            if future_covariates is not None:
                X_decoder[:, :, 2:2 + extra] = future_covariates
            if persistence_gate:
                X_decoder[:, :, -1] = last_observed[:, None]
        else:
            raise ValueError(f"Unknown decoder mode: {decoder_mode}")

        if target_mode == "residual":
            y = y - last_observed[:, None]
        elif target_mode != "absolute":
            raise ValueError(f"Unknown target mode: {target_mode}")

        return X_encoder, X_decoder, y.reshape(n_sequences, output_steps, 1)

    def _future_covariates_from_index(self, index, output_steps):
        cov = super()._future_covariates_from_index(
            index, output_steps * self.POOL
        )
        return cov.reshape(output_steps, self.POOL, cov.shape[-1]).mean(axis=1)


class DetailMixin:

    POOL = 6

    def __init__(self):
        self.name = "Seq2Seq_Detail"
        self.head = None
        self.hint_noise = 0.0

    def attach_head(self, head, hint_override=None):
        """Anexa a head 1 congelada (wrapper já preparado, com ``model``
        carregado). ``hint_override`` = (n_origins, output_steps) em
        unidades — hints PRÉ-COMPUTADOS para a primeira chamada de
        ``prepare_data``, consumidos uma única vez."""
        self.head = head
        self._hint_override = hint_override

    def _head_hint_units(self, data):
        """Predições da head 1 (unidades do ws100) para cada origem de
        *data*. Com ``hint_noise`` > 0, ruído gaussiano com esse σ é
        somado aos hints (DART-style; DESLIGAR na avaliação)."""
        if getattr(self, "_hint_override", None) is not None:
            hints = np.asarray(self._hint_override, dtype=np.float32)
            self._hint_override = None
            return hints
        head = self.head
        prepared, _ = head.prepare_data(
            data, self.input_steps, head.output_steps, head.target_col,
            scaler_target=head.scaler_target,
            scaler_other=head.scaler_other,
            denoise=head.denoise, decoder_mode="direct",
            target_mode="absolute",
        )
        means_scaled = head.model.predict(
            [prepared["X_encoder"], prepared["X_decoder"]], verbose=0
        )[:, :, 0]
        hint_units = head.scaler_target.inverse_transform(
            means_scaled.reshape(-1, 1)
        ).reshape(means_scaled.shape)
        if self.hint_noise > 0:
            rng = np.random.default_rng(42)
            hint_units = hint_units + rng.normal(
                0.0, self.hint_noise, hint_units.shape
            ).astype(np.float32)
        return hint_units

    def prepare_data(self, data, input_steps, output_steps, target_col,
                     scaler_target=None, scaler_other=None, **kwargs):
        if self.head is None:
            return super().prepare_data(
                data, input_steps, output_steps, target_col,
                scaler_target=scaler_target, scaler_other=scaler_other,
                **kwargs
            )
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("cascata requer DataFrame com DatetimeIndex")
        if scaler_target is None:
            scaler_target = MinMaxScaler().fit(data[[target_col]].to_numpy())
            self._time_reference = (
                data.index[0], data.index[-1] + pd.Timedelta(days=15)
            )
        hint_units = self._head_hint_units(data)
        self._hint_scaled = scaler_target.transform(
            hint_units.reshape(-1, 1)
        ).reshape(hint_units.shape).astype(np.float32)
        return super().prepare_data(
            data, input_steps, output_steps, target_col,
            scaler_target=scaler_target, scaler_other=scaler_other,
            **kwargs
        )

    def create_sequences(
        self,
        data,
        input_steps,
        output_steps,
        target_col_index,
        decoder_mode="direct",
        target_mode="absolute",
        persistence_gate=False,
        future_indices=None,
    ):
        if self.head is None:
            return Seq2SeqWrapper.create_sequences(
                data, input_steps, output_steps, target_col_index,
                decoder_mode=decoder_mode, target_mode=target_mode,
                persistence_gate=persistence_gate,
                future_indices=future_indices,
            )
        if decoder_mode != "direct":
            raise NotImplementedError("cascata suporta apenas decoder direto")
        if target_mode != "absolute":
            raise NotImplementedError("cascata suporta apenas target absolute")
        if persistence_gate:
            raise NotImplementedError("persistence_gate não suportado na cascata")

        data = np.asarray(data, dtype=np.float32)
        n_sequences = len(data) - input_steps - output_steps + 1
        if n_sequences <= 0:
            raise ValueError(
                f"Not enough rows ({len(data)}) for input_steps={input_steps} "
                f"and output_steps={output_steps}."
            )
        hint = np.repeat(self._hint_scaled[:n_sequences], self.POOL, axis=1)

        encoder_windows = sliding_window_view(
            data, (input_steps, data.shape[1])
        )[:n_sequences, 0]
        X_encoder = np.ascontiguousarray(encoder_windows)

        target_windows = sliding_window_view(
            data[:, target_col_index], input_steps + output_steps
        )[:n_sequences]
        last_observed = target_windows[:, input_steps - 1]

        covariates = None
        if future_indices:
            covariates = np.stack(
                [
                    sliding_window_view(
                        data[:, idx], input_steps + output_steps
                    )[:n_sequences, input_steps:]
                    for idx in future_indices
                ],
                axis=-1,
            ).astype(np.float32)

        extra = covariates.shape[-1] if covariates is not None else 0
        width = 3 + extra
        X_decoder = np.empty((n_sequences, output_steps, width), dtype=np.float32)
        X_decoder[:, :, 0] = last_observed[:, None]
        X_decoder[:, :, 1] = np.arange(1, output_steps + 1) / output_steps
        X_decoder[:, :, 2] = hint - last_observed[:, None]
        if covariates is not None:
            X_decoder[:, :, 3:3 + extra] = covariates

        y = (target_windows[:, input_steps:] - hint).astype(np.float32)
        return X_encoder, X_decoder, y.reshape(n_sequences, output_steps, 1)

    def predict(self, input_data):
        """Horizonte output_steps a partir de exatamente input_steps linhas.
        Com head anexada, a saída é o DETALHE (resíduo em unidades) — a
        composição broadcast(médias) + detalhe fica a cargo do chamador."""
        data_scaled, target_idx = self._to_scaled_array(input_data)
        if len(data_scaled) != self.input_steps:
            raise ValueError(
                f"predict expects exactly {self.input_steps} rows, "
                f"got {len(data_scaled)}."
            )
        encoder_input = data_scaled.reshape(
            1, self.input_steps, data_scaled.shape[1]
        )
        future_covariates = None
        if getattr(self, "future_indices", None):
            future_covariates = self._future_covariates_from_index(
                input_data.index, self.output_steps
            )[None]
        if self.head is None:
            decoder_input, last_observed = self._build_decoder_input(
                encoder_input, self.output_steps, self.decoder_mode,
                target_idx, persistence_gate=self.persistence_gate,
                future_covariates=future_covariates,
            )
        else:
            if self.persistence_gate:
                raise NotImplementedError(
                    "persistence_gate não suportado na cascata"
                )
            hint_units = self.head.predict(input_data).reshape(
                1, self.head.output_steps
            )
            hint_scaled = self.scaler_target.transform(
                hint_units.reshape(-1, 1)
            ).reshape(hint_units.shape)
            extra = (
                future_covariates.shape[-1]
                if future_covariates is not None else 0
            )
            decoder_input = np.empty(
                (1, self.output_steps, 3 + extra), dtype=np.float32
            )
            decoder_input[:, :, 0] = encoder_input[:, -1, target_idx][:, None]
            decoder_input[:, :, 1] = (
                np.arange(1, self.output_steps + 1) / self.output_steps
            )
            decoder_input[:, :, 2] = (
                np.repeat(hint_scaled, self.POOL, axis=1)
                - encoder_input[:, -1, target_idx][:, None]
            )
            if future_covariates is not None:
                decoder_input[:, :, 3:3 + extra] = future_covariates
            last_observed = None

        prediction = self.model.predict(
            [encoder_input, decoder_input], verbose=0
        )
        if self.target_mode == "residual":
            prediction = prediction + last_observed
        return self.scaler_target.inverse_transform(
            prediction.reshape(-1, 1)
        ).flatten()

    def rolling_forecast(self, data, test_start=None, batch_size=256):
        if self.head is not None:
            raise NotImplementedError(
                "rolling_forecast não definido para a cascata "
                "(usar predict_all_horizons)"
            )
        return super().rolling_forecast(
            data, test_start=test_start, batch_size=batch_size
        )
