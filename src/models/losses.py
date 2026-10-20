"""Training losses shared by the Seq2Seq wrappers.

The LSTM wrappers train against the horizon-weighted MSE so that the model
prioritises the longer forecast horizons. A partir do round 3 a curva é
piecewise-linear: peso 1,0 no primeiro passo, caindo para 0,8 em h=12
(primeiras 2 h, onde a persistência domina) e subindo para 1,2 no último
passo do horizonte — o otimizador foca o fim da janela sem descartar o
começo. Os wrappers TCN podem treinar com MSE puro (loss por modelo).
"""

import numpy as np
import tensorflow as tf

LOSS_NAME = "horizon_weighted_mse"


def horizon_weights(output_steps):
    """(output_steps,) float32 weights (curva round 3).

    Piecewise-linear passando por (h=1, 1,0), (h=12, 0,8) e
    (h=output_steps, 1,2). A média fica ~0,97 (sem renormalização, para
    preservar os valores pedidos). Para horizontes curtos (< 12 passos) o
    joelho coincide com o fim e a curva fica decrescente.
    """
    steps = np.arange(1, output_steps + 1, dtype=np.float32)
    knee = min(12, output_steps)
    w = np.empty_like(steps)
    first = steps <= knee
    w[first] = 1.0 + (0.8 - 1.0) * (steps[first] - 1.0) / max(knee - 1, 1)
    last = ~first
    if last.any():
        w[last] = 0.8 + (1.2 - 0.8) * (steps[last] - knee) \
            / max(output_steps - knee, 1)
    return w


def horizon_weighted_mse(output_steps):
    """Keras loss: weighted MSE over the output horizon (weight grows with h)."""
    weights = tf.constant(
        horizon_weights(output_steps)[None, :, None], dtype=tf.float32
    )

    def loss(y_true, y_pred):
        return tf.reduce_mean(weights * tf.square(y_pred - y_true))

    return loss


# ---------------------------------------------------------------------------
# ITILDE-Q (loss shape-aware) — Lee et al. (2022), arXiv:2210.15050,
# Eqs. (5)–(8); ITILDE-Q (MSE + termo shape-aware) per Dong Sheng (2026),
# J. Ocean Univ. China, Eq. (15). Porta as três invariantes de distorção:
# amplitude-shift (softmax do desvio assinado), phase-shift (coeficientes de
# Fourier das frequências dominantes) e uniform-amplification
# (cross-correlação normalizada por lag).
ITILDE_Q_NAME = "itilde_q"


def _norm_xcorr(a, b):
    """Cross-correlação normalizada por lag (lag 0..T-1) de duas séries (B,T)."""
    a = a - tf.reduce_mean(a, axis=-1, keepdims=True)
    b = b - tf.reduce_mean(b, axis=-1, keepdims=True)
    t = tf.shape(a)[-1]
    fa = tf.signal.rfft(a, fft_length=[2 * t])
    fb = tf.signal.rfft(b, fft_length=[2 * t])
    cc = tf.signal.irfft(fa * tf.math.conj(fb), fft_length=[2 * t])[:, :t]
    denom = tf.sqrt(tf.reduce_sum(a * a, axis=-1)
                    * tf.reduce_sum(b * b, axis=-1)) + 1e-8
    return cc / denom[:, None]


def itilde_q(output_steps, alpha=0.3, beta=0.5, gamma=0.1):
    """Keras loss ITILDE-Q: L_MSE + α·(β·L_a.shift + (1−β)·L_phase + γ·L_amp).

    α, β, γ são hiperparâmetros (defaults escolhidos aqui; sem valores
    oficiais nos textos disponíveis — podem ser adicionados ao Optuna).

    Observação: a perda NÃO zera para predição perfeita — o termo de fase
    penaliza a energia nas frequências não-dominantes (‖F(ŷ)‖), que é a
    regularização anti-ruído prevista no artigo (Eq. 6), não um bug.
    """
    def loss(y_true, y_pred):
        yt = tf.squeeze(y_true, -1)          # (B, T)
        yp = tf.squeeze(y_pred, -1)
        length = tf.cast(tf.shape(yt)[-1], tf.float32)

        l_mse = tf.reduce_mean(tf.square(yt - yp))

        # amplitude shifting (Eq. 5): T·Σ|1/T − softmax(d)|, d = desvio assinado
        d = yt - yp
        sm = tf.nn.softmax(d, axis=-1)
        l_ashift = tf.reduce_mean(
            length * tf.reduce_sum(tf.abs(1.0 / length - sm), axis=-1))

        # phase shifting (Eq. 6): dominantes = k = ⌈√T⌉ maiores |F(y)|
        fy = tf.signal.rfft(yt)
        fp = tf.signal.rfft(yp)
        k = int(np.ceil(np.sqrt(output_steps)))
        mag = tf.abs(fy)
        top = tf.math.top_k(mag, k=min(k, int(mag.shape[-1]))).indices
        mask = tf.reduce_max(
            tf.one_hot(top, depth=tf.shape(mag)[-1]), axis=1) > 0.5
        diff = tf.abs(fy - fp)
        norm_non = tf.abs(fp)
        l_phase = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
            tf.where(mask, diff, norm_non) ** 2, axis=-1)))

        # uniform amplification (Eq. 7): ||R(y,y) − R(y,ŷ)||_2 por lags
        l_amp = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
            (_norm_xcorr(yt, yt) - _norm_xcorr(yt, yp)) ** 2, axis=-1)))

        return l_mse + alpha * (beta * l_ashift
                                + (1.0 - beta) * l_phase + gamma * l_amp)

    return loss