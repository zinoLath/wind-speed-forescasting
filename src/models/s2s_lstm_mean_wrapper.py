"""Seq2Seq LSTM de resolução reduzida: prevê a MÉDIA de cada hora futura.

Mesma arquitetura do wrapper ``lstm`` (encoder LSTM + atenção, decoder
direto), mas a saída é agrupada: em vez de 36 passos de 10 min, prevê
``output_steps`` médias, cada uma = média das ``POOL`` amostras de 10 min
do bloco futuro correspondente (``POOL=6`` → 1 hora). A entrada segue em
resolução de 10 min. Lógica de agrupamento em ``s2s_cascade_mixins.py``.
"""

from .s2s_cascade_mixins import PooledMixin
from .s2s_lstm_wrapper import S2SLSTMWrapper


class Seq2SeqLSTMMean(PooledMixin, S2SLSTMWrapper):
    """LSTM seq2seq com saída agregada por blocos de POOL amostras."""

    def __init__(self):
        super().__init__()
        self.name = "Seq2Seq_LSTM_Mean"
