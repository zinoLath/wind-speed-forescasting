"""Seq2Seq TCN de resolução reduzida: prevê a MÉDIA de cada hora futura.

Mesma arquitetura do wrapper ``tcn`` (encoder/decoder TCN + atenção,
decoder direto), com a saída agrupada: ``output_steps`` médias, cada uma
= média das ``POOL`` amostras de 10 min do bloco futuro correspondente
(``POOL=6`` → 1 hora). Entrada em 10 min. Lógica de agrupamento em
``s2s_cascade_mixins.py``.
"""

from .s2s_cascade_mixins import PooledMixin
from .s2s_tcn_wrapper import S2STCNWrapper


class Seq2SeqTCNMean(PooledMixin, S2STCNWrapper):
    """TCN seq2seq com saída agregada por blocos de POOL amostras."""

    def __init__(self):
        super().__init__()
        self.name = "Seq2Seq_TCN_Mean"
