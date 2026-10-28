"""Head 2 da cascata multi-resolução (TCN): detalhe condicionado na head 1.

A head 1 (``tcn_mean``/``Seq2SeqTCNMean``) prevê a média de cada hora
futura; esta head prevê o **detalhe** de 10 min — o resíduo
``ws100_bruto − broadcast(médias_previstas)`` — recebendo no decoder um
canal extra LEVEL-FREE com a predição da head 1 relativa à última
observação. A predição final da cascata = broadcast(médias) + detalhe.

Sem head anexada (``attach_head`` nunca chamado) o wrapper degrada para
um TCN seq2seq padrão treinado no próprio ``target_col`` — é o
**controle** do experimento de cascata. Lógica compartilhada em
``s2s_cascade_mixins.py`` (DetailMixin).
"""

from .s2s_cascade_mixins import DetailMixin
from .s2s_tcn_wrapper import S2STCNWrapper


class Seq2SeqTCNDetail(DetailMixin, S2STCNWrapper):

    def __init__(self):
        super().__init__()
        self.name = "Seq2Seq_TCN_Detail"
