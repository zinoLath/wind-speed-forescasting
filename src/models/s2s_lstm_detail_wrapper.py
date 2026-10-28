"""Head 2 da cascata multi-resolução (LSTM): detalhe condicionado na head 1.

A head 1 (``lstm_mean``/``Seq2SeqLSTMMean``) prevê a média de cada hora
futura; esta head prevê o **detalhe** de 10 min — o resíduo
``ws100_bruto − broadcast(médias_previstas)`` — recebendo no decoder um
canal extra LEVEL-FREE com a predição da head 1 relativa à última
observação. A predição final da cascata = broadcast(médias) + detalhe.

Sem head anexada (``attach_head`` nunca chamado) o wrapper degrada para
uma LSTM seq2seq padrão treinada no próprio ``target_col`` — é o
**controle** do experimento de cascata (mesma arquitetura/HPs/seed, sem
o canal de dica). Lógica compartilhada em ``s2s_cascade_mixins.py``
(DetailMixin); restrições: decoder direto, target absolute, sem
persistence_gate; ``rolling_forecast`` com guard quando a head está
anexada.
"""

from .s2s_cascade_mixins import DetailMixin
from .s2s_lstm_wrapper import S2SLSTMWrapper


class Seq2SeqLSTMDetail(DetailMixin, S2SLSTMWrapper):

    def __init__(self):
        super().__init__()
        self.name = "Seq2Seq_LSTM_Detail"
