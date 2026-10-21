# Roteiro de redação — Teste DM no artigo SBSR (§1.2 do kanban)

> **Roteiro de conteúdo, não texto final.** A redação é do autor (norma
> SBSR: prosa impessoal, sem cópia de IA). Abaixo: o que o parágrafo
> precisa declarar, os números já calculados e as citações certas.

## 1. O que o parágrafo do §Avaliação (no `03`) precisa declarar

- Que a **significância das diferenças de precisão preditiva** entre os
  modelos foi avaliada com o teste de **Diebold–Mariano** (cite:
  Diebold & Mariano, 1995, *JBES* 13(3));
- Que as perdas comparadas são `d_t = L_A,t − L_B,t` **por origem do
  rolling forecast** (perda quadrática), com **H₀: E[d_t] = 0**;
- Que a variância de `d_t` foi estimada de forma **HAC (Newey–West)** —
  necessário porque as perdas do rolling forecast são serialmente
  correlacionadas — com a **correção de amostras pequenas de
  Harvey–Leybourne–Newbold** (cite: HLN, 1997, *IJF* 13(2));
- O **complemento por bootstrap em blocos** (pedido explícito da
  Vanessa): blocos contíguos de `d_t` preservam a dependência temporal;
  implementação com `mean_block = 36` (1 horizonte completo) e
  `n_boot = 2000`;
- O nível de significância adotado (α = 0,05; o MCS usou α = 0,1 —
  declarar o que reportar).

## 2. Números já calculados (rodada canônica `report_8020_itildeq`,
restaurada em 21/10/2026 — mesmas origens p/ DL e baselines, 1.477)

Fonte: `data/results/report_8020_itildeq/mcs.json` (gerado por
`tests/relatorio_resultados.py`; n_boot 2000, bloco 36).

**Model Confidence Set (α = 0,1)** — conjunto de modelos
estatisticamente indistinguíveis do melhor:

- **Dentro do MCS:** TCN, TCN-Bi, Persistência, ARIMA, Random Forest;
- **Excluídos (p de exclusão):** LSTM p = 0,0295 · LSTM-Bi p = 0,0105
  (exclusão significativa ⇒ piores com significância estatística);
- RF p = 0,192 (não excluído).

**MSE rolling (mesma grade de origens):** TCN **1,301** (melhor geral,
à frente do ARIMA 1,329) · TCN-Bi 1,379 · Persistência 1,497 · RF 1,572 ·
LSTM-Bi 1,815 · LSTM 1,844.

**Leitura suportada pelos números** (o autor decide como apresentar):
a superioridade das TCNs sobre as LSTMs é estatisticamente suportada
(exclusão das LSTMs do MCS); a TCN supera o ARIMA no agregado do rolling;
nos horizontes longos a vantagem DL se amplia (Tabela h=36).

## 3. Onde entram no artigo

- **§Avaliação (`03`)**: o parágrafo-método (item 1) — placeholder já
  marcado no `03` ([METODO-A1]);
- **§Resultados (`04`)**: o placeholder `%Teste estatístico` + a
  alegação "TCNs superiores às LSTMs" (par. final) passa a citar os
  p-valores de exclusão do MCS (item 2);
- Tabela de p-valores opcional (pares TCN×LSTM etc.): se quiser DM
  pareado completo por par, regenerar
  `reports/lidar/8020-itildeq-metodos/` (DM ajustado HLN+HAC+Holm).

## 4. Checklist antes de fechar o cartão

- [ ] Parágrafo-método redigido pelo autor no §Avaliação;
- [ ] Números citados conferem com `mcs.json` da rodada final;
- [ ] Cites criados no bib: Diebold & Mariano 1995; HLN 1997;
- [ ] Parágrafo final dos Resultados reescrito com os números do
      retreino (0,08/0,04 mudam com a tabela nova);
- [ ] Remover o comentário [METODO-A1] após aplicar.
