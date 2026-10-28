---
title: Artigo SBR (SBSR) — estado e pendências
area: artigos
updated: 2026-10-22
---

# Artigo SBR (SBSR) — estado e pendências

`articles/sbr/LaTeX_Template_SBSR_Portugues/` — clone git do Overleaf
(remote `origin` = git bridge). Estrutura: `00_Template_SBSR.tex` +
`01_Resumo` … `06_Conclusoes` + `07_SBSR.bib` + `figuras/` + `sbsr.sty`.

## Escopo do artigo

Comparação **TCN vs LSTM** em seq2seq com atenção + impacto da
bidirecionalidade no encoder, com dados LiDAR do litoral maranhense.
Resumo cita R² **0.502** (all) e **0.416** no horizonte de 6 h, ambos
TCN-Bi.

## Estado das tabelas/figuras (checado em 2026-10-19)

- **Tabelas** (`tab:error_all`, `tab:error_h36` em `04_Resultados.tex`)
  trazem números de uma **rodada anterior**: TCN-Bi 0.502/0.416, TCN
  0.486/0.415, LSTM-Bi 0.460/0.388, LSTM 0.406/0.359, ARIMA 0.508/0.316,
  RF 0.418/0.290, persistência 0.388/0.122; texto fala de 1513 amostras de
  teste e 1406 rolling predictions.
- **Figuras** já vêm da rodada canônica `report_8020` (via
  `articles/sbr/sync.json`: `fig_mse_heatmap.png`, `fig_r2_heatmap.png`,
  `fig_dataset_division.png` ← `data/results/report_8020/images/`).
- **Inconsistência conhecida**: tabela (rodada antiga) ≠ figuras/números
  atuais do report_8020 (TCN 0.514 all / 0.403 rolling, TCN-Bi 0.403,
  rolling 1477 amostras). Corrigir via skill **sbsr-integrate** (atualiza
  as duas tabelas, números do Resumo/texto, descrição da loss e figuras,
  com verificação de consistência); as tabelas novas já podem ser geradas
  prontas com `scripts/sbsr_tables.py` (finding `tabelas-sbsr-geradas`;
  artefato em `reports/lidar/8020/tabelas_sbsr.tex`).

## Regras

- Sync de figuras: `python scripts/articles_sync.py sync sbr` — **só
  comita figuras, nunca .tex do autor** (ver finding `overleaf-sync`).
- Ao trocar a rodada que alimenta o paper: usar a skill sbsr-integrate
  (variância implícita uniforme; figura = tabela da mesma rodada).

## Sessão de revisão 20/10/2026

- **Dado ANEEL atualizado** (thread `6ace4915`): Introdução agora cita o
  **SIGA/ANEEL base diária** (`@misc{aneelSiga}` no bib; snapshot
  20-10-2026; arquivo `data/siga-empreendimentos-geracao-diario.csv`):
  eólica **35 GW em operação (outorgada), 2ª fonte nacional, 15,8%**
  (hídrica 110,3 GW); ranking global 2023 (6º) segue com
  `arcilanVentoSul`. O texto antigo ("22 GW na geração, 3º lugar")
  misturava capacidade com geração.
- **Comentários dos revisores nos `.tex`**: rodada Overleaf inserida
  como blocos `% COMENTÁRIOS DE REVISÃO` em `02/03/04/06` (marcadores
  `[V-…]/[F-…]` + ID do thread; remover antes de submeter).
- **Consolidação da sessão**: relatório favoritado
  `reports/protocolo/sessao-comentarios-sbsr-1/` (kanban 4 colunas + 3
  seções + roteiro de estudo DM/Theil U/Friedman); varredura de testes
  estatísticos da coleção em `docs/revisions/sbr/kbscout/result-5.md`.
- **Push Overleaf 20/10/2026 (`75d2ba0`, branch `main`)** — rebased sobre
  o "Update on Overleaf" (`a6ea1cf`) deles, que JÁ endereçava: verbos
  pessoais (Intro impessoal), números 0,08/0,04, "família TCN mostrou",
  contexto do negócio (down-ramps/Vento Sul), legenda "Fonte:
  \cite{EOSOLAR}" e placeholder `%Teste estatístico` no `04` (onde entra
  o DM). **Chaves do bib**: `arcilanVentoSul` = artigo novo (Sea
  breeze/down-ramps, Energy 294, 2024); `EOSOLAR` = projeto (2022);
  `aneelSiga` = SIGA/ANEEL (nosso). Comment blocks [V-/F-] pushados —
  remover antes de submeter.

## Sessão 21/10/2026 — plano E/I/P e terreno das figuras

- **Kanban com E/I/P**: cartões do relatório da sessão agora têm badges
  **P1** (independe de números, fazer já) / **P2** (após rodada final) /
  **P3** (fechamento) + importância, e títulos **linkados por âncora**
  (`#sec-*`) à seção que descreve a demanda. Plano consolidado na §6 do
  relatório (~8–10 h totais; ≈2,5 h de P1 executável durante o HPO).
- **Dependência de HPO**: LSTM-Bi está em **Optuna** (resgatar ITILDE-Q
  fraca: LSTM 0,318/LSTM-Bi 0,329 vs TCN 0,519) ⇒ Resultados/Discussão,
  tabela de hiperparâmetros, parágrafo da loss e Resumo são
  **provisórios** até a rodada final.
- **Contagem 7632×7561 FECHADA** (`V-6ace7f52`): **não é duplicata**
  (0 timestamps duplicados, 0 NaN); 7561 linhas (16/09 18:00 → 08/11
  14:29); grade do próprio intervalo = **7611 → 50 ausentes (0,7%)**
  (linhas inteiras ausentes); 7632 = 53 dias cheios (intervalo não
  alinhado a dia). **`dataset.csv` é o bruto da campanha** (sem coluna
  `imputed`; imputação wavelet é `src/impute_wind_data.py` →
  `wind_data_imputed.csv`, fora da cadeia do modelo). Falta só redigir
  no `03` (texto pronto na §1.4 do relatório).
- **Figuras — terreno pronto** (`V-6ace71fd/992b`): fontsizes dos
  heatmaps (`plot_horizon_*_heatmap`) 16–22 (efetivo ≥6–9 pt no artigo,
  que entra a `\linewidth` de coluna ≈0,37×); fig. 1 P1 ampliada
  `0.5\linewidth` → `\linewidth` no `03`. Regen só na rodada final +
  `articles_sync.py sync sbr` (comentário no `04` atualizado).
- **`sbsr_tables.py` validado**: o ValueError do vigia
  (`watch_retreino_itildeq.log`) já estava corrigido (`.resolve()` no
  cabeçalho); roda rc=0 sobre `report_8020_itildeq`.
- **GRU não será abordada** (decisão do autor) — da antiga §5.5 do doc
  de revisão resta só `\cite{vaswani2017attention}` na menção a
  Transformers nas Conclusões (chave já no bib).

## Auditoria de reescritas (21/10) — histórico git do Overleaf

- **Toda a reescrita editorial do artigo é de UMA pessoa**: Vanessa
  Souza, commit `a6ea1cf` (13/10 21:08, "Update on Overleaf") — 115+/107−
  em 7 arquivos: autores/afiliações (Arcilan entra como 3º autor),
  Introdução reescrita, nova subseção "Entendimento do negócio"
  (down-ramps/Vento Sul, 8 eventos), "Entendimento dos dados" reescrita
  (WindCube V2, legenda Fonte, "não apresenta valores ausentes" — frase
  depois desmentida pela análise do §1.4), Modelagem expandida
  (seq2seq/encoder-decoder, LSTM/TCN com cites, baselines), Resultados
  reestruturados (ARIMA vence agregado; placeholder `%Teste
  estatístico`), typos das Conclusões, bib (EOSOLAR, bai2018,
  arcilanVentoSul atualizado). Sessões locais de 20–21/10 só rebasaram
  por cima (ANEEL + typos + comentários) — **nenhuma prosa local foi
  reescrita** (diff provado).
- **Parágrafo down-ramps/Vento Sul do `03`**: autoria Vanessa
  (`a6ea1cf`), resposta ao thread `6ace64b6`.
- **Bloco `\begin{comment}` no `03`** (linhas ~95–134): antiga
  Modelagem (autoria João/Zino, 10–12/10) envelopada pela Vanessa;
  contém a única `\cite{vaswani2017attention}` do corpo — remover na
  limpeza pré-submissão (cartão c-31) e cuidar do cite (cartão c-29).
- **Contagem 7632×7561 definitivamente fechada** (`V-6ace7f52`): não é
  duplicata (0 dup, 0 NaN); grade arredondada 10 min = **7612 slots →
  51 ausentes (0,67%) em 47 lacunas** (43 de 1 passo, 4 de 2 passos;
  máx 20 min; 40 no treino, 11 no teste — densidades ~iguais). Imputados
  por interpolação temporal e **excluídos das métricas** via
  `observed_timestamps`. Frase pronta no relatório da sessão §1.4.

## Resultados featurex x baselines (contexto p/ Discussão)

- TCNs empatam com o ARIMA no agregado (0,49–0,52 vs 0,508; DM sem par
  significativo) e vencem em h24+ (heatmap); **MCS α=0,10** manteve só a
  TCN na featurex e todos os 7 no híbrido. Baseline ARIMA forte em série
  diurna regular é **padrão na literatura da coleção** (Liu et al.
  SARIMA > GRU/LSTM em vento offshore horário; citado em
  `shengxiang2022deep`/`nascimentotransformer`).

## Fontes

- `articles/sbr/LaTeX_Template_SBSR_Portugues/{01_Resumo,04_Resultados}.tex`
- `articles/sbr/sync.json`; skill **sbsr-integrate**;
  finding `resultados-canonicos`.
