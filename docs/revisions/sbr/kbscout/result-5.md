Varredura concluída (61 arquivos markdown, 16 padrões de busca + leituras de contexto). Sessão de 20/10/2026 — foco: Theil's U e padrões de testes estatísticos (consolidação cruzada com os lotes 1–4).

# LOTE 5 — Achados metodológicos

## 1. Theil's U como métrica em previsão de vento ⭐
- **Artigo (a):** DL + hybrid time series decomposition + multi-objective optimization (Applied Energy 2022) — bib: `lin2022deep` (arquivo: `Deep learning combined wind speed forecasting...md`)
  - Trecho: *"Other related performance measures including the median of absolute percentage error (MdAPE) [4] and Theil U statistics [57] are also considered in many wind speed forecasting researches"* + formaliza o DM por erros quadráticos (Eq. 33).
  - Gancho de citação mais direto para justificar Theil U + DM no SBSR.
- **Artigo (b):** MSFSC — model selection forecasting system (Energy 2024) — bib: `chen2024research` (arquivo: `Research and application of a Model selection...md`)
  - Trecho: *"seven performance evaluation criteria are introduced: MAE, RMSE, STDAPE, directional accuracy (DA), Theil coefficient (TIC), MAPE, goodness of fit (R2), and classification accuracy (ACC)"*
  - Usa o TIC (variante do U de Theil) ao lado de DA (acurácia direcional — métrica adicional barata para o SBSR).

## 2. DM — padrão dominante (7 artigos de vento da base)
- **Ajuste de Harvey + Newey–West:** `gomez2026predicting` (*"the DM test used a two-sided hypothesis test with the Harvey small-sample adjustment and Newey–West variance estimation"* + KSPA/ECDF) — protocolo mais completo da base; ver lote 1.
- **DM contra múltiplos benchmarks + 5 runs com IC95%:** `xing2026secondary` (SD-TCGRU; *"the DM statistic was negative across all datasets, with p-values significantly below 0.05"*).
- **DM + Friedman por estação do ano:** `lin2026whole` (Applied Energy; Tabelas 19–22: uma tabela DM por estação; honestidade de reportar casos não-significativos — ex. DM = −2,09 vs Informer-BiGRU no verão).
- **DM por horizonte com 10 runs:** `12026s2b` (DWT+1D-CNN+Bi-LSTM+ARSO; Tabela 9(a)-(c): stat, p-valor e magnitude por horizonte de 30/40/60 min).
- **DM em ablação:** `wan2026multiscale` (VMD-MSI; IC95% entre seeds + DM validando ganho incremental de módulo).
- **Matriz DM completa no suplementar:** `data2026driven` (benchmark fixo, matriz stat+p-valor).

## 3. Giacomini–White como alternativa condicional ao DM ⭐
- **Artigo:** `12026s2a` (SKLD-RR-VMD-D-GRU/LSTM, Advanced Engineering Informatics) — e também `jie2026designing` (lote 4)
  - Trecho: *"the DM test primarily evaluates unconditional predictive ability and does not account for model performance variability under different conditions... the GW test... evaluates whether two forecasting models exhibit statistically significant differences in conditional predictive ability, given a specific information set"*
  - GW com 3 conjuntos de informação: geral / não-extremo (<13,9 m/s) / extremo (≥13,9 m/s); valor crítico χ². Encaixa com os estratos regime/banda do metodos.json.

## 4. Bateria não-paramétrica completa
- **Artigo:** `bio2026inspired` / `biomimetics11` (DWT–BiLSTM; = `bendes2026bio` do lote 1)
  - *"First, the Friedman test was used... pairwise comparisons were performed using the Wilcoxon signed-rank test, and the inflation of Type I errors due to multiple comparisons was controlled using the Holm correction... the effect size measures Cohen's d and the rank-based Cliff delta were reported"*; IC bootstrap 95%; 30 runs; skill vs. persistência em unidades físicas.
- **Friedman com horizontes como blocos:** `leongomez2026multi` (RFF-RNN; *"average ranks (mean ± standard deviation) computed over all the horizons"* — 36 passos = 36 blocos, encaixa direto).
- **Friedman sobre RMSE de subtarefas:** `zhao2027mcbformer` (teste em blocos temporais do conjunto de teste); `12026s2` (ECM; *"a non-parametric Friedman rank test is used... It does not assume normality"*).
- **Wilcoxon pareado (2 artigos de vento):** `transformerdeep`/`12023introduction` (α=0,05 por torre); `laeeq2026physics` (VORTEX; heatmap dataset × métrica de p-valores).
- **t-Student pareado sobre runs repetidos (4 artigos):** `anto2026anbarasu` (n=5), `yaoyao2026probabilistic` (12 sazonalidades), `kandemir2026explainable` (10 inicializações), `sridhar2026predictive` (IC95% + t pareado) — alternativa paramétrica; menos adequada com caudas pesadas.

## 5. SEM ocorrência na base (diferenciais potenciais)
- **Model Confidence Set (MCS)** — nenhuma ocorrência (o repo já produz `mcs.json`).
- **Nemenyi / diagrama de diferença crítica** — nenhum uso (pós-hoc usado: Wilcoxon+Holm apenas).
- **Ljung–Box nos resíduos** como avaliação — nenhum uso metodológico (só menção a Box–Jenkins e ACF de dados brutos em `saxena2021offshore`).
- **KSPA** — só `gomez2026predicting` (o repo já calcula).

## Síntese
DM domina (7 artigos), tipicamente stat + p-valor por par; upgrades baratos: Harvey+Newey–West, DM por horizonte, DM em ablação, matriz contra benchmark. Complementos: Friedman com blocos + Wilcoxon/Holm; GW condicional por regime; Theil U2/TIC + DA como métricas relativas (precedentes `lin2022deep`, `chen2024research`); MCS e Ljung–Box são exclusividades do repo face à coleção.
