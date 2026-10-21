Recebi o briefing do escoteiro (.opencode/agent/kb-scout.md). Vou varrer o lote 1. Primeiro, tamanho dos arquivos e mapeamento das chaves bib:
Mapeamento bib ok (12 diretos + o stub `ieee-ssci2020-multidimcnn.md` que aparenta apontar para o fulltext — vou conferir). Agora greps dirigidos por grupos de palavras-chave metodológicas:
Ótimos sinais em shengxiang2022, wei2026, bendes2026, gomez2026, aodi2026. Agora detalho os arquivos ainda obscuros (zhu2019, suzuki2021, trebing stub, ramu2026) e confirmo trechos-chave:
O stub do Trebing é só abstract (nada extraível). Agora confirmo seções e detalhes dos achados fortes (bendes, wei, gomez, xianfeng, aodi):
Varredura concluída. Achados (ordenados por relevância para o SBSR):

## Teste DM com correção de pequena amostra (Harvey + Newey–West)
- Artigo: Predicting medium-term wind speed (TCN-sLSTM-MHA + TL) — bib: `gomez2026predicting` (arquivo: 1-s2.0-S0378779626002385-main.md)
- Onde no artigo: Seção 3.3 (Table 5, Fig. 6)
- Como usado: "the DM test used a two-sided hypothesis test with the Harvey small-sample adjustment and Newey–West variance estimation", significância avaliada pela t de Student a 1%; reporta estatística ± SE por horizonte e por baseline, aceitando DM negativo/insignificante em horizontes longos.
- Custo p/ adotar: baixo (pós-modelo, só previsões já geradas)
- Encaixe no SBSR: Avaliação — upgrade direto do DM já planejado (bootstrap em blocos + HAC como robustez)

## Friedman + post-hoc Wilcoxon pareado com Holm + tamanhos de efeito
- Artigo: Bio-Inspired Metaheuristic Optimization (benchmark estatístico DWT–BiLSTM) — bib: `bendes2026bio` (arquivo: biomimetics-11-00568.md)
- Onde no artigo: Seção 3.7 (Evaluation Criteria and Statistical Analysis), aplicado na 4.1
- Como usado: 30 execuções por algoritmo; Friedman para diferença global, "pairwise comparisons were performed using the Wilcoxon signed-rank test... controlled using the Holm correction", com Cohen's d e Cliff's δ para significância prática, α=0,05.
- Custo p/ adotar: médio (exige múltiplas seeds por modelo — ver achado de repetições abaixo)
- Encaixe no SBSR: Avaliação/Resultados — comparação conjunta das 4 arquiteturas + baselines

## ECDF do erro absoluto + teste KSPA
- Artigo: Predicting medium-term wind speed — bib: `gomez2026predicting` (arquivo: 1-s2.0-S0378779626002385-main.md)
- Onde no artigo: Seção 3.3 (Fig. 6a–b)
- Como usado: "analyze empirical cumulative distribution functions (ECDFs) of absolute forecasting errors and perform Kolmogorov–Smirnov predictive accuracy (KSPA) tests" — compara distribuições de erro inteiras entre pares de modelos, não só médias.
- Custo p/ adotar: baixo (pós-modelo)
- Encaixe no SBSR: Resultados — complemento visual + estatístico ao DM

## Rolling-origin aninhado com separação de variâncias (seed vs. escolha de modelo)
- Artigo: Bio-Inspired Metaheuristic Optimization — bib: `bendes2026bio` (arquivo: biomimetics-11-00568.md)
- Onde no artigo: Seções 3.6 e 4.3
- Como usado: "each origin is trained only on past years, early-stopped on the subsequent year, and evaluated on a later, completely untouched test year that enters neither hyperparameter selection nor training. Each selected configuration is retrained under multiple neural-network seeds so that training-induced variance is estimated separately from metaheuristic-induced variance"; robustez multi-janela via Kruskal–Wallis + Mann–Whitney U com Holm.
- Custo p/ adotar: médio (andar walk-forward já existe no pipeline; faltam as múltiplas seeds)
- Encaixe no SBSR: Avaliação — protocolo walkforward do pipeline

## Skill score vs. persistência em unidades físicas + relato transparente de resultado negativo
- Artigo: Bio-Inspired Metaheuristic Optimization — bib: `bendes2026bio` (arquivo: biomimetics-11-00568.md)
- Onde no artigo: Seções 3.8 e 4.5
- Como usado: persistência como "recommended minimum reference baseline"; skill computado após inversão de escala: "the reconstructed forecast does not exceed a naive persistence baseline (skill ≈ −0.5 in m/s versus +0.44 in normalized component space), a discrepancy... that we report transparently"; IC 95% via bootstrap.
- Custo p/ adotar: baixo (persistência já é baseline do SBSR)
- Encaixe no SBSR: Resultados/Discussão — skill por horizonte contra persistência

## Repetições independentes (5×) com desvio-padrão e IC 95% das métricas
- Artigo: Secondary decomposition... strong wind prediction (HSR) — bib: `wei2026secondary` (arquivo: 1-s2.0-S0952197625032695-main.md)
- Onde no artigo: Seção 4.2 (setup) e Sec. 4.6 (Table 10)
- Como usado: "each experiment was independently repeated five times. The standard deviation and 95% confidence intervals of key performance metrics were calculated and reported to evaluate the repeatability and robustness".
- Custo p/ adotar: médio (5 treinos por modelo; dados existem)
- Encaixe no SBSR: Avaliação — média±desvio nas tabelas de métricas

## Kruskal–Wallallis (módulo) + Dunn pareado (nível) para significância de componentes
- Artigo: Deep learning combined... multi-objective optimization — bib: `shengxiang2022deep` (arquivo: Deep learning combined wind speed forecasting with hybrid time series decomposition and multi-objective parameter optimization - ScienceDirect.md)
- Onde no artigo: Seção 5 (5.4.1 módulos, 5.4.2 níveis; Tables 15–16)
- Como usado: "module significance test is realized by Chi-Squared test with a significance level of 0.05" e "level significance test is realized by Dunn pairwise comparison test" — quantifica contribuição de cada componente do pipeline sobre a acurácia via ranks médios.
- Custo p/ adotar: médio (requer matriz de execuções por fator; ablações de perda já previstas)
- Encaixe no SBSR: Resultados — significância da ablação (encoder LSTM×TCN, uni×bi, atenção)

## Análise de sensibilidade por perturbação ±30% dos hiperparâmetros
- Artigo: Deep learning combined... — bib: `shengxiang2022deep` (arquivo: Deep learning combined wind speed forecasting... - ScienceDirect.md)
- Onde no artigo: Seção 4 (Table 8); variante grid no wei2026secondary Sec. 4.5 via MSE de validação
- Como usado: valores ótimos tratados como baseline (0%) e variados em "[−30%, 30%] with a step length of 10%" para mostrar robustez da escolha; wei2026 usa grid sistemático com critério de seleção explícito.
- Custo p/ adotar: médio (retreinos; trials Optuna podem aproximar)
- Encaixe no SBSR: Resultados/Apêndice

## Avaliação de eventos extremos como classificação binária (precisão/recall + desbalanceamento)
- Artigo: Secondary decomposition... strong wind — bib: `wei2026secondary` (arquivo: 1-s2.0-S0952197625032695-main.md)
- Onde no artigo: Sec. 4.x (Table 11) e Discussão
- Como usado: "Strong wind events are labeled as 1... The model's overall prediction accuracy... exceeded 97.50%... However, the recall... are merely 25.00%, 54.78%, and 35.85%" — relata honestamente o fracasso em eventos raros causado pelo desbalanceamento.
- Custo p/ adotar: baixo (pós-modelo, threshold sobre ws100)
- Encaixe no SBSR: Resultados — análise de eventos de vento alto no LiDAR

## Detecção de breakpoints (BFAST) + atribuição do erro a mudanças estruturais
- Artigo: Optimized deep learning... coastal/offshore (tufões) — bib: `aodi2026optimized` (arquivo: s40722-025-00446-1.md)
- Onde no artigo: Sec. 4.2 (Fig. 10) — análise pós-previsão
- Como usado: "the BFAST method was employed to detect breakpoints in the actual values during the prediction phase... all statistically significant at the 95% confidence level", conectando erros grandes dos modelos aos pontos de mudança (turning points vs. steady-state).
- Custo p/ adotar: baixo/médio (pós-modelo; implementação do BFAST)
- Encaixe no SBSR: Discussão — explicar onde (e por que regime) o modelo erra mais

## Erro estratificado por faixa de intensidade + boxplots de erro + t-test pareado
- Artigo: Physics-informed wind power forecasting (PI-BiLSTM) — bib: `xianfeng2026data` (arquivo: 1-s2.0-S2213138826005576-main.md)
- Onde no artigo: Table 3 (faixas de velocidade 0–3/3–13/13–25 m/s), Fig. A3
- Como usado: métricas por faixa operacional ("performs significantly better... in the shutdown and full-load zones"), distribuições de erro em boxplot ("box plots reveal... significantly lower median errors") e "Paired t-test results (p < 0.05) further confirm that the observed improvements are statistically significant"; MBE (viés) reportado junto.
- Custo p/ adotar: baixo (pós-modelo)
- Encaixe no SBSR: Resultados — erro por regime de intensidade do vento

## Diagnóstico estatístico do dataset (não-gaussianidade, ACF/PACF, curtose)
- Artigo: Secondary decomposition... strong wind — bib: `wei2026secondary` (arquivo: 1-s2.0-S0952197625032695-main.md)
- Onde no artigo: Sec. 4.1 (Table 1, Fig. 5–6)
- Como usado: tabela com média/máx/desvio/p-valor de normalidade/skewness/kurtosis + "autocorrelation functions (ACF) and partial autocorrelation functions (PACF)" para motivar o desafio de previsão (dependência de longo prazo).
- Custo p/ adotar: baixo (dados LiDAR existem; saxena2021offshore usa prática similar com IC 95% no ACF)
- Encaixe no SBSR: Dados/Métodos — caracterização da série ws100

Notas do lote: `ieee-ssci2020-multidimcnn.md` é stub de abstract (12 linhas, sem full text) — nada extraível; `3147-7723-1-PB.md` (ramu2026short), `saxena2021-lidar-eemd.md`, `1-s2.0-S1364682626000234-main.md` (chen2026hybrid), `ieee-tste2020-pstn-fulltext.md` (zhu2019learning) e `suzuki2021deep` não geraram achados acima do corte (práticas já cobertas: scatter y=x/boxplot de erro, métricas por horizonte, RMSE mensal por estação).

LOTE 1 CONCLUIDO - 13 arquivos varridos, 12 achados.
