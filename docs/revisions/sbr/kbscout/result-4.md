Chaves bib localizadas no index.md. Agora vou mapear os cabeçalhos de cada arquivo para direcionar a leitura das seções de metodologia/experimentos/resultados.
Os 5 stubs IEEE são abstract-level (full text não disponível) — extraídos. Agora leio as seções metodológicas dos artigos completos, em lotes.
Três artigos varridos (jie2026, wan2026, sheng2026). Próximo lote: Quiñones (microgrids), Zhang (JSupercomputing) e Shafiullah (smartcities).
Mais dois varridos. Próximo lote: Shafiullah (smartcities), Wang (iScience) e Huang (Model selection/MSFSC).
Cinco varridos. Próximo lote: Huang (Model selection MSFSC) e Gharehbaghi (CEEMDAN comparativo).
Sete varridos. Último lote: IJE (Mezaache), Bai (CNN-Seq2Seq) e Ozdemir (Ensemble ANN).
Varredura completa. Segue o relatório consolidado do lote 4.

---

# LOTE 4 — Achados metodológicos (17 arquivos)

## 1. `1-s2.0-S1474034626003435-main.md` — bib: `jie2026designing`
*(Designing robust wind speed forecasting under extreme Events, AEI 2026)*
- **Testes estatísticos**: **Giacomini-White (GW) test** de habilidade preditiva condicional (α=0.05, χ²) em 240 instâncias, com 3 conjuntos de informação (geral; geral+não-extremos; geral+extremos) — justifica escolha do GW sobre o DM (condicional vs incondicional); exceções relatadas honestamente (2/240).
- **Protocolo**: split 7:2:1 (treino/validação/teste); lookback ótimo via distância DTW média (10 sequências aleatórias do treino); nº de modos VMD via razão SKLD só no treino; unidades GRU {32,64,128,256} por MAPE na validação; extremos definidos por threshold físico (Beaufort ≥13.9 m/s) e **validados contra relatos reais de eventos extremos**.
- **Comparação/relato**: métricas em 4 horizontes (1/2/3/12 passos); **ACCI** (melhoria % vs benchmark para MAPE/MAE/MSE); **bateria replicada trocando GRU→LSTM** para testar generalidade do framework; comparação com SOTA (Transformer/Autoformer/SCINet); experimento de robustez com dados reorganizados (zeros mantidos, MSE como loss) para checar se conclusões sobrevivem à pré-seleção de dados.
- **Pós-modelo**: boxplots de erros por modelo; scatter previsto×observado; análise de skewness/kurtosis com/sem extremos; custo computacional; imputação linear documentada com estatísticas antes/depois.

## 2. `1-s2.0-S0378779626006851-main.md` — bib: `wan2026multiscale`
*(Multiscale VMD + SHAP, EPSR 2026)*
- **Testes estatísticos**: **DM test pareado** para significância do ganho incremental do módulo GTTS; métricas como **mean±CI sobre múltiplas seeds**.
- **Protocolo**: split **cronológico** 70/10/20 "para evitar vazamento temporal"; **normalização ajustada só no treino**; early stopping; seed fixa; tabela completa de implementação (reprodutibilidade); baselines ajustados individualmente sob mesmo split/features.
- **Comparação/relato**: ablação incremental GRU→VMD-GRU→VMD-MSI-GRU→VMD-MSI-GTTS em 4 turbinas; baselines fortes (CEEMDAN-Hybrid, Informer, TCN-ECABiLSTM) com mean±std + tempo de treino; boxplots de distribuição de erro pós-inverse-scaling.
- **Pós-modelo**: **SHAP global por ramo** (normalização intra-ramo; contribuição de ramo com valores absolutos); importância temporal agregada (recency effect); **consistência cross-turbina quantificada por correlação de postos de Spearman entre rankings SHAP** (todos >0.92); robustez a missingness (0–5%) + ruído (STD 0–0.02) com **aumento relativo de RMSE/MAE** vs baseline limpo; sensibilidade do K do VMD com tradeoff RMSE×tempo; latência/throughput GPU/CPU por batch.

## 3. `Shape-Aware Seq2Seq...md` — bib: `sheng2026shape`
- **Protocolo**: 3 datasets (buoys geograficamente distintos) para generalização; 70/10/20 com papéis exclusivos explícitos; normalização por estatísticas do treino; horizontes 3/6/9/12 com input = 2× horizonte.
- **Comparação/relato**: métricas MAE/RMSE/MAPE + **MDTW (mean DTW) como métrica de similaridade de forma**; **PMAE/PRMSE/PMAPE** (melhoria % vs benchmark); scatterplots com slope da reta de ajuste; análise qualitativa de **defasagem de fase** e sub/superestimação de picos/vales; degradação por horizonte (vantagem cresce com horizonte).
- **Pós-modelo**: **ablação de loss functions** (WYM-1 sem perda especial → WYM-2 MSE → WYM-3 TILDE-Q → WYM ITILDE-Q), isolando encoder-decoder / denoising / perda shape-aware; limitações admitidas (denoising remove extremos locais).

## 4. `Towards smart energy management...md` — bib: `nomenclature2023` *(entrada do index que aponta o markdown; existe também `quinonestowards` sem link de texto)*
- **Protocolo**: hiperparâmetros via **BO-GP (Bayesian Optimization + Gaussian Process, 25 iterações)** com espaço tabulado; **lookback estudado separadamente (k=168/330/720 h)** para preservar arquitetura; busca independente por variável-alvo.
- **Métricas probabilísticas**: ensemble profundo (M redes, iniciações distintas) para incerteza **epistêmica** + NLL Gaussiano para **aleatórica** (soma de variâncias); PICP/MPIW/CRPS; PIs a 80% e 95%.
- **Pós-modelo (análise fatorial)**: efeito do lookback, do horizonte (1/3/6/24 h), da estação (anual+4 estações) e 95% vs 80% — incluindo **R² negativo no inverno 24 h (pior que média) relatado**; tradeoff acurácia×tempo de computação (GRU recomendado como modelo prático); seção de limitações.

## 5. `s11227-026-08589-0.md` — bib: `zhang2026novel`
- **Protocolo**: 3 datasets (DK/NL/US); splits idênticos aos artigos originais para dados públicos (comparabilidade) + 7:1:2 cronológico para o dataset próprio; early stopping; **baselines em 3 categorias progressivas** (CNN / GCN grafo pré-definido / adaptive graph), configs conforme publicações originais; comparações citadas dos papers-fonte quando re-treino inviável.
- **Comparação/relato**: coluna **"Improve" % vs segundo melhor** por horizonte; média entre horizontes; relato honesto de derrota pontual (2 h Dutch, dataset com poucos nós) com hipótese explicativa.
- **Pós-modelo**: ablação de 4 variantes; **visualização dos grafos aprendidos por camada** (estático vs dinâmico, esparsidade); complexidade teórica O(·) por módulo + runtime empírico (s/epoch, inferência do test set); degradação por horizonte com análise de lag/over-smoothing.

## 6. `smartcities-09-00059.md` — bib: `shafiullah2026advanced`
- **Protocolo**: split por anos (ano 1 treino, ano 2 validação/hiperparâmetros, ano 3 teste); grid search; **100 epochs fixos para todos "para atribuir diferenças somente aos inputs"** (controle experimental).
- **Comparação/relato**: 3 configurações de input (endógena; +meteorológicas; +sinais cíclicos sin/cos) × 5 DL — **ablação de features sem mudar arquitetura**; MAPE deliberadamente não usado para GHI (quase-zeros); horizontes 12 h/1 d/3 d.
- **Pós-modelo**: robustez por **regime de clima via clear-sky index** (cloudy/parcial/sunny/clear) e por **12 dias representativos (1/mês)**; perfis de % erro (viés); curvas de loss treino/validação (diagnóstico over/underfit); scatter; tabela comparativa com R² da literatura.

## 7. `1-s2.0-S2589004226019346-main.md` — bib: `wang2026adaptive`
- **Testes estatísticos**: **DM test** vs benchmarks com estatística, SE e p-valor (p<0.01), em 2 cenários (10-day VALHALL; 7-day Pacific) — exceção relatada (CNN-Trans.-MLP p=0.14).
- **Protocolo**: cross-dataset (2 estações); horizontes médios; feature selection **CRIA** (exclui variáveis colineares); imputação forward/backward-mean.
- **Comparação/relato**: avaliação **sazonal particionada** (admite derrota no outono); ablação de submodelos em 2 prediction lengths; melhorias % relatadas.
- **Pós-modelo**: pipeline pré/in/pós-modelo: atenção dinâmica por feature; scores de dependência temporal/espacial in-model; **LIME-ElasticNet com validação quantitativa da interpretação (NMSE/NMAE entre importâncias)**; **modelo físico log-law como verificação da predição do AI** (u*, z0); parâmetros treináveis + latência por amostra vs Transformers; código no Zenodo.

## 8. `Research and application of a Model selection...md` — bib: `huang2024research`
- **Testes estatísticos**: **Mann-Whitney U** (por modelo, real vs previsto, p-valores na tabela) e **Kruskal-Wallis** (múltiplos modelos, justificado pela não-normalidade do vento); IC 95% com EDMR (diferença de postos médios).
- **Métricas**: 8 critérios — MAE, RMSE, **STDAPE** (estabilidade), **DA (directional accuracy)**, TIC, MAPE, R², ACC (acurácia do classificador de seleção).
- **Incerteza pós-modelo**: **t-location-scale** para distribuição dos erros; intervalos via **FICP/FINAW/AWD**; boxplots.
- **Práticas**: seleção de modelo ótimo por ponto temporal (critério de erro mínimo) com classificador; curva de potência com **correção de densidade do ar** e bandas upper/standard/lower de Cp; análise de viés MPE (subestimação) com implicação operacional de rede; análise de curtailment (full-load hours); tradeoff acurácia×tempo de treino; relato honesto de que a seleção não atinge 100%.

## 9. `s41748-025-00714-y.md` — bib: `gharehbaghi2026comparative`
- **Protocolo**: feature selection por **PCC (limiar 0.5)**; split **cronológico 80/20 sem randomização**; normalização z-score; nº de árvores RF até estabilizar MSE.
- **Comparação/relato**: cadeia incremental de 6 modelos (RF→CEEMDAN-RF→+LSTM/+GRU/(LSTM-GRU)/(LSTM+GRU)); métricas R²/RMSE/MAE + **AICc** (qualidade vs complexidade) + convergence time; **TLP (nº de parâmetros treináveis) como diagnóstico de over/underfitting** em mesmo NHN.
- **Pós-modelo**: violin plots observado vs previsto; scatter com reta Y=X; relato de melhor desempenho em calibração que em teste.

## 10. `10.5829_ije.2026.39.02b.03.md` — bib: `image2026quality` *(⚠ index com título errado — "Image Quality Enhancement…"; conteúdo real: Mezaache et al., ICEEMDAN wind speed, IJE 2026)*
- **Protocolo**: ICEEMDAN + **Sample Entropy para seleção de IMFs** (tabela de SE por IMF); z-score; 2 sites com resoluções distintas (30/15 min).
- **Comparação/relato**: 2 abordagens (com/sem decomposição) × 4 DNNs + agregação por **classificação por região do espaço de entrada** (modelo ótimo por segmento); histogramas de RMSE entre abordagens; comparação com números da literatura (RMSE/MAPE/MAE por resolução).
- Nota crítica: MABE do modelo combinado é *pior* que individuais embora RMSE/MAPE/R² sejam melhores — inconsistência relatada sem teste estatístico; sem protocolo de split explícito (apenas "finetuning e cross-validation" mencionados de passagem).

## 11. `Short-time multi-energy load forecasting...md` — bib: `xiaoqing2021short`
- **Protocolo**: feature selection por **Spearman** (limiar 0.1, justificativa sobre Kendall/Pearson); min-max (preserva zeros); split 75/15/10 dentro de conjuntos sazonais segregados (cooling/heating) por característica física; **método da variável de controle** para dropout e lr (tabelas de sensibilidade com WMA).
- **Métricas/práticas**: **WMA (acurácia ponderada por importância da tarefa** — pesos pela capacidade de carga); **remoção de zeros no MAPE** e troca para RMSE quando muitos zeros (não-working days); comparação separada dia útil vs não útil; **ablação multi-task vs single-task** (efeito do acoplamento, sem mudar arquitetura).

## 12. `A Novel Ensemble Wind Speed Forecasting...md` — bib: `novel2024ensemble`
- **Práticas**: ANN única vs ensemble por site (configs dedicadas por local); seleção de configs por trial-and-error com verificação em holdout 30%; boxplots exploratórios por site; MAE/MAPE/RMSE.
- Nota crítica (armadilha): **teste com 2592 dados aleatórios de 1998–2022** apesar de série temporal — risco de vazamento; "acurácia" de verificação usada como critério de seleção; sem teste estatístico.

## 13–17. Stubs IEEE (abstract-level, full text não disponível no repo)
- `ieee-scems2020-visualized-matrix.md` — `liu2020deep`: comparação com LSTM; robustez em períodos de forte flutuação (claim só no abstract).
- `ieee-gucon2020-predictive-analysis.md` — `saini2020predictive`: split 70/30; MAE/MSE/RMSE reportados **em treino e teste**; XGBoost vs ARIMA/LSTM/RF.
- `ieee-tste2020-pstn.md` — `zhu2019learning` *(index aponta o fulltext `ieee-tste2020-pstn-fulltext.md`)*: treino end-to-end com loss única; dados reais multi-sítio.
- `ieee-ropec2020-comparison.md` — `penagallardo2020comparison`: benchmark vs ARIMA, 1-step e multi-step.
- `ieee-icist2020-lstm.md` — `ehsan2000wind` *(index aponta o fulltext `ieee-icist2020-lstm-fulltext.md`)*: 12 algoritmos comparados; "acurácia 97.8%" sem métrica de erro definida.

---

**Padrões transversais úteis para o SBSR**: GW/DM com múltiplos conjuntos de informação e p-valores (`jie2026designing`, `wang2026adaptive`, `wan2026multiscale`); métricas mean±CI multi-seed (`wan2026multiscale`); métricas de forma (MDTW) e de direção (DA/STDAPE) além de MAE/RMSE (`sheng2026shape`, `huang2024research`); análises por estação/regime de clima (`nomenclature2023`, `shafiullah2026advanced`, `wang2026adaptive`); robustez a ruído/missing (`wan2026multiscale`); pós-modelo SHAP com consistência quantificada por Spearman (`wan2026multiscale`); verificação física da predição (`wang2026adaptive`, `huang2024research`).

LOTE 4 CONCLUIDO - 17 arquivos varridos, 49 achados.
