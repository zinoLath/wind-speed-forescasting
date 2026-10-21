Todos os 13 arquivos existem. Vou varrer cada um priorizando seções de metodologia/experimentos/resultados. Primeiro, localizo as chaves bib no índice:
Todas as 13 chaves bib mapeadas. Agora localizo a estrutura de seções dos arquivos para focar em metodologia/experimentos/resultados:
Metade dos arquivos usa seções em negrito/numeradas sem `#`. Localizo essas seções:
Estrutura mapeada. Agora varro palavras-chave de testes estatísticos em todo o lote e leio as seções-alvo em paralelo:
O output foi truncado. Vejo o restante das ocorrências de testes estatísticos nos demais arquivos:
Testes estatísticos mapeados. Agora leio as seções de metodologia/resultados de cada arquivo:
Dois arquivos varridos a fundo. Continuo com os próximos:
Varredura completa. Segue o consolidado do lote 2.

---

# LOTE 2 — Achados metodológicos (13 arquivos)

## 1. `1-s2.0-S030626192600718X-main.md` → `binrong2026whole`
*Whole-process interpretable wind speed combination forecasting (Applied Energy 2026)*

- **[anti-vazamento]** Rolling GWO-VMD com janela causal estrita ("no lookahead"); identifica o forward bias da decomposição global pré-split como lacuna generalizado na literatura (l.130, 140, 164; Gap 1)
- **[protocolo]** Split 8:1:1 treino/validação/teste em 4 datasets sazonais; previsão single-step 1h (l.473, 686)
- **[comparação justa]** PLO otimiza hiperparâmetros uniformemente de submodelos e baselines (l.445, 686); decomposições comparadas com o mesmo otimizador GWO aplicado a EMD/SVMD/CEEMDAN/ICEEMDAN/VMD (l.1018)
- **[seleção de features]** Dois estágios mRMR (132→10) + TreeSHAP (10→5), validada por ablação (l.439, 523–542, 943); controle de qualidade de IMFs por MI + Adaptive Information Distance com critérios alta-correlação/baixa-redundância/estabilidade-de-endpoint (l.214)
- **[testes estatísticos]** DM test + Friedman sobre sequências de erro salvas (52 comparações × 4 estações, Tabelas 19–22); relato honesto de p não-significativos e DM negativo (verão −2,09; p=0,036) com explicação pelo peso do sub-modelo dominante (l.1197–1357)
- **[limitação de teste discutida]** DM só examina média da perda — não captura estabilidade da distribuição; complementa com IQR de resíduos (l.1355)
- **[multiobjetivo]** Pareto MAPE + variância do erro; compara 8 estratégias de ponderação (equal/SAO/Bayesian/DE/random/Q-learning/CEO/NSGA-II) sob inputs idênticos; seleção por distância ponderada 0,8/0,2 (l.1026, 834, 1170)
- **[estratificação]** Subsets por volatilidade Δ_v com limiares P25/P75/P90 (janela ±6h); **MAPE calculado excluindo amostras <0,5 m/s** (guarda contra instabilidade); IR cresce monotonicamente com volatilidade (l.1365–1382)
- **[pós-modelo]** Cadeia de interpretabilidade em 4 camadas (TreeSHAP→LRP→interação residual→Kernel SHAP) + AOPC, perturbação contrafactual, ECA, IMI inter-modelo, SHAP rolante no tempo (l.562–578, 1439–1524)
- **[ablação]** Remoção de cada sub-modelo degrada consistente; ganhos atribuídos à complementaridade, não dominância (l.1122)

## 2. `1-s2.0-S0952197626015824-main.md` → `cao2026probabilistic`
*Probabilistic multi-step prediction with online feature extraction (EAAI 2026)*

- **[protocolo]** 12 test sets (3 áreas × 4 estações); treino/validação usam **apenas observações anteriores ao período de teste** (ordem temporal estrita) (l.310)
- **[anti-vazamento]** Normalização min-max calculada só do treino; verificação de completude pré-treino (l.306)
- **[comparação justa]** Grid search no validation + early stopping para **todos** os métodos, ranges tabelados (l.379–388)
- **[métricas probabilísticas]** QL + CRPS; diagramas de confiabilidade PIT em QQ plots; métrica de cobertura R^(1−2ν) ideal=0 (l.396–514, 632)
- **[detecção de drift]** CSQL (soma cumulativa da quantile loss, inspirada em Page-1954) com limiar adaptativo μ+kσ + teste Page-Hinckley para caracterizar drift sazonal; diferencia drifts críticos de não-críticos (l.136–156, 308, 640)
- **[análise condicionada]** QL avaliado nas vizinhanças dos pontos de drift; erro concentrado em torno do drift (l.472, 662)
- **[teste estatístico]** Paired t-test nos 12 test sets para RMSE e CRPS, 95% confiança (l.688)
- **[custo]** Comparação de runtime com justificativa (reseleção intermitente, não contínua) (l.795)

## 3. `1-s2.0-S036054422601529X-main.md` → `liu2026robust`
*Multi-kernel correntropy + adaptive residual error correction (Energy 2026)*

- **[caracterização pré-modelo]** Séries descritas com banda IQR, histograma, média vs mediana e assimetria direita para motivar perda robusta a não-Gaussiano (l.397–405)
- **[anti-vazamento explícito]** "Todo ajuste de hiperparâmetros e formulação de regras conduzido apenas no treino; VMD e limiar θ calibrados só com treino; testes estritamente excluídos" (l.409); split 7:1:2
- **[seleção de decomposição]** Grid VMD (K∈{4..12}, α∈{500..4000}) avaliado por 3 métricas de qualidade — ortogonalidade, spectral flatness do resíduo, ACF energy do resíduo — com discussão de trade-off (l.413–472)
- **[sensibilidade de perda]** K kernels 1–7 + esquemas de pesos via radar chart; princípio da parcimônia e máxima entropia (l.474–476)
- **[gating pré-treino]** Score de previsibilidade = autocorrelation energy + spectral flatness → tabela manter/descartar componentes **antes** de treinar; resíduo descartado como banda larga (l.478–547)
- **[correção pós-modelo]** AREC: diagnóstico estatístico+espectral dos resíduos → seleciona regressão harmônica (resíduo periódico) ou MKC-LSTM auxiliar (não-linear) (l.199, 345–369)
- **[métricas originais anti-lag]** L_max (lag da correlação cruzada máxima) e D_phase (desvio médio entre extremos) para substituir inspeção visual subjetiva de fase (l.775–789)
- **[stress test de dados]** Treino truncado a 40/60/80/100% com conjunto de teste idêntico entre cenários (l.803–807)
- **[eficiência]** Parâmetros, inferência/step, tempo de treino no mesmo hardware; justifica pelo ciclo de despacho de 10–15 min (l.811–848)
- **[métricas]** MSE/RMSE/MAE/MAPE/sMAPE/R² (Tabela 4)

## 4. `1-s2.0-S1568494625015972-main.md` → `geng2026long`
*Long/short-term feature fusion network (Applied Soft Computing 2026)*

- **[protocolo]** Split sequencial temporal 6:2:2 (l.396)
- **[seleção data-driven]** Comprimentos de sequência definidos por Transformada de Fourier (periodicidade inerente), **explicitamente não por otimização de performance** (l.458–476)
- **[comparação justa]** Baselines alimentados separadamente com sequências curtas e longas (2 conjuntos de tabelas); modo single-scale do modelo para igualar escala de entrada dos lineares (l.486)
- **[relato]** Melhor em negrito/segundo sublinhado; tabelas multi-horizonte (1–7 dias, 1–6 semanas); afirmação "97,4% de 38 cenários comparativos" (l.29, 508–543)
- **[escopo de claim]** Ablação LASH-Net^attn (módulo trocado por PatchTST) com ressalva: vale para um tipo de atenção, não generaliza para toda atenção (l.93)
- **[análises]** Sensibilidade (comprimentos, taxa de retenção, patch), contagem de parâmetros + verificação de complexidade linear (l.366, 660–714)

## 5. `1-s2.0-S0306261926007464-main.md` → `aslam2026physics`
*VORTEX physics-informed extreme wind prediction (Applied Energy 2026)*

- **[protocolo]** Split cronológico por local; janela 24h→horizonte 6h, stride 6h; 4 sites geograficamente diversos (l.272, 307, 322)
- **[comparação justa]** TPE para todos os modelos; baselines com MSE, proposto com VAH-Loss — seguido de estudo com 5 variantes de perda sob protocolo idêntico (l.326, 380)
- **[sensibilidade com disclaimer]** Protocolo curto de validação (10 épocas), perturbação um-por-vez, ΔRMSE relativo; valores rotulados como "intermediários de exploração do espaço de busca, não resultados finais de teste" (l.376–382)
- **[reprodutibilidade]** Estabilidade semente-a-semente: 5 seeds com mean±std e faixa; semente representativa = mais próxima da mediana multiseed (l.402)
- **[teste estatístico]** Wilcoxon signed-rank p<0.01 em todas as métricas × todos os datasets (l.430, 470)
- **[estratificação]** Avaliação por **quartis de volatilidade** (σ da velocidade real no horizonte, do ground truth): Q1–Q4 com RMSE/MAE/R²; ganho relativo cresce com a volatilidade (Q4 até 13,3%) (l.452–505)
- **[evolução por horizonte]** Métricas hora a hora no horizonte de 6h (l.428, Fig. 6)
- **[seleção de baseline justificada]** Cobertura de paradigmas (attention/recurrent/physics/hybrid): "melhora reflete valor do método, não fraqueza dos baselines" (l.406–418)
- **[custo]** Tabela de params/FLOPs/latência/memória + viabilidade de deploy (l.472–487)

## 6. `1-s2.0-S0360544226006948-main.md` → `zhang2026wind`
*GADSG-CL wind farm cluster continual learning (Energy 2026)*

- **[limpeza com regras físicas]** Zeros corrigidos; saída "unknown" sinalizada por 2 condições físicas (potência≤0 com vento>2,5 m/s; pitch>89°), **com evidência estatística**: scatter plots e distribuição condicional mostrando só 0,6% dos casos zero-potência com vento alto (l.365–377)
- **[decisão de escala]** Métricas reportadas em escala normalizada para comparabilidade entre datasets — inversão introduziria valores escala-dependentes (l.385)
- **[protocolo]** Split 6:2:2; early stopping patience 7; Adam/MSE (l.385)
- **[seleção de hiperparâmetros]** threshold por grid 0,1–0,9 (melhor R²); nº de clusters via coeficiente Tanimoto de similaridade de grafos, maximizando distinção entre clusters (l.421–427)
- **[comparação controlada]** Baselines em 3 camadas + variantes GA (grafo proposto injetado em backbones inalterados) para isolar a contribuição da construção de entrada vs preditor (l.433–470)
- **[relato honesto]** R² negativos mantidos nas tabelas (ex. STGCN −0,3212); capacidade instalada não reportada por indisponibilidade de metadado (l.335, 461)
- **[multi-horizonte]** 12/24/36 passos em 3 benchmarks; ablação inclui análise sazonal (l.381, 480–535)

## 7. `1-s2.0-S0378779626006425-main.md` → `dong2026graphdynnet`
*GraphDynNet offshore continuous prediction (EPSR 2026)*

- **[pré-processamento]** Imputação pela média de outliers/faltantes; MinMax [−1,1]; IMFs como canais extras (l.241)
- **[comparação padronizada]** 7 baselines no mesmo dataset; MSE/RMSE/MAE/R²; Adam + ReduceLROnPlateau/CosineAnnealing + gradient clipping 1.0 + weight decay 1e-5 (l.249)
- **[ablação]** 4 variantes de remoção de módulo com quantificação (ex. MSE +112,4% sem ODE) (l.305–339)
- **[generalização zero-shot]** Treino em 1 estação-mestre, aplicação direta em 12 estações **sem fine-tuning**, justificado como benchmark mais estrito e operacionalmente relevante; análise por estação de distribution shift e limites de aplicabilidade (l.343–400)
- **[caveat de métrica]** Discussão explícita: R² = 1 − MSE/Var(y) pode mascarar erro absoluto crescente — padrão "erro alto mas R² aceitável" (l.388)
- **[protocolo de evento extremo]** Janela de tufão via HURDAT2 (±24h da aproximação máxima), estratificação dual-subset (teste completo vs subset extremo), degradação quantificada em +%, razão de amostras extremas reportada por evento; **tufões reais, não sintéticos**; análise "graceful degradation" e gargalo de pico (l.402–471)

## 8. `processes-14-01162.md` → `jiajun2026collaborative`
*VPP + HVDC collaborative optimization (Processes 2026)*

- **[métricas de previsão]** MAPE/NRMSE/R² reportados separadamente para OWP e PV vs RNN/CNN/LSTM clássico; NRMSE normalizada pela capacidade instalada (l.324–352)
- **[comparação de otimizadores]** Taxas de corte + custo total operacional vs GA/PSO/GW (l.360–377)
- **[eficiência computacional]** Iterações, tempo/iteração, tempo total; trade-off explícito "rápido mas não ótimo" vs "rápido e bom" (l.388–415)
- **[limitações declaradas]** HVDC simplificado, framework determinístico ignorando incerteza de previsão, VPP único, preços exógenos, escalabilidade não testada (l.417)

## 9. `s12046-026-03222-5.md` → `altun2026hybrid`
*Hybrid transformer wind power (Sadhana 2026)*

- **[EDA pré-modelo]** Tabela descritiva completa (count/média/std/quartis/máx), heatmap de correlação, wind rose por altura (l.181–228)
- **[relato comparativo]** Tabela hora-a-hora 24h com todos os modelos + métricas MSE/MAE/RMSE; ganhos relativos de MSE 74,8–95,7% vs 3 híbridos (l.19, 275–305)
- **[custo computacional]** Tempo de execução/inferência/memória dos 4 modelos (l.250–257)
- **[fraqueza de protocolo — registrar como exemplo negativo]** Avaliação em janela única de 24h; sem split cronológico ou teste estatístico relatados (l.236–265)

## 10. `1-s2.0-S2590005626000950-main.md` → `sridhar2026predictive`
*Bi-directional permutation LSTM-RNN on SCADA (Array 2026)*

- **[protocolo]** Split sequencial temporal 70/15/15 com validação/teste cronologicamente posteriores, "para simular cenários reais e prevenir vazamento" (l.206–210, 228)
- **[tuning]** Bayesian optimization 50 trials minimizando MAE de validação (proposto) vs grid search (baselines); parâmetros e ranges declarados (l.258)
- **[regularização]** Dropout + early stopping patience 15 + L1/L2 (λ₁=0,01; λ₂=0,005); Huber loss; Adam (β₁/β₂), cyclical LR (l.132, 222, 248–278)
- **[testes estatísticos]** Paired t-test nos erros vs baselines em test splits idênticos com **IC 95%** (p<0,001 vs RF; p<0,05 geral) (l.360–378)
- **[bateria de baselines]** 7 modelos do ML clássico ao DL com hiperparâmetros listados (l.270–290)
- **[eficiência]** Latência, complexidade e tamanho do modelo (seções 5.4–5.5)
- **[reprodutibilidade]** Matriz de permutação softmax diferenciável + Xavier init; hardware/software reportados (l.260, 296)

## 11. `139961N.md` → `chang2021ai`
*AI-driven correction of NWP wind in Kumtag Desert (SPIE 2021)*

- **[sensibilidade pré-treino]** 5 configurações de canais de entrada (31c/22c/19c/10c/4c) testadas para selecionar o conjunto ótimo de features antes da modelagem final (l.148–174)
- **[anti-vazamento na normalização]** Média/variância calculadas de 20.000 amostras de treino apenas (l.154)
- **[estratificação por lead time]** 3 grupos (3–24h, 27–48h, 51–72h, 13.740 amostras cada); split 7:2:1 com papéis explícitos (val = seleção de configuração; teste = avaliação final) (l.195)
- **[relato honesto]** Razão de melhora vs NWP cru **incluindo pioras** (valores negativos, ex. k-NN −81,53% MSE; Linear Reg −8856,92% em 51–72h) (l.203–270)
- **[avaliação de evento extremo]** Vento forte (≥ Beaufort 5) tratado como classificação: acurácia/precisão/recall/F1 antes vs depois da correção; % de amostras de alto erro (>5 m/s) (l.138–142, 274–294)
- **[importância de features]** PFI: permutação por canal, MSE médio de 10 repetições, % de aumento como score; U-10m/V-10m/V-100m >80% da contribuição (l.300–316)
- **[goodness-of-fit]** R², r de Pearson, scatter vs linha 1:1, bias quantificado por faixa de velocidade antes/depois (l.318–346)
- **[limitações]** Adaptação limitada dos CNNs e treino curto reconhecidos nas conclusões (l.352–354)

## 12. `energies-19-01071-v2.md` → `chengcheng2026wave`
*Wave power density with wind conditions (Energies 2026)*

- **[justificativa de métrica]** RMSE escolhido sobre MAPE/R²: unidade alinhada à densidade de potência e "evita explosões matemáticas que afligem percentuais" em 40 anos de dados (l.77)
- **[pré-processamento]** Anos incompletos deletados (1986, 1989, 1990, 2004, 2008); z-score; onda calculada só quando profundidade > λ/2 (l.59–67)
- **[features espaciais]** Grade 9-célula (alvo + 8 vizinhos) = 18 features; matriz 71×107 (l.89)
- **[protocolo temporal]** Treino jan/1979 (248 passos de 3h), teste fev/1979; avaliação adicional em 1979-08, 2018-02, 2018-08 (l.97–138)
- **[amostragem de teste]** 6 pontos aleatórios cobrindo tipos de local (costeiro/offshore/estreito/plataforma) (l.130)
- **[visualização de erro]** Mapas de RMSE no Golfo inteiro por período; RMSE mensal por local; erro estratificado por tipo de área (l.159–175)

## 13. `ieee-icist2020-lstm-fulltext.md` → `ehsan2000wind`
*Wind speed prediction & visualization LSTM (ICIST 2020)*

- **[bateria de comparação]** 12 algoritmos (9 ML + DNN/CNN/LSTM), métricas MAE/MSE/MedAE/R² no teste, tabela única com hiperparâmetros (l.86, 150–163)
- **[métrica robusta]** MedAE incluído explicitamente pela robustez a outliers (l.98)
- **[checagem de normalidade]** Gráfico Qui-Quadrado Q-Q para decidir necessidade de normalização pré-ajuste (l.120)
- **[diagnóstico de resíduos]** Residual plots (resíduo vs ajustado) como validação de feature fit (l.104, 124); curvas de loss/MSE por época checadas contra over/underfitting (l.128)
- **[fraqueza de protocolo — exemplo negativo]** Split 80/20 aleatório (não cronológico) em série temporal; "acurácia" usada como sinônimo de R² (l.114, 134)

---

**Síntese transversal para o artigo SBSR:** os processos mais transferíveis do lote são (i) testes de significância sobre sequências de erro — DM+Friedman (`binrong2026whole`), Wilcoxon (`aslam2026physics`), paired t-test (`cao2026probabilistic`, `sridhar2026predictive`); (ii) estratificação da avaliação por volatilidade — quartis Q1–Q4 (`aslam2026physics`) e subsets P25/P75/P90 com exclusão de amostras <0,5 m/s no MAPE (`binrong2026whole`); (iii) protocolos de evento extremo com janelas reais (HURDAT2 ±24h, `dong2026graphdynnet`) e classificação de vento forte com recall/F1 (`chang2021ai`); (iv) salvaguardas anti-vazamento explícitas (rolling VMD, tuning só no treino, normalização só do treino); (v) métricas pós-modelo sem mudança de arquitetura (L_max/D_phase anti-lag, IQR de resíduos, PIT/CRPS, PFI, SHAP/AOPC).

LOTE 2 CONCLUIDO - 13 arquivos varridos, 91 achados.
