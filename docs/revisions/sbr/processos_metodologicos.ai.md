# Processos metodológicos candidatos — varredura da base de conhecimento

> **Status (14/10/2026):** implementados A1–A6 + B3/B4/B6/B8 em
> `studies/metodos_complementares.py` → resultados em
> `data/results/metodos_complementares/` · explicações e fontes em
> `guia_metricas.ai.md` · sugestões de texto para o artigo como
> comentários `%` em `03_Material_e_metodos.tex`. Pendentes de GPU:
> A3 (5 seeds), B5, B9; B7 (PFI) parcial.

> **Autoria:** consolidação por IA (opencode/GLM-5.3) sobre 4 sub-relatórios
> produzidos por subagents `kb-scout` rodando **GLM 5.3 Flash** via
> `opencode run` (14/10/2026). 60 artigos varridos em
> `articles/knowledge/markdown/`, 234 achados brutos, 40 chaves bib
> verificadas em `articles/knowledge/references.bib`. Relatórios brutos:
> [`kbscout/result-1.md` … `result-4.md`](kbscout/) — **não editar**.
> Cada processo abaixo lista as fontes (chave bib + arquivo + seção) para
> permitir o referenciamento direto quando o processo for adotado.

Critério de curadoria: passos que **melhoram a escrita/defesa do artigo
SBSR sem mudar a arquitetura dos modelos** — testes comparativos,
protocolos de validação, estratificações, métricas pós-modelo e práticas
de relato. Ligação com o parecer: `parecer_revisor.ai.md` (M1–M10).

---

## Tier A — recomendados (baixo custo, alto valor, cabem no artigo curto)

### A1. DM com correção de Harvey (pequena amostra) + variância Newey–West

O teste DM já planejado, mas na forma que resiste a revisão: ajuste de
Harvey–Leybourne–Newbold + estimador HAC, estatística ± EP e p-valor
**por horizonte e por par**, com os não-significativos relatados.

- **Por quê:** é o pedido explícito da Vanessa, e a forma "nua" do DM em
  rolling forecast autocoracionado é alvo fácil de crítica.
- **Custo:** baixo — `reports/src/relatorio_resultados.py` já implementa o DM
  (agregado + por horizonte, blocos por origem); falta reportar
  estatística/por-par e o ajuste de Harvey.
- **Encaixe:** §Avaliação (1 frase de definição) + Resultados (tabela
  compacta de p-valores dos pares-chave).
- **Fontes:**
  - `gomez2026predicting` — arquivo `1-s2.0-S0378779626002385-main.md`, Seção 3.3 (Table 5, Fig. 6): "DM test used a two-sided hypothesis test with the Harvey small-sample adjustment and Newey–West variance estimation".
  - `wang2026adaptive` — `1-s2.0-S2589004226019346-main.md`: DM com estatística/SE/p (p<0,01) e **exceção relatada** (CNN-Trans.-MLP p=0,14).
  - `binrong2026whole` — `1-s2.0-S030626192600718X-main.md` (l.1197–1357): DM+Friedman com p não-significativo e DM negativo discutidos; nota de que DM só compara **média** da perda → complementar com dispersão.

### A2. Skill score vs. persistência por horizonte (unidades físicas) + IC por bootstrap

Percentual de melhora sobre a persistência em m/s por horizonte — com IC
95% — e disposição a reportar skill negativo onde existir.

- **Por quê:** transforma a tabela de métricas em declaração operacional
  ("X% melhor que o ingênuo em 6 h"); a persistência já é baseline do
  artigo. O relatório interno já computa skill vs. persistência.
- **Custo:** baixo (pós-modelo).
- **Encaixe:** Resultados (coluna extra ou mini-tabela) / Discussão.
- **Fontes:**
  - `bendes2026bio` — `biomimetics-11-00568.md`, Seções 3.8/4.5: persistência como "recommended minimum reference baseline"; skill ≈ −0,5 em m/s reportado transparentemente com IC 95% bootstrap, contrastando com +0,44 no espaço normalizado.
  - `trebing2020wind` — `ieee-ssci2020-multidimcnn-fulltext.md`: persistência presente em **todos** os experimentos/horizontes como referência mínima.

### A3. Múltiplas execuções (5 seeds) → média ± desvio + IC 95%

Repertir o treino de cada arquitetura (5×) sob seeds distintas e reportar
média±desvio (e IC) das métricas; run "representativa" = mais próxima da
mediana.

- **Por quê:** mata o M3 do parecer (1 treino/arquitetura, zero
  incerteza) e viabiliza qualquer teste estatístico entre arquiteturas.
  É a prática mais repetida em toda a varredura.
- **Custo:** médio — ~17 min de GPU por rodada dos 4 modelos
  (skill `lidar-retrain`) → ~1,5 h para 5 seeds; sem mudança de código.
- **Encaixe:** Avaliação (protocolo) + tabelas de Resultados (±).
- **Fontes:**
  - `wei2026secondary` — `1-s2.0-S0952197625032695-main.md`, Sec. 4.2/4.6 (Table 10): "independently repeated five times… standard deviation and 95% confidence intervals".
  - `aslam2026physics` — `1-s2.0-S0306261926007464-main.md` (l.402): 5 seeds com mean±std e faixa; representante = mediana multiseed.
  - `singh2026new` — `1-s2.0-S2666519026000622-main.md`: 10 runs, mean±std+IC, **boxplots de MSE** (estabilidade).
  - `kandemir2026explainable` — `1-s2.0-S2590174526003843-main.md`: 10 inicializações; t-test pareado **sobre as runs** (BiLSTM vs LSTM t=2,83, p=0,0475).
  - `xiang2026seqbinder` — `1-s2.0-S0360544226022462-main.md`: seeds fixas 42–46 com ±std.
  - `jiang2026data` — `1-s2.0-S096014812601075X-main.md`: 5 execuções idênticas por modelo.

### A4. Erro estratificado por regime (diurno/noturno) e/ou faixa de intensidade

Métricas por faixa operacional (ex.: 0–3 / 3–13 / >13 m/s) e por regime
diurno vs. noturno — o EDA do artigo já mostra ciclo semidiurno
(quedas 06h/18h), então a estratificação usa uma característica já
declarada.

- **Por quê:** alimenta a Discussão (M8) com **onde** o modelo erra;
  responde "para que serve" (rampas, despacho) sem custo de treino.
- **Custo:** baixo (pós-modelo, só predições + timestamps).
- **Encaixe:** Resultados (1 figura ou tabela) / Discussão.
- **Fontes:**
  - `xianfeng2026data` — `1-s2.0-S2213138826005576-main.md`, Table 3/Fig. A3: faixas 0–3/3–13/13–25 m/s, boxplots + t-test pareado (p<0,05) + MBE (viés) por faixa.
  - `aslam2026physics` — quartis de volatilidade Q1–Q4 (σ do horizonte); ganho relativo cresce com volatilidade (l.452–505).
  - `shafiullah2026advanced` — `smartcities-09-00059.md`: regime por *clear-sky index* (análogo: regime pelo ciclo diurno do vento) + perfis de viés por dia representativo.
  - `binrong2026whole` — subsets por volatilidade P25/P75/P90 (l.1365–1382).

### A5. Métricas de cauda por horizonte: P95, MaxAE (+ MedAE)

Além de MAE/MSE: percentil 95 e máximo do erro absoluto por horizonte;
MedAE como métrica robusta a outliers.

- **Por quê:** barato, uma linha por modelo; muda a leitura de
  "erro médio" para "pior caso" — relevante para vento/rampas.
- **Custo:** baixíssimo (pós-modelo).
- **Encaixe:** tabelas de Resultados (colunas extras).
- **Fontes:**
  - `xiang2026seqbinder` — P95 e MaxAE por horizonte.
  - `ehsan2000wind` — `ieee-icist2020-lstm-fulltext.md`: MedAE incluído pela robustez a outliers.
  - `binrong2026whole` — IQR dos resíduos como complemento ao DM (dispersão, não só média).

### A6. Tabela de custo computacional (parâmetros, tempo de treino, inferência/janela)

Já pedido pela Vanessa; padronizar como os artigos da área fazem.

- **Custo:** baixíssimo — `evaluation.json` já guarda `params`,
  `size_mb`, `eval_sec`, timing por janela.
- **Encaixe:** Modelagem/Resultados (tabela única).
- **Fontes:**
  - `liu2026robust` — `1-s2.0-S036054422601529X-main.md` (l.811–848): parâmetros, inferência/step, tempo de treino **no mesmo hardware**, justificado pelo ciclo de despacho.
  - `jiang2026data` — TT (treino) e PT (predição) de todos os modelos.
  - `aslam2026physics` — params/FLOPs/latência/memória + viabilidade de deploy.
  - `zhang2026novel` — `s11227-026-08589-0.md`: complexidade O(·) por módulo + runtime empírico.

### A7. Declarações anti-vazamento explícitas (escrita, custo zero)

Dizer no texto: min-max ajustado **só no treino**; suavização aplicada
**por parte** após o split (já feito, já dito — estender o padrão);
features auxiliares **defasadas** quando houver; tuning restrito ao
treino/validação.

- **Por quê:** na varredura, todo artigo sólido declara isso; é uma
  frase e blinda o protocolo.
- **Fontes:**
  - `venkateswaran2026hybrid` — `1-s2.0-S0167610526000772-main.md`: IQR (outliers) computado **somente no treino**, teste intacto; apenas variáveis *lagged* como auxiliares.
  - `cao2026probabilistic` — `1-s2.0-S0952197626015824-main.md` (l.306): min-max só do treino + verificação de completude pré-treino.
  - `binrong2026whole` — decomposição *rolling* causal ("no lookahead"); mapeia o viés da decomposição global pré-split como **lacuna generalizada na literatura** (Gap 1) — a suavização pós-split do SBSR está do lado certo, vale citar.
  - `jiang2026data` — normalização fit no treino aplicada ao teste.

### A8. Baselines com orçamento de ajuste igual e declarado

Descrever (e tratar) ARIMA/RF com o mesmo rigor de tuning dos modelos
próprios: grid de AIC para ordem do ARIMA, nº de árvores até estabilizar
MSE para RF, ranges tabelados.

- **Por quê:** elimina a objeção "baseline mal ajustado inflata o ganho
  do DL" — especialmente importante porque no SBSR **o ARIMA vence no
  agregado**.
- **Custo:** baixo (os artefatos já têm order (1,1,3) por AIC e RF 200
  árvores/janela 72 — falta declarar no texto).
- **Fontes:**
  - `leongomez2026multi` — `technologies-14-00238-v2.md`: Optuna/TPE aplicado **também aos baselines**, espaços explícitos.
  - `binrong2026whole` — otimizador uniforme para submodelos e baselines (l.445, 686).
  - `nascimentotransformer` — `A transformer-based deep neural network...md`: baseline LSTM tunado com Hyperband — "baseline tunado, não cru".
  - `trebing2020wind` — paridade de nº de parâmetros como critério de comparabilidade.
  - `gharehbaghi2026comparative` — `s41748-025-00714-y.md`: nº de árvores RF até estabilizar MSE.

---

## Tier B — valiosos, custo médio ou espaço disputado

### B1. Friedman multi-modelo (+ ranks médios) como Omniabus test

Se A3 (multi-seed) for adotado: Friedman sobre as runs de todas as
arquiteturas + post-hoc pareado (Wilcoxon/Dunn) com correção de Holm e
tamanho de efeito (Cohen's d / Cliff's δ) — o "teste de família" antes
dos pares.

- **Fontes:** `bendes2026bio` (Sec. 3.7/4.1 — Friedman+Wilcoxon+Holm+d/δ, 30 runs); `neshat2026physics` (15 modelos, ranks médios em radar); `leongomez2026multi` (Friedman χ² por dataset×métrica com ranks±DP); `zhao2027mcbformer` (ano de teste dividido em 24 subtarefas); `huang2024research` (Kruskal–Wallis + Mann–Whitney justificados pela não-normalidade do vento).
- **Encaixe:** Avaliação; só vale com múltiplas runs.

### B2. Teste de habilidade condicional (Giacomini–White)

Alternativa/upgrada ao DM que testa habilidade **condicional** (por
regime/informação), não só incondicional — 3 conjuntos de informação
(geral; não-extremos; extremos) no artigo-fonte.

- **Fonte:** `jie2026designing` — `1-s2.0-S1474034626003435-main.md`:
  GW (α=0,05, χ²) em 240 instâncias, com justificativa GW>DM e exceções
  relatadas (2/240).
- **Custo:** médio (implementação nova).

### B3. KSPA / ECDF das distribuições de erro

Comparar **distribuições inteiras** de erro entre pares (Kolmogorov–
Smirnov predictive accuracy) + curvas ECDF — complemento visual barato
ao DM.

- **Fonte:** `gomez2026predicting` — Seção 3.3 (Fig. 6a–b).

### B4. Métricas de forma/fase: MDTW, L_max/D_phase, acurácia direcional

O fenômeno tem ciclo semidiurno; defasagem de fase (prever o pico
atrasado) é falha típica que MAE/MSE disfarçam.

- **Fontes:** `sheng2026shape` — `Shape-Aware Seq2Seq...md`: MDTW
  (similaridade de forma), análise qualitativa de defasagem e
  sub/superestimação de picos/vales; `liu2026robust` — métricas
  **L_max** (lag da correlação cruzada máxima) e **D_phase** (desvio
  entre extremos) criadas para substituir inspeção visual subjetiva
  (l.775–789); `huang2024research` — DA (directional accuracy) entre 8
  critérios; `nascimentotransformer` — Fac2 + ρ de Pearson.
- **Custo:** baixo-médio (implementação das métricas).

### B5. Ablação de grupos de features (endógena / +meteo / +engenharia)

3–4 configs de entrada sob arquitetura e épocas fixas — isola o valor
**dos dados** sem tocar no modelo; casa com a tabela de dataset pedida
pela Vanessa.

- **Fontes:** `shafiullah2026advanced` — 3 configurações de input × 5 DL
  com 100 épocas fixas "para atribuir diferenças somente aos inputs";
  `chang2021ai` — `139961N.md` — 5 configs de canais (31c→4c) antes da
  modelagem final; `mengshi2020ultra` — LSTM¹ (só vento) vs LSTM³
  (+temp/pressão).
- **Custo:** médio (retreinos por config).

### B6. Ablação de perda formalizada (com significância)

As rodadas `report_8020` (ponderada) vs `report_8020_mseloss` (MSE) já
existem — tratar como ablação de perda com teste, em vez de confusão.

- **Fontes:** `sheng2026shape` — cadeia WYM-1/2/3/ITILDE isolando o
  efeito da perda shape-aware; `aslam2026physics` — 5 variantes de perda
  sob protocolo idêntico; `singh2026new` — cadeia incremental
  baseline→+DWT→+HPO.
- **Custo:** baixo (dados existem; falta o teste + 1 parágrafo).

### B7. Importância de features pós-modelo (PFI; SHAP se quiser ir além)

Permutation Feature Importance sobre o modelo treinado (10 repetições,
% de aumento de MSE por feature/grupo) — justifica a engenharia de
atributos declarada no artigo.

- **Fontes:** `chang2021ai` — PFI por canal, MSE médio de 10 repetições,
  contribuição % (U/V-10m/100m >80%); `kandemir2026explainable` —
  protocolo de amostragem SHAP (explainer 1000 treino / 200 teste),
  bee swarm + mean|SHAP| + waterfall; `wan2026multiscale` —
  `1-s2.0-S0378779626006851-main.md`: consistência de rankings SHAP entre
  turbinas quantificada por **Spearman** (>0,92) + robustez a ruído/
  missingness (aumento relativo de RMSE); `zhao2027mcbformer` — SHAP
  pós-hoc p/ variáveis meteorológicas; `binrong2026whole` — TreeSHAP→AOPC.
- **Custo:** PFI baixo; SHAP médio.

### B8. Diagnóstico de resíduos + atribuição de erro a breakpoints

Q-Q/histograma de resíduos; detectar mudanças estruturais na série de
teste (BFAST) e mostrar que o erro se concentra nelas — explica **por
que** erra, não só quanto.

- **Fontes:** `ehsan2000wind` — Q-Q + residual plots; `venkateswaran2026hybrid`
  — resíduos vs ajustado + histograma (viés sistemático);
  `aodi2026optimized` — `s40722-025-00446-1.md`: BFAST detecta
  breakpoints significativos a 95%, erro conectado a turning points;
  `cao2026probabilistic` — drift via soma cumulativa da quantile loss
  (Page–Hinckley), erro avaliado nas vizinhanças do drift.
- **Custo:** baixo-médio.

### B9. Stress test de volume de dados (treino truncado 40/60/80/100%)

Teste idêntico, treino truncado — quantifica se ~2 meses bastam;
transforma a limitação "série curta" em experimento.

- **Fonte:** `liu2026robust` — l.803–807, teste idêntico entre cenários.
- **Custo:** médio (retreinos).

---

## Tier C — anotados para trabalho futuro (não cabem no SBSR curto)

| Processo | Fonte(s) | Nota |
|---|---|---|
| Previsão probabilística: CRPS, PICP/MPIW, PIT/diagramas de confiabilidade, regressão quantílica | `cao2026probabilistic` (QL+CRPS+PIT+R^(1−2ν)); `nomenclature2023` (ensemble profundo: epistêmica+aleatórica, PICP/MPIW/CRPS); `nascimentotransformer` (quantílica q=0,95 calibrada na validação); `huang2024research` (FICP/FINAW/AWD, t-location-scale dos erros) | Próximo artigo/TCC longo |
| Generalização multi-sítio / zero-shot | `dong2026graphdynnet` (12 estações sem fine-tuning); `zhang2026novel` (3 datasets, splits dos papers originais); `geng2026long` | Depende de dados de outros sítios |
| Benchmark multi-site com hp unificados (sem tuning por dataset) | `jiang2026data` | Resultados "comparativos, não ótimos" |
| Sensibilidade ±30% dos hiperparâmetros | `shengxiang2022deep` (Table 8; grade ±30% passo 10%) — trials do Optuna aproximam | Apêndice/relatório |
| Métricas por estação/safra | `jiang2026data` (4 estações, heatmaps); `wang2026adaptive` (derrota no outono admitida) | Série de 2 meses não cobre estações |
| Exclusão de amostras ~0 do MAPE | `binrong2026whole` (exclui <0,5 m/s) | Só se MAPE for adotado |
| Verificação física da predição (log-law, curva de potência) | `wang2026adaptive` (log-law u*, z0); `huang2024research` (correção densidade do ar); `neshat2026physics` (consistência direcional de esteiras) | Requer metadados de parque |
| Honestidade de relato como prática transversal | `zhang2026wind` (R² negativos mantidos); `chang2021ai` (razões incluem pioras, até −8856%); `binrong2026whole`/`wang2026adaptive` (DM não-significativo relatado); `bendes2026bio` (skill negativo publicado) | Filosofia; custo zero |

---

## Roadmap sugerido (menor esforço → maior defesa)

1. **Quase grátis (pós-modelo, 1–2 dias):** A1 (DM+Harvey) · A2 (skill) ·
   A5 (P95/MaxAE) · A6 (custo) · A7 (declarações) · A8 (baselines
   declarados) — resolves M3 parcial, M4 e itens da Vanessa.
2. **Maior ganho científico (~2 h GPU):** A3 (5 seeds ± IC) → destrava
   B1 (Friedman) e o t-test pareado sobre runs.
3. **Alimenta a Discussão (pós-modelo):** A4 (regime/intensidade) · B4
   (fase/MDTW) · B8 (breakpoints).
4. **Se sobrar espaço/paciência:** B5/B6 (ablações) · B7 (PFI).

## Correções pendentes no catálogo (detectadas na varredura)

- `image2026quality` — título errado no índice ("Image Quality
  Enhancement…"); conteúdo real: **Mezaache et al., ICEEMDAN wind speed,
  IJE 2026** (`10.5829_ije.2026.39.02b.03.md`).
- `nomenclature2023` aponta o markdown de Quinones; a entrada
  `quinonestowards` fica sem link de texto (duplicidade).
- `jiang2026data` tem duplicata `jiang1` no índice.
- Stubs (abstract-only, sem full text): `ieee-ssci2020-multidimcnn.md` e
  os 5 stubs IEEE do lote 3 — sem achados extraíveis.
