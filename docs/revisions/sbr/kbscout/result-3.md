Os 5 arquivos IEEE de aproximadamente 2KB são provavelmente stubs. Vou lê-los diretamente e usar grep nos arquivos maiores em busca de cabeçalhos de seção/marcadores de metodologia em paralelo.
Os cinco arquivos IEEE são resumos (sem texto completo) — extrairei o que estiver disponível a partir deles. Agora, lendo as seções de metodologia/resultados dos artigos completos em paralelo.
One last check on the remaining sections of the jiang2026data review (post-4.5):
Varredura concluída. Achados metodológicos (sem arquiteturas/híbridos) por arquivo, com chave bib localizada em `articles/knowledge/index.md`:

---

**1. `1-s2.0-S019689042600943X-main.md`** — bib: `neshat2026physics`
- Remoção de outliers physics-informed: envelope de quantis por bin de velocidade (q5/q95) + restrição de monotonicidade na região sub-rated da curva de potência (Algorithm 1 "OR-QEMC"); retém low-power@high-wind legítimo (curtailment)
- Ablação componente-a-componente (grafo LLM, encoding contextual, multi-scale temporal, pré-processamento físico) com degradação % em MAE/TPC
- Validação física pós-modelo: consistência direcional das arestas do grafo dinâmico vs. direção esperada de propagação de esteira (wake), sem avaliar acurácia
- Análise de sensibilidade/robustez por regime de direção do vento (8 setores; N por setor) e por velocidade×direção combinados
- Teste de Friedman (não-paramétrico): 15 modelos ranqueados em 4 métricas (R, MAE, ARBS, TPC) por horizonte; ranks médios em radar plot
- Análise de complexidade/runtime: s por matriz de adjacência (GPT), s/época, O(N·k) esparsa vs O(N²); métricas compostas (TPC, ARBS, EVS, MedAE) além de MAE/RMSE

**2. `1-s2.0-S096014812601075X-main.md`** — bib: `jiang2026data` (duplicata `jiang1`)
- Benchmark multi-site (4 datasets: offshore/onshore China, deepwater Noruega, GEFCom2014) com hiperparâmetros UNIFICADOS, sem tuning por dataset — resultados declarados como comparativos, não ótimos por modelo
- Normalização fit apenas no treino, aplicada ao teste (anti-vazamento explícito)
- 5 execuções independentes por modelo, mesmo split e pré-processamento
- Teste Diebold-Mariano com modelo benchmark (BDWPT+GPR) para validar melhorias
- Multi-step (3h/6h) com reduções % médias e cross-site (faixas 65–83% / 48–69%)
- Análise sazonal: split por 4 estações, mesmas configs, heatmaps
- Complexidade: tempo de treino (TT) e predição (PT) de 40 modelos
- Limitações de cobertura climática declaradas (sem tufões; >25 m/s fora do teste)

**3. `technologies-14-00238-v2.md`** — bib: `leongomez2026multi`
- Partição cronológica 70/10/20 (treino/val/teste) para prevenir vazamento; sliding window stride 1; sem data augmentation (atribuição de ganhos ao método)
- Fair tuning: Bayesian Optimization TPE (Optuna) aplicado também aos baselines Transformer, com espaços de busca explícitos; objetivo = val MSE
- Benchmark sintético controlado para isolar o efeito do encoder espectral
- Friedman χ² por dataset×métrica com ranks médios±DP por horizonte + p-valores (todos ≪0.05)
- Pré-benchmark: KDEs e inspeção de não-estacionariedade por partição para justificar protocolo
- Seção de limitações com custo computacional da busca TPE

**4. `1-s2.0-S2666519026000622-main.md`** — bib: `singh2026new`
- 10 runs independentes por modelo; métricas reportadas como mean±std + IC 95%
- Teste DM completo vs. 5 baselines em 3 horizontes: estatística, p-valor, magnitude do erro, best/worst/mean MSE e % de melhora
- Boxplots de MSE sobre as 10 runs (mediana + IQR = estabilidade)
- Ablação sucessiva DWT (denoising) e ARSO (HPO): baseline → +DWT → +ARSO
- Feature selection via correlação de Pearson + boxplots para detecção de outliers
- Split 80/20 com validação interna 90/10; min-max; DWT db3 nível 7 justificado por literatura

**5. `1-s2.0-S0360544226022462-main.md`** — bib: `xiang2026seqbinder`
- 5 seeds fixas (42–46) para estimar incerteza; métricas com ±std
- Dois datasets de fazendas a 1.319 km com climas distintos (generalização de domínio), split 80/20
- Subconjunto de eventos extremos alinhado 1:1 por timestamp (datas específicas listadas; 68/46 amostras), avaliado separadamente com MAE/MSE
- Métricas de cauda: P95 e MaxAE por horizonte — além de médias
- Violin plots de distribuição de erro (concentração no zero, IQR, caudas)
- NRE (erro relativa normalizada) para padronizar comparação entre datasets

**6. `1-s2.0-S0957417426030617-main.md`** — bib: `zhao2027mcbformer`
- Friedman test com o ano de teste dividido em 24 subtarefas; RMSE mean±std, estatística e p por modelo vs. proposto
- Ablação MFTE/CBformer/BiTCN com % de melhora relativa (P_RMSE, P_MAE…) reportada
- Experimento de generalização externa (fazenda em Xinjiang, 15 min) com settings idênticos
- Métrica composta proposta HEDS (fusão harmônica de NDE+RE com termo de CV) para avaliar escala, proporção e dinâmica simultaneamente
- Prática de relato: interpretar múltiplas métricas em paralelo — MAPE sozinho gera ilusão em regiões de baixa potência; priorizar RMSE/R²
- SHAP pós-hoc para quantificar contribuição das variáveis meteorológicas
- Análise de complexidade FLOPs/parâmetros variando profundidade, dim e comprimento de entrada

**7. `1-s2.0-S2590174526003843-main.md`** — bib: `kandemir2026explainable`
- 10 inicializações aleatórias por arquitetura; RMSE/MAE/R² mean±std
- Paired-sample t-test sobre RMSEs das 10 runs (BiLSTM vs LSTM: t=2.83, p=0.0475)
- SHAP em 3 visões: bee swarm (local), mean|SHAP| (global), waterfall por instância com decomposição aditiva ŷ=E[f]+Σφ
- Protocolo de amostragem SHAP: explainer com 1000 amostras de treino, explicação em 200 de teste
- Matrizes de correlação Kendall + Pearson + Spearman em paralelo (consistência de ranking)
- Randomized search de hiperparâmetros com top-5 configurações reportadas por val RMSE

**8. `A transformer-based deep neural network...ScienceDirect.md`** — bib: `nascimentotransformer` (a entrada de texto no índice usa `12023introduction`)
- Split temporal explícito: 550 amostras treino (70/30 validação) + teste de 194 h nunca visto; 9 séries (3 torres × 3 alturas)
- Early stopping patience=30; min-max scaling
- HPO do baseline LSTM com Hyperband (Keras Tuner) — baseline tunado, não "cru"
- Teste de Wilcoxon signed-rank (α=0.05) entre modelos
- Regressão quantílica (q=0.95) para intervalos de predição, calibrada na validação e aplicada ao teste
- Métricas NMSE, ρ (Pearson) e Fac2 (fração dentro de fator 2) além de MAE/MSE/RMSE
- Baseline de persistência incluído na comparação
- Relato de tempo de treino E inferência (quantitativo + qualitativo)
- Ablação incremental: 6 simulações com níveis de decomposição wavelet 1–5

**9. `ultrashort-morphology-lstm.md`** — bib: `mengshi2020ultra`
- Design fatorial modelo×decomposição: LS-SVM/BPNN/LSTM cada um com e sem MM (MM-X vs X) — isola o efeito da decomposição
- Comparação direta de métodos de decomposição MM vs WT (db3) sob protocolo idêntico
- Ablação de entradas: LSTM¹ (só vento) vs LSTM³ (+temp/pressão) vs MM-LSTM⁴
- Tabela MAE/RMSE para 12 horizontes (2–24 passos à frente) — relato horizon-wise completo
- Split 80/20 treino/teste; sem teste estatístico (prática de 2020)

**10. `1-s2.0-S0167610526000772-main.md`** — bib: `venkateswaran2026hybrid`
- IQR outlier removal (Q1−1.5·IQR, Q3+1.5·IQR) computado SOMENTE no treino; teste intacto — evita vazamento e distribution shift
- Prevenção de mapeamento estático: apenas valores defasados (lagged) das features SCADA auxiliares; nenhuma variável do instante alvo
- Diagnóstico pós-modelo: residual plots (balanceamento em torno de zero) + histogramas de erro (viés sistemático)
- Curvas treino/validação para verificar overfitting
- Comparação com 8 baselines em 4 horizontes (1/2/3/6 h)
- Split cronológico 80/20, janela 24, min-max

**11. `document.md`** — bib: `fennane2026enhancing`
- Callbacks: EarlyStopping (patience=5, restore_best_weights=True) + ReduceLROnPlateau (patience=3, factor=0.5)
- Validação como split interno de 10% sobre treino+validação combinados
- Checagens de robustez declaradas: cross-validation + teste em dataset de outro site (anti-overfitting/viés de dataset)
- Tabela comparativa com literatura 2019–2025 (RMSE/MAE/R² por estudo) como prática de posicionamento
- Regression plots predito×observado como avaliação visual global; sem testes estatísticos

**12. `ieee-ssci2020-multidimcnn-fulltext.md`** — bib: `trebing2020wind`
- Paridade de capacidade: nº de parâmetros como critério principal de comparabilidade entre os 5 modelos (em vez de otimização individual) — limitação declarada explicitamente
- Baseline de persistência em todos os experimentos e horizontes
- Relato granular: MAE/MSE por horizonte E por cidade-alvo, além da média
- Min-max scaler treinado apenas no treino
- Reprodutibilidade: código (GitHub) e dados públicos
- Contraste de resultados entre 2 datasets (DK/NL) como teste de generalização

**13–17. Stubs IEEE (abstract-only; texto completo não disponível)**
- `ieee-access2020-dwpt-blstm.md` — bib: `dolatabadi2020hybrid` — 2 estudos de caso (NREL M2 Colorado + Edmonton, Canadá) vs. shallow e SOTA; estado de espaço reconstruído como input
- `ieee-access2020-convlstm-bpnn.md` — bib: `chen2020short` — 3 datasets, comparação com 5 modelos; MAPE/RMSE como matriz de avaliação
- `ieee-icue2020-convlstm.md` — bib: `nahid2020very` — combina previsão pontual + probabilística; 2,5 anos de dados em 3 casos; RMSE/MAPE/MAE/correlação; baselines CNN e LSTM
- `ieee-epec2020-bayesian-ann.md` — bib: `rahaman2020bayesian` — seleção de hiperparâmetros por Bayesian optimization; comparado a SVM/RF/decision tree em MAE/RMSE
- `ieee-npsc2020-eemd-lstm.md` — bib: `santhosh2020ensemble` — decomposição aplicada só ao treino; validação com dados reais vs. benchmarks por erros estatísticos

---

Padrões transversais úteis ao SBSR: testes de significância (Friedman multi-modelo; DM e Wilcoxon par-a-par; paired t-test), multi-run com mean±std/IC (5–10 seeds), splits cronológicos com scaler/outlier-fit só no treino, ablações de pré-processamento separadas de arquitetura, avaliação por regime/extremos/sazonalidade, métricas de cauda (P95/MaxAE) e custo computacional (TT/PT, FLOPs).

LOTE 3 CONCLUIDO - 17 arquivos varridos, 82 achados.
