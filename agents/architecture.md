# agents/architecture.md — arquitetura dos modelos

> Base canônica: `findings/knowledge/visao-geral.md`,
> `wrappers-arquiteturas.md`, `losses-decoder.md` (skill `canonical-findings`).

Seq2Seq wind speed forecasting (Keras/TensorFlow). Todos os modelos herdam
de `Seq2SeqWrapper` (`src/models/seq2seq_wrapper.py`), que provê
`prepare()`, `build()`, `fit()`, `predict()` e `rolling_forecast()` em lote
(janelas são funções puras do histórico conhecido; um único `model.predict`).

Registro de modelos em `src/common.py:WRAPPERS` com imports lazy; chaves:
`lstm`, `lstm_bi`, `gru`, `gru_bi`, `lstm_cnn`, `tcn`, `tcn_bi`,
`tcn_lstm`, `transformer` (Pre-LN). `validate_wrapper_names()` falha rápido
em `.name` duplicados (names viram nomes de diretórios de resultados).

- A loss de treino é a **MSE ponderada por horizonte**
  (`src/models/losses.py`), favorecendo horizontes longos; a chave `loss`
  do config é mantida por compatibilidade, mas os wrappers fixam a loss.
- Entradas do decoder usam a convenção **direta (válida em deployment)** —
  último alvo observado + fração de horizonte, sem teacher forcing e sem
  vazamento — então val_loss/early stopping e o objetivo do Optuna (MSE
  ponderada por horizonte em blocos de validação) medem o comportamento
  real de deployment.
- Métricas de cabeçalho comparam contra o `ws100` BRUTO; métricas contra o
  alvo denoised ficam em colunas secundárias `*_denoised`.
- Espaço de busca do TCN é compartilhado e reduzido:
  `src/models/tcn_hp.py:tcn_hyperparameters()` — um único `filters` alimenta
  encoder e decoder, sem dropout pós-bloco, kernels pequenos (2–3) e
  dilatações rasas. O TCN bidirecional usa mapeamento direto
  encoder→decoder (sem pooling de contexto).

`src/common.py` é a fonte única de verdade para setup de GPU, carga/split do
dataset, adaptadores de hp (`FixedHyperParameters`, `OptunaHyperParameters`),
métricas e predição em lote (`predict_all_horizons`). Os estágios do pipeline
o importam; lógica compartilhada nova vai lá.

Denoise wavelet: `src/utils.py:wavelet_denoising()` — wavelet `sym18`,
nível 2 por padrão.

Imputação `src/impute/`: `rf.py`, `lightgbm.py`, `knn.py`, `kalman.py` +
`base.py` (loaders, sin/cos de direção, validação por injeção de lacunas,
métricas, plots). Registrados em `pipeline/step_impute.py:METHODS`.
