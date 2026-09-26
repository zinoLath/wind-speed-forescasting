# Pipeline de Treinamento e Avaliação

Pipeline em estágios para otimizar, treinar, avaliar, imputar e validar por
walk-forward os modelos Seq2Seq de previsão de velocidade do vento.

Cada estágio é um script independente que lê os artefatos salvos pelos
estágios anteriores e produz os seus próprios artefatos em
`pipeline/tmp/` (sempre sobrescritos na execução seguinte). Os estágios
podem ser executados em sequência com um único comando ou isoladamente.

## Estrutura

```
pipeline/
├── pipeline.py             # orquestrador: roda os estágios em sequência
├── pipeline.default.json   # valores padrão de todos os parâmetros
├── pipeline.json           # configuração explícita do usuário
├── common.py               # helpers compartilhados (GPU, dados, métricas, wrappers)
├── config.py               # leitura/merge das configurações
├── step_optuna.py          # estágio 1: otimização de hiperparâmetros
├── step_train.py           # estágio 2: treinamento do melhor modelo
├── step_evaluate.py        # estágio 3: avaliação do modelo salvo
├── step_impute.py          # estágio 4: imputação de dados
├── step_walkforward.py     # estágio 5: validação walk-forward
├── tmp/                    # artefatos de cada estágio (gitignored, sobrescritos)
└── README.md
```

A imputação é implementada em `src/impute/`:

```
src/impute/
├── __init__.py
├── base.py                 # helpers comuns (dados, direções circulares, gaps, métricas, gráficos)
├── rf.py                   # imputação com Random Forest (IterativeImputer)
├── lightgbm.py             # imputação com LightGBM (IterativeImputer)
├── knn.py                  # imputação com KNNImputer
└── kalman.py               # imputação coluna-a-coluna (interpolação + Kalman)
```

## Configuração

Toda a configuração fica em `pipeline/pipeline.json` (o arquivo explícito do
usuário) e `pipeline/pipeline.default.json` (os padrões). O `pipeline.json` é
sobreposto ao default, então pode declarar apenas o que mudar. Se o arquivo do
usuário não existir, apenas os padrões são usados.

Os caminhos (`paths`) são relativos à raiz do projeto.

### Seção `common`

Valores compartilhados pelos estágos vivem em `"common"` e são herdados por
cada estágio cujo valor não esteja explicitamente definido — assim
`input_steps`, `output_steps`, `target_col`, `denoise`, `denoise_level`,
`seed`, `batch_size`, `loss`, `patience`, `train_ratio`/`val_ratio` e
`params_source`/`hyperparameters` são declarados uma única vez. A precedência
é sempre: **valor do estágio > valor do usuário em `common` > default do
estágio > default em `common`**.

Exemplo — trocar a seed de tudo e a paciência apenas do walk-forward:

```json
{
  "common": {"seed": 7},
  "walkforward": {"patience": 20}
}
```

## Como executar

### Pipeline completa (em sequência)

```bash
source .venv/bin/activate
source ./scripts/tf_gpu_env.sh   # Arch Linux + NVIDIA

python pipeline/pipeline.py
```

### Apenas alguns estágios

```bash
python pipeline/pipeline.py --stage train evaluate
```

### Estágios individuais (standalone)

Cada estágio pode ser executado sozinho e reutiliza os artefatos salvos nos
`tmp/`:

```bash
python pipeline/step_optuna.py
python pipeline/step_train.py
python pipeline/step_evaluate.py
python pipeline/step_impute.py
python pipeline/step_walkforward.py
```

Todos aceitam `--config <caminho>` para usar um arquivo de configuração
alternativo:

```bash
python pipeline/step_walkforward.py --config pipeline/pipeline.json
```

A rodada distribuída de Optuna (`pipeline/pipeline.optuna_round2.json`,
estudo `round2` com sqlite por wrapper em `pipeline/tmp/optuna_round2/`) é
orquestrada por `scripts/colab_round2.py` e `scripts/colab_monitor.py` —
wrappers pesados em notebooks do Google Colab, leves na GPU local. O runbook
completo (incluindo o playbook de erros) está na skill `optuna-colab`
(`.opencode/skills/optuna-colab/SKILL.md`) e resumido no `AGENTS.md`.

## Estágios

### 1. `optuna` — otimização de hiperparâmetros

Para cada wrapper configurado em `optuna.wrappers`, roda `n_trials` trials e
salva o melhor resultado em `pipeline/tmp/optuna/<WrapperName>/`:

- `best_trial.json` — hiperparâmetros vencedores (objetivo: **val RMSE**
  medido com o decoder de inferência, punindo trial com erros grandes)
- `optuna.db` — storage sqlite do estudo (permite retomar buscas)
- `trials.csv` — tabela com todos os trials

### 2. `train` — treinamento do melhor modelo

Treina o wrapper `train.wrapper` com os hiperparâmetros de
`train.params_source`:

- `optuna`: usa o melhor resultado salvo pelo estágio 1
- `explicit`: usa `train.hyperparameters`
- `default`: usa hiperparâmetros embutidos (fallback)

A função de perda é `train.loss` (padrão `mse`, que pune erros grandes;
alternativas: `mae`, `huber`) e fica registrada no `model.json`.

Salva em `models/best_model/`:

- `model.keras` — o modelo treinado
- `model.json` — contexto completo do modelo (classe, hiperparâmetros,
  loss, timestamp de treinamento, épocas rodadas/solicitadas, tempo de
  treino, loss/val_loss finais, split do dataset)

Ambos também são copiados para `pipeline/tmp/` para os estágios seguintes.

### 3. `evaluate` — avaliação do modelo salvo

Carrega `pipeline/tmp/model.keras` + `model.json` por padrão. Para avaliar
outra versão do modelo, defina `evaluate.model_path` / `evaluate.model_json_path`
no config (opção de *overload*).

Gera, em `pipeline/tmp/evaluate/<WrapperName>/`:

- `predictions_all_horizons.csv` — previsão de todos os horizontes (origin,
  timestamp, horizon, actual, actual_raw, predicted, persistence)
- `predictions_rolling.csv` — forecast rolante no horizonte do modelo
- `per_horizon_metrics.csv` — MAE/RMSE/R² por horizonte
- `metrics.json` — todas as métricas agregadas (inclui baseline de persistência)
- gráficos: série temporal, métricas por horizonte, histograma de erros e
  scatter predicted vs actual

Todos os dados dos gráficos ficam disponíveis em CSV/JSON, permitindo
regenerar ou reagregar gráficos sem reexecutar o código.

**Convenções de métricas:** as métricas principais comparam as previsões
contra o **ws100 cru** (sem suavização wavelet), evitando que o smoothing
mascare o erro; as métricas contra a série denoised ficam como colunas
secundárias (`*_denoised`). O forecast rolante usa apenas janelas de
histórico conhecido (`[i-input_steps, i)`), sem valores futuros.

### 4. `impute` — imputação de dados

Para cada método em `impute.methods`, imputa `data/wind_data.csv` e valida
injetando gaps artificiais. Salva em `data/impute/<metodo>/`:

- `<metodo>_imputed.csv` — dataset imputado (com flag `imputed`)
- `detailed.csv` — erros por amostra (por gap/posição/coluna)
- `summary.csv` — métricas agregadas por coluna e tamanho de gap
- `metrics.json` — métricas globais + configuração
- `metrics_by_gap_length.png`, `mae_by_column.png`, `mae_heatmap.png`

Uma cópia é espelhada em `pipeline/tmp/imputed/<metodo>/` para o
walk-forward consumir.

Para adicionar um novo método, crie um módulo em `src/impute/` com uma função
`impute_dataframe(df, ...)` e registre-o em `pipeline/step_impute.py:METHODS`.

### 5. `walkforward` — validação por walk-forward

Simulação com re-treino periódico: para cada janela o modelo é treinado nos
últimos `train_window_days` (padrão 60 dias = 2 meses) e avaliado nos
`test_window_days` seguintes (padrão 30 dias = 1 mês). A janela avança por
`step_days`; quando `step_days` é menor que `test_window_days` os períodos de
avaliação se sobrepõem (a janela pode iterar sobre pontos já avaliados).

Por padrão usa o dataset imputado mais recente em `pipeline/tmp/imputed/`; se
não existir, usa `data/dataset.csv`. Pode ser forçado com `walkforward.input`.

Salva em `pipeline/tmp/walkforward/<WrapperName>/`:

- `predictions_window_<NNNN>.csv` — previsões por janela
- `predictions_all.csv` — todas as janelas concatenadas
- `metrics.csv` — métricas por janela
- `metrics.json` — configuração + métricas globais e por horizonte

`walkforward.max_windows` limita o número de janelas (útil para testes).

## Dependências entre estágios

```
optuna ──> train ──> evaluate
             │
impute ─────┴──> walkforward
```

- `optuna` produz os hiperparâmetros que `train` e `walkforward` consomem.
- `train` produz o modelo que `evaluate` avalia.
- `impute` produz o dataset que `walkforward` usa por padrão.
- Cada estágio funciona sozinho desde que os artefatos que ele consome já
  existam (ou que `params_source` esteja em `default`/`explicit`).

## `pipeline/tmp/`

Contém tudo que cada estágio retorna, sempre sobrescrito na próxima execução:

```
pipeline/tmp/
├── optuna/<WrapperName>/best_trial.json, trials.csv
├── model.keras, model.json          # último modelo treinado
├── evaluate/<WrapperName>/...       # avaliação do último modelo
├── imputed/<metodo>/...             # último dataset imputado + validação
├── walkforward/<WrapperName>/...    # última validação walk-forward
└── pipeline_run.json                # relatório da última execução
```

## Notas de qualidade

O código foi escrito para ser direto e legível, sem abstrações desnecessárias:

- importações do TensorFlow/Keras são locais a cada estágio (nada de efeitos
  colaterais no import dos módulos);
- lógica compartilhada entre estágios vive em `pipeline/common.py`
  (e.g. `predict_all_horizons` é usado por evaluate e walkforward);
- métricas e helpers de imputação vivem em `src/impute/base.py`;
- complexidade ciclomática média **A** (radon), sem função acima do grau B.