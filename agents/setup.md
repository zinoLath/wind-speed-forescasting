# agents/setup.md — ambiente e comandos-chave

> Base canônica: `findings/knowledge/ambiente-execucao.md`,
> `pipeline-estagios.md` (skill `canonical-findings`).

## Setup

```bash
./setup_environment.sh        # cria .venv (Python 3.10), instala deps
source .venv/bin/activate
source ./scripts/tf_gpu_env.sh  # OBRIGATÓRIO no Arch Linux com NVIDIA GPU
```

A venv usa Python 3.10 especificamente. TensorFlow 2.21 com CUDA via pip.
`markdown` (render dos relatórios) e `pymupdf` (metadados de PDF no
knowledge) estão no ambiente; `markdown>=3.5` está no requirements.txt.

## Comandos-chave

```bash
# Pipeline em estágios (optuna -> train -> evaluate -> impute -> walkforward)
python pipeline/pipeline.py                  # pipeline completo
python pipeline/pipeline.py --stage train evaluate   # só alguns estágios
python pipeline/step_train.py                # um estágio isolado
# Estágios aceitam --config <arquivo> (mesclado sobre
# pipeline/pipeline.default.json). Valores compartilhados na seção "common"
# do config; valores de estágio vencem. Ver pipeline/README.md.

# Testes de regressão (scripts standalone, sem pytest)
python tests/test_wrappers_build.py              # build+predict de todos wrappers
python tests/test_create_sequences_equivalence.py  # vetorizado vs loop oráculo

# Dashboard de treino no terminal (nvtop-style; rode em outro pane do tmux
# enquanto qualquer wrapper.fit() treina). Cada fit acrescenta métricas por
# época em pipeline/tmp/train_dashboard/<run>.jsonl (TRAIN_DASHBOARD=0
# desliga; TRAIN_DASHBOARD_DIR muda a pasta).
python scripts/train_dashboard.py               # segue o run ativo mais novo
python scripts/train_dashboard.py --list        # lista runs (ativo/parado)
python scripts/train_dashboard.py --run tcn     # fixa um run por substring
python scripts/train_dashboard.py --once        # snapshot estático

# A/B: alvo cru (ws100) vs wavelet-denoised
python tests/ab_raw_vs_wavelet.py --epochs 15

# Benchmark de orçamento fixo dos wrappers
python tests/benchmark_transformer.py --wrapper lstm_bi --epochs 40

# Driver Optuna legado (referência; pipeline/step_optuna.py é o primário)
python tests/optuna_all_hyperparameters.py --n-trials 100 --epochs 60

# Imputer de colunas focais tunado por Optuna
python tests/compare_rf_lgbm_impute_detailed.py --optuna

# Fatias de período completo (data/series_list/)
python src/periods.py --min-length 108

# Relatórios: ver agents/reports.md (hub, builder, servidor)
# Artigos/Overleaf: ver agents/articles.md (sync, commit point)
```

Treinos LiDAR (protocolo de reamostragem 80/20 e variantes) — runbook
completo na skill **lidar-retrain**:

```bash
python scripts/lidar_7525_train.py --train-ratio 0.80   # 80/20 -> trained_8020
python scripts/lidar_7525_train.py --train-ratio 0.75   # 75/25 -> trained_7525
python scripts/lidar_8020_mseloss_train.py              # ablação de loss
python scripts/lidar_8020_startweight_train.py          # loss invertida
# Regenerar relatório de resultados (figuras + evaluation.json, ~5 min GPU):
python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020 \
    --out-dir data/results/report_8020 --no-baselines
```

## Lint / testes

Não há linter/formatter/type-checker/test runner configurados nem CI.
Para lint durante o desenvolvimento, `pyflakes` e `radon` são úteis (não
estão no requirements.txt). Os arquivos em `tests/` NÃO são pytest/unittest —
são scripts standalone executados com `python` direto.
