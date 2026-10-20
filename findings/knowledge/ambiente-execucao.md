---
title: Ambiente e execução (venv, GPU, testes)
area: infra
updated: 2026-10-19
---

# Ambiente e execução

- `./setup_environment.sh` cria `.venv` com **Python 3.10** e instala
  `requirements.txt` (TF >= 2.21 com CUDA via pip, optuna >= 4,
  keras-tcn >= 3.5.6, lightgbm >= 4, statsmodels >= 0.14, PyWavelets,
  plotext, markdown, jupyterlab).
- **Obrigatório no Arch Linux + NVIDIA**: `source ./scripts/tf_gpu_env.sh`
  (existe variante `.fish`) antes de qualquer treino/eval em GPU.
- Sem CI, sem linter/formatter/type-checker configurados; `pyflakes` e
  `radon` úteis no dev (fora do requirements).
- `tests/` **não é pytest**: scripts standalone (`python
  tests/test_wrappers_build.py` etc.). Principais:
  `test_wrappers_build.py` (build+predict de todos wrappers),
  `test_create_sequences_equivalence.py` (vetorizado vs loop oráculo),
  `ab_raw_vs_wavelet.py`, `benchmark_transformer.py`.
- Dashboard de treino no terminal: `python scripts/train_dashboard.py`
  (segue o run ativo; `--list`, `--run <substr>`, `--once`); grava por época
  em `pipeline/tmp/train_dashboard/<run>.jsonl` (`TRAIN_DASHBOARD=0`
  desliga).
- Jobs longos em background: `setsid ... < /dev/null & disown` (o
  `nohup ... &` simples morre com o timeout da sessão do agente). Ao matar
  processos, padrão pgrep não pode casar com o próprio shell
  (ex.: `lidar_[7]525_train`).
- Notebooks do TCC: `./run_notebook.sh`.

## Fontes

- `agents/setup.md`; `agents/gotchas.md`; `requirements.txt`;
  `setup_environment.sh`; `scripts/tf_gpu_env.sh`.
