---
title: Armadilhas de execução (gotchas)
area: processo
updated: 2026-10-19
---

# Armadilhas de execução (gotchas)

Consolidado de `agents/gotchas.md` — os pontos que já quebraram algo antes:

- **`src/common.py` sem efeitos colaterais de import**: TF é configurado
  via `common.setup_tensorflow()` ANTES dos imports de modelo (lazy). Não
  adicionar import de keras/tf no nível de módulo.
- **Bootstrap sys.path**: arquivos sob `pipeline/` e `reports/*/report.py`
  precisam inserir a raiz do projeto em `sys.path[0]` antes de
  `from pipeline import ...` (senão `pipeline/pipeline.py` sombreia o
  pacote).
- **SQLite do Optuna**: estudos resumem com `load_if_exists=True`; apagar
  o `.db` (`pipeline/tmp/optuna*/<name>/optuna.db`) para recomeçar.
- **pgrep/pkill**: padrão não pode casar com o próprio shell (usar
  colunas: `lidar_[7]525_train`).
- **Background jobs**: `setsid ... < /dev/null &` com `disown`;
  `nohup ... &` morre quando a sessão do agente encerra com timeout.
- **Overleaf**: pushes best-effort; `paper/.overleaf-clone/` é cache.
- **Gerados, nunca editar**: `reports/_site/`, `reports/state.json`,
  `articles/knowledge/index.md`, `articles/knowledge/references.bib`,
  `notebooks/colab_round2/`.
- **`OLD/`**: código aposentado (mantido, não mantido) — testes em
  `OLD/tests/`, wrappers em `OLD/models/`, notebooks legados.
- **Métricas históricas** (MAE 0.1589): protocolo antigo com vazamento —
  números honestos em `reports/` e `docs/`.

## Fontes

- `agents/gotchas.md`; skills `lidar-retrain`/`optuna-colab` (regras).
