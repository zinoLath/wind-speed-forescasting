---
title: Armadilhas de execução (gotchas)
area: processo
updated: 2026-10-21
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

## Kaggle: corrida de versionamento do dataset

Kernels restauram a **última versão processada** do Dataset no momento em
que rodam. `datasets version` retorna antes do processamento assíncrono
terminar — pushar o kernel na sequência pode restaurar a versão ANTERIOR
(caso real: `can't open file .../itildeq_terms_search_train.py`, duas
vezes, em 21/10/2026). **Fluxo correto**: versionar → esperar o
processamento → pushar kernels. Conferir pelo `creationDate` de um
arquivo qualquer da listagem (`kaggle datasets files <id>`); com
`--dir-mode zip` a listagem mostra os arquivos **extraídos** do zip (não
o nome do zip) e é **paginada** — grep de caminho pode falhar por
paginação (use a data de criação, em **UTC**).

## Patch de coeficientes ITILDE-Q: binding de import

Os wrappers fazem `from .losses import itilde_q` **no import** — patchar
`losses.itilde_q` DEPOIS não afeta o wrapper (binding antigo). Para variar
α/β/γ é preciso patchar o **módulo do wrapper**
(`src.models.s2s_lstm_wrapper.itilde_q = ...`, como faz
`scripts/itildeq_terms_search_train.py:patch_lstm_itildeq`) ou patchar
ANTES do primeiro import do wrapper. **Impacto nos resultados até
21/10/2026: nenhum** — os args (0,3/0,5/0,1) eram iguais aos defaults da
perda, então o binding "errado" produzia os mesmos valores. Busca de
coeficientes em `scripts/itildeq_terms_kaggle.py` (Kaggle).

## Fontes

- `agents/gotchas.md`; skills `lidar-retrain`/`optuna-colab` (regras).
