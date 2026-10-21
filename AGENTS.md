# AGENTS.md

Seq2Seq wind speed forecasting (Keras/TensorFlow) em dados LiDAR. Este
arquivo é o **índice do contexto**; os detalhes vivem em `agents/*.md`
(leia o arquivo correspondente antes de trabalhar em cada área).

## Mapa do contexto

| Arquivo | Conteúdo |
|---|---|
| `agents/setup.md` | venv Python 3.10 + GPU, comandos-chave (pipeline, testes, treinos LiDAR), lint |
| `agents/architecture.md` | wrappers seq2seq (`src/common.py:WRAPPERS`), losses, decoder direto, TCN hp, imputação |
| `agents/data.md` | datasets, saídas do pipeline, `data/results/report*`, dashboard jsonl |
| `agents/reports.md` | **framework `reports/`**: hub, componentes com popup LaTeX (steam/sbsr), builder, servidor, regenerar/retreinar — skill `reports-hub` |
| `agents/articles.md` | **`articles/`**: steam/sbr/knowledge, catálogo bibtex com abstract (skill `bibtex-articles`), sincronização Overleaf (skill `articles-overleaf`), integração de report no artigo SBSR (skill `sbsr-integrate`) |
| `agents/paper.md` | sync Overleaf do paper + hook post-commit (ponto de commit dos artigos) |
| `agents/optuna-colab.md` | rodada distribuída Optuna (local+Colab+Kaggle); runbook completo na skill `optuna-colab` |
| `agents/gotchas.md` | armadilhas (imports, bootstrap sys.path, sqlite Optuna, jobs em background) |
| `findings/` | **base canônica de conhecimento** (`INDEX.md` + `knowledge/` + `playbooks/` + scripts) — fatos, números, decisões e estados consolidados; skill `canonical-findings` |

## Setup rápido

```bash
./setup_environment.sh           # cria .venv (Python 3.10), instala deps
source .venv/bin/activate
source ./scripts/tf_gpu_env.sh   # OBRIGATÓRIO no Arch Linux com NVIDIA GPU
```

## Comandos essenciais

```bash
python pipeline/pipeline.py                       # pipeline completo (estágios: optuna -> train -> evaluate -> impute -> walkforward)
python pipeline/pipeline.py --stage train evaluate
python tests/test_wrappers_build.py               # smoke test dos wrappers (tests/ são scripts standalone, sem pytest)

# Relatórios (hub em reports/_site/ — sempre reconstruído a cada build)
python reports/src/builder.py build               # gera hub + todas as páginas
python reports/src/server.py                      # http://127.0.0.1:8765/ + API regenerar/retreinar
python reports/lidar/8020/report.py               # reconstrói o site a partir de um relatório

# Treino sob a reamostragem 80/20 + relatório de resultados (skill lidar-retrain)
python scripts/lidar_7525_train.py --train-ratio 0.80
python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020 --out-dir data/results/report_8020 --no-baselines

# Catálogo de artigos (índice + references.bib com abstract)
python articles/knowledge/tools/knowledge.py build|list|search|show|bibtex

# Ponto de commit/push Overleaf dos artigos (figuras)
python scripts/articles_sync.py list|sync|init|push-all
```

Não há linter/formatter/type-checker/test runner configurados nem CI;
`pyflakes` e `radon` são úteis no desenvolvimento (fora do requirements.txt).

## Visão das pastas novas

- **`reports/`** — relatórios padronizados por categoria (treinamento,
  lidar, resultados, imputação, protocolo). Cada pasta tem o código gerador
  + arquivos usados. Todo inserível (figura/tabela) tem botão **LaTeX**
  (popup com formatos steam/sbsr/genérico, label `fig:`/`tab:` e legenda
  editáveis). Detalhes: `agents/reports.md`.
- **`articles/`** — `steam/` (relatório STEAM, classe imc), `sbr/` (artigo
  SBSR, clone git Overleaf), `knowledge/` (coleção completa de artigos da
  máquina + trabalhos próprios, com `index.md` e `references.bib` gerados —
  bib sempre com abstract; skill `bibtex-articles` para consultas).
  Detalhes: `agents/articles.md`.
- **`agents/`** — este arquivo registra o contexto completo; os arquivos
  acima são a fonte detalhada por área.

## Regras de ouro (resumo — detalhes em agents/gotchas.md)

- **Consulte e mantenha a base canônica** (skill `canonical-findings`):
  antes de responder/decidir com fatos do projeto, comece por
  `findings/INDEX.md`; todo conhecimento importante descoberto vira/atualiza
  um finding em `findings/knowledge/` + linha no INDEX; valide com
  `python findings/scripts/findings.py check` antes de encerrar.

- **Todo relatório vive em `reports/<categoria>/<nome>/`** (página, análise
  pós-modelo, `report.py`/`report.md`/`report.html` + figuras). Nunca criar
  relatório em `docs/`, na raiz ou em `data/results/` — `docs/` é
  documentação e `data/results/` guarda artefatos de dados, não relatórios.
  Geradores (`tests/`, `scripts/`) que produzem relatório escrevem
  direto em `reports/`.
- **Estado dos relatórios**: `reports/state.json` é **gerado** (não editar
  à mão) — criado/regenerado/favorito/título por relatório; datas no hub,
  favoritos (★) e renomear (✏) via servidor (`/api/state`,
  `/api/favorite`, `/api/rename`) com fallback localStorage.
- `src/common.py` não pode ter efeitos colaterais de import (TF lazy).
- Arquivos sob `pipeline/` e `reports/` precisam do bootstrap `sys.path`
  para a raiz do projeto.
- `reports/_site/`, `articles/knowledge/index.md` e
  `articles/knowledge/references.bib` são **gerados** — regenere, nunca
  edite à mão.
- Pushes Overleaf são best-effort e nunca falham um commit.
- Métricas históricas antigas (ex.: MAE 0.1589) vêm de protocolo com
  vazamento; números honestos estão em `reports/` e `docs/`.
