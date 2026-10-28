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
python tests/test_wrappers_build.py               # smoke test dos wrappers (tests/ = só smoke tests, sem pytest)

# Persistência de configuração (canônico nunca é editado por experimento)
python pipeline/config.py show                    # config resolvida (default+canônico+local)
python pipeline/config.py set train.epochs=5      # overlay local (pipeline.local.json, gitignored)
python pipeline/config.py reset                   # volta ao canônico
python pipeline/config.py profile-save NAME|profile-use NAME

# Relatórios (hub em reports/_site/ — sempre reconstruído a cada build)
python reports/src/builder.py build               # gera hub + todas as páginas
python reports/src/server.py                      # http://127.0.0.1:8765/ + API regenerar/retreinar
python reports/lidar/8020/report.py               # reconstrói o site a partir de um relatório

# Rodada canônica do artigo (reval 72,5/12,5/15 — treinadores em scripts/ raiz)
python scripts/lidar_reval_sbsr_train.py          # lineup SBSR (HPs do snapshot versionado)
python scripts/lidar_reval_h160_train.py          # perfil 40–160 m (rodada atual do paper)
python reports/src/relatorio_resultados.py --models-dir pipeline/tmp/trained_reval_sbsr \
    --out-dir data/results/report_reval_sbsr --trim-boundary-origins 72

# Treinos/estudos históricos: scripts/experiments/ (80/20, ablações, wfo, itildeq)
python scripts/experiments/lidar_7525_train.py --train-ratio 0.80

# Catálogo de artigos (índice + references.bib com abstract)
python articles/knowledge/tools/knowledge.py build|list|search|show|bibtex

# Banco de runs de predição (data/predictions.db — findings/knowledge/predicoes-db.md)
python -m src.predictions_db list|show|categories|homologar|prune

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
- **`studies/`** — estudos, ablações, A/Bs, benchmarks e ferramentas
  estatísticas (ex-`tests/`, que agora só tem smoke tests `test_*.py`).
- **`scripts/`** — raiz = treinadores da rodada canônica do artigo
  (`lidar_reval_*`, `lidar_8020_sbsr_*`, `sbsr_seeds_kaggle`),
  ferramentas do paper (`articles_sync`, `paper_sync`, `sbsr_tables`) e
  utilitários vivos; `scripts/experiments/` = rodadas concluídas/ablações;
  `scripts/dist/` = automação distribuída (Colab/Kaggle, watchers,
  notificações). Bootstraps em subpasta usam `parents[2]`.
- **`articles/`** — `steam/` (relatório STEAM, classe imc), `sbr/` (artigo
  SBSR, clone git Overleaf), `knowledge/` (coleção completa de artigos da
  máquina + trabalhos próprios, com `index.md` e `references.bib` gerados —
  bib sempre com abstract; skill `bibtex-articles` para consultas).
  Detalhes: `agents/articles.md`.
- **`agents/`** — este arquivo registra o contexto completo; os arquivos
  acima são a fonte detalhada por área.

## Regras de ouro (resumo — detalhes em agents/gotchas.md)

- **A rodada canônica é sempre o default da pipeline.**
  `pipeline/pipeline.json` reflete a rodada canônica (hoje: protocolo
  reval 72,5/12,5/15 + ES/RLRP, 60 épocas, MSE pura, seed 42) e **nunca é
  editado por experimento**. Ajustes de rodada/ambiente usam a camada de
  persistência (`python pipeline/config.py set train.epochs=5` →
  `pipeline.local.json`, gitignored; perfis em `pipeline/profiles/` via
  `config.py profile-save/profile-use`; `config.py reset` volta ao
  canônico; `--config` explícito ignora o local por design). Os HPs
  canônicos dos treinadores vivem versionados em
  `pipeline/configs/canonical_hparams_8020_itildeq.json`.

- **Consulte e mantenha a base canônica** (skill `canonical-findings`):
  antes de responder/decidir com fatos do projeto, comece por
  `findings/INDEX.md`; todo conhecimento importante descoberto vira/atualiza
  um finding em `findings/knowledge/` + linha no INDEX; valide com
  `python findings/scripts/findings.py check` antes de encerrar.

- **Todo relatório vive em `reports/<categoria>/<nome>/`** (página, análise
  pós-modelo, `report.py`/`report.md`/`report.html` + figuras). Nunca criar
  relatório em `docs/`, na raiz ou em `data/results/` — `docs/` é
  documentação e `data/results/` guarda artefatos de dados, não relatórios.
  Geradores (`studies/`, `scripts/`) que produzem relatório escrevem
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
