# agents/articles.md — artigos (steam, sbr, knowledge) e Overleaf

> Skills relacionadas: **bibtex-articles** (catálogo/abstracts) e
> **articles-overleaf** (sync/commit point), em `.opencode/skills/`.
> Base canônica: `findings/knowledge/artigo-sbr.md`,
> `artigo-steam-knowledge.md`, `overleaf-sync.md` +
> `findings/playbooks/` (skill `canonical-findings`).

## Organização

```
articles/               # inteiramente gitignored no repo principal
  steam/                # relatório STEAM (classe imc/UNIFEI); ver sync abaixo
  sbr/                  # artigo SBSR — clone do git Overleaf (origin remoto)
  knowledge/            # coleção de artigos + trabalhos próprios
    index.md            # índice gerado apontando para todos os arquivos
    references.bib      # BibTeX de todos os artigos, COM abstract
    pdf/                # PDFs (raiz, documents/ da máquina, own-work/)
    markdown/           # extrações de texto (fonte dos metadados/abstracts)
    meta/ieee/          # metadados oficiais do IEEE Xplore (JSON)
    pages/              # páginas HTML salvas dos downloads
    notes/              # anotações de trabalho (inventários, roteiros)
    figures/            # figuras extraídas de artigos
    tools/knowledge.py  # gerador do índice + bib, com busca
```

Projetos antigos também vivem em knowledge: `overleaf-paper/` (projeto
diferente — detecção de objetos urbanos) e `revisao_narrativa/` (revisão
narrativa de vento, clone Overleaf). Os três clones Overleaf têm `.git`
próprio e remoto `origin` (git bridge do Overleaf).

## Catálogo do knowledge (índice + bibtex com abstract)

```bash
.venv/bin/python articles/knowledge/tools/knowledge.py build   # regenera
.venv/bin/python articles/knowledge/tools/knowledge.py list    # títulos+chaves
.venv/bin/python articles/knowledge/tools/knowledge.py search transformer
.venv/bin/python articles/knowledge/tools/knowledge.py show <chave|título>
.venv/bin/python articles/knowledge/tools/knowledge.py bibtex <termo>
```

- `build` varre `pdf/` (raiz + `documents/` + `own-work/`), `markdown/`,
  `meta/ieee/` e `pages/`, cruza por DOI/título/stem, e escreve `index.md`
  e `references.bib` (com abstract quando extraído; entradas sem abstract
  ficam marcadas com ⚠ no índice).
- **Nunca editar `index.md`/`references.bib` à mão** — corrija os fontes
  (`markdown/`, `meta/ieee/`) e regenere.
- A skill **bibtex-articles** (`.opencode/skills/bibtex-articles/SKILL.md`)
  descreve o fluxo para achar/construir entradas bibtex lendo os downloads
  (PDFs/PDF→markdown) e completando abstracts faltantes.
- Artigos novos da máquina: `pdf/documents/` é um **symlink** para
  `~/documents/papers` — basta salvar o PDF lá e rodar `build` (o scan
  segue o link; dedupe por checksum evita duplicatas).

## Ponto de commit/push Overleaf dos artigos

```bash
python scripts/articles_sync.py list              # estado dos artigos
python scripts/articles_sync.py sync sbr          # figuras -> commit -> push
python scripts/articles_sync.py sync sbr --no-push
python scripts/articles_sync.py init steam https://git.overleaf.com/<id>
python scripts/articles_sync.py push-all --auto   # best-effort (hook)
```

- Cada artigo descreve o mapeamento em `articles/<artigo>/sync.json`
  (`figures_dir` + `sources`: nome-no-overleaf → caminho local). Já
  configurados: `sbr` (heatmaps/divisão do relatório 80/20 → figuras do
  template) e `steam` (rolling/inferência do homologado → `figs/`).
- O sync **só comita figuras** (nunca .tex do autor) e usa o remote
  `origin` do próprio repo do artigo. Credenciais: `OVERLEAF_USER`/
  `OVERLEAF_TOKEN` (ver paper/README.md) — sem elas o push falha sem
  quebrar o fluxo.
- O hook `scripts/hooks/post-commit` (ativo via `core.hooksPath`) roda
  `articles_sync.py push-all --auto` a cada commit do repo principal —
  este é o **ponto de commit** que leva imagens novas/adições ao Overleaf.
- `steam/` ainda não tem `.git`: quando souber o project-id do Overleaf,
  rode `init steam <url>` e depois `git fetch origin && git reset
  origin/master` dentro de `articles/steam/` para casar com o Overleaf.

## Tabelas e figuras do artigo SBSR (fontes por tipo)

- **Rodada canônica**: `report_8020_itildeq` (mista — finding
  `resultados-canonicos`). Qualquer tabela/figura citada no artigo tem
  que vir dela.
- **Tabelas de métricas** (agregado + h=36): `scripts/sbsr_tables.py
  --round-dir data/results/report_8020_itildeq` → tabelas LaTeX prontas
  (caption+label); finding `tabelas-sbsr-geradas`.
- **Tabelas de hiperparâmetros/tempo de treino** (pedida pela revisão):
  `data/results/report_8020_itildeq/evaluation.json` (params/loss/
  timing), `data/results/optuna/<modelo>/trials.csv`,
  `data/results/*/summary.json` (`training_time_sec`).
- **Tabela descritiva do dataset**: colunas de `data/dataset.csv`
  (coletados `ws/dir/v`, `press/humid/temp`; computados `cis*`,
  `disp*`).
- **Figuras**: `articles/sbr/sync.json` (heatmaps/split ← relatório da
  rodada canônica); regeneração das imagens em
  `reports/src/relatorio_resultados.py`.
- **Teste DM/p-valores**: regeneração da canônica gera `mcs.json`;
  análise ajustada em `reports/lidar/8020-itildeq-metodos/`.
- Integração de tudo no texto: skill **sbsr-integrate** (verifica
  consistência: var implícita uniforme, figura = tabela da mesma
  rodada).
