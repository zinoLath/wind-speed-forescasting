---
title: Artigo STEAM e catálogo knowledge
area: artigos
updated: 2026-10-19
---

# Artigo STEAM e catálogo knowledge

## articles/steam/

Relatório STEAM (classe **imc/UNIFEI**), `articles/steam/` — **ainda sem
`.git`**; para ligar ao Overleaf:
`python scripts/articles_sync.py init steam <url-do-projeto>` e depois,
dentro de `articles/steam/`, `git fetch origin && git reset origin/master`.
Mapeamento de figuras já configurado no `sync.json` (rolling/inferência do
**report_homologado** → `figs/`).

## articles/knowledge/

Coleção completa de artigos da máquina + trabalhos próprios:

- `pdf/` (raiz + `documents/` — symlink para `~/documents/papers` — +
  `own-work/`), `markdown/` (extrações), `meta/ieee/` (JSON oficial do
  IEEE Xplore), `pages/`, `notes/`, `figures/`.
- Gerador: `articles/knowledge/tools/knowledge.py` com subcomandos
  `build|list|search|show|bibtex` — varre as pastas, cruza por
  DOI/título/stem e escreve `index.md` + `references.bib` (**sempre com
  abstract**; sem abstract = ⚠ no índice).
- `index.md`/`references.bib` são **GERADOS** — nunca editar à mão;
  corrigir os fontes e rodar `build`. Dedupe de PDFs por checksum.
- Projetos antigos no knowledge: `overleaf-paper/` (detecção de objetos
  urbanos) e `revisao_narrativa/` (revisão narrativa de vento) — clones
  Overleaf com `.git` e remote `origin`.
- Skill **bibtex-articles** descreve o fluxo para achar/construir entradas
  bibtex completas (PDF → markdown → abstract).

## Fontes

- `agents/articles.md`; `articles/README.md`;
  `articles/knowledge/tools/knowledge.py`.
