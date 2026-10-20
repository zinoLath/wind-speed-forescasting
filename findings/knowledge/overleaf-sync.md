---
title: Sync Overleaf (paper + artigos) e hook post-commit
area: artigos
updated: 2026-10-19
---

# Sync Overleaf e hook post-commit

Dois mecanismos complementares, ambos **best-effort** (sem credenciais o
commit segue normalmente) e disparados pelo hook
`scripts/hooks/post-commit` (ativo via `git config core.hooksPath
scripts/hooks`):

1. **Paper** (`paper/figures/` → projeto do paper):
   `python scripts/paper_sync.py {init-manifest|list|copy|push}`;
   `paper/figures_manifest.json` mapeia fontes
   (data/results/report_homologado, baselines, EDA) → nomes canônicos;
   remote em `paper/overleaf_remote.txt` (gitignored) ou
   `$OVERLEAF_REMOTE`.
2. **Artigos** (`articles/steam`, `articles/sbr`):
   `python scripts/articles_sync.py {list|sync|init|push-all}`; mapeamento
   em `articles/<artigo>/sync.json` (`figures_dir` + `sources`:
   nome-no-overleaf → caminho local). **Só figuras são comitadas** nos
   repos dos artigos (nunca .tex do autor).

Credenciais: `OVERLEAF_USER`/`OVERLEAF_TOKEN` (ver `paper/README.md`).

Fluxo típico pós-experimento:

```bash
python reports/lidar/homologado/report.py        # regenera dados+site
python scripts/paper_sync.py copy                # figuras -> paper/figures
git add paper/figures && git commit -m "figuras"  # hook empurra ao Overleaf
python scripts/articles_sync.py sync sbr         # figuras -> artigo SBSR
```

Armadilhas: `paper/.overleaf-clone/` é cache (não editar); pushes nunca
falham um commit.

## Fontes

- `agents/paper.md`; `agents/articles.md`; `scripts/paper_sync.py`;
  `scripts/articles_sync.py`; `scripts/hooks/post-commit`.
