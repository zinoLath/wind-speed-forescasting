# agents/paper.md — sincronização com Overleaf (paper + artigos)

> Base canônica: `findings/knowledge/overleaf-sync.md` +
> `findings/playbooks/sync-overleaf.md` (skill `canonical-findings`).

Dois mecanismos complementares; ambos best-effort (sem credenciais o commit
segue normalmente) e disparados pelo hook `scripts/hooks/post-commit`
(ativo via `git config core.hooksPath scripts/hooks`):

1. **Paper** (`paper/figures/` → projeto Overleaf do paper):
   `python scripts/paper_sync.py {init-manifest|list|copy|push}`.
   `paper/figures_manifest.json` mapeia fontes (data/results/report_homologado,
   baselines, EDA) → nomes canônicos. Remote em
   `paper/overleaf_remote.txt` (gitignored) ou `$OVERLEAF_REMOTE`;
   credenciais `OVERLEAF_USER`/`OVERLEAF_TOKEN`.

2. **Artigos** (`articles/steam`, `articles/sbr` → projetos Overleaf de cada
   artigo): `python scripts/articles_sync.py {list|sync|init|push-all}`.
   Mapeamento por artigo em `articles/<artigo>/sync.json`; só figuras são
   comitadas nos repos dos artigos. Detalhes em agents/articles.md.

Fluxo típico pós-experimento:

```bash
python reports/lidar/homologado/report.py        # regenera dados+site
python scripts/paper_sync.py copy                # figuras -> paper/figures
git add paper/figures && git commit -m "figuras"  # hook empurra p/ Overleaf
python scripts/articles_sync.py sync sbr         # figuras -> artigo SBSR
```
