---
title: Sincronizar figuras com Overleaf (paper + artigos)
area: playbook
updated: 2026-10-19
---

# Playbook: sync Overleaf

Fonte: finding `overleaf-sync` + skills **articles-overleaf**/**sbsr-integrate**.

```bash
# estado
python scripts/articles_sync.py list
python scripts/paper_sync.py list

# figuras -> commit -> push (por artigo)
python scripts/articles_sync.py sync sbr            # ou steam
python scripts/articles_sync.py sync sbr --no-push  # só commit local
python scripts/paper_sync.py copy && git add paper/figures && git commit -m "figuras"

# ligar artigo sem .git ao Overleaf (ex.: steam)
python scripts/articles_sync.py init steam https://git.overleaf.com/<id>
# dentro de articles/steam/: git fetch origin && git reset origin/master
```

Regras:

- **Só figuras** são comitadas nos repos dos artigos (nunca .tex do
  autor); mapeamento em `articles/<artigo>/sync.json`.
- Hook `post-commit` roda `push-all --auto` a cada commit do repo
  principal — o commit no repo principal É o ponto de push.
- Best-effort: sem `OVERLEAF_USER`/`OVERLEAF_TOKEN` o push falha sem
  quebrar o fluxo. `paper/.overleaf-clone/` é cache (não editar).
