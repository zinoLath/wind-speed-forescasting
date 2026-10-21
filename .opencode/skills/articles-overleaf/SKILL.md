---
name: articles-overleaf
description: Ponto de commit e push Overleaf dos artigos (articles/steam, articles/sbr) — sincroniza figuras dos relatórios, comita nos repos dos artigos e empurra via git bridge; também liga um artigo local ao Overleaf (init). Use quando o usuário mencionar "overleaf", "empurrar figuras", "sync sbr", "sync steam", "sincronizar artigo", "commit dos artigos", ou comandos `scripts/articles_sync.py`.
---

# Artigos ↔ Overleaf (steam / sbr)

Cada artigo é um repo git próprio dentro de `articles/<artigo>/`
(gitignored no repo principal). O mapeamento de figuras vive em
`articles/<artigo>/sync.json` (`figures_dir` + `sources`:
nome-no-overleaf → caminho local). Detalhes: `agents/articles.md`.

```bash
python scripts/articles_sync.py list              # estado dos artigos
python scripts/articles_sync.py sync sbr          # copia figuras + commit + push
python scripts/articles_sync.py sync sbr --no-push
python scripts/articles_sync.py init steam https://git.overleaf.com/<id>
python scripts/articles_sync.py push-all --auto   # best-effort (hook post-commit)
```

## Regras

- O sync **só comita figuras** (`git add` restrito a `figures_dir`) — nunca
  .tex nem rascunhos do autor. Sem mudanças nas figuras, nada é comitado.
- Push precisa de credenciais: `OVERLEAF_USER`/`OVERLEAF_TOKEN` (Git
  integration token) ou URL com credenciais embutidas — mesmo esquema do
  paper (ver `paper/README.md`). Sem credenciais o push falha com aviso,
  o fluxo segue (best-effort).
- `steam/` ainda não tem `.git`: quando tiver o project-id, rode `init
  steam <url>` e depois, dentro de `articles/steam/`:
  `git fetch origin && git reset origin/master` (mantém os arquivos locais
  no working tree para casar com o Overleaf).
- O hook `scripts/hooks/post-commit` (ativo via
  `git config core.hooksPath scripts/hooks`) roda `push-all --auto` a cada
  commit do repo principal — é o ponto automático de levar figuras novas ao
  Overleaf. Nunca deixe o hook falhar um commit.
- Fluxo típico pós-experimento: regenerar relatório
  (`reports/lidar/8020/report.py` ou botão) → `sync sbr` (ou confiar no
  hook) → conferir no Overleaf.
- Figuras atuais no `sync.json` do sbr: heatmaps MSE/R² + divisão temporal
  do relatório 80/20 → `LaTeX_Template_SBSR_Portugues/figuras/`; do steam:
  rolling + inferência do homologado → `figs/`. Para figuras novas, adicione
  entradas em `sources` (nome no Overleaf = nome usado no `\includegraphics`).
