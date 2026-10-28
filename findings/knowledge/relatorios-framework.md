---
title: Framework de relatórios (reports/)
area: relatorios
updated: 2026-10-19
---

# Framework de relatórios (reports/)

Hub padronizado com site estático + servidor de ações. Regras de ouro e
arquitetura:

- **Todo relatório vive em `reports/<categoria>/<nome>/`** (página,
  análise pós-modelo, report.py/md/html + figuras + meta.json). NUNCA
  criar relatório em `docs/`, na raiz ou em `data/results/`. Geradores
  (`studies/*.py`, `scripts/**/*.py`) escrevem direto em `reports/`.
- Categorias: `treinamento/` (curvas de treino), `lidar/` (resultados por
  protocolo: 7525, 8020, 8020-itildeq*, 8020-lstm-lossab, 8020-mseloss,
  8020-optuna, duel-loss, homologado, lstm-bi-origins, nonorm),
  `resultados/` (temas: investigacao, features_study, context-mode-ablation),
  `imputacao/` (wfo-*), `protocolo/` (metodos-complementares,
  persistence-gate-ab).
- `reports/src/`: `components.py` (figure/table/cards/section/markdown),
  `latexfmt.py` (snippets LaTeX por formato: **steam, sbsr, generic**),
  `resultados.py` (renderizador padrão dos relatórios LiDAR, com síntese
  textual, tabelas com `sub=` interpretativo e temas de figuras
  `FIGURE_THEMES`), `builder.py` (descoberta + hub + `_site/`),
  `server.py` (http.server + API).
- **`reports/_site/` e `reports/state.json` são GERADOS** — nunca editar à
  mão; regenerar com `python reports/src/builder.py build`.
- `state.json` guarda por relatório: `created`, `regenerated`,
  `favorite`, `title` (via servidor: `/api/state`, `/api/favorite`,
  `/api/rename`; fallback localStorage `hubstate.v1`).
- Servidor (`python reports/src/server.py`, porta 8765): `POST
  /api/rebuild`; `POST /api/regenerate/<cat>/<nome>` (síncrono);
  `POST /api/retrain/<cat>/<nome>` (background, log em
  `<relatório>/retrain.log`, estado em `pipeline/tmp/reports_jobs/`;
  encadeia regeneração ao terminar); `GET /api/job`, `/api/meta`,
  `/api/state`.
- Todo inserível tem botão **LaTeX** (popup: formatos steam/sbsr/generic,
  `\label` com prefixo `fig:`/`tab:` forçado, legenda editável, copiar);
  figura tem lightbox; grades usam thumbnails (máx. 2 cards por linha).
- Padrão `report.py`: bootstrap sys.path (3 níveis até a raiz) +
  `report_module(...)` de `reports/src/resultados.py` (relatórios LiDAR) e
  `builder.run_standalone(__file__)` no `__main__` (reconstrói o site
  inteiro).

## Fontes

- `agents/reports.md` (detalhe completo); `reports/README.md`;
  `reports/src/*.py`; skill **reports-hub**.
