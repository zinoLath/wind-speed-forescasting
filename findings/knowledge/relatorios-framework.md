---
title: Framework de relatórios (reports/)
area: relatorios
updated: 2026-10-28
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
- **Figuras de erro em RMSE + CD Wilcoxon-Holm (desde 28/10/2026)**: o
  heatmap por modelo × horizonte dos relatórios de resultados saiu de MSE
  para **RMSE (m/s)** (arquivo `06_heatmap_rmse_horizonte.png`, antigo
  `06_heatmap_mse_…` removido) e o diagrama de diferença crítica da
  rodada do paper (`reports/lidar/reval-h160/friedman_horizonte.py` +
  `ab-alturas/cd_horizonte.py` para h160) usa **agrupamento
  Wilcoxon-Holm** (Wilcoxon signed-rank pareado por origem, Holm entre os
  pares; cliques ligados por barras) com perda RMSE — não mais Nemenyi/CD.
  RMSE é monotônico por bloco: ranks/χ² idênticos ao critério MSE.
  Números da rodada h160 (72 origens): Friedman agregado 6 métodos
  χ² = 255,5, p = 3,6e-53; global 7 métodos (com RF, perdas MSE,
  `testes_hipotese.json`) χ² = 251,2, p = 2,2e-51; Friedman por horizonte
  35/36 significativos após Holm. Wilcoxon-Holm separa 14/15 pares — só
  {TCN, ARIMA} não se separa (Nemenyi separava menos). DM/MCS/Wilcoxon em
  `testes_hipotese.json` seguem em MSE por origem. Rodadas antigas
  (reval_sbsr, 8020-sbsr) não foram regeneradas — ainda exibem MSE/Nemenyi
  até a próxima regeneração.
- Servidor aceita `--host` (ex.: `0.0.0.0` para acesso pelo celular na
  LAN; sem firewall ativo na máquina, porta 8765).

## Fontes

- `agents/reports.md` (detalhe completo); `reports/README.md`;
  `reports/src/*.py`; skill **reports-hub**.
