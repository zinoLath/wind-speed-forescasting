---
title: Gerar/regenerar relatórios do hub
area: playbook
updated: 2026-10-19
---

# Playbook: geração de relatórios

Fonte: skill **reports-hub**. Resumo operacional:

## Regenerar dados de um relatório LiDAR (~5 min GPU)

```bash
python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020 \
    --out-dir data/results/report_8020 --no-baselines
```

## Reconstruir o site (sempre após mudar qualquer relatório)

```bash
python reports/src/builder.py build      # hub + todas as páginas
```

## Servir com botões de ação

```bash
python reports/src/server.py             # http://127.0.0.1:8765/
# botões: Regenerar (síncrono) e Regenerar com retreino (background +
# regeneração encadeada; log em reports/<cat>/<nome>/retrain.log)
```

## Adicionar um relatório novo

1. Criar `reports/<categoria>/<nome>/report.py` (padrão: bootstrap
   sys.path + `report_module(...)` para LiDAR; ou `report.md` +
   `meta.json`; ou `report.html` auto-contido) — **nunca fora de
   `reports/`**;
2. Rodar `python reports/<categoria>/<nome>/report.py` — o site inteiro é
   reconstruído e o relatório entra no hub/state;
3. Configurar `regenerate=[...]`/`retrain=[...]` no `REPORT` para os
   botões funcionarem;
4. Figura nova em relatório LiDAR: adicionar stem ao tema certo
   (`FIGURE_THEMES`) e caption em `FIGURE_CAPTIONS` — nada em "Outras
   figuras";
5. Rodar `python findings/scripts/findings.py check` se algo do
   conhecimento mudou (números, conclusões).

## Consultar estado/favoritos/renomear

- Programático: `builder.state_for/state_update/state_save/state_all`;
  HTTP: `/api/state`, `/api/favorite`, `/api/rename` (instantâneos, sem
  rebuild).
