---
name: reports-hub
description: Constrói, serve e estende o hub de relatórios padronizados (reports/) — builder, servidor com botões regenerar/retreinar, popup LaTeX (formatos steam/sbsr/genérico com label fig:/tab: e legenda editável). Use quando o usuário mencionar "hub de relatórios", "reports", "regenerar relatório", "retreinar pelo site", "popup latex", "adicionar relatório", ou comandos `reports/src/builder.py`, `reports/src/server.py`, `reports/lidar/*/report.py`, `reports/treinamento/*/report.py`.
---

# Hub de relatórios (reports/)

Framework em `reports/src/` (Python puro + stdlib http.server + `markdown`).
Site gerado em `reports/_site/` — **GERADO, nunca editar à mão**. Detalhes
completos: `agents/reports.md`.

Layout: `reports/<categoria>/<nome>/` com `report.py` (código gerador) ou
`report.md`/`report.html` (+ `meta.json` com title/description). Categorias
atuais: `treinamento`, `lidar`, `resultados`, `imputacao`, `protocolo`.

## Comandos

```bash
.venv/bin/python reports/src/builder.py build    # gera o site inteiro (hub + páginas)
.venv/bin/python reports/src/server.py [--port 8765]  # http://127.0.0.1:8765/ + API
.venv/bin/python reports/<cat>/<nome>/report.py  # reconstrói o site a partir de um relatório
```

API do servidor: `POST /api/rebuild`, `POST /api/regenerate/<cat>/<nome>`
(síncrono, roda `REPORT["regenerate"]`, **carimba `regenerated` no
state.json** e reconstrói), `POST
/api/retrain/<cat>/<nome>` (background, log em
`reports/<cat>/<nome>/retrain.log`, estado em
`pipeline/tmp/reports_jobs/`; ao terminar **encadeia a regeneração**
automaticamente e carimba `regenerated`), `GET /api/job/<cat>/<nome>`,
`GET /api/state` (dump do state.json), `POST /api/favorite/<cat>/<nome>`
(toggle ou `{"favorite": bool}`), `POST /api/rename/<cat>/<nome>`
(`{"title": "..."}`, vazio restaura; NÃO dispara rebuild — o JS aplica ao
vivo e o próximo build assa no estático), `GET /api/meta/<cat>/<nome>`
(inclui created/regenerated/favorite/title).

## Estado (reports/state.json)

GERADO — builder + servidor escrevem; nunca editar à mão. Por relatório:
`created` (1ª vez visto), `regenerated` (última regeneração ok; `null`
se nunca), `favorite`, `title` (sobrescrita do botão renomear — muda só
o título de exibição, nunca pastas/caminhos). Helpers no builder:
`state_for/state_update/state_save/state_all/fmt_date`. Sem servidor,
favorito/renomear caem no localStorage (`hubstate.v1`) via `app.js`.

## Regras

- **Localização obrigatória**: todo relatório (página, análise pós-modelo,
  `report.py`/`report.md`/`report.html` + figuras) fica em
  `reports/<categoria>/<nome>/`. Nunca criar relatório em `docs/`, na raiz
  ou em `data/results/` — `docs/` é documentação e `data/results/` guarda
  artefatos de dados. Geradores (`studies/`, `scripts/`) que produzem
  relatório escrevem DIRETO em `reports/` (os que escreviam em
  `data/results/` já foram migrados — ver agents/reports.md).
- **HTML standalone**: `report.html` que é um documento completo
  (`<!doctype`/`<html>` — ex.: relatórios auto-contidos migrados de
  data/results) é publicado como veio, sem a shell do hub.
- **Popup LaTeX**: todo inserível é criado com
  `reports/src/components.py` (`C.figure`, `C.table`) — os botões e o
  popup (formatos steam/sbsr/generic, `\label` com prefixo `fig:`/`tab:`
  forçado, legenda/largura/posição editáveis) vêm dos data-attributes +
  `assets/app.js`. Os templates LaTeX moram APENAS em
  `reports/src/latexfmt.py` (fonte única, embutida na página em
  `<script id="latex-templates">`). Figuras clicadas abrem **lightbox**
  (imagem ampliada; fecha ao clicar fora).
- **Texto explicativo**: relatórios LiDAR estruturam a análise em
  `reports/src/resultados.py` — síntese gerada dos dados, `sub=`
  interpretativo nas tabelas e temas de figuras em `FIGURE_THEMES`
  (figura nova entra no tema certo + caption em `FIGURE_CAPTIONS`; nada
  em "Outras figuras").
- **report.py novo**: bootstrap obrigatório `sys.path` para a raiz do
  projeto (parents[3] em relatórios de 2º nível), `REPORT = {...}` com
  `regenerate`/`retrain` (argv list) e `if __name__ == "__main__"` chamando
  `builder.run_standalone(__file__)`. Relatórios LiDAR usam a fábrica
  `reports/src/resultados.py:report_module`.
- **Ordem no hub/dropdown** = ordem alfabética das pastas; sobrescreva com
  `REPORT["order"]`/meta `"order"`.
- `regenerate` dos relatórios LiDAR roda `reports/src/relatorio_resultados.py`
  (~5 min de GPU). `retrain` sobrescreve modelos canônicos — confira
  backup em `pipeline/tmp/trained_8020_backup_*` antes (skill
  `lidar-retrain` cobre o fluxo completo de retreino).
- Ao mudar qualquer coisa em `reports/src/`, rode `build` e confira o hub
  e uma página py + uma md; `pyflakes reports/src/*.py` deve ficar limpo.
