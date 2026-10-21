# agents/reports.md — framework de relatórios (reports/)

> Runbook operacional na skill **reports-hub**
> (`.opencode/skills/reports-hub/SKILL.md`). Base canônica:
> `findings/knowledge/relatorios-framework.md` +
> `findings/playbooks/geracao-relatorios.md` (skill `canonical-findings`).

## Layout

```
reports/
  src/                    # framework (nada de relatório aqui)
    components.py         # helpers: figure/table/cards/section/markdown
    latexfmt.py           # snippets LaTeX por formato: steam, sbsr, generic
    resultados.py         # renderizador padrão dos relatórios LiDAR
    builder.py            # descoberta + shell + hub + site estático (_site/)
    server.py             # http.server + API regenerate/retrain/rebuild
    assets/               # theme.css + app.js (popup LaTeX, dropdowns, ações)
  treinamento/<exp>/report.py   # relatórios de treino (reamostragem 80/20)
  lidar/<exp>/report.py         # resultados por protocolo (8020, 7525, ...)
  resultados/<tema>/report.md   # relatórios markdown migrados de docs/
  imputacao/<tema>/report.md
  protocolo/<tema>/report.py | report.md | report.html
  state.json            # ESTADO (gerado): created/regenerated/favorite/title
  _site/                # GERADO — hub + páginas; nunca editar à mão
```

Cada pasta de relatório contém **o código para gerar o relatório e todos
os arquivos que ele usa** (report.py/report.md/report.html + figures/ ou
img/ copiadas no build + meta.json com título/descrição).

> **Regra:** todo relatório — página, análise pós-modelo, `report.py`,
> `report.md`/`report.html` e suas figuras — **tem que ficar em
> `reports/<categoria>/<nome>/`**. Nunca criar relatório em `docs/`, na
> raiz, em `data/results/` ou em qualquer outra pasta. `docs/` é
> documentação; `data/results/` guarda artefatos de dados, não relatórios.
> Geradores (`tests/*.py`, `scripts/*.py`) que produzem relatório escrevem
> DIRETAMENTE em `reports/<categoria>/<nome>/` — nunca em `data/results/`.

## Estado por relatório (reports/state.json)

`reports/state.json` é **gerado** (builder + servidor escrevem; não editar
à mão) e guarda, por chave `"categoria/nome"`:

- **`created`** — data de criação (carimbada na 1ª vez que o relatório é
  visto; fallback = mtime do arquivo principal);
- **`regenerated`** — data da última regeneração bem-sucedida (carimbada
  pelo servidor após `regenerate` ou `retrain`+regeneração; `null` se
  nunca);
- **`favorite`** — favorito (bool);
- **`title`** — sobrescrita de título (botão renomear; ausente = título do
  meta/REPORT).

O builder expõe `state_for/state_update/state_save/state_all` e as
propriedades `Report.created/.regenerated/.favorite/.title`; datas no hub
e no painel de metadados via `builder.fmt_date`. As páginas do hub mostram
"criado / regenerado / atualizado" por cartão (atualizado = mtime real dos
arquivos; criado/regenerado vêm do state).

## Favoritos e renomear (organização pelo usuário)

- **★ favorito** — botão nos cartões do hub e na topbar de cada página.
  Sem servidor: fica no `localStorage` (`hubstate.v1`); com servidor:
  `POST /api/favorite/<cat>/<nome>` grava no `state.json`. Favoritos
  aparecem na seção "★ Favoritos" do hub e com estrela nos dropdowns.
- **✏ renomear** — muda só o **título de exibição** (nunca a pasta nem os
  caminhos): `POST /api/rename/<cat>/<nome>` com `{"title": "..."}`;
  título vazio restaura o original. Aplicado ao vivo via JS e assado no
  próximo build (títulos do hub/nav/páginas).
- Os dois endpoints NÃO disparam rebuild (instantâneos); o JS aplica na
  hora e o próximo build persiste no estático.

## Padrão de relatório (report.py)

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # relatórios LiDAR

REPORT, build = report_module(
    name="8020", title="...", description="...",
    source_dir="data/results/report_8020",
    regenerate=[...argv...],   # comando do botão "Regenerar"
    retrain=[...argv...] or None,  # botão "Regenerar com retreino"
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)   # reconstrói o site todo
```

Relatórios arbitrários importam `reports.src.components` e retornam HTML
de `build(ctx)`; `ctx.dir` é a pasta do relatório, `ctx.published` o destino
no `_site/`. Relatórios estáticos: `report.md` (+ `meta.json` opcional) ou
`report.html`.

## Inseríveis com popup LaTeX (exigência central)

Todo inserível (figura/tabela) ganha botão **LaTeX** que abre um popup com:

- **formatos**: `steam` (classe imc/UNIFEI), `sbsr` (template SBSR) e
  `generic`; convenções copiadas dos projetos reais (`latexfmt.py` é a
  fonte única — templates também embutidos na página em
  `<script id="latex-templates">`);
- **textboxes** editáveis: `\label` (prefixo `fig:`/`tab:` forçado),
  legenda (`\caption`), arquivo, largura, posição flutuante;
- **Copiar** com clipboard API + fallback.

```python
C.figure("figures/x.png", "Legenda...", label="fig:x", width=0.8)
C.table(cols, rows, "Legenda...", label="tab:y", highlight_rows=[0])
```

Toda figura também abre um **lightbox** ao ser clicada (imagem ampliada,
fecha ao clicar fora — `assets/app.js` + `dialog.lightbox` no CSS). Nas
grades, imagens são **thumbnails** (máx. 2 cards por linha, caption/label
abaixo da imagem) com o texto da seção ANTES da grade.

## Sidebars (layout de 3 colunas)

- **Esquerda**: TOC gerada por `builder.build_toc()` — seções
  (`components.section` auto-gera `id="sec-<slug>"`) + cada figura/tabela
  (`C.figure`/`C.table` geram âncora `fig-…`/`tab-…`); destaque da seção
  ativa por scroll (`app.js:wireToc`).
- **Direita**: metadados — painel estático de `builder.build_meta_panel`
  (`REPORT["info"]` = pares [rótulo, valor]) + bloco vivo `#meta-live`
  preenchido por JS via `GET /api/meta/<cat>/<nome>` (job rodando/ocioso,
  início, tail do log, comandos configurados; polling a cada 5 s).
  Layout some abaixo de 1380px.

## Análise estruturada (exigência de texto explicativo)

Os relatórios LiDAR não são só figuras: `reports/src/resultados.py` monta

1. **Síntese da análise** — parágrafo gerado dos dados (melhor modelo,
   MSE/R², custo do rolling) + instruções de leitura;
2. **Tabelas com `sub=` interpretativo** (o que a métrica significa, o que
   observar);
3. **Análise gráfica temática** — `FIGURE_THEMES` agrupa as figuras
   (protocolo, comparação global, por horizonte/DM, trajetórias,
   diagnósticos, custos) e cada grupo traz o texto que explica como ler
   aquelas figuras. Figura nova = adicionar stem ao tema certo e caption
   em `FIGURE_CAPTIONS`; nada deve cair em "Outras figuras".

## Hub, navegação e ações

- `python reports/src/builder.py build` → gera `_site/` inteiro (hub com
  cartões por categoria sempre atualizado + páginas com topbar de dropdowns
  por categoria, na ordem das pastas).
- `python reports/src/server.py [--port 8765]` → serve o `_site/` em
  http://127.0.0.1:8765/ e expõe:
  - `POST /api/rebuild` — reconstrói o site (botão "Atualizar hub");
  - `POST /api/regenerate/<cat>/<nome>` — roda `REPORT["regenerate"]`
    (síncrono), **carimba `regenerated` no state** e reconstrói;
  - `POST /api/retrain/<cat>/<nome>` — roda `REPORT["retrain"]` em
    background (log em `<relatório>/retrain.log`, estado em
    `pipeline/tmp/reports_jobs/`); ao terminar **encadeia a regeneração**
    automaticamente e carimba `regenerated`;
  - `GET /api/job/<cat>/<nome>` — polling do job;
  - `GET /api/state` — dump do `reports/state.json` (favoritos, títulos,
    datas — o JS sincroniza);
  - `POST /api/favorite/<cat>/<nome>` — alterna/define favorito
    (`{"favorite": true|false}`, vazio = alternar);
  - `POST /api/rename/<cat>/<nome>` — `{"title": "..."}` (vazio restaura);
  - `GET /api/meta/<cat>/<nome>` — inclui `created`, `regenerated`,
    `favorite` e `title` (painel vivo da direita).
- Sem servidor, os botões avisam "sem servidor — rode:
  python reports/src/server.py" e favorito/renomear caem no localStorage.

## Regenerar vs retreinar (relatórios LiDAR)

- **Regenerar** = `tests/relatorio_resultados.py --models-dir ... --out-dir
  ...` (~5 min GPU: reavalia os pesos e recria figures/evaluation.json)
  e depois rebuild do site.
- **Regenerar com retreino** = treina de novo
  (`scripts/lidar_7525_train.py --train-ratio 0.80` →
  `pipeline/tmp/trained_8020`) e depois regenera. Retreino sobrescreve os
  modelos canônicos — existe backup em `pipeline/tmp/trained_8020_backup_*`.
- O relatório `treinamento/8020` re-gera as curvas a cada build (lê
  `pipeline/tmp/train_dashboard/*.jsonl`, o `model.json` e o
  `data/results/lidar_8020_features/all.json`).

## Adicionar um relatório

1. Crie `reports/<categoria>/<nome>/report.py` (ou `report.md` +
   `meta.json`; `report.html` auto-contido também vale — documentos
   completos são publicados sem a shell do hub);
2. Rode `python reports/<categoria>/<nome>/report.py` — o site inteiro é
   reconstruído, o hub já lista o relatório (ordem = ordem das pastas) e
   ele ganha entrada no `reports/state.json` (created);
3. Para regeneração automática, configure `regenerate`/`retrain` no `REPORT`.

## Relatórios migrados de data/results (out/2026)

Estavam soltos em `data/results/` e hoje vivem no hub (geradores
atualizados para escrever direto lá):

- `resultados/investigacao` (era `data/results/investigacao/`;
  `tests/build_investigacao.py`);
- `resultados/features_study` (era `data/results/features_study/`;
  `tests/feature_study.py` + `tests/build_feature_study_report.py`);
- `resultados/context-mode-ablation` (era `context_mode_ablation.md`);
- `imputacao/wfo-lightgbm` e `imputacao/wfo-lightgbm-2m` (eram
  `data/results/wfo_lightgbm{,_2m}/relatorio.html`;
  `scripts/wfo_lightgbm_report.py`);
- `protocolo/persistence-gate-ab` (era
  `data/results/report/ab_persistence_gate.html`;
  `tests/ab_persistence_gate.py`).

Os HTML legados `relatorio_resultados.html` (em `data/results/report*/`)
e `relatorio_metodos.{md,html}` foram **removidos**: as páginas do hub
(`reports/lidar/*`, `reports/protocolo/metodos-complementares`) os
substituem e os geradores não os produzem mais.
