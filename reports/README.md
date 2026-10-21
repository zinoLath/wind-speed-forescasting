# Relatórios padronizados (hub)

Site estático + micro-servidor para todos os relatórios do projeto,
organizados por categoria em pastas — **cada pasta de relatório contém o
código que o gera e todos os arquivos que ele usa**.

```
reports/<categoria>/<nome>/
  report.py   (gera o HTML; ou report.md/report.html estático + meta.json)
  figures/    (imagens usadas/copiadas pelo gerador)
  retrain.log / regenerate.log / job.json  (criados pelo servidor)
```

Categorias atuais: `treinamento/`, `lidar/`, `resultados/`, `imputacao/`,
`protocolo/`. O **hub** (`_site/index.html`, sempre reconstruído a cada
build) lista tudo; cada página tem topbar com dropdowns por categoria
(segundo a ordem das pastas).

## Uso rápido

```bash
python reports/src/builder.py build     # gera/regenera o site inteiro
python reports/src/server.py            # http://127.0.0.1:8765/ + API
python reports/lidar/8020/report.py     # reconstrói a partir de um relatório
```

## Popup LaTeX

Todo inserível (figura/tabela) tem botão **LaTeX** com:

- formatos **steam** (classe imc/UNIFEI), **sbsr** (template do simpósio) e
  genérico — convenções reais dos projetos (`src/latexfmt.py` é a fonte
  única, embutida na página para o JS);
- textboxes para `\label` (prefixo `fig:`/`tab:` garantido), legenda,
  arquivo, largura e posição flutuante;
- botão de copiar (clipboard com fallback).

## Regenerar / retreinar

Com o servidor ativo, cada página tem:

- **↻ Regenerar** — roda o comando `regenerate` do relatório (ex.:
  `tests/relatorio_resultados.py` para reavaliar os modelos e refazer as
  figuras) e reconstrói o site;
- **▶ Regenerar com retreino** — roda o `retrain` em background (ex.:
  `scripts/lidar_7525_train.py --train-ratio 0.80`), com log em
  `retrain.log`; ao terminar, a página recarrega com os números novos
  (para os relatórios LiDAR, o retreino dispara a regeneração do
  relatório de resultados; atenção: sobrescreve os modelos canônicos —
  há backups `pipeline/tmp/trained_8020_backup_*`);
- **⟳ Reconstruir página** — só re-renderiza o HTML a partir dos dados
  atuais (rápido, sem GPU).

Framework em `reports/src/` — detalhes para agentes/IA em `agents/reports.md`
e para humanos no código (`builder.py`, `components.py`, `server.py`).
