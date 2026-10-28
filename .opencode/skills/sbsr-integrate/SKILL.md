---
name: sbsr-integrate
description: Integra um relatório de resultados (data/results/report_*) no artigo SBSR — atualiza as duas tabelas de métricas (tab:error_all, tab:error_h36), os números citados no texto/Resumo, a descrição da função de perda e as figuras sincronizadas, com verificação de consistência (variância implícita uniforme, figura=tabela da mesma rodada). Use quando o usuário mencionar "integrar report", "integra o report X no artigo", "atualizar o artigo/tabelas/métricas/gráficos do SBSR", ou trocar a rodada que alimenta o paper.
---

# Integrar um report no artigo SBSR

Dado um relatório canônico `data/results/<REPORT>/` (ex.: `report_8020`,
`report_8020_mseloss`), atualizar o artigo
(`articles/sbr/LaTeX_Template_SBSR_Portugues/`) para que **tabelas,
texto e figuras venham todos dessa única rodada** — o erro clássico que
este fluxo existe para eliminar é tabela de uma rodada e figura de
outra.

## 0) Pré-checagens

```bash
ls data/results/<REPORT>/          # precisa ter: evaluation.json,
                                   # relatorio_resultados.html, images/, mcs.json
python -c "import json;d=json.load(open('data/results/<REPORT>/evaluation.json'));print({k:(v.get('loss'),v['all']['r2']) for k,v in d.items()})"
```

- Artefatos faltando (ou baselines ausentes na rodada): **regenerar
  primeiro** com `python reports/src/relatorio_resultados.py --models-dir
  pipeline/tmp/trained_8020 --out-dir data/results/<REPORT>` **sem**
  `--no-baselines` (assim baselines e DL saem na mesma grade de
  origens; fluxo de treino completo = skill `lidar-retrain`).
- Anote o campo `loss` de cada modelo: se **todos iguais**, a frase da
  perda no `03_Material_e_metodos.tex` deve descrever esse loss único;
  se mistos, descrever por família (decisão com o usuário).

## 1) Extrair os números

**Tabelas de métricas** (agregado e h=36) — estão prontas no HTML do
relatório, já formatadas com vírgula decimal. Parser stdlib
(pandas.read_html falha — sem lxml no venv):

```python
from html.parser import HTMLParser
class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.t, self.row, self.cell = [], None, None, None
    def handle_starttag(self, tag, attrs):
        if tag == 'table': self.t = []
        elif tag == 'tr' and self.t is not None: self.row = []
        elif tag in ('td','th') and self.row is not None: self.cell = ''
    def handle_data(self, data):
        if self.cell is not None: self.cell += data
    def handle_endtag(self, tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(self.cell.strip()); self.cell = None
        elif tag == 'tr' and self.row is not None:
            if self.row: self.t.append(self.row)
            self.row = None
        elif tag == 'table' and self.t is not None:
            self.tables.append(self.t); self.t = None

p = Tables()
p.feed(open('data/results/<REPORT>/relatorio_resultados.html').read())
mets = [t for t in p.tables if t and t[0][:2] == ['Método', 'R²']]
# mets[0] = agregado (todos os horizontes), mets[1] = h=36
```

- Se o relatório foi gerado **com** baselines, as tabelas já têm as 7
  linhas (baselines com sufixo "(baseline)" — no artigo o rótulo é sem
  sufixo).
- Se gerado **com `--no-baselines`**: completar os baselines a partir
  de `data/results/baselines/arima_rf/metrics.json` (`arima`,
  `persistence` em `all_horizons`; RF em
  `data/results/gate_test_8020/summary.json`) e, para o h=36,
  `data/results/baselines/arima_rf/per_horizon_metrics.csv`
  (última linha; `mse = rmse²`). **Só é válido se o gate da variância
  (passo 2) passar** — senão regenere com baselines.

## 2) Gates de consistência (não editar o .tex antes de passar)

1. **Variância implícita uniforme** (achado da Vanessa): para **cada**
   linha da tabela agregada, `Var = MSE/(1−R²)` deve ser igual dentro
   de ~1% entre DL e baselines. Se os baselines mesclados divergirem
   (ex.: 2,70 vs 2,78), os conjuntos de origens diferem → **regenerar
   o relatório com baselines** em vez de mesclar.
2. **Figura = tabela**: as fontes em `articles/sbr/sync.json` devem
   apontar para `data/results/<REPORT>/images/` (passo 4). O heatmap
   só é consistente com as tabelas novas se vier da mesma rodada.
3. `mcs.json` presente (validação DM disponível); se ausente, avisar.

## 3) Editar o artigo

### `04_Resultados.tex`
- **`tab:error_all`** e **`tab:error_h36`**: substituir o corpo das
  tabelas pelos valores extraídos. Regras: manter a estrutura LaTeX
  (`|c|c|c|c|c|`, `\hline`); rótulos canônicos `ARIMA`, `TCN - Bi`,
  `TCN`, `LSTM - Bi`, `Random Forest`, `LSTM`, `Persistência`;
  **ordenar por MSE crescente** (≈ R² decrescente); manter vírgula
  decimal e 3 casas.
- **Par. 1**: atualizar o nº de predições ("1406 predições") =
  `evaluation.json → all.samples / 36` da rodada (a grade de mesma
  origem muda a contagem; ~1477). Conferir também "1513 amostras"
  (fixo do split 80/20 de 7561).
- **Par. 2** (ARIMA degrada, DL melhor em horizontes distantes):
  verificar contra a tabela h36 nova; só reescrever se contradito —
  senão deixar.
- **Par. 3**: recomputar as "vantagens de R²" (diferenças
  unidirecionais e bidirecionais) e a direção das comparações (ex.: na
  rodada ponderada TCN puro ganha do bidirecional — a frase atual
  "vantagem de 0,8 / 0,4" já estava errada).

### `01_Resumo.tex` (PT **e** EN)
- "R² de até X, e até Y no horizonte de 6 h, alcançado com
  <arquitetura>": X = melhor R² agregado, Y = melhor R² no h=36,
  arquitetura = qual modelo alcançou (pode deixar de ser TCN-Bi!).

### `03_Material_e_metodos.tex`
- Parágrafo da função de perda conforme o campo `loss` da rodada
  (`report_8020` = `horizon_weighted_mse` em todos → "todos os
  modelos usam a perda ponderada"; `report_8020_mseloss` = MSE em
  todos). É mudança semântica: **confirmar com o usuário** antes de
  reescrever.

### Não tocar automaticamente
- `06_Conclusoes.tex` (alegações qualitativas sobre bidirecionalidade
  — listar no resumo final para o usuário reavaliar), `07_SBSR.bib`,
  labels/estrutura das tabelas, figuras estáticas
  (`fig1_local_estudo`, `fig_model_struct`, `fig_lstm_cell`,
  `fig_convolutions`).

## 4) Figuras

1. Atualizar `articles/sbr/sync.json`: as 3 fontes
   (`fig_mse_heatmap.png`, `fig_r2_heatmap.png`,
   `fig_dataset_division.png`) apontando para
   `data/results/<REPORT>/images/…` (mesmos nomes de arquivo).
2. Copiar localmente (SEM commit):
   `cp data/results/<REPORT>/images/06_heatmap_mse_horizonte.png
   articles/sbr/LaTeX_Template_SBSR_Portugues/figuras/fig_mse_heatmap.png`
   (idem `06_heatmap_r2_horizonte.png` → `fig_r2_heatmap.png`,
   `01_split_temporal.png` → `fig_dataset_division.png`).
3. Commit+push Overleaf (`python scripts/articles_sync.py sync sbr`)
   **somente se o usuário pedir** — é o ponto de commit dos artigos.

## 5) Fechamento

Apresentar ao usuário: tabela antigo→novo de cada número trocado, os
gates que passaram, e a lista de "alegações qualitativas a reavaliar"
(Conclusões par. 2–3, par. 3 dos Resultados, frase da perda). Compilação
local falha por falta de `babel-spanish` — validação visual final é no
Overleaf.

## Armadilhas

- `evaluation.json` atual **não tem** métricas por horizonte — por isso
  o h=36 vem do HTML (ou do CSV de baselines). Não inventar: se faltar
  fonte para h36, regenere o relatório.
- O relatório ordena por MSE; rótulos de baseline no HTML vêm com
  "(baseline)" — normalizar para os rótulos canônicos do artigo.
- `scripts/articles_sync.py sync sbr` **commita e empurra**; usar
  `--no-push` ou cp manual quando o usuário só quis ver a integração.
- Números do Resumo PT e EN precisam ficar idênticos (um em vírgula,
  outro em ponto).
