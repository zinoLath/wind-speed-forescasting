---
title: Banco SQLite de runs de predição (data/predictions.db)
area: dados
updated: 2026-10-22
---

# Banco SQLite de runs de predição (data/predictions.db)

Fonte canônica das execuções de avaliação/predição — substitui o rastro
solto de `evaluation.json` (sem controle do que foi executado) como base
dos relatórios. Módulo: `src/predictions_db.py`; gerado por
`tests/relatorio_resultados.py` (gravação automática ao fim de cada
avaliação; `--no-db` para pular).

## Esquema (7 tabelas)

- `run_category` — tipo geral da predição (ex.: `lidar 8020`); tabela
  externa referenciada pelas runs.
- `runs` — uma execução: `title`, `description`, **`report_id`** (chave do
  hub, ex.: `lidar/8020`), `models_dir`, split (train/val ratio),
  `denoise_level`, **`source_file`** (código gerador), **`source_cmd`**
  (linha de comando completa), **`out_file`** (arquivo de saída),
  `created_at` — o trio fonte/comando/saída + models_dir torna a run
  totalmente reprodutível.
- `run_model` — cada modelo da run: `name`, `class_name` (wrapper),
  `wrapper_key`, **`seed`**, **`homologado` + `homologado_em`** (tracking
  de quando foi homologado), loss, params, size_mb, color,
  timing/training em JSON.
- `run_model_hp` — hiperparâmetros (nome/valor, JSON quando estruturado).
- `run_model_feature` — colunas de entrada efetivas do modelo.
- `run_model_stath` — estatísticas **por horizonte** (`horizon = 0` é o
  agregado total; 1..36 por horizonte), com `regime` (`all`/`rolling`) e
  `target` (`raw`/`denoised`) — 4 linhas no horizonte 0 + 72 por modelo.
- `run_model_prediction` — cada linha predita: `prediction_id` (janela de
  origem), `timestamp` (amostra base), `origin`, `horizon`, `actual`
  (ws100 bruto real), `predicted`; `mode` = `dense` (todos os horizontes)
  ou `rolling` (autônomo, horizon = output_steps). Rodada 8020 canônica:
  218.596 linhas, DB ≈ 22 MB.

## Semântica operacional

- **PROCESSO CANÔNICO (sempre assim)**: toda avaliação de resultados grava
  a run no banco — é o comportamento padrão de
  `tests/relatorio_resultados.py` (nada a mais a fazer). `--no-db` é só
  para depuração e não deve entrar em runbooks. Regeneração de figuras
  quando a run já está no banco: `--from-db` (sem GPU). A run de um
  `report_id` é substituída a cada nova avaliação (run canônica única por
  relatório); `--keep-prior` preserva a anterior.
- **Reprodutibilidade**: `runs.source_file` + `runs.source_cmd` +
  `runs.out_file` + `runs.models_dir` + split/seed identificam exatamente
  como recriar a run; a nota de fonte nos relatórios do hub exibe esses
  campos.

- **Uma run canônica por relatório**: gravar com `report_id` igual
  substitui a run anterior (cascade); `--keep-prior` preserva. Runs sem
  `report_id` (avulsas) acumulam como histórico.
- **Relatórios leem do banco primeiro**: `reports/src/resultados.py`
  resolve `<categoria>/<nome>` em `runs.report_id`; fallback =
  `evaluation.json` (rodadas antigas). Badge "banco: run #N" + nota de
  fonte na página quando servida do banco.
- **`--from-db <id|report_id>`** regenera figuras/DM/MCS/evaluation.json
  a partir das predições gravadas, **sem GPU** (alvo wavelet e
  persistência são recomputados do dataset — determinísticos; métricas
  reproduzem a avaliação GPU bit a bit). Diagramas `15_arquitetura_*` e
  `16_wavelet_vs_real` não são regenerados nesse modo (persistem na pasta
  `images/` da fonte).
- `evaluation.json` continua sendo escrito (cache p/ figuras,
  `scripts/sbsr_tables.py`), mas não é mais a fonte primária.
- Timestamps gravados como `YYYY-MM-DD HH:MM:SS` (grade de 10 min
  regularizada; só amostras observadas, mesmo filtro da avaliação).

## CLI de controle

```bash
python -m src.predictions_db list                          # runs gravadas
python -m src.predictions_db show lidar/8020               # detalhe (id ou report)
python -m src.predictions_db categories
python -m src.predictions_db homologar lidar/8020 --model tcn  # (+ --remove)
python -m src.predictions_db prune <id>                    # remove a run
```

## Números de referência (run #1, 22/10/2026)

- `lidar/8020` rodada canônica mista (TCN MSE + LSTM ITILDE-Q): TCN MSE
  1.3142 / R² 0.514; LSTM 1.9481/0.279; LSTM-Bi 2.0941/0.225; TCN-Bi
  1.6150/0.403 (horizonte 0, alvo raw) — idênticos a
  `data/results/report_8020/evaluation.json` da mesma rodada. Seed 42 em
  todos os modelos da rodada (coluna `run_model.seed`).

## Fontes

- `src/predictions_db.py` (schema + API + CLI);
  `tests/relatorio_resultados.py` (`record_run`, `run_from_database`);
  `reports/src/resultados.py` (`_run_from_db`).
- Validação 22/10/2026: avaliação GPU → DB (218.596 predições) → rebuild
  do hub lendo do banco → `--from-db lidar/8020` reproduz métricas com
  diferença 0.00e+00 (all, rolling e 36 horizontes).
