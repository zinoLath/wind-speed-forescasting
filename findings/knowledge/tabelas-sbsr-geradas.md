---
title: Gerador de tabelas SBSR (tab:error_all / tab:error_h36)
area: artigos
updated: 2026-10-19
---

# Gerador de tabelas SBSR (`scripts/sbsr_tables.py`)

Gera `tab:error_all` e `tab:error_h36` prontas para colar no artigo — já
com **caption e `\label` configurados**, linhas ordenadas por R²
decrescente, colunas Método/R²/MSE/MAE/RMSE e separador decimal do
template (vírgula para sbsr/steam). Usa `reports/src/latexfmt.py`
(formatos sbsr/steam/generic).

## Semântica das tabelas (validada contra o artigo e os artefatos)

- **tab:error_all**: agregado de todas as origens/horizontes — seq2seq do
  bloco `all` do `evaluation.json` + baselines do bloco `all_horizons` de
  `data/results/baselines/arima_rf/metrics.json`.
- **tab:error_h36**: horizonte 36 — seq2seq de `per_horizon` (h=36) do
  `evaluation.json` (≈ rolling; diferenças < 0.002 MSE) + baselines do
  bloco `rolling` (= h36 exato; alternativa `per_horizon_metrics.csv`).
- Números de baselines batem 1:1 com as tabelas atuais do artigo
  (ARIMA 0.508/0.316, RF 0.418/0.290, Persistência 0.388/0.122).

## Uso

```bash
python scripts/sbsr_tables.py                                    # stdout sbsr
python scripts/sbsr_tables.py --out reports/lidar/8020/tabelas_sbsr.tex
python scripts/sbsr_tables.py --format steam | --format generic
python scripts/sbsr_tables.py --round-dir data/results/report_<rodada>
python scripts/sbsr_tables.py --trim-boundary-origins 72         # baselines na grade cortada
python scripts/sbsr_tables.py --no-baselines --decimal point --nd 2
python scripts/sbsr_tables.py --strict                           # aborta se variância implícita divergir
# labels/captions sobrescrevíveis: --label-all/--caption-all/--label-h36/--caption-h36
```

## Vigia de variância implícita (23/10/2026)

`check_implied_variance` roda sobre as duas tabelas: se `MSE/(1−R²)`
divergir >3% entre linhas, avisa (ou **aborta com `--strict`**) — é o
safeguard contra o problema das tabelas misturando rodadas
(`variancia-implicita.md`). Testado: flagou TCN-Bi injetada com var 2,70
vs 2,78 do resto. Desligável com `--no-variance-guard`.

## Persistência de per_horizon

- `reports/src/relatorio_resultados.py` agora grava `per_horizon`/`per_horizon_den`
  (records) em cada modelo do `evaluation.json` — rodadas novas já nascem
  completas.
- Rodadas antigas sem `per_horizon`: o script **recomputa dos pesos**
  (`pipeline/tmp/trained_<suffix>`; exige venv + `tf_gpu_env.sh`, ~30 s)
  e persiste de volta no `evaluation.json` (`--no-persist` desliga).
- Artefato canônico corrente: `reports/lidar/8020/tabelas_sbsr.tex`
  (rodada report_8020; TCN lidera as duas tabelas — 0.514 all / 0.403 h36).

## Fontes

- `scripts/sbsr_tables.py`; `reports/src/latexfmt.py`;
  `data/results/baselines/arima_rf/metrics.json`;
  `reports/src/relatorio_resultados.py` (persistência per_horizon);
  `articles/sbr/LaTeX_Template_SBSR_Portugues/04_Resultados.tex` (convenção).
