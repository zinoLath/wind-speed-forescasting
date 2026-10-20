---
title: Estudos e ablações concluídos (mapa de evidências)
area: resultados
updated: 2026-10-19
---

# Estudos e ablações concluídos (mapa de evidências)

Índice dos estudos que sustentam as decisões atuais — onde procurar
evidência antes de re-abrir uma questão.

- **Alvo cru vs wavelet**: `tests/ab_raw_vs_wavelet.py` →
  `docs/ab_target_denoise.md` (denoised venceu; base da escolha do alvo).
- **Loss**: ablações 80/20 — MSE pura (`reports/lidar/8020-mseloss`), loss
  invertida (`lidar_8020_startweight_train.py`), duel loss
  (`reports/lidar/duel-loss`); conclusão: MSE ponderada por horizonte
  permanece.
- **HPO dedicado**: `data/results/{lstm_hpo_8020, lstm_bi_hpo_8020,
  lstm_bihp_8020, optuna_round4*, optuna_round5, attention_dropout_trials}`,
  relatório `reports/lidar/8020-optuna`; LSTM units sweep
  (`data/results/lstm_u{192,224,256}_8020`) + scripts
  `lidar_lstm_units_*.py`.
- **Seeds/robustez**: `lidar_8020_itildeq_seeds_train.py` →
  `data/results/itildeq_seeds_8020` + `reports/lidar/8020-itildeq-seeds`;
  sem normalização: `reports/lidar/nonorm`.
- **Gate/persistência**: `tests/lidar_gate_test.py` →
  `data/results/gate_test_8020{,_static}`; `pipeline.train_compare_*`.
- **TCN dilatações**: `data/results/dm_{dilations,tcn_dec5,tcn_full,
  tcn_r4b,tcn_r4blocal,vs_hpo}` (comparações Diebold-Mariano de variantes
  TCN).
- **Origens LSTM-Bi**: `reports/lidar/lstm-bi-origins`
  (`lidar_lstm_bihp_ab_train.py`, `data/results/lstm_bihp_ab_8020`).
- **Features study**: `tests/feature_study.py` +
  `tests/build_feature_study_report.py` → `reports/resultados/features_study`;
  investigação de baselines: `reports/resultados/investigacao`.
- **WFO LightGBM/RF**: `scripts/wfo_*` → `reports/imputacao/wfo-*`.
- Código aposentado: `OLD/` (ver README lá; `run_simulation.sh` referencia
  `src/simulation.py` que nunca existiu).

## Fontes

- `data/results/` (listagem); `reports/` (categorias); `agents/gotchas.md`.
