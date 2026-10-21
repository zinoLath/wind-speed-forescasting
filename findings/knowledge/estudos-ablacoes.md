---
title: Estudos e ablações concluídos (mapa de evidências)
area: resultados
updated: 2026-10-20
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
  **resultado coletado do Kaggle (20/10/2026, 10 seeds 42–51)**: LSTM
  R² all 0,309±0,070 (melhor seed 47: 0,392; bootstrap rank 2,35),
  LSTM-Bi 0,314±0,045 (melhores 43/46, indistinguíveis; CV 6,6% =
  mais estável); **seed 42 canônica NÃO é a melhor** (rank ~4,4–4,7);
  **persistência ganha de todas as seeds em `all`** (0,446) mas perde
  em rolling p/ várias seeds (ex. LSTM-Bi 50: 0,324 vs 0,181).
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

- **Termos ITILDE-Q (busca curta no Kaggle, 21/10/2026)**: 9 combos
  (α,β,γ) em LSTM com HPs canônicos, 1 treino/combo (seed 42) —
  `data/results/itildeq_terms_search/terms_search.json` +
  `scripts/itildeq_terms_kaggle.py`. **MSE pura (α=0) ganhou de todos**
  (R² all 0,42 vs 0,26 do default 0,3/0,5/0,1 — 8º de 9); α menor =
  melhor. **Com hweight na análise**: ranking LSTM = MSE pura 0,42 >
  ITILDE-Q ~0,32±0,06 (3 runs) > hweight ~0,27±0,01 (2 runs);
  único pareado formal (lossab, DM n=1203): ITILDE-Q > hweight
  (ΔMSE −0,33, p=0,0105); vs baselines: MSE pura ≈ RF no agregado e
  supera TODAS no h36. Ressalva: runs únicos (ruído de seed ±0,07);
  sem pareado MSE pura × ITILDE-Q. **LSTM-Bi + MSE pura** (21/10,
  HPs canônicos): R² all 0,276/MSE 1,948, h36 0,167 — NÃO supera
  itilde-q (0,3285 canônica; 0,314±0,045 em seeds) nem hweight (0,292):
  para a Bi, a perda é indiferente dentro do ruído (diferente da LSTM
  uni, onde MSE pura venceu com folga). Nota: com HPs errados (lr da
  uni) o treino Bi+MSE divergiu (−1,39) — instabilidade real. Impacto em
  aberto: narrativa de loss do artigo (itilde-q × MSE nos LSTM).

## Fontes

- `data/results/` (listagem); `reports/` (categorias); `agents/gotchas.md`.
