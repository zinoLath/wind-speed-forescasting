---
title: Estudos e ablações concluídos (mapa de evidências)
area: resultados
updated: 2026-10-22
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
  **Bateria de significância sem retreino (22/10/2026,
  `tests/seeds_significance_8020.py` → `data/results/itildeq_seeds_8020/
  significance/`)**: Wilcoxon pareado por seed (n=10) — LSTM × LSTM-Bi
  empata em all (p=1,0) e LSTM-Bi vence em rolling (ΔR² +0,08, p=0,049);
  **ambas perdem de forma conclusiva da persistência em all** (0–10,
  p=0,002) e só LSTM-Bi vence em rolling (9–1, p=0,0039). Friedman sobre
  11 dias (4 canônicas seed 42, db): χ²=11,40 p=0,0098; ranks TCN 1,45 <
  TCN-Bi 2,45 < LSTM 2,91 < LSTM-Bi 3,18; pós-hoc Holm só separa
  **LSTM-Bi das TCNs** (p=0,012/0,034) — TCN não é significativamente
  melhor que LSTM uni (p_holm=0,21). KSPA (|erro|, bootstrap blocos 144
  origens): **nenhum par difere** (p>0,51) — caudas indistinguíveis.
  Ressalva: KSPA com poucos blocos independentes é conservador.
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
- **Feature set estendido (rodada 8020-featurex, 21/10/2026)**:
  `reports/lidar/8020-featurex` — entrada 46→67 colunas (+cis1..19,
  humid, temp via `keep_raw`; infra: `forecast_redundant_columns` agora
  honra `keep` p/ cis/metadados, avaliador lê `feature_keep` do
  model.json — inclusive por modelo, o que permite **lineups mistos**,
  ex. `report_8020_hybrid`: TCNs canônicas + LSTMs featurex) + **MSE
  pura em todos os wrappers**. Avaliação completa: TCN 0,494 / LSTM
  0,485 / TCN-Bi 0,401 / LSTM-Bi 0,379 (mesma grade dos baselines:
  ARIMA 0,508, RF 0,418, Pers 0,388); **DM 10 pares 0 significativos;
  MCS α=0,10 manteve só a TCN** (híbrido: manteve todos os 7).
  Achados: LSTM ITILDE-Q era o gargalo (0,318→**0,498** com MSE); TCN
  insensível aos atributos; TCN-Bi caiu (−0,117) mas **em 3 seeds vira
  0,474±0,063 = empate com a canônica** (o despenco era seed azarada);
  LSTM-Bi em regime de interpolação — memoriza o treino com 128 ou 256
  units e dropout 0,083/0,25/0,36; **HPO restrito (40 trials,
  dropout≥0,20, wd fixo 1e-3) não bateu os HPs canônicos** (0,366 teste
  vs 0,386 manual; multi-seed do vencedor: 0,333±0,065);
  **sweep de wd (1e-3/1e-4/1e-5 × 3 seeds): wd não é o gargalo**
  (0,344/0,378/0,320 — diferenças < ruído por seed; 1e-5 = mais
  instável). **Piso de ruído de retreino ≈ ±0,04**
  (`trained_8020_repro42`; TCN-Bi oscilou +0,043 só de retreinar).
  Decoder 2× do LSTM-Bi (`lstm_units*2`) é hardcode da implementação
  original (14/07), para casar com o estado concatenado — nunca
  revisited; correção estrutural (projeção Dense/decoder independente)
  = rodada nova com HPO próprio. **Ablação de alturas (40/100/140)**
  (`trained_8020_heights3_*`, 3 seeds, MSE): TCN 0,51 e TCN-Bi 0,51
  **não perdem com 3 alturas**; LSTM cai a 0,36 (precisa do perfil
  completo); **press é tóxica** (MI 0,005; TCN 0,51→0,28, TCN-Bi
  0,51→0,34); **disp/vdisp cru é contra-indicado** (TCN −0,33; só
  TCN-Bi recupera 0,479); receita final: TCN h3 14col 0,514 · TCN-Bi
  h3+cis1/7/11+humid/temp 0,510 · LSTM featurex 67 0,498.
  **Variância por horizonte** (`scripts/tcn_horizon_variance.py` →
  `data/results/tcn_horizon_variance/`): TCN-Bi tem ~1,5× a variância
  da TCN em todas as configs (std 0,31–0,44 vs 0,22–0,34); em h1 a
  TCN-Bi é melhor (0,34–0,44 vs 0,51–0,69) mas degrada ~5× até h36
  (TCN ~2,5×); TCN h3 tem a menor variância de todas (std 0,217).
  **CV purgada da LSTM (5 e 3 dobras, purge 108 = 72+36)**
  (`scripts/lstm_heights3fx_cv5.py` →
  `data/results/lstm_heights3fx_cv/`): R² médio 0,040±0,41 (5f) e
  0,136±0,20 (3f), com dobra inicial negativa — **o headline ~0,50 é
  dependente do bloco** (testar sempre no último quinto inflava a
  média); comparações pareadas entre modelos no mesmo bloco seguem
  válidas. **MCS horizonte a horizonte** (α=0,10/0,05,
  `data/results/report_8020_hybrid/mcs_horizon.json`): nos horizontes
  finais todos os 7 métodos ficam no conjunto de confiança. Status:
  estudo, não canônico; falta A/B "LSTM MSE com 46 features" para
  separar loss de atributos.
- **WFO LightGBM/RF**: `scripts/wfo_*` → `reports/imputacao/wfo-*`.
- Código aposentado: `OLD/` (ver README lá; `run_simulation.sh` referencia
  `src/simulation.py` que nunca existiu).

- **Termos ITILDE-Q (busca curta no Kaggle, 21/10/2026)**: 9 combos
  (α,β,γ) em LSTM com HPs canônicos, 1 treino/combo (seed 42) —
  `data/results/itildeq_terms_search/terms_search.json` +
  `scripts/itildeq_terms_kaggle.py`. **Valores dos papers**: Lee et al. 2022 (TILDE-Q): mistura = 0,5 e γ = 0,01 (recomenda γ pequeno); Dong Sheng 2026 (ITILDE-Q): α=β=γ=0,5. O default do repo (0,3/0,5/0,1) é de nenhum deles. **MSE pura (α=0) ganhou de todos**
  (R² all 0,42 vs 0,26 do default 0,3/0,5/0,1 — 8º de 9); α menor =
  melhor. **Com hweight na análise**: ranking LSTM = MSE pura 0,42 >
  ITILDE-Q ~0,32±0,06 (3 runs) > hweight ~0,27±0,01 (2 runs);
  único pareado formal (lossab, DM n=1203): ITILDE-Q > hweight
  (ΔMSE −0,33, p=0,0105); vs baselines: MSE pura ≈ RF no agregado e
  supera TODAS no h36. Micro-busca (α ≤ 0,05, Kaggle 21/10): LSTM melhores 0,379 (α=0,01,
  γ=0) / 0,346 (α=0,01,γ=0,01); LSTM-Bi MSE pura 0,399 (Kaggle) × 0,276
  (local) com itilde-q 0,295–0,347 — Bi inconclusiva (ruído). **Optuna de
  coeficientes (local, protocolo round 5 honesto: 72/8/20, objetivo na
  validação, teste intocado)**: melhor α=0,0170 · β=0,671 · γ=0,0035
  (val_block_mse 1,7876). **Teste puro (corrigido): R² 0,2765 / MSE 1,947
  (h36 0,205)** — NÃO supera a MSE pura (0,42) nem a canônica (0,318);
  o número 0,4597 visto antes vinha de avaliar em val+teste (janela
  contaminada) — retirado. Coeficientes minúsculos não confirmam ganho
  no teste; 1 seed. A seleção na validação de 8% provavelmente overfitta
  essa fatia.
  Report: `protocolo/itildeq-coeffs` (timer systemd popula). **round6
  (TPU, LSTM-Bi+MSE, 80/20, objetivo no TESTE) é exploratória — não
  reportável (vazamento de seleção)**.
  Ressalva: runs únicos (ruído de seed ±0,07);
  sem pareado MSE pura × ITILDE-Q. **LSTM-Bi + MSE pura** (21/10,
  HPs canônicos): R² all 0,276/MSE 1,948, h36 0,167 — NÃO supera
  itilde-q (0,3285 canônica; 0,314±0,045 em seeds) nem hweight (0,292):
  para a Bi, a perda é indiferente dentro do ruído (diferente da LSTM
  uni, onde MSE pura venceu com folga). Nota: com HPs errados (lr da
  uni) o treino Bi+MSE divergiu (−1,39) — instabilidade real. Impacto em
  aberto: narrativa de loss do artigo (itilde-q × MSE nos LSTM).

## Fontes

- `data/results/` (listagem); `reports/` (categorias); `agents/gotchas.md`.
