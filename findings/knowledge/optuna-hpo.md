---
title: HPO com Optuna e rodadas distribuídas
area: metodos
updated: 2026-10-23
---

# HPO com Optuna e rodadas distribuídas

Um estudo por modelo; protocolo metodológico:

- **TPESampler multivariado com agrupamento** (`group=True`), 20 ensaios
  iniciais aleatórios, 64 candidatos por iteração; **MedianPruner** (10
  ensaios de aquecimento, poda a partir da época 8) via callback Keras.
- **Objetivo: RMSE em validação com decoder em modo inferência** (não a
  val_loss teacher-forced); RMSE preferido ao MAE para penalizar erros
  grandes.
- Espaços: lr log-uniforme [1e-5, 1e-2] (LSTMs) ou [1e-4, 1e-2] (demais);
  unidades LSTM 64–512 (passo 32); dropout 0–0.5; TCN espaço compartilhado
  (filtros 32–256, kernel 2–3, 1–2 stacks, dilatação 1–5 → potências de 2);
  Transformer d_model 32–256, cabeças 1–8, camadas 1–4, ff_dim 64–512,
  dropout 0–0.3.
- Reprodutibilidade: seed 42, `K.clear_session()` entre ensaios, dados
  preparados uma única vez, contrato `hp.*` abstraído por adaptadores
  (`OptunaHyperParameters`/`FixedHyperParameters`).
- **Storage sqlite retomável**: `pipeline/tmp/optuna/<name>/optuna.db`,
  `load_if_exists=True` — apagar o `.db` para recomeçar.

## Rodadas distribuídas (round2+)

- `pipeline/pipeline.optuna_round*.json` define estudos (round2: lstm,
  lstm_bi, tcn, tcn_bi; um sqlite por wrapper em
  `pipeline/tmp/optuna_round2/`).
- `scripts/dist/colab_round2.py` planeja/empacota (pesados → notebooks Colab e
  kernels Kaggle; leves → watchdog local `scripts/dist/optuna_watchdog.sh`);
  notebooks em `notebooks/colab_round2/` — **nunca editar à mão, regenerar**.
- **Round 5 (COLETADA E PROMOVIDA em 21/10/2026)**: 2 kernels Kaggle,
  404 trials (LSTM 104), split 72/8/20, objetivo = skill bloco k=4 na
  validação (571 origens), combinação canônica (LSTM→ITILDE-Q,
  TCN→MSE) + dilatações C′. Best skill (val): LSTM-Bi **+0,378** >
  LSTM +0,348 > TCN +0,320 > TCN-Bi +0,312. Promovida para
  `pipeline/tmp/optuna/` (backups: `optuna_backup_*`,
  `Seq2Seq_TCN.bak_r4`); relatório `reports/resultados/optuna-round5/`.
  Coletor automático: `scripts/dist/watch_round5_collect.sh`.
- Heartbeats/snapshots no Drive (`MyDrive/wind-speed-colab/round2/`);
  Kaggle via `scripts/dist/kaggle_round2_worker.py` + `kaggle_runner.py`
  (snapshot a cada 10 min); `scripts/dist/colab_monitor.py` consolida
  (exit 3 = erro/stale). RAM: watchdog calibrado por ambiente (local 85%
  do MemTotal; Colab/Kaggle 82%).
- Runbook completo: skill **optuna-colab**.

## Composição do espaço: merge, nunca substituição

O espaço de busca da round 5 **não é superset** dos HPs canônicos: campos
estruturais (`context_pooling`, `encoder/decoder_dilation_rate`,
`batch_size`) não estão no espaço — e no retreino 80/20 eles **caíram
para os defaults do wrapper** em vez de herdar os canônicos, junto com um
split de HPO diferente (72/8/20 × 80/20). Resultado: TCN 0,52 → 0,29
(round canônica restaurada; rodada round5 mantida como referência em
`report_8020_round5`). **Regra**: ao treinar com HPs de uma rodada,
mesclar `base canônica + override apenas dos campos buscados` (ou
garantir que o espaço inclua todos os campos estruturais).

## Diagnóstico: por que a seleção retorna configs agressivas/instáveis (23/10/2026)

Análise dos trials round5 (`pipeline/tmp/optuna_round5/*/trials.csv`):

- **Ruído ≥ resolução da seleção**: piso de ruído de retreino ±0,04 R² e σ
  por seed 0,045–0,07 superam a separação dos top-10 trials (0,02–0,04 de
  skill). `best_trial` ponto = outlier de sorte, e o canto agressivo
  (lr alto/dropout 0) é o de maior variância ⇒ maior sorte máxima.
  Evidência: melhores HPs lr 0,0059 (LSTM) / 0,0066 (LSTM-Bi); TCN/TCN-Bi
  dropout 0,00–0,05 e wd≈0; mediana lr dos top-10 ≈ 0,006 (LSTMs).
- **MedianPruner viés contra learners lentos**: poda por `val_mae`
  teacher-forced escalado a partir da época 5 (startup 8) ⇒ 63–67% dos
  trials TCN podados; lr baixo melhora devagar e é podado, sobrevivem os
  rápidos/agressivos. Sinal de poda ≠ objetivo (skill com decoder
  inferência).
- **Espaço expõe o canto agressivo**: lr até 1,2e-2, dropout floor 0,0
  (TCN), wd até 0 — sem custo de instabilidade no objetivo.
- **Validação de 8% (571 origens) overfitta** (precedente itildeq coeffs:
  0,46 val → 0,28 teste; já registrado em estudos-ablacoes).
- **Instabilidade não mensurada**: `skill_blocks` gravado mas fora do
  objetivo; bloco 3 dos melhores trials = 0,13–0,23 vs 0,28–0,50 nos
  demais blocos.
- **Sem gradient clipping** em LSTM/TCN (só Transformer tem clipnorm
  opcional) com lr ~6e-3 — precedente de divergência (Bi+MSE → R² −1,39).
- **Transferência**: retreino 80/20 com HPs round5 promovidos = TCN 0,293
  vs 0,514 canônica (`data/results/report_8020_round5/evaluation.json`;
  agrava o bug de merge já registrado).

### Mudanças recomendadas (com confiança)

1. Reavaliar top-k (≈10) com 2 seeds extras antes de promover (90).
2. Gate de promoção: retreino 80/20 do top-2 com merge canônico; só
   promove se ≥ canônica − ruído (90).
3. Gradient clipping (clipnorm 1,0 ou HP [0,5, 1,0]) em todos os
   wrappers (85).
4. Seleção 1-SE: escolher dentro dos trials a ≤1 SE do melhor o de
   maior regularização/menor lr (85).
5. Cortar o canto do espaço: lr ≤ 2–3e-3, dropout floor 0,05–0,1 (TCN),
  wd incluir 1e-3 (80; TCN) / (70; LSTM).
6. Pruner: warmup ≥ 10 épocas, startup 15–20, reportar objetivo proxy
   com decoder inferência em vez de val_mae teacher-forced (80).
7. Objetivo com termo de estabilidade: skill_médio − λ·std(blocos),
   λ≈0,5 (ou skill mínimo) (70).
8. Ampliar validação de HPO para 12–15% ou folds purgados alternados (75).
9. Anchor trials conservadores enfileirados + n_startup 30 (65).

### Evidências quantitativas dos dbs round5 (23/10/2026)

- **Sinal de poda fraco em TODAS as épocas**: Spearman(val_mae@época e,
  objetivo final) entre COMPLETE = 0,07–0,51 para e∈{5,8,10,12,15,20};
  na época 12 ≈ +0,07 (LSTM) / −0,05 (TCN-Bi). O desacordo vem do
  mismatch teacher-forcing × decoder inferência — corrigir a métrica
  reportada ao pruner importa mais que o warmup.
- **Pruning anti-lr-baixo (TCN/TCN-Bi)**: frac de trials com lr<1e-3
  entre PRUNED = 0,38/0,43 vs COMPLETE = 0,26/0,24. PRUNED morreram na
  mediana na época 5; COMPLETE treinaram mediana 20 (LSTM) – 41 (TCN)
  épocas (ReduceLROnPlateau patience 6 ⇒ primeiro decaimento ~ép 12–18).
- **Importância fANOVA**: wd = 0,06–0,10 (nível de ruído; sweep 3 seeds
  já mostrara diferenças < ruído) ⇒ **fixar wd=1e-4** é seguro. TCN:
  dropouts dominam (0,66 somados); LSTM: lr 0,53.
- **Over-exploitation do TPE**: params duplicados exatos: LSTM 3, TCN 4,
  TCN-Bi 12/100, LSTM-Bi 0 (duplicação concentrada nos estudos TCN).
- **Custos (Kaggle+XLA, 4 wrappers)**: trial COMPLETE mediano 46s (LSTM)
  / 66s (LSTM-Bi) / 204s (TCN) / 388s (TCN-Bi); busca round5-like ≈ 6,9h
  GPU (só COMPLETE); re-eval top-10×2 seeds = 3,9h (top-5×2 = 2,0h);
  gate top-2×3 seeds×4 wrappers (retreino 80/20 ≈ 17 min) = 6,8h. Cabe em
  2 kernels Kaggle (~5h e ~6,7h) + gate na GPU local, dentro da cota de
  30h/semana.
- **Caveat wd**: importância fANOVA do wd é 0,06–0,10 nos demais, mas
  **0,19 na LSTM-Bi**; decisão de fixar wd=1e-4 mantida (sweep multi-seed
  não mostrou efeito), com mini-ablação {1e-4, 3e-4}×2 seeds se a LSTM-Bi
  da próxima rodada regredir.
- **Backtest do gate** (único contrafactual disponível, round5→80/20):
  TCN 0,293 < 0,514−0,04 → rejeita ✓; TCN-Bi 0,317 < ~0,42 → rejeita ✓;
  LSTM 0,345 ≥ 0,269 → aceita ✓; LSTM-Bi 0,311 ≥ 0,274 → aceita ✓
  (decisão correta nos 4). SE(média 3 seeds) = 0,026 (LSTM-Bi σ=0,045) a
  0,040 (LSTM σ=0,07) ⇒ limiar −0,04 ≈ teste de não-inferioridade a 1 SE.
- **Multiobjetivo (NSGA-II) × pruner incompatível** (verificado optuna
  4.9): `trial.report()` lança NotImplementedError em estudos
  multi-direção ⇒ NSGA-II = sem poda ⇒ busca sem poda dobra o compute
  (~7h→~14h). Alternativa sem custo: Pareto pós-hoc sobre o top-10
  reavaliado (skill × std de seed já gravados).

### Veredito pruner/sampler (validação 23/10/2026, optuna 4.9)

- **Pruner**: trocar MedianPruner por **SuccessiveHalvingPruner
  (min_resource=12, η=3 ⇒ rungs 12/36/60)** — poda em 2 decisões maduras
  em vez de ~55 por época, alinhado ao pós-plateau (~ép 12) e à mediana
  de épocas dos sobreviventes (20–41). Confiança 75. **HyperbandPruner
  não**: fragmenta 100 trials em brackets e o bracket agressivo
  reintroduz a poda precoce que a evidência condena; hedge de
  min_resource é desnecessário (já sabemos que ép. iniciais não informam).
- **Sampler**: manter TPE multivariado com **n_startup_trials 30** e
  **constant_liar=True se >1 worker por estudo**; se duplicados exatos
  >5% numa rodada nova, cair para QMCSampler (Sobol; espaço estático
  por wrapper comporta). CmaEsSampler inadequado (categóricos no espaço);
  GPSampler persegue outliers sob ruído alto; NSGAIISampler só se
  estabilidade virar 2º objetivo (Pareto skill×std). Confiança 70.

### Pacote decidido para a próxima rodada (23/10/2026, decisões do autor)

- **Sem re-eval multi-seed do top-k** (sem budget; foco no resultado) e
  **sem multiobjetivo** (escalar). Gate top-2×3 seeds mantido.
- Objetivo escalar com estabilidade: `skill_médio − λ·std(blocos)`,
  **λ = 0,3–0,5** (regra prática: λ×std_típico(≈0,10–0,15) deve ficar na
  faixa do spread dos top-10, 0,02–0,04 ⇒ λ≈0,3; 0,5 se quiser punir mais).
- Sem multi-seed, o conjunto 1-SE usa **SE de bloco** (std(blocos)/2,
  n=4) e o desempate continua "mais conservador" (menor lr/maior
  dropout) dentro do conjunto.
- Resto do pacote: SHP(min_resource=12, η=3) com métrica reportada =
  objetivo proxy (decoder inferência, unidades invertidas); lr ≤2e-3
  (TCN) /3e-3 (LSTM), dropout floor 0,05–0,1, **wd fixo 1e-4**
  (mini-ablação 3e-4 só se a LSTM-Bi regredir no gate); clipnorm 1,0 em
  todos; TPE n_startup 30; merge canônico na promoção.

### Round 7 (preparada 23/10/2026 — NÃO executada)

Config `pipeline/pipeline.optuna_round7.json` (estudo `round7`, storage
`pipeline/tmp/optuna_round7/`), mudanças de código:

- `pipeline/step_optuna.py`: objetivo com λ (`_skill_metrics` —
  value = −(skill − λ·std(blocos), ddof=1)); pruner configurável
  (`pruner.type=s successive_halving`; default continua Median p/ rounds
  antigas); `pruning_metric="objective"` reporta o PRÓPRIO objetivo
  (decoder inferência) nos rungs 12/36 em vez de `val_mae` teacher-forced;
  `sampler_n_startup_trials`/`sampler_constant_liar` no config.
- Wrappers: `clipnorm=1.0` default (sobrescritível por instância) em
  todos os Adam/AdamW (11 arquivos).
- `scripts/experiments/optuna_select_1se.py`: seleção 1-SE (SE = std(blocos do
  argmin)/2) + desempate conservador (lr asc, dropout desc, tamanho asc);
  escreve `best_trial.json` (1-SE) + `best_trial_raw.json` (argmin).
  **Dry-run na round5**: escolheria lr 0,00055–0,00074 (vs 0,0059/0,0066
  promovidos) com conjuntos 1-SE de 14–33 trials — evidência de que a
  regra empurra para a região estável.
- Espaço: lr [5e-4, 3e-3] (LSTM/Bi) / [5e-4, 2e-3] (TCN/Bi); dropout
  floors 0,05; **wd fixo 1e-4**; warm starts da round5 clipados aos novos
  limites + anchor conservador (lr 1e-3, dropout 0,15).
- **Loss = MSE em todos** (decisão rodada-final-sbsr). Referências do
  gate: TCN 0,514 · TCN-Bi ~0,46 · LSTM MSE ~0,42 (1 seed) · LSTM-Bi
  0,276–0,399 (ruído).
- Cache ARIMA reaproveitado da round5 (mesma fatia 72/8; 574 origens).
- **Distribuição** (assign no config; plano em
  `pipeline/tmp/colab_round2/plan.json`): kaggle-gpu1 = tcn (~5–7h),
  kaggle-gpu2 = tcn_bi (~8–10h, caminho crítico, resume via snapshot),
  local 1660 = lstm+lstm_bi (~5h). Kernels compartilham o MESMO template
  (`kaggle_round2_worker.py`), só muda `--wrappers`. Fluxo:
  `colab_round2.py kaggle` → `local --start` → `colab_monitor.py --watch
  60` → `collect --kaggle` → **`optuna_select_1se.py`** → promote.

### Round 7: lançamento e lições operacionais (24/10/2026)

- **Local (GTX 1660) CONCLUÍDO em ~2,7h**: LSTM 102 trials (35 COMPLETE),
  LSTM-Bi 100 (33 COMPLETE). 1-SE dry-run escolheu **lr 0,0005–0,0009,
  dropout até 0,31** (região conservadora; argmin cru era lr de novo
  alto). SHP poda ~2/3 no rung 12 (esperado, η=3).
- **2 falhas de push Kaggle antes do sucesso** (kernels versão 1 e 2):
  1. corrida de attach do dataset — kernels anexaram a versão ANTERIOR
     (sem `pipeline.optuna_round7.json`). Mitigação aplicada: **versionar
     dataset → confirmar conteúdo em `kaggle datasets files` → só então
     `kernels push`** (o wait "ready" do cmd_kaggle não basta).
  2. **zip do pacote saiu sem o cache ARIMA** (2,1 MB vs 2,5 MB) — o
     primeiro `package` rodou antes do cache existir e o nome
     `colab_round2_package_<dia>.zip` mascarou o stale. Regra: **validar
     `unzip -l dist/colab_round2_package_*.zip | grep arima_val_cache`
     antes de qualquer push**.
- Estado no lançamento: gpu1 (tcn) e gpu2 (tcn_bi) RUNNING >12 min
  (falhas anteriores matavam em <3 min). O zip novo também embarca o
  progresso local (dbs lstm/lstm_bi) — sem conflito, 1 wrapper/sqlite.
- Fluxo de encerramento: `collect --kaggle` → `optuna_select_1se.py`
  (todos os 4) → promote manual. Relatório do retreino criado:
  `reports/lidar/8020-optuna7/` (fábrica resultados.py; "Sem dados" até o
  promote+retreino; retrain/regenerate wired).

### Round 7: 1-SE aplicado + teste 80/20 dos candidatos (24/10/2026)

- **1-SE escrito no storage da rodada** (`optuna_select_1se.py` sem
  dry-run; NADA promovido — `pipeline/tmp/optuna/` intocado): lstm →
  **trial 10** (lr 5,0e-4 piso, 96u, enc 0,085 / dec 0,309; conjunto de
  25 trials, SE 0,0802), lstm_bi → **trial 26** (lr 9,0e-4, 160u, enc
  0,160 / dec 0,130; conjunto de 21, SE 0,0815; empate técnico com o
  argmin #70: −0,3170 vs −0,3185). Argmin cru preservado em
  `best_trial_raw.json` (o `best_trial.json` que existia antes era o
  argmin escrito pelo watchdog/collect).
- **Treino 80/20 dos 1-SE** (`scripts/experiments/lidar_8020_optuna7_train.py` —
  lê HPs DIRETO do storage round7, MSE em todos, nunca toca os
  promovidos): lstm MSE 1,8685 / skill vs persistência **−0,265**;
  lstm_bi MSE 1,8566 / **−0,257** (persistência MSE 1,4773 no teste de
  1523 origens). **Alerta: a região conservadora do 1-SE perde da
  persistência no teste 20%** — a skill positiva da validação (0,21–0,27
  vs ARIMA, 72/8) não transferiu para o teste 80/20.
- **Sweep diagnóstico do conjunto 1-SE** (`scripts/experiments/lidar_8020_optuna7_sweep.py`):
  treina TODOS os 46 candidatos (25 lstm + 21 lstm_bi) com resume
  (`data/results/lidar_8020_optuna7_sweep/sweep.json|.csv`; modelos em
  `trained_8020_optuna7_sweep/`). Diagnóstico, não seleção — escolher
  trial pelo teste seria vazamento.
- `reports/lidar/8020-optuna7/` retrain agora aponta para
  `lidar_8020_optuna7_train.py` (independe de promoção).
- **Diagnóstico do sweep (23/46 trials, 24/10/2026): o HPO round 7 não
  transfere para o teste.** r(val-obj × MSE teste) = **0,036** — o
  objetivo (skill vs ARIMA + λ·std na fatia 72/8, 4 blocos) não prevê o
  teste 20%; banda 1-SE varia 1,57–2,00 de MSE com val quase igual. O
  melhor da banda (1,5714, trial #28) ainda perde da persistência
  (1,4773). Controles na mesma grade: LSTM do artigo (trained_8020_sbsr,
  featurex 67 col) MSE **1,3887** / skill +0,060. Causas: (a) espaço
  cortado (lr ≤ 3e-3) **excluiu a região do campeão do artigo**
  (lr 11,1e-3, batch 64, 192u — refuta a premissa conservadora da
  rodada); (b) 1-SE degenerou (SE 0,080 de 4 blocos → banda = 71% dos
  COMPLETE → desempate conservador escolheu canto underfit: lr piso
  5e-4, 96u, dec drop 0,31); (c) features 46 col vs 67 featurex do
  artigo. **Não promover a round 7** sem reabrir espaço/objetivo
  (lr até ~1,2e-2, featurex p/ LSTM, MSE absoluto ou skill vs
  persistência como objetivo, mais blocos de val). Controles e tabela
  ao vivo: `reports/lidar/8020-optuna7/` (seção Sweep 1-SE).
- **Validação do fix com os dados do sweep (n=25, banda LSTM)**: nenhum
  critério de val correlaciona com o teste dentro da banda
  (r = +0,052 objetivo round7 / −0,089 skill cru / +0,030 MSE absoluto)
  — o ruído (val de 574 origens, 4 blocos, 1 seed, 60 épocas) domina;
  trials empatados na val (1,4929 vs 1,4931!) divergem 1,57 vs 1,82 no
  teste. Ainda assim o argmin por **MSE absoluto** (round 2) teria
  levado 1,5744 (#80) ao promote vs 1,8204 (#82) do objetivo round 7.
  Fix recomendado: alinhar HPO ao protocolo final (featurex, batch no
  espaço, lr ≤ 1,2e-2), objetivo = MSE absoluto por bloco, 8–10 blocos
  de val, promote por média/estabilidade entre 3 seeds no top-k da val
  (teste intocado), ou busca local ao redor do campeão do artigo.

### Round 8 (design 24/10/2026 — pós-diagnóstico do sweep; NÃO executada)

Reorganização do HPO para otimizar o que **transfere** para o teste (sem
usar o teste como critério). Config `pipeline/pipeline.optuna_round8.json`
(gerada por /tmp/opencode/gen_round8.py com HPs exatos dos campeões do
artigo); mudanças de código em `pipeline/step_optuna.py` e
`scripts/experiments/optuna_select_1se.py` (fumaca testada em CPU, 2 trials OK):

- **Objetivo novo `block_mse`**: MSE absoluto por bloco de origens na
  grade de inferência (predict_all_horizons, alvo bruto, sem horizon
  weighting) — mesma unidade/mecânica da métrica do teste do artigo.
  **k=8 blocos** (antes 4). Skill vs ARIMA continua gravada como
  user_attr de diagnóstico (cache da round 7, mesma fatia 72/8) mas NÃO
  entra no valor.
- **Espaços reabertos cobrindo os campeões do artigo** (a round 7 os
  excluía — lr dos campeões: lstm 11,1e-3, lstm_bi 3,4e-3, tcn 9,7e-3):
  lstm [1e-3, 1,5e-2], lstm_bi [5e-4, 8e-3], tcn [1e-3, 1,5e-2], tcn_bi
  [3e-4, 3e-3]; dropouts até 0,35; wd fixo 1e-4.
- **Warm starts = HPs exatos dos campeões** (`trained_8020_sbsr/*/model.json`)
  + 1 âncora conservadora — TPE explora ao redor deles.
- **Features alinhadas ao lineup final** via `keep_raw_by_wrapper`:
  LSTM/Bi featurex 67 col (humid/temp + cis1..19), TCN/Bi meteo 48 col.
  `batch_size_by_wrapper` = batches dos campeões (lstm 64, demais 32).
- **Seletor 1-SE atualizado**: blocos do objetivo (`val_mse_blocks`,
  precedência sobre `skill_blocks` diagnóstico) e `--tiebreak argmin`
  (o desempate conservador escolheu canto underfit na round 7).
- Pruner SHP (rungs 12/36 pelo objetivo real) e sampler TPE mantidos.
- Comando local: `python pipeline/step_optuna.py --config
  pipeline/pipeline.optuna_round8.json --wrappers lstm lstm_bi`;
  seleção: `optuna_select_1se.py --storage pipeline/tmp/optuna_round8
  --study round8 --tiebreak argmin`. Distribuição Kaggle: regenerar
  plan/package (skill optuna-colab) com a config nova.
- **Pendente de design** (não implementado): promote final com top-k×3
  seeds na val (média/estabilidade) antes de copiar para optuna/.
- **Revisão de design pré-lançamento (24/10/2026)** — 2 desalinhamentos
  reais achados e corrigidos, 3 verificados OK:
  1. **CORRIGIDO — callbacks**: o HPO rodava com
     `EarlyStopping(val_loss, patience 12, restore_best)` +
     `ReduceLROnPlateau` (`default_callbacks`), regime que o treino
     final 80/20 NUNCA usa (60 épocas secas, callbacks=None). Flag nova
     `final_protocol_callbacks: true` (round 8) remove ambos; resta só
     o pruning por rung. Custo: trial ruim não morre sozinho — o SHP
     continua podando nos rungs 12/36.
  2. **CORRIGIDO — reseed por trial**: `set_random_seed(seed)` dentro
     do objective() (antes só 1× em run()) — trials comparáveis entre
     si e o escolhido reproduz o treino final.
  3. **Verificado OK — LR schedule**: constante em TODOS os lugares
     (`schedule_mode_override` nunca é setado fora de ablações; o
     `schedule_total_steps` dos treinos finais é ignorado pelo wrapper).
  4. **Verificado OK — warm starts TCN no espaço**: kernel 2–3,
     stacks 1–2, depths enc 4–7/dec 3–5, filters na grade 48–208/16
     cobrem os campeões (tcn 48f 2/1/7/5; tcn_bi 144f 2/1+dec_k3/7/5).
  5. **Verificado OK — cache ARIMA** (diagnóstico): fatia 72/8 da
     round 7 é a mesma da round 8 (mesmos ratios) — cache reutilizado.
  Smoke re-testado (1 época, 1 trial, sem ES) — determinístico e ok.
- **Teste de regime (argmin lstm #82, 24/10/2026)**: mesmos HPs, treino
  no regime do HPO (72/8 + ES patience 12 + RLRP) → ES parou na época 27
  e o teste de 20% cai de **1,8204 (regime final seco, sweep) para
  1,4878** (skill −0,232 → −0,012, ~empate com a persistência). Duas
  conclusões: (a) o regime final seco overfitta esses HPs — os pesos da
  melhor época generalizam melhor; (b) o teto dos HPs round 7 é o nível
  da persistência mesmo no melhor regime (artigo: 1,3887/+0,060). E a
  val de 8% segue otimista: skill vs persistência +0,523 na val vs
  −0,012 no teste. **Questão aberta p/ a round 8**: manter
  `final_protocol_callbacks` (alinhamento seco) ou adotar ES+RLRP no
  protocolo final também — mudaria o protocolo do artigo. Script:
  `scripts/experiments/lidar_optuna7_regime_test.py`; modelo em
  `pipeline/tmp/trained_optuna7_hpo_regime/lstm/`.
- **Revalidação SBSR (reval-sbsr, 24/10/2026)**: lineup canônico do
  artigo com split 72,5/12,5/15 + ES(patience 12)+RLRP — LSTM MSE
  **1,2455 / R² 0,626**, à frente do ARIMA (1,3199) pela primeira vez em
  rodada com val real; TCN 1,3353; tcn_bi 1,3925; lstm_bi 1,4634
  (report_reval_sbsr, trim 72). O regime com validação supera o seco —
  reforça o papel do EarlyStopping (pesos da melhor época). Análises por
  horizonte no report: Friedman (H₀ rejeitada 36/36; LSTM domina h ≥ 18)
  e CD de Wilcoxon-Holm por horizonte (LSTM melhor rank em ~27/36,
  TCN-Bi no curtíssimo prazo; persistência última de h ≥ 23). Relatório
  `reports/lidar/reval-sbsr/` com 3 figuras estáticas
  (`figures_static/` — figures/ é wipada a cada build, resultados.py
  recopia de images/).

## Fontes


- `docs/metodologia.md` (seção 4); `agents/optuna-colab.md`;
  `pipeline/step_optuna.py`; `pipeline/pipeline.optuna_round2.json`.
