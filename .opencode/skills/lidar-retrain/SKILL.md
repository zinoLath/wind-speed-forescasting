---
name: lidar-retrain
description: Retreina os modelos seq2seq LiDAR sob o protocolo de reamostragem 80/20 — RODADA CANÔNICA = mista 8020-itildeq (TCN MSE + LSTM ITILDE-Q) — ou variantes (ponderada, 75/25, loss ablations), regenerando os relatórios padronizados. Use quando o usuário mencionar "retreinar", "treinar de novo", "reamostragem", "80/20", "novo treino lidar", ou comandos `scripts/lidar_8020_itildeq_train.py`, `scripts/lidar_7525_train.py`, `tests/relatorio_resultados.py`, `reports/treinamento/8020`.
---

# Retreino LiDAR sob a reamostragem (80/20)

Protocolo: split temporal 80/20 **sem validação**, orçamento fixo de 60
épocas, seed 42, janela 72 passos (12 h) → horizonte 36 passos (6 h) de
10 min, alvo `ws100_wavelet` (sym18 nível 2), métricas vs `ws100` bruto.

**Rodada canônica (decidida em 20/10/2026): MISTA `8020-itildeq`** —
TCN/TCN-Bi com HPs canônicos + dilatações estáticas até 36 + perda MSE;
LSTM/LSTM-Bi com HPs do HPO 80/20 + perda ITILDE-Q. Modelos em
`pipeline/tmp/trained_8020_itildeq/`.

## Fluxo completo (canônico — rodada mista)

```bash
# 0) ambiente (obrigatório p/ TF/GPU)
source .venv/bin/activate && source scripts/tf_gpu_env.sh

# 1) BACKUP da rodada mista anterior (o treino SOBRESCREVE)
cp -r pipeline/tmp/trained_8020_itildeq \
      pipeline/tmp/trained_8020_itildeq_backup_$(date +%Y%m%d%H%M)

# 2) treino da rodada mista (losses/HPs por modelo já embutidos no script)
python scripts/lidar_8020_itildeq_train.py

# 3) regenerar figuras + evaluation.json + DM/MCS — SEM --no-baselines:
#    DL e baselines (ARIMA/RF/persistência) saem na MESMA grade de
#    origens (var implícita única — critério de consistência do artigo)
python tests/relatorio_resultados.py \
    --models-dir pipeline/tmp/trained_8020_itildeq \
    --out-dir data/results/report_8020_itildeq

# 4) tabelas LaTeX do artigo (agregado + h=36) da rodada canônica
python scripts/sbsr_tables.py --round-dir data/results/report_8020_itildeq

# 5) reconstruir o site (lidar/8020-itildeq e afins pegam os números novos)
python reports/src/builder.py build
```

Depois do treino: `python scripts/articles_sync.py sync sbr` propaga as
figuras da rodada canônica ao artigo (`articles/sbr/sync.json` já aponta
para `data/results/report_8020_itildeq/`); integração de tabelas/números
no texto pela skill **sbsr-integrate**.

## Rodada ponderada (legado — mesma mecânica, outra loss)

```bash
python scripts/lidar_7525_train.py --train-ratio 0.80   # -> trained_8020
python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020 \
    --out-dir data/results/report_8020 --no-baselines
# ablações: scripts/lidar_8020_mseloss_train.py | lidar_8020_startweight_train.py
```

Alternativa 2–4 automática: com o servidor do hub ativo
(`python reports/src/server.py`), o botão **“Regenerar com retreino”** em
`reports/lidar/8020-itildeq/` (ou o relatório da rodada escolhida) roda
treino → regeneração → rebuild, com log em
`reports/<cat>/<nome>/retrain.log`.

Sessão de pós-re treino armada: `scripts/watch_retreino_itildeq.sh
[MAX_H]` espera o treino começar/terminar e sozinho calcula os pontos
(regeneração com baselines → tabelas → build) e notifica. Rodar com
`setsid ... < /dev/null & disown` (log em
`pipeline/tmp/watch_retreino_itildeq.log`); não sobrevive a reboot.

## Regras / armadilhas

- **Sempre fazer backup no passo 1** — não há undo do treino; os backups
  existentes ficam em `pipeline/tmp/trained_8020*_backup_*`.
- **Nunca usar `--no-baselines` na rodada canônica**: a Tabela 1 do
  artigo precisa de DL e baselines na mesma grade de origens (revisão
  SBSR — inconsistência R²×MSE).
- Treino em background (se o usuário pedir p/ rodar e continuar outra
  coisa): usar `setsid ... < /dev/null & disown` com log em
  `pipeline/tmp/<nome>.log` — `nohup ... &` simples morre com o timeout da
  sessão. Ao matar processos, padrão de pgrep não pode casar com o próprio
  shell (ex.: `lidar_[7]525_train`).
- Resultados variam entre execuções (sem early stopping/validação); não
  tratar diferenças de ±0.3 MSE como regressão — compare com o relatório
  anterior em `reports/lidar/8020-itildeq/` e comente a variância.
- Curvas por época vêm de `pipeline/tmp/train_dashboard/<run>.jsonl` (o
  relatório `treinamento/8020` usa o run mais novo COMPLETO por wrapper).
- Avaliação inline do treino fica em
  `data/results/lidar_8020_features/all.json` (skill vs persistência).
