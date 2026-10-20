---
name: lidar-retrain
description: Retreina os modelos seq2seq LiDAR sob o protocolo de reamostragem 80/20 (ou variantes 75/25, loss ablations) e regenera os relatórios padronizados com os números novos. Use quando o usuário mencionar "retreinar", "treinar de novo", "reamostragem", "80/20", "novo treino lidar", ou comandos `scripts/lidar_7525_train.py`, `tests/relatorio_resultados.py`, `reports/treinamento/8020`.
---

# Retreino LiDAR sob a reamostragem (80/20)

Protocolo canônico: split temporal 80/20 **sem validação**, orçamento fixo
de 60 épocas, seed 42, janela 72 passos (12 h) → horizonte 36 passos (6 h)
de 10 min, alvo `ws100_wavelet` (sym18 nível 2), métricas vs `ws100` bruto.
Modelos em `pipeline/tmp/trained_8020/` (lstm, lstm_bi, tcn, tcn_bi).

## Fluxo completo

```bash
# 0) ambiente (obrigatório p/ TF/GPU)
source .venv/bin/activate && source scripts/tf_gpu_env.sh

# 1) BACKUP dos modelos canônicos (o treino SOBRESCREVE trained_8020)
cp -r pipeline/tmp/trained_8020 pipeline/tmp/trained_8020_backup_$(date +%Y%m%d%H%M)

# 2) treino (~17 min de GPU: lstm 2min, lstm_bi 5min, tcn 3min, tcn_bi 7min)
python scripts/lidar_7525_train.py --train-ratio 0.80
#   variante 75/25: --train-ratio 0.75  ->  trained_7525
#   ablações: scripts/lidar_8020_mseloss_train.py (MSE pura),
#             scripts/lidar_8020_startweight_train.py (loss invertida)

# 3) regenerar figuras + evaluation.json (~5 min GPU)
python tests/relatorio_resultados.py \
    --models-dir pipeline/tmp/trained_8020 \
    --out-dir data/results/report_8020 --no-baselines

# 4) reconstruir o site (treinamento + lidar/8020 pegam os números novos)
python reports/src/builder.py build
```

Alternativa 2–4 automática: com o servidor do hub ativo
(`python reports/src/server.py`), o botão **“Regenerar com retreino”** em
`reports/lidar/8020/` ou `reports/treinamento/8020/` roda treino →
regeneração → rebuild, com log em `reports/<cat>/<nome>/retrain.log`.

## Regras / armadilhas

- **Sempre fazer backup no passo 1** — não há undo do treino; os backups
  existentes ficam em `pipeline/tmp/trained_8020_backup_*`.
- Treino em background (se o usuário pedir p/ rodar e continuar outra
  coisa): usar `setsid ... < /dev/null & disown` com log em
  `pipeline/tmp/<nome>.log` — `nohup ... &` simples morre com o timeout da
  sessão. Ao matar processos, padrão de pgrep não pode casar com o próprio
  shell (ex.: `lidar_[7]525_train`).
- Resultados variam entre execuções (sem early stopping/validação); não
  tratar diferenças de ±0.3 MSE como regressão — compare com o relatório
  anterior em `reports/lidar/8020/` e comente a variância.
- Curvas por época vêm de `pipeline/tmp/train_dashboard/<run>.jsonl` (o
  relatório `treinamento/8020` usa o run mais novo COMPLETO por wrapper).
- Avaliação inline do treino fica em
  `data/results/lidar_8020_features/all.json` (skill vs persistência).
