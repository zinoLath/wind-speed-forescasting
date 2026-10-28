# agents/gotchas.md — armadilhas importantes

> Base canônica: `findings/knowledge/gotchas.md` (skill `canonical-findings`).

- **`src/common.py` deve continuar sem efeitos colaterais de import**: o
  TensorFlow é configurado via `common.setup_tensorflow()` ANTES dos imports
  de modelo (lazy nos estágios). Não adicionar imports de keras/tf no nível
  de módulo lá.
- **pacote `pipeline/` vs módulo `pipeline/pipeline.py`**: todo arquivo sob
  `pipeline/` insere a raiz do projeto em `sys.path[0]` antes dos imports
  `from pipeline import ...` — senão `pipeline/pipeline.py` sombreia o
  pacote. Manter o bloco de bootstrap ao adicionar estágios. O mesmo vale
  para os `report.py` em `reports/` (bootstrap para a raiz do projeto).
  Profundidade do `parents[n]`: raiz de `scripts/` e `studies/` = 1;
  `scripts/experiments/`, `scripts/dist/` e `reports/src/` = 2.
- **Estudos Optuna são persistidos em sqlite**
  (`pipeline/tmp/optuna/<name>/optuna.db`) e resumem com
  `load_if_exists=True`; apague o `.db` para recomeçar do zero.
- **pgrep/pkill com padrão de comando**: ao matar treinos, cuide para o
  padrão não casar com o próprio shell (use colunas de caractere, ex.
  `lidar_[7]525_train`).
- **Jobs em background no shell**: usar `setsid ... < /dev/null &` com
  `disown` — `nohup ... &` simples morre quando a sessão do agente encerra
  o processo com timeout.
- **`OLD/`** contém código aposentado (mantido, não mantido): testes
  arquivados em `OLD/tests/` (ver README lá para substitutos), wrappers
  aposentados em `OLD/models/`, notebooks legados em `OLD/notebooks/`,
  `run_simulation.sh` (cujo `src/simulation.py` nunca existiu).
- **Métricas históricas** em docs/commits antigos (ex.: MAE 0.1589) vêm de
  um dataset antigo e de um protocolo rolling com vazamento de linhas
  futuras; os números honestos atuais estão documentados em `docs/` e nos
  relatórios de `reports/protocolo/`.
- **Overleaf/figuras**: pushes são best-effort (sem credenciais o commit
  segue normalmente). `paper/.overleaf-clone/` é cache — não editar.
- **reports/_site/** é gerado — nunca editar à mão; regenere com
  `python reports/src/builder.py build` ou o botão do servidor.
