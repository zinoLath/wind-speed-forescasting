# INDEX — base canônica de findings

> Ponto de entrada do conhecimento consolidado do repositório: código,
> dados, protocolos, resultados, artigos e processos. O *how-to
> operacional* por área vive em `agents/`; runbooks de tarefas nas skills
> (`.opencode/skills/`) e em `findings/playbooks/`.
>
> Regras de manutenção: `findings/README.md`. Agente: skill
> **canonical-findings**. Validação: `python findings/scripts/findings.py check`.

### arquitetura

- [visao-geral](knowledge/visao-geral.md) — projeto/TCC: vento LiDAR Delta do Maranhão, seq2seq direct 72→36 passos de 10 min, Keras/TF + Optuna, dois protocolos (pipeline 75/20/5 e LiDAR 80/20).
- [wrappers-arquiteturas](knowledge/wrappers-arquiteturas.md) — `Seq2SeqWrapper` + registry `src/common.py:WRAPPERS` (9 chaves), atenção Luong, TCN hp compartilhado, Transformer Pre-LN.
- [losses-decoder](knowledge/losses-decoder.md) — MSE ponderada por horizonte (wrappers fixam a loss); decoder direto na validação/inferência (sem teacher forcing/vazamento).

### dados

- [dados-map](knowledge/dados-map.md) — mapa completo de entradas/saídas/artefatos (`data/`, `pipeline/tmp/`, report*/), regras de gitignore e rastreabilidade `model.json`.

### protocolo

- [protocolo-avaliacao](knowledge/protocolo-avaliacao.md) — protocolos: pipeline 75/20/5 com validação vs LiDAR 80/20 sem validação (60 épocas, seed 42); métricas vs ws100 bruto; rolling sem vazamento.
- [wavelet-denoise](knowledge/wavelet-denoise.md) — alvo `ws100_wavelet` = sym18 nível 2, VisuShrink soft-threshold; escolha validada por A/B (`docs/ab_target_denoise.md`).

### resultados

- [resultados-canonicos](knowledge/resultados-canonicos.md) — números de referência: report_8020 (TCN R² 0.514), report_homologado (antigo, alimenta steam/paper), tabelas do artigo SBR desatualizadas, métricas históricas inválidas (MAE 0.1589).
- [estudos-ablacoes](knowledge/estudos-ablacoes.md) — mapa de evidências: A/B alvo, ablações de loss, HPO dedicado, seeds, gate/persistência, dilatações TCN, features study, WFO.

### metodos

- [imputacao](knowledge/imputacao.md) — RF/LightGBM (IterativeImputer, 3k linhas), KNN, Kalman; sin/cos de direção; validação por injeção de lacunas; flag `imputed`; linha WFO LightGBM.
- [optuna-hpo](knowledge/optuna-hpo.md) — TPE multivariado+group, MedianPruner, objetivo RMSE com decoder inferência, sqlite retomável; rodadas distribuídas round2+ (Colab/Kaggle/local).
- [metodos-complementares](knowledge/metodos-complementares.md) — baselines ARIMA/RF/persistência; DM/KSPA com bootstrap, loss_ab, breakpoints, estratos regime/banda; persistence gate A/B.

### relatorios

- [relatorios-framework](knowledge/relatorios-framework.md) — hub `reports/` (categorias, builder, server + API, state.json gerado, popup LaTeX steam/sbsr/generic, análise textual); relatório só em `reports/<cat>/<nome>/`.

### artigos

- [artigo-sbr](knowledge/artigo-sbr.md) — estado do artigo SBSR: escopo TCN vs LSTM; tabelas em rodada antiga ≠ figuras/números report_8020 (integração pendente via sbsr-integrate).
- [tabelas-sbsr-geradas](knowledge/tabelas-sbsr-geradas.md) — `scripts/sbsr_tables.py`: tabelas LaTeX prontas (caption+label) de uma rodada + baselines; semântica all/h36 validada; per_horizon persistido no evaluation.json.
- [artigo-steam-knowledge](knowledge/artigo-steam-knowledge.md) — steam (classe imc/UNIFEI, sem .git, figures do homologado) + knowledge (catálogo bibtex com abstract, gerador `knowledge.py`, index/bib GERADOS).
- [overleaf-sync](knowledge/overleaf-sync.md) — paper_sync + articles_sync, sync.json só de figuras, hook post-commit = ponto de push, credenciais/best-effort.

### infra

- [pipeline-estagios](knowledge/pipeline-estagios.md) — estágios optuna→train→evaluate→impute→walkforward, configs mesclados, `src/common.py` fonte única sem efeitos colaterais, bootstrap sys.path.
- [ambiente-execucao](knowledge/ambiente-execucao.md) — venv py3.10, TF 2.21+CUDA (`tf_gpu_env.sh` obrigatório), tests/ standalone, train_dashboard, setsid para background.

### processo

- [gotchas](knowledge/gotchas.md) — armadilhas consolidadas: imports lazy, sys.path, sqlite Optuna, pgrep, background jobs, arquivos gerados, OLD/, métricas com vazamento.

### playbook

- [retreino-lidar](playbooks/retreino-lidar.md) — treino 80/20 (backup → treino 17 min → regeneração 5 min → build) + variantes/ablações.
- [geracao-relatorios](playbooks/geracao-relatorios.md) — regenerar dados de relatório, rebuild do site, servidor com botões, adicionar relatório novo.
- [integracao-sbr](playbooks/integracao-sbr.md) — trocar a rodada que alimenta o artigo SBSR (tabelas, Resumo, loss, figuras, consistência).
- [sync-overleaf](playbooks/sync-overleaf.md) — comandos de sincronização de figuras com Overleaf (paper + artigos) e init de artigo novo.
- [optuna-distribuida](playbooks/optuna-distribuida.md) — planejar/submeter/monitorar/coletar rodada Colab+Kaggle+local.
- [registrar-finding](playbooks/registrar-finding.md) — fluxo canônico para registrar/atualizar findings e validar o INDEX.
