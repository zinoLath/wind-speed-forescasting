---
title: Armadilhas de execução (gotchas)
area: processo
updated: 2026-10-21
---

# Armadilhas de execução (gotchas)

Consolidado de `agents/gotchas.md` — os pontos que já quebraram algo antes:

- **`src/common.py` sem efeitos colaterais de import**: TF é configurado
  via `common.setup_tensorflow()` ANTES dos imports de modelo (lazy). Não
  adicionar import de keras/tf no nível de módulo.
- **Bootstrap sys.path**: arquivos sob `pipeline/` e `reports/*/report.py`
  precisam inserir a raiz do projeto em `sys.path[0]` antes de
  `from pipeline import ...` (senão `pipeline/pipeline.py` sombreia o
  pacote). **Profundidade**: raiz de `scripts/` e `studies/` usam
  `parents[1]`; subpastas de `scripts/` (`experiments/`, `dist/`) e
  `reports/src/` usam `parents[2]` (reorganização 28/10/2026) — conferir
  ao criar/mover scripts.
- **SQLite do Optuna**: estudos resumem com `load_if_exists=True`; apagar
  o `.db` (`pipeline/tmp/optuna*/<name>/optuna.db`) para recomeçar.
- **pgrep/pkill**: padrão não pode casar com o próprio shell (usar
  colunas: `lidar_[7]525_train`).
- **Background jobs**: `setsid ... < /dev/null &` com `disown`;
  `nohup ... &` morre quando a sessão do agente encerra com timeout.
- **Overleaf**: pushes best-effort; `paper/.overleaf-clone/` é cache.
- **Gerados, nunca editar**: `reports/_site/`, `reports/state.json`,
  `articles/knowledge/index.md`, `articles/knowledge/references.bib`,
  `notebooks/colab_round2/`.
- **`OLD/`**: código aposentado (mantido, não mantido) — testes em
  `OLD/tests/`, wrappers em `OLD/models/`, notebooks legados.
- **Métricas históricas** (MAE 0.1589): protocolo antigo com vazamento —
  números honestos em `reports/` e `docs/`.

## Janela de avaliação: nunca incluir a validação no "teste"

Em protocolos com split 72/8/20, a avaliação final tem de começar **depois
de train+val**. Avaliar a partir do fim do treino (incluindo o bloco de
validação) infla o R² — caso real (coeficientes ITILDE-Q, 21/10):
0,4597 (val+teste) → **0,2765** (teste puro), pois a validação tinha sido
usada para selecionar a config e tinha outra variância. Regra: corte =
`len(train)+len(val)`; contexto do encoder pode vir do fim da val.

## Kaggle TPU: acelerador nem sempre é anexado

`enable_tpu: true` no kernel-metadata é aceito pela CLI (kaggle 1.7.4.5)
e preservado pelo servidor, mas **não garante TPU**: sem acesso/quota de
TPU na conta o runtime cai na imagem padrão (CPU) — sinais: `TPU_NAME`
vazio e `TPUs not found in the cluster / No OpKernel ... ConfigureDistributedTPU`.
O smoke `itildeq-tpu-smoke` (21/10) rodou em CPU (3 épocas×1500 amostras
= 23 s). Templates TPU: `scripts/dist/kaggle_tpu_push.py` (sem
tensorflow[and-cuda]; usa libtpu da VM) + `scripts/dist/tpu_smoke_test.py`.
TPU no Kaggle exige verificação por telefone; quota é semanal e separada
da GPU. FFT (`tf.signal.rfft`, termo de fase da ITILDE-Q) não roda em
TPU — perdas com FFT ficam fora.

## Kaggle: corrida de versionamento do dataset

Kernels restauram a **última versão processada** do Dataset no momento em
que rodam. `datasets version` retorna antes do processamento assíncrono
terminar — pushar o kernel na sequência pode restaurar a versão ANTERIOR
(caso real: `can't open file .../itildeq_terms_search_train.py`, duas
vezes, em 21/10/2026). **Fluxo correto**: versionar → esperar o
processamento → pushar kernels. Conferir pelo `creationDate` de um
arquivo qualquer da listagem (`kaggle datasets files <id>`); com
`--dir-mode zip` a listagem mostra os arquivos **extraídos** do zip (não
o nome do zip) e é **paginada** — grep de caminho pode falhar por
paginação (use a data de criação, em **UTC**).

## Patch de coeficientes ITILDE-Q: binding de import

Os wrappers fazem `from .losses import itilde_q` **no import** — patchar
`losses.itilde_q` DEPOIS não afeta o wrapper (binding antigo). Para variar
α/β/γ é preciso patchar o **módulo do wrapper**
(`src.models.s2s_lstm_wrapper.itilde_q = ...`, como faz
`scripts/experiments/itildeq_terms_search_train.py:patch_lstm_itildeq`) ou patchar
ANTES do primeiro import do wrapper. **Impacto nos resultados até
21/10/2026: nenhum** — os args (0,3/0,5/0,1) eram iguais aos defaults da
perda, então o binding "errado" produzia os mesmos valores. Busca de
coeficientes em `scripts/experiments/itildeq_terms_kaggle.py` (Kaggle).

## Popup LaTeX: strip de chaves no snippet (corrigido 23/10/2026)

O `snippetText()` em `reports/src/assets/app.js` tinha um segundo
`.replace(/\{(\w+)\}/g, "$1")` após a substituição de placeholders que
**apagava chaves legítimas do LaTeX** (`\begin{tabular}` →
`\begintabular`, `\caption{...}` → `\caption...`, `\rowcolor[HTML]{...}`
→ sem chaves). **Todo snippet copiado do popup antes dessa data é
inválido** — revisar blocos LaTeX colados no artigo steam/SBSR
(figuras inclusive: `\begin{figure}` → `\beginfigure`). Correção:
removido o replace extra (`12111b9`); placeholder desconhecido já era
preservado pelo primeiro replace. Combinado com o default antigo em
steam (`\rowcolor` precisa de colortbl, ausente no sbsr.sty), popups
agora abrem em `sbsr|steam|generic` (resultados.py `FORMATS`).

## Fontes

- `agents/gotchas.md`; skills `lidar-retrain`/`optuna-colab` (regras).
