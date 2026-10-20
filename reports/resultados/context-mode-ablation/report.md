# Ablação: modo de injeção de contexto no decoder TCN

Pergunta: o `RepeatVector` (broadcast do contexto de 416-d igual para os 36
passos) estava atrapalhando? Protocolo: TCN sem persistence gate, mesmos
hiperparâmetros Optuna (best_trial do estudo com gate), seed 42, protocolo
step_evaluate, 272 origens de teste.

| context_mode | Descrição | Denso MAE | R² denso | Rolling MAE | Épocas | Params |
|---|---|---|---|---|---|---|
| `repeat` | contexto repetido igual em todos os passos (histórico) | 1,122 | −0,217 | **1,056** | 33 | 1.747.329 |
| `none` | sem contexto; condição só via atenção | 1,109 | −0,026 | 3,818 | 28 | 1.481.089 |
| `horizon` | contexto repetido + canal posicional h/36 | **0,970** | **+0,116** | 1,192 | 44 | 1.747.969 |

## MAE denso por horizonte

| h | repeat | none | horizon |
|---|---|---|---|
| 1 | 1,187 | 0,678 | **0,661** |
| 3 | 1,047 | 0,686 | 0,746 |
| 6 | 1,040 | 0,888 | 0,880 |
| 12 | 1,117 | 1,028 | **0,927** |
| 18 | 1,133 | 1,134 | **0,989** |
| 24 | 1,136 | 1,188 | **1,015** |
| 30 | 1,143 | 1,282 | **1,065** |
| 36 | 1,165 | 1,423 | **1,119** |

## Interpretação

1. **O broadcast uniforme cega o decoder para o horizonte**: o perfil de MAE do
   `repeat` é quase plano (1,19 em h1 → 1,17 em h36) — o decoder não distingue
   "prever 10 min à frente" de "prever 6 h à frente". O canal posicional
   (`horizon`) corrige a raiz do problema: h1 cai 44% (1,19 → 0,66) e o denso
   total cai 13,5% (1,12 → 0,97) com +640 parâmetros.
2. **O contexto em si é essencial no regime autônomo**: sem ele (`none`) o
   rolling de 6 h colapsa (MAE 3,82) — no decoder sem estado, o vetor de
   contexto carrega praticamente toda a informação da janela.
3. **O gate estático mascarava a patologia**: no run com gate, o h1 do TCN era
   0,47 porque a âncora cobria o curto prazo; sem gate, o problema aflorou
   (1,19) e a correção estrutural (canal de horizonte) resolve sem gate.
4. **Trade-off restante**: `repeat` mantém o melhor rolling (1,06 vs 1,19) —
   o perfil "achatado" pode atuar como regularizador no regime autônomo.
   `horizon` domina no denso.

Ressalva: semente única e hiperparâmetros ajustados para `repeat`; o `horizon`
tende a melhorar ainda mais com busca própria (também há espaço para tunar o
formato do canal posicional).

Implementação: `S2STCNWrapper.build` com `context_mode` em
`repeat | none | horizon` (config/metadata `context_mode`, default `repeat`).

Artefatos: `pipeline/tmp/trained_compare_{nogate,ctxnone,ctxhorizon}/`.
