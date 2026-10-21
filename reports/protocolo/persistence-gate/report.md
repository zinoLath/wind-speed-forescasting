# Roteiro: explicando o Persistence Gate (estático → dinâmico)

Roteiro de apresentação para o mecanismo de *persistence gate* dos modelos
seq2seq de previsão de vento (LiDAR, resolução de 10 min, horizontes h1–h36).
Números do experimento A/B: `data/results/report/ab_persistence_gate.html`.

## Bloco 0 — Gancho (30 s)

> "Nosso alvo é a velocidade do vento a 100 m em resolução de 10 minutos. Nessa
> escala, a atmosfera é extremamente **persistente**: o melhor preditor de
> 'daqui a 10 minutos' é simplesmente 'o valor de agora'. Nossa EDA mostra
> autocorrelação de 0,76 em 1 hora. Nosso teste confirma: no horizonte h1, a
> persistência tem MAE de **0,28 m/s** e nenhum modelo chega perto — o LSTM
> erra 0,48. A pergunta que motiva o gate é: **se a persistência é insuperável
> nos passos curtos, por que forçar a rede a competir com ela em vez de
> ensiná-la a usá-la?**"

## Bloco 1 — O mecanismo (2 min)

**A equação central** (slide sozinha):

    ŷ[t,h] = g_h · pred[t,h] + (1 − g_h) · y_obs[t]
             └── g_h ∈ (0,1): sigmoid de um logit treinável por horizonte ──┐

**Pontos de fala, nesta ordem:**

1. **"É uma média ponderada aprendida entre dois experts"** — a previsão da
   rede e o último valor observado. Nada mais que isso.
2. **"O gate g é um sigmoid de um logit treinável por horizonte"** — 36
   parâmetros, inicializados em zero (g = 0,5). A rede *descobre* o peso de
   cada fonte; nós não fixamos nada.
3. **"A persistência entra pela estrutura de dados"** — quando o gate está
   ativo, o decoder ganha um canal extra de entrada repetindo o último valor
   observado em todos os passos. É a mesma informação que o baseline usa —
   estamos dando à rede acesso ao concorrente.
4. **"Custo zero em operação"** — não há segunda rede nem consulta externa: é
   uma camada de 36 parâmetros no fim do grafo.

**Analogia que funciona bem:** *um mixer de áudio com 36 faders* — um por
horizonte de previsão. O engenheiro de som (treino por gradiente) abaixa o
fader da rede e sobe o da persistência onde ela é melhor.

## Bloco 2 — O que o experimento mostrou (2 min)

A história em 3 atos, com os números do A/B (sem gate vs gate estático):

| Ato | Fala |
|---|---|
| **Atos funcionam** | "Com o gate, o ganho nos horizontes curtos é enorme: sem ele, o TCN erra **+0,72 m/s** a mais em h1 e o LSTM_BI **+0,48**. E o gate não custa nada no treino — mesmas épocas de early stopping." |
| **Mas é um compromisso global** | "O g_h é **igual para toda amostra**: o mesmo 'fader' vale para vento estável ou rajada. Com o tempo, a âncora envelhece — em h36 ela referencia uma observação de 6 h atrás. Resultado: para os LSTMs o gate vira desvantagem a partir de **h8** (LSTM) e **h18** (LSTM_BI), e no regime autônomo de 6 h ele **atrapalha todos os 4 modelos** (LSTM_BI: 2,23 → 1,76 sem gate)." |
| **Diagnóstico** | "O problema não é a ideia da mistura — é o gate ser **estático**: ele não pode decidir *quando* ancorar." |

## Bloco 3 — A evolução: gate dinâmico (2 min)

**Transição:** "Se o gate estático é um termostato com temperatura fixa, o
próximo passo é um termostato **com sensor**."

**Nova equação:**

    g[t,h] = sigmoid( W · f[t,h] + b_h )

**Pontos de fala:**

1. **"O vetor f é o estado do decoder"** — as features combinadas (saída do
   LSTM decoder + atenção, ~320 dimensões) que já alimentam o
   `TimeDistributed(Dense)`. O gate passa a ler o mesmo contexto que o preditor.
2. **"b_h preserva o prior por horizonte"** — a intuição estática continua lá
   (ancorar em h1, soltar em h36); o termo **W·f adiciona a condição**: a rede
   pode aprender, por exemplo, *"quando o histórico mostra vento estável,
   ancoro; quando há transição de regime, prevejo"*.
3. **"Subsume o gate estático"** — se W = 0, recuperamos exatamente o modelo
   anterior. É um superespaço: a melhor solução estática continua alcançável.
4. **"Custo desprezível"** — ~356 parâmetros a mais (320×1 + 36), frente a
   240 mil–1,7 milhão dos modelos.
5. **"E gera um diagnóstico que o estático não permite"** — plotar a média de
   σ(g) por horizonte no teste: a assinatura aprendida. Hipótese: curva
   crescente de ~0 (persistência) em h1 até ~1 (rede) em h36.

## Bloco 4 — Perguntas antecipadas (defesa / apresentação)

| Pergunta provável | Resposta curta |
|---|---|
| "O gate não é 'roubar' do baseline?" | "É **destilação do baseline na arquitetura**. A avaliação é contra o ws100 bruto, igual para todos. O baseline continua reportado — e ainda ganha de todos em h1–h4, o que o gate apenas mitiga." |
| "Por que sigmoid e não peso livre?" | "Peso livre pode extrapolar fora da convexidade dos dois experts e desestabiliza o treino. σ garante g ∈ (0,1) — combinação convexa — com gradiente saudável." |
| "E se a última observação for ruidosa?" | "O termo W·f permite exatamente isso: o modelo pode detectar janelas de má qualidade (sinal CIS, volatilidade) e reduzir a âncora — impossível no gate estático." |
| "Por que não aplicar o gate só no rolling?" | "O gate vive na arquitetura, não no protocolo. E o A/B mostrou que ele também ajuda no denso — o dinâmico promete o melhor dos dois mundos." |
| "Overhead computacional?" | "Inferência ~0,3–0,5 ms/janela já medido com o gate estático; o dinâmico adiciona uma multiplicação 320×1 por passo." |

## Bloco 5 — Fechamento (20 s)

> "Em resumo: transformamos o baseline — o adversário que não conseguimos
> bater nos passos curtos — em **um componente da arquitetura**. A versão
> estática provou o conceito e revelou o limite: âncora global não serve para
> vento que muda de regime. A versão dinâmica dá à rede a decisão de **quanto
> de persistência usar, em cada horizonte, em cada situação** — e é isso que
> quantificamos no experimento a seguir."
