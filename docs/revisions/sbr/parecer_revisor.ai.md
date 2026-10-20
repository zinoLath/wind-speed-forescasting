# Parecer de revisão — artigo SBSR

> **Autoria:** gerado por IA (opencode/GLM) em 14/10/2026, a pedido do
> autor, no perfil de revisor de periódico. Complementa (não substitui) o
> levantamento de comentários em `comentarios_revisao_sbsr.md`, que mapeia
> a rodada de revisão da Vanessa/Flávio.

**Síntese do artigo:** compara 4 arquiteturas seq2seq (LSTM/TCN ×
uni/bidirecional) + 3 baselines na previsão de vento a 100 m (LiDAR, 6 h
à frente), série de ~2 meses, split 80/20, avaliação rolling forecast.

**Recomendação geral: major revision.** O núcleo experimental é
interessante e o problema é relevante, mas há problemas de
reprodutibilidade, framing de resultados e completude metodológica que um
revisor de periódico barraria.

---

## Questões maiores (estruturais/metodológicas)

**M1. Procedência dos resultados não verificável (o mais grave).** A
Tabela 1 (DL) não corresponde a nenhuma rodada nos artefatos do projeto;
o heatmap (`fig_mse_heatmap`) vem de **outra rodada** (loss ponderada,
onde TCN puro vence) e **contradiz a tabela** (TCN-Bi vence). Além disso,
DL e baselines foram agregados em **grades de origens diferentes**
(1406 vs 1478), o que produz variâncias implícitas de y distintas (2,78
vs 2,70) e torna os R² das duas metades da Tabela 1 **não comparáveis
entre si** — a própria Eq. (1) do artigo denuncia isso. Tabela, figura e
texto precisam sair de uma única rodada (a skill `sbsr-integrate` existe
para isso).

**M2. Resumo omite o resultado central e seleciona número favorável.**
"R² de até 0,502" (TCN-Bi) esconde que o **ARIMA atinge 0,508 no
agregado** — ou seja, o DL *não* vence no geral; vence no horizonte de
6 h (0,416 vs 0,316). O resumo não menciona baselines em momento algum.
O framing honesto e mais forte é: *modelos de referência vencem no
agregado; DL vence em horizontes longos, que é o caso de uso*. Da mesma
forma, 04:6 diz "os modelos de baseline perdem" sem qualificar que é só
no h=36.

**M3. Ausência de validação estatística e de incerteza.** Alegações de
superioridade (TCN > LSTM, 04:9; "não afetou significativamente",
Conclusões par. 2) sem teste de significância, sem intervalo, com **um
único treino por arquitetura** (sem variabilidade de seed). O teste DM +
bootstrap em blocos (pedido da Vanessa) já está implementado no repo —
falta entrar no artigo.

**M4. Protocolo experimental opaco.** Não há nada sobre: épocas/
otimizador/LR (treino fixo de 60 épocas sem validação — sequ
mencionar), busca de hiperparâmetros (Optuna usado e **nunca citado**,
embora esteja no bib), quais atributos entraram nos modelos (a tabela de
features pedida pela Vanessa), como os baselines foram ajustados (ordem
do ARIMA, features/janela do RF), e — crucial — **que o alvo treinado é
a série suavizada por wavelet** (ws100_wavelet), com métricas calculadas
contra o valor bruto. Isso é uma decisão metodológica central e o artigo
só diz "suavização do dado" genericamente (03:42). Um revisor pergunta:
"vocês preveem o sinal denotado — contra o que exatamente as métricas
são calculadas?"

**M5. Escopo da CRISP-DM prejudica a estrutura.** O parágrafo 03:4 e os
nomes das subseções ("Entendimento do negócio", "Entendimento dos
dados") são andaimes que não entregam conteúdo — a subseção "negócio"
tem **uma frase** de negócio (Vanessa já apontou). Ou remove a CRISP-DM
ou renomeia as seções pelo conteúdo.

**M6. Sem trabalhos relacionados.** A Introdução cita ~5 trabalhos; o
bib tem prontas **~12 entradas nunca citadas**, incluindo
`compareLSTMnTCN` (comparação LSTM×TCN — exatamente o tema!),
`zhang2021cnnseq2seq`, `nascimento2023transformer`, `sari2021dclstm`.
Sem posicionar o trabalho, a lacuna ("comparações TCN vs LSTM em seq2seq
com LiDAR multinível em costa equatorial são escassas") fica implícita.

**M7. Narrativa dos resultados internamente inconsistente.** "vantagem
de R² de 0,8... e 0,4" (04:9) — a tabela dá 0,08/0,04; o par. 3 afirma
LSTM-Bi > LSTM, o que o heatmap atual contradiz; "1406 predições" (04:4)
não bate com a grade de mesma origem (~1477–1478); o heatmap omite
baselines enquanto as tabelas os incluem.

**M8. "Discussão" existe só no título.** Não há análise de *por quê*
(TCN vence — campo receptivo dilatado casado com ACF de 6 h?
paralelismo? caminho de gradiente?), nem comparação com valores da
literatura, nem análise de regimes (o relatório interno tem skill por
regime diurno/noturno — seria 1 frase forte). As Conclusões repetem
método ("resolução de 10 minutos por amostra", já dito 2×).

**M9. Dataset mal caracterizado.** 7561 amostras vs ~7632 esperadas sem
justificar (lacunas do instrumento; Vanessa apontou); imputação não
mencionada; controle de qualidade ausente (o pipeline exclui timestamps
interpolados da avaliação — fato relevante e não declarado); "temporada
seca setembro a novembro" vs datas reais 16/09–08/11.

**M10. Figuras.** fig1: P1 ilegível a 0.5\linewidth (2 threads da
Vanessa); heatmap: fonte pequena; `fig_lstm_cell` é diagrama de
livro-texto que nada agrega (ver opiniões).

---

## Questões menores (com localização)

- 01:3 "é dificultoso"; "Os resultados obtidos aprofundaram os
  conhecimentos sobre os assuntos abordados" — frase vazia; "camada de
  bidirecionalidade" → "codificador bidirecional". EN: "in a seq2seq
  with attention mechanism approach", "approached topics" — traduções
  literais.
- 02:3 dado ANEEL 2023 com cite errado (`arcilanVentoSul` não é fonte de
  capacidade instalada — Vanessa pediu ANEEL); "Então" como conector 2×
  em 3 linhas (02:3, 02:5).
- 02:5 verbos pessoais ("realizarmos", "utilizamos").
- 03:9 "Através de estações" → "Por meio de"; "nominalmente de setembro
  a novembro" — "nominalmente" mal usado; 03:23 "possui...
  possuindo" na mesma frase; "100m" vs "100 m" inconsistente; 03:24
  "reincidência" → "recorrência"; 03:28 "conjunto de testagem" →
  "conjunto de teste"; "impossibilitando o embaralhamento do dado"
  coloquial.
- 03:60 "quatro variações de modelos" → "quatro arquiteturas" (Flávio).
- 03:76 notação do campo receptivo `[t_{1d}, t_{2d}, ...]` é
  não-padrão/imprecisa — usar a fórmula clássica (k = tamanho do
  filtro, dilatações dᵢ): r = 1 + (k−1)·Σᵢ dᵢ.
- 03:78 antecedente de "A estrutura recorrente também degrada"
  ambíguo; e a divisão de perdas LSTM/TCN não bate com nenhuma rodada
  (M1/M4).
- 03:90 frase sintaticamente incompleta ("onde uma janela deslizante de
  tamanho i+o, em que... e o da saída" — sem verbo principal; e "o"
  artigo se confunde com o símbolo *o*).
- 03:117 **erro factual**: "Persistência é a repetição do último valor
  **previsto**" → **observado**; e "derivada da série" para o termo I do
  ARIMA → "diferenciação".
- 04: legendas das tabelas sem escopo nem unidades ("Métricas de erro
  observadas" → "Métricas por todos os horizontes (MSE/RMSE em
  m/s...)" e "... no horizonte h=36").
- 06: "foi concluído com êxito, contrastando o funcionamento" — regência
  + "funcionamento" vago (sugestões já no doc de comentários).

---

## Opinião — o remover, reescrever e resumir (e onde ganhar espaço)

**1. Cortar a CRISP-DM inteira** (03:4 + nomes de subseção). Subseções
novas: *Área de estudo e dados* · *Preparação dos dados* ·
*Arquiteturas* · *Protocolo experimental*. Resolve estruturalmente o
comentário da Vanessa: a frase de negócio (2 linhas: geração/eólica/
complexo Delta + por que prever importa) entra na abertura da seção de
dados, e o resto do conteúdo de coleta fica onde sempre deveria ter
estado. Ganho: ~10 linhas + coerência.

**2. Resumir o livro-texto de LSTM/TCN de 5 parágrafos para 2**
(03:62–78). Público SBSR conhece LSTM; o que o leitor precisa: 2 frases
por arquitetura conectando o mecanismo *ao problema* (LSTM: portas →
dependências longas, decai em horizontes longos; TCN: convolução causal
dilatada → campo receptivo exponencial, treino paralelo, gradiente
estável). **Remover `fig_lstm_cell`** (diagrama padrão, ~⅓ de coluna
desperdiçada); manter `fig_convolutions` (dilatação é o conceito menos
familiar) ou, se faltar espaço, cortar também e deixar só a fórmula
correta do campo receptivo. Manter `fig_model_struct` (é a contribuição
estrutural do artigo). Ganho: ~⅔ de coluna.

**3. Reescrever o Resumo** nas duas línguas: matar as duas frases vazias
(início "dificultoso" e "aprofundaram os conhecimentos") e colocar o
resultado concreto *com baselines*: "…a TCN superou a LSTM em ambas as
variantes; no agregado o ARIMA foi competitivo (R² 0,508), mas no
horizonte de 6 h as arquiteturas propostas venceram (R² 0,416 vs
0,316)." — é mais forte e é verdade.

**4. Introdução em 4 parágrafos:** (i) motivação com dado ANEEL novo;
(ii) **trabalhos relacionados** usando as entradas já no bib
(`compareLSTMnTCN`, `zhang2021cnnseq2seq`, `sari2021dclstm`,
`nascimento2023transformer`, `lv2022hybrid`, `ozdemir2024ensemble`) — 4
frases fecham a lacuna; (iii) objetivo (par. atual 02:7 está bom, só
enxugar); (iv) 1 linha de contribuições. Custo: ~8 linhas — pagas pelos
cortes 1–2.

**5. Preparação dos dados: conectar a EDA à escolha experimental.** A
ACF de 6 h (03:24) *é* a justificativa do horizonte de 36 passos — hoje
a conexão é implícita e a seção Modelagem nunca a invoca. Uma frase: "o
horizonte de 6 h foi escolhido por corresponder ao alcance da
autocorrelação". Além disso, declarar: alvo suavizado + avaliação contra
bruto (M4), e o que entrou de features (tabela pedida pela Vanessa, ou 2
linhas resumindo grupos: 20 níveis × ws/dir/vert + meteo + derivados).

**6. Resultados: reordenar a narrativa** — (i) protocolo (nº de origens
*da mesma grade*); (ii) agregado com o tradeoff honesto ARIMA×DL e **o
horizonte de cruzamento** (dá para ler no heatmap — citar o valor
exato); (iii) comparação arquitetural com números certos + p-valores
DM. Incluir baselines no heatmap (consistência figura×tabela, M1/M7).

**7. Escrever uma Discussão de verdade** (3–5 frases): mecanismo
provável da vitória da TCN (receptivo dilatado ↔ dependência de 6 h;
treino paralelo; gradiente), neutralidade/instabilidade da
bidirecionalidade (janela fixa de 12 h limita o ganho do contexto
"futuro" no codificador — apresentar como hipótese), 1 comparação com
literatura, 1 limitação. Se não quiser discutir, renomeie a seção para
"Resultados" — mas aí o artigo perde o que mais o salvaria em avaliação
por pares.

**8. Conclusões em 2 parágrafos:** (i) objetivo + números-chave
(agregado e h36, com significância DM); (ii) limitações (sítio/período
únicos, treino único sem IC, sem validação cruzada temporal) + futuros
com cites (Transformer/GRU — Transformer já foi testado no repo e foi
mal no walk-forward, o que rende 1 frase honesta). Cortar repetições de
método.

**Orçamento de espaço:** cortes 1–2–8 (~1 coluna) pagam as adições
obrigatórias: relacionados, ferramentas/versões, protocolo de treino,
detalhes de baselines, DM, tabela de hiperparâmetros (4 linhas) e a
justificativa de amostras. Se apertar, corte `fig_convolutions`.

**Não cortar:** `fig_model_struct`, as duas tabelas, Eq. (1), o parágrafo
da wavelet (com a transparência do alvo), a justificativa do split
temporal com cite.

---

## Referência cruzada com a revisão Vanessa/Flávio

| Achado aqui | Correspondente na rodada anterior |
|---|---|
| M1 tabela×figura×texto de rodadas diferentes | inconsistência R²×MSE (Vanessa), "tá correto?" |
| M3 sem validação estatística | DM + bootstrap (Vanessa), "validar isso estatisticamente" |
| M4 protocolo/ferramentas opacas | bibliotecas/versões, tabelas hp/dataset (Vanessa) |
| M5 CRISP-DM / seção negócio | "falar do negócio" (Vanessa) |
| M9 contagem de amostras | 7632 (Vanessa) |
| M10 figuras | P1 invisível, fonte pequena (Vanessa) |
| m. 02:5 verbos pessoais, 03:60 "arquitetura" | Flávio |
