# Guia das métricas e processos metodológicos — o que são, o que implicam, fontes

> **Autoria:** IA (opencode/GLM-5.3), 14/10/2026. Acompanha os resultados
> calculados em `data/results/metodos_complementares/` (script:
> `studies/metodos_complementares.py`; rodada: `trained_8020`, 1196 origens
> comuns × 36 horizontes = 43.056 amostras avaliadas, α=0,05).
>
> **Política de fontes:** todo link/DOI abaixo foi **verificado por
> acesso direto** (webfetch em 14/10/2026) ou consta da coleção local
> (`articles/knowledge/`, com DOI no `references.bib`). Nenhuma referência
> de memória. Clássicos sem DOI verificado são citados apenas pela
> referência completa contida **dentro** de um artigo da coleção.

---

## A1 — Teste de Diebold–Mariano (com HAC + correção de pequenas amostras)

**O que é.** Compara a precisão preditiva de dois modelos por meio da
série de diferenças de perda por origem de previsão, d_t = L(e_A,t) −
L(e_B,t). H₀: E[d_t] = 0 (mesma precisão). A estatística usa variância
HAC (autocovariâncias até h−1 — Newey–West) porque as janelas de rolling
forecast são sobrepostas e correlacionadas, e a correção de
Harvey–Leybourne–Newbold ajusta o viés de pequenas amostras, avaliada
contra t(n−1).

**O que implica.** Um p baixo diz "a diferença de desempenho entre A e B
dificilmente é sorte". Sem isso, frases como "X superou Y" são
anedota. **Caveats:** o DM compara a *média* da perda (não a estabilidade
— binrong2026whole complementa com IQR dos resíduos); com 1 única seed
por modelo, ele testa a diferença *destes treinos*, não *das
arquiteturas* (por isso o item A3/seeds importa).

**Resultado obtido** (`metodos.json:dm`, tabela A1 do
`relatorio_metodos.md`):
- TCN > LSTM (p=0,012) e TCN > LSTM-Bi (p=0,0002) — superioridade da TCN
  **é** significativa;
- TCN > TCN-Bi (p=0,024) — a bidirecionalidade **piora** a TCN
  (contraria a narrativa atual do artigo);
- **TCN vs ARIMA: p=0,85 — empate estatístico** no agregado;
- LSTM-Bi **pior que a persistência** (p=0,026);
- LSTM vs LSTM-Bi não significativo (p=0,30).

**Fontes:**
- Diebold, F.X.; Mariano, R.S. (1995). *Comparing Predictive Accuracy*.
  JBES 13(3), 253–263. DOI verificado:
  <https://doi.org/10.1080/07350015.1995.10524599>
- Harvey, D.; Leybourne, S.; Newbold, P. (1997). *Testing the equality of
  prediction mean squared errors*. IJF 13(2), 281–291. DOI verificado:
  <https://doi.org/10.1016/S0169-2070(96)00719-4>
- Uso na coleção: `gomez2026predicting` ("Harvey small-sample adjustment
  and Newey–West variance estimation", Seção 3.3);
  `wang2026adaptive` (DM com estatística/SE/p e exceção relatada);
  `binrong2026whole` (DM+Friedman, discussão dos não-significativos).

---

## A2 — Skill score vs. persistência (por horizonte, IC 95%)

**O que é.** SS = 1 − MSE_modelo,h / MSE_persistência,h. Skill 0 = igual
à predição ingênua; positivo = melhor; negativo = **pior que repetir o
último valor**. O IC vem de bootstrap estacionário (blocos de
comprimento aleatório geométrico, média 24 origens) sobre as origens,
preservando a dependência temporal.

**O que implica.** Traduz o erro para a pergunta operacional: "vale a
pena este modelo em vez do ingênuo?". É a prática de benchmark
recomendada por Hyndman & Athanasopoulos (FPP3, §5.2/§5.9: métodos
simples servem de referência mínima; comparações *scale-free* usam skill
scores). Reportar skill **negativo** é honestidade metodológica
(bendes2026bio publica skill ≈ −0,5 e discute o contraste com a métrica
normalizada).

**Resultado obtido** (`skill_por_horizonte.csv`):
- No agregado: só ARIMA (+0,057) tem skill positivo; TCN ≈ −0,05;
  LSTMs e TCN-Bi negativos;
- No h36 (6 h): **TCN +0,27**, ARIMA +0,18, TCN-Bi +0,22 — a história
  do artigo mora no horizonte longo, exatamente como a Tabela 2 sugeria.

**Fontes:**
- Hyndman, R.J.; Athanasopoulos, G. *Forecasting: Principles and
  Practice* (3ª ed.), §5.8–5.9 — <https://otexts.com/fpp3/accuracy.html>
- Politis, D.N.; Romano, J.P. (1994). *The Stationary Bootstrap*. JASA
  89(428), 1303–1313. DOI verificado:
  <https://doi.org/10.1080/01621459.1994.10476870>
- Uso na coleção: `bendes2026bio` (persistência como referência mínima,
  skill com IC 95%); `trebing2020wind` (persistência em todos os
  experimentos).

---

## A4 — Erro estratificado por regime e por faixa de intensidade (+ MBE)

**O que é.** Repartir o erro por bloco homogêneo: regime diurno
(06–17h) vs. noturno (18–05h) e faixas de intensidade do valor real
(0–3 / 3–13 / ≥13 m/s — as faixas operacionais usadas por
xianfeng2026data). MBE = média(pred − real): **viés** sistemático
(positivo = superestima).

**O que implica.** MAE/RMSE globais escondem *onde* o modelo falha:
regimes com dinâmicas distintas (brisas diurnas vs. noturnas) e eventos
de vento alto são os casos que importam para energia. MBE por extrato
revela calibração condicional (ex.: subestimar extremos é o erro
clássico em séries turbulentas — wei2026secondary relata recall de 25%
em eventos fortes).

**Resultado obtido** (tabelas A4):
- Faixa 0–3 m/s: **vazia** nesta série (não há calmaria no sítio —
  informa o leitor sobre o regime local);
- Faixa ≥13 m/s: só 36 amostras, mas **MBE ≈ −4 m/s em todos os
  modelos** — subestimação sistemática severa dos extremos (achado
  forte para a Discussão);
- Regime: os modelos LSTM/TCN-Bi têm MBE **positivo de dia e negativo
  à noite** (troca de sinal do viés); TCN é o mais estável (+0,07/−0,03);
  todos erram mais de dia.

**Fontes:**
- `xianfeng2026data` — faixas 0–3/3–13/13–25 m/s com boxplots, t-test
  pareado e MBE por faixa (Table 3, Fig. A3);
- `aslam2026physics` — quartis de volatilidade Q1–Q4 (alternativa quando
  intensidade não é o corte natural);
- `wei2026secondary` — eventos fortes como classificação, com recall
  reportado honestamente (25–55%).

---

## A5 — Métricas de cauda: MedAE, P95, MaxAE (por horizonte)

**O que é.** Percentis do erro absoluto: MedAE (mediana — robusto a
outliers), P95 (pior caso típico), MaxAE (pior caso absoluto).

**O que implica.** MAE/MSE falam da média; despacho e segurança de
grade precisam do **pior caso**. Dois modelos com mesmo MAE podem ter
P95 muito diferentes. Ehsan2000wind inclui MedAE explicitamente pela
robustez; xiang2026seqbinder reporta P95/MaxAE por horizonte.

**Resultado obtido** (tabela A5 + `caudas_por_horizonte.csv`): TCN tem
as melhores caudas entre os DL (MedAE 0,68; P95 2,45; Max 4,86) —
ligeiramente melhores que ARIMA em MedAE, piores em P95.

**Fontes:**
- `ehsan2000wind` (MedAE por robustez a outliers, Seção de métricas);
- `xiang2026seqbinder` (P95 e MaxAE por horizonte).

---

## A6 — Tabela de custo computacional

**O que é.** Parâmetros, tamanho, épocas, tempo de treino e inferência
por janela, no mesmo hardware (pedido explícito da Vanessa).

**O que implica.** Viabiliza o trade-off acuráça×custo (TCN-Bi: 1,6M
params, 443 s de treino, e é *pior* que a TCN — argumento forte);
permite ao leitor estimar despacho. Prática padrão: liu2026robust
justifica pelo ciclo de despacho de 10–15 min.

**Resultado obtido** (tabela A6): LSTM 159k/136 s; LSTM-Bi 970k/272 s;
TCN 110k/201 s; TCN-Bi 1,6M/443 s; inferência 0,10–0,23 ms/janela.

**Fontes:** `liu2026robust` (l.811–848); `jiang2026data` (TT/PT de 40
modelos); `aslam2026physics` (params/FLOPs/latência).

---

## B3 — Teste KS sobre as distribuições de |erro| (KSPA-style) + ECDF

**O que é.** Kolmogorov–Smirnov duas amostras sobre os erros absolutos
de dois modelos: H₀ = mesma distribuição. A estatística D é a distância
máxima entre as ECDFs (efeito observável no gráfico).

**O que implica.** O DM compara *médias*; o KS compara a distribuição
inteira — um modelo pode empatar na média e ainda assim ter caudas
piores. **Caveat importante (nosso n=43k):** com amostras grandes **todo**
p sai ≈ 0 — reporte o **D** (tamanho de efeito) e o gráfico ECDF, não o
p isolado.

**Resultado obtido** (tabela B3 + `images/ecdf_erro.png`): todos os
pares p≈0; os maiores D são LSTM×TCN (D=0,11) — a hierarquia por D
espelha a do DM.

**Fontes:**
- scipy.stats.ks_2samp (implementação e hipóteses; verificado):
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html>
- Uso na coleção: `gomez2026predicting` (ECDF + KSPA, Seção 3.3, Fig. 6).

---

## B4 — Métricas de fase: L_max, D_phase, DA

**O que é.** L_max: lag (passos) da correlação cruzada máxima entre
predição e real por trajetória — defasagem sistemática. D_phase:
|instante do pico previsto − do pico real| dentro da janela de 6 h.
DA: fração de passos em que o sinal da variação bate (subiu/mesmo caiu).

**O que implica.** Séries com ciclo semidiurno (o nosso caso, quedas
06h/18h) premiam modelos que "acertam o relógio": um pico previsto com
2 h de atraso tem MAE razoável e valor operacional baixo.
liu2026robust criou L_max/D_phase exatamente para substituir inspeção
visual subjetiva de fase.

**Resultado obtido** (tabela B4): L_max médio ≈ 0 (sem defasagem
sistemática); D_phase mediano 7–10 passos (70–100 min de erro de timing
do pico); **DA ≈ 0,50 em todos** — ou seja, na escala de 10 min nenhum
modelo tem informação direcional relevante; a informação está no
*padrão* da janela, não no incremento passo-a-passo (caveat de
interpretação, não defeito).

**Fontes:** `liu2026robust` (L_max/D_phase, l.775–789);
`huang2024research` (directional accuracy entre 8 critérios).

---

## B6 — A/B de função de perda (ponderada pelo horizonte vs. MSE) com DM

**O que é.** Mesma arquitetura, mesma seed, duas perdas; DM na
diferença de perdas. A ablação que formaliza o parágrafo de perdas do
artigo (que hoje não bate com nenhuma rodada).

**O que implica.** Sem isso, a frase "a perda ponderada foi desenvolvida
para as LSTM" é afirmação sem evidência — e a evidência que existe é
fraca: **só a TCN-Bi difere significativamente, a favor do MSE puro**
(p=0,028); lstm/lstm_bi/tcn: n.s. (p=0,12–0,30). Recomendação para o
texto: ou descrever a perda como avaliada em ablação sem ganho
estatístico, ou omitir a motivação específica.

**Fontes:** prática de ablação de perda em `sheng2026shape`
(cadeia WYM-1/2/3) e `aslam2026physics` (5 perdas, protocolo idêntico).

---

## B8 — Diagnóstico de resíduos + detecção de mudanças (Page–Hinkley)

**O que é.** Assimetria/curtose/Jarque–Bera dos resíduos (afastamento da
Gaussianidade → intervalos normais não valem; FPP3 §5.4); Page–Hinkley
(CUSUM dois lados) detecta mudanças de média na série real; compara-se o
erro perto vs. longe das mudanças.

**O que implica.** Mostra *onde* e *por que* erra. Nosso resultado foi
**hipótese refutada com honestidade**: o erro perto dos breakpoints
(diários) é ≈ ao erro longe (1,02 vs 1,10 no LSTM) — a falha não está
nas transições de média diárias; resíduos de todos os modelos são
leptocúrticos e enviesados (JB p≈0), reforçando o uso de bootstrap
(ítems A1/A2) em vez de pressupostos Gaussianos.

**Fontes:**
- Page, E.S. (1954). *Continuous inspection schemes*. Biometrika
  41(1/2), 100–115 — referência completa conforme citada **dentro** de
  `cao2026probabilistic` (Seção de drift detection);
- FPP3 §5.4 (diagnóstico de resíduos): <https://otexts.com/fpp3/accuracy.html>
- Uso correlato: `aodi2026optimized` (BFAST, breakpoints com IC 95%);
  `cao2026probabilistic` (drift via soma cumulativa, erro nas
  vizinhanças do drift).

---

## Recomendados ainda não executados (com fonte)

| Processo | O que implica | Fonte verificada |
|---|---|---|
| **A3** — 5 seeds → média±IC | transforma alegações sobre *arquiteturas* (não sobre treinos); destrava Friedman | prática em `wei2026secondary`, `aslam2026physics`, `singh2026new`, `kandemir2026explainable`, `jiang2026data` |
| **B1** — Friedman + post-hoc Wilcoxon/Holm + tamanho de efeito | teste "de família" antes dos pares; correção de multiplicidade | `bendes2026bio` (Sec. 3.7: Friedman, Wilcoxon+Holm, Cohen d/Cliff δ) |
| **B2** — Giacomini–White (habilidade condicional) | DM testa média; GW testa por regime/informação — casa com A4 | Giacomini & White (2006), Econometrica 74(6), 1545–1578, DOI verificado: <https://doi.org/10.1111/j.1468-0262.2006.00718.x>; uso: `jie2026designing` |
| **B7** — Permutation Feature Importance | importância de grupos de features sem retrinar; warning sobre features correlacionadas (nosso caso: 20 alturas!) | scikit-learn (verificado): <https://scikit-learn.org/stable/modules/permutation_importance.html>; base: Breiman (2001), DOI citado lá: 10.1023/A:1010933404324; uso: `chang2021ai` |
| **B5/B9** — ablação de features / truncamento de treino | valor dos dados vs. do modelo; quantifica se 2 meses bastam | `shafiullah2026advanced`; `liu2026robust` (l.803–807) |

## Achados operacionais desta execução (para corrigir no pipeline)

1. **Bug latente no relatório canônico:** `data/results/report_8020/mcs.json`
   tem `n_origins: 163` — as origens dos baselines
   (`predictions_long.csv`) têm drift de microssegundos (ex.:
   `00:39:59.999998`) e não casam com a grade regularizada do dataset;
   qualquer análise que alinhe DL×baselines por `origin` colapsa para
   ~163. Correção: `origin.dt.round("min")` antes de alinhar (aplicado
   em `studies/metodos_complementares.py`; **falta aplicar em
   `reports/src/relatorio_resultados.py`/`src`**).
2. **Trajetórias incompletas:** o filtro de timestamps observados deixa
   ~18% das origens sem algum horizonte; este relatório usa só as 1196
   origens completas (43.056 amostras).
3. A frase de perdas do `03_Material_e_metodos.tex` segue inconsistente
   com as rodadas (B6 acima) — já apontado em comentário no .tex.
