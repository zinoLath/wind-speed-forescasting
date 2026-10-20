# Comentários de revisão — Profª. Dra. Vanessa Cristina Oliveira de Souza

> Fonte: comentários inline de revisão no artigo SBSR
> (`articles/sbr/LaTeX_Template_SBSR_Portugues/`), seções
> `06_Conclusoes.tex` e `03_Material_e_metodos.tex`
> (commit `151ac29` "Revisao de Resultados e Conclusoes (comentarios)").
> Convenção usada no texto: `[TEXTO]` = sugestão pontual de redação;
> `[REF]` = referência sugerida. **Remover os comentários antes de submeter.**

## Conclusões (06_Conclusoes.tex)

### Parágrafo 1 — objetivo
- **[TEXTO]** "O objetivo do trabalho foi **concluído** com êxito" →
  "foi **alcançado** com êxito" (regência mais usual; opcional).
- **[TEXTO]** "**contrastando** o **funcionamento** das redes" →
  "**comparando** o **desempenho** das redes" (mesmo sentido).

### Parágrafo 2 — resultados
- **[TEXTO]** "**Contanto**" → "**Contudo**" (typo).
- **[TEXTO]** "a TCN, tanto unidirecional quanto bidirecional, **mostrou**" →
  "**mostraram**" (concordância com sujeito composto) — ou, se preferir
  singular, "a **família TCN** mostrou".

### Parágrafo 3 — limitações
- **[TEXTO]** "**limtação**" → "**limitação**" (typo).
- **[TEXTO]** "**depreciando** a confiabilidade" →
  "**comprometendo** a confiabilidade" (mesmo sentido).
- **[REF]** Validar a crítica com **validação cruzada de estrutura temporal**:
  citar `crossvalidation` (Roberts et al. 2016) — **já presente** em
  `07_SBSR.bib` — respaldando a limitação apontada e a proposta futura.

### Parágrafo 4 — trabalhos futuros
- **[TEXTO]** "redes diferentes no **codificador e codificador**" →
  "no **codificador e no decodificador**" (typo).
- **[REF]** **Transformers**: citar `vaswani2017attention` (já no bib).
- **[REF]** **GRU**: se adotar, criar entrada — Cho, K. et al. "Learning
  Phrase Representations using RNN Encoder-Decoder for Statistical Machine
  Translation", EMNLP 2014 (**dados não conferidos no bib local; verificar
  antes de adicionar**).

## Material e métodos (03_Material_e_metodos.tex)

### Baselines (seção de métricas/modelos)
- **[REF]** Baselines sem citação — sugestões para criar entradas no bib
  (com dados verificados; nada inventado):
  - **ARIMA**: Box, G. E. P.; Jenkins, G. M. "Time Series Analysis:
    Forecasting and Control" (livro clássico).
  - **Random Forest**: Breiman, L. "Random Forests", Machine Learning
    45(1), 5–32, 2001.
  - **Persistência** (baseline ingênuo): Hyndman & Athanasopoulos,
    "Forecasting: Principles and Practice" (OTexts).
- **[TEXTO]** "implementado**s** através de componentes" →
  "implementado** ** através de componentes" — o sujeito é
  "O ARIMA", singular.

## Pendências para submissão

- [ ] Aplicar as correções de texto (typos: Contanto, limtação,
      codificador/codificador).
- [ ] Decidir concordância TCN (mostraram vs família TCN mostrou).
- [ ] Adicionar citações dos baselines (ARIMA, RF, Persistência) no bib.
- [ ] Conferir entrada da GRU (Cho et al. 2014) antes de adicionar.
- [ ] Citar `crossvalidation` e `vaswani2017attention` no corpo do texto.
- [ ] Remover os blocos `% COMENTÁRIOS DE REVISÃO` dos `.tex`.
