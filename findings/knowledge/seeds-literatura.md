---
title: Manejo de seeds na literatura da base (articles/knowledge)
area: artigos
updated: 2026-10-22
---

# Seeds na literatura da base (articles/knowledge)

Mapeamento de como os artigos da base tratam aleatoriedade de treino
(seeds/inicializações) e replicação estatística. Fonte: varredura grep em
`articles/knowledge/markdown/` por `seed`, `independent run`, `random
initialization`, `Friedman/Wilcoxon/Diebold-Mariano` etc. (2026-10-22).

## Panorama (4 padrões)

1. **Multi-seed explícito + estatística por modelo** (~12 artigos; o
   padrão mais rigoroso, todos 2025–2026): N runs independentes com seeds
   diferentes (N = 3, 5, 10 ou 30), resultados como **mean ± std** (às
   vezes IC 95%) e teste estatístico **sobre os N valores por modelo**.
   Detalhes notáveis:
   - **bendes2026bio** (biomimetics-11-00568, DWT–BiLSTM): o mais
     completo — 30 runs/algoritmo; Friedman + Wilcoxon pareado com Holm +
     Cohen's d + Cliff δ; **mesmo conjunto de seeds compartilhado entre
     condições** (joint vs sequential, ablações) para desenho pareado;
     seeds determinísticas 42001–42030 nas janelas Y2–Y5 vs aleatórias no
     Stage 1 (força Mann–Whitney em vez de Wilcoxon pareado); seeds
     persistidas em SQLite; rolling-origin com 10 training seeds.
   - **xiang2026seqbinder** (SeqBinder): 5 seeds explícitas (42–46);
     mean ± std; compara modelos também pela magnitude do std.
   - **aodi2026optimized** (ARSO): 10 runs/modelo; mean ± std + IC 95%;
     boxplots; teste DM calculado sobre as 10 runs por horizonte.
   - **kandemir2026explainable** (SHAP-BiLSTM): 10 inicializações;
     mean ± std; t-teste pareado sobre os RMSE das 10 runs (BiLSTM vs LSTM).
   - **wei2026secondary** (SD-TCGRU): 5 repetições; std + IC 95%; DM test.
   - **jiang2026data** (review+benchmark): 5 runs/modelo, mesmo split.
   - S2213138826005576 (PI-*, SETA): TODOS os números = mean ± std sobre
     5 seeds ("training and data partitioning").
   - **sridhar2026predictive**: múltiplas inicializações; mean ± std;
     t-testes pareados por split idêntico.
   - VORTEX (Applied Energy S0306261926007464): estabilidade seed-a-seed
     da config final (5 seeds: mean/std/range); "representative seed" =
     run mais próxima da mediana multi-seed; Wilcoxon para significância.
   - **dong2026graphdynnet**: multi-seed grid search de HPO — 3 seeds
     (42, 52, 62) por combinação, seleção por RMSE médio entre seeds.
   - VMD-MSI-GTTS (EPSR S0378779626006851): híbrido — tabela lista seed
     fixa 42, mas reporta ICs entre múltiplas runs com seeds diferentes
     + DM pareado.
   - **neshat2026physics** (LLXSTG): seeds fixas + confissão explícita de
     não-determinismo de CUDA ("within reported standard deviations").
2. **Teste estatístico sem replicar seed** (1 run; amostra = tempo/
   subtarefas/estações): DM, Friedman, Wilcoxon, KSPA, Kruskal–Wallis
   calculados sobre erros/predições — ex.: **binrong2026whole** (DM+
   Friedman por estação), S0957417426030617 (Friedman em 24 subtarefas),
   **leongomez2026multi** (Friedman/ranks entre horizontes), nascimento­
   transformer (Wilcoxon LSTM vs transformer sobre amostras), 
   shengxiang2022deep (DM + Kruskal–Wallis módulo/nível).
3. **Seeds como ensemble, não como comparação**: **quinonestowards**
   (microgrids) — múltiplas inicializações do seq2seq para gerar
   intervalos de predição 80/95% (incerteza epistêmica).
4. **Sem menção a seed** (maioria; provável 1 run): clássicos
   pré-2024 em geral — penagallardo2020comparison, chen2020short,
   dolatabadi2020hybrid, suzuki2021deep, zhu2019learning, nahid2020very,
   trebing2020wind, ehsan2000wind, liu2020deep, geng2026long etc.

## Resposta direta

- "Uma seed só?" — Na maioria da base, sim (não documentado).
- "Várias seeds + análise estatística por modelo?" — Sim, mas só na
  minoria recente (2025–2026): o teste usa os N valores por modelo
  (t pareado, Wilcoxon pareado, DM sobre runs); a prática de ouro
  (bendes2026bio, SeqBinder) fixa o **mesmo conjunto de seeds para todos
  os modelos** para permitir pareamento.
- N mais comuns: 5 e 10; 30 só no benchmark de metaheurísticas.

## Implicação para o projeto

- O protocolo multi-seed do repo (ver `estudos-ablacoes.md`, seção seeds)
  está alinhado com a minoria rigorosa da literatura; citar bendes2026bio
  e xiang2026seqbinder como precedentes de seeds compartilhadas + teste
  pareado no SBSR.
- Contraste útil: a maioria da literatura reporta 1 run — argumento de
  venda do nosso protocolo (e do benchmark honesto em `reports/`).
