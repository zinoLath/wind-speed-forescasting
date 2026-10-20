# Sessão de Comentários SBSR #1

Sessão de 20/10/2026. Consolidação mecânica do estado da revisão do
artigo SBSR (`articles/sbr/LaTeX_Template_SBSR_Portugues/`), a partir de:

- `docs/revisions/sbr/comentarios_revisao_sbsr.md` — análise comentário
  a comentário da rodada Overleaf (13–14/10/2026; 18 threads, Vanessa +
  Flávio) + pendências herdadas;
- `docs/comentarios_vanessa.md` — rodada do commit `151ac29` (typos e
  citações);
- comentários inseridos nos `.tex` nesta sessão (blocos
  `% COMENTÁRIOS DE REVISÃO` em `02`, `03`, `04`, `06`).

**Convenções:** IDs (`6ace…`) = threads do export Overleaf. `[V-…]` =
Vanessa, `[F-…]` = Flávio. Este relatório **não contém prosa de artigo** —
textos prontos para colar vivem no doc de revisão; aqui estão estado,
números, comandos e material de estudo.

Panorama: **5 em aberto · 9 com decisão humana · 6 resolvidos**.

---

## 1. Comentários em aberto (trabalho determinístico pendente)

Ordem por impacto. Nenhum item aqui exige opinião — são passos mecânicos
com critério de pronto verificável.

### 1.1 [CRÍTICO] Rodada canônica: Tabela 1 ≠ figuras ≠ nenhuma rodada local — `V-6ace944b`

- **Evidência verificada:** pela eq. (1) do artigo,
  `Var(y) implícita = MSE/(1−R²)`. Na Tabela 1 (`tab:error_all`):
  - baselines → 1,329/(1−0,508) = **2,702** (= metrics.json dos
    baselines: 1478 origens × 36 = 53208 amostras);
  - DL → 1,384/(1−0,502) = **2,779** (≈ 1406 origens × 36 = 50616).
  Origens diferentes ⇒ R² da mesma tabela não comparáveis entre si.
- **Agravante:** o heatmap publicado vem da rodada `report_8020`
  (ponderada), cuja ordenação (TCN 0,514 melhor DL) **contradiz** a
  Tabela 1 (TCN-Bi 0,502 melhor DL). Nenhuma rodada local reproduz a
  tabela DL do artigo.
- **Passos (mecânicos, depois de escolhida a rodada — ver §2):**
  1. Backup: `cp -r pipeline/tmp/trained_8020 pipeline/tmp/trained_8020_backup_$(date +%Y%m%d)`;
  2. `python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020 --out-dir data/results/report_8020` (sem `--no-baselines` ⇒ DL e baselines nas mesmas origens);
  3. Critério de pronto: `MSE/(1−R²)` **igual para as 7 linhas** (uma única variância implícita);
  4. Tabelas LaTeX prontas: `python scripts/sbsr_tables.py` (finding `tabelas-sbsr-geradas`; artefato `reports/lidar/8020/tabelas_sbsr.tex`);
  5. Atualizar Tabela 1, Tabela 2 (h=36), Resumo e frases com números — fluxo da skill `sbsr-integrate` (inclui verificação de consistência).

### 1.2 Teste DM no artigo — `V-6ace9381` + `V-6ace9991`

- **Estado:** ferramenta existe (`tests/relatorio_resultados.py`,
  `diebold_mariano_loss_diff`; saída `mcs.json`) e a Vanessa pediu
  explicitamente "DM complementado por bootstrap em blocos". Falta
  **usar** no artigo.
- **Passos:**
  1. Rodar o comando do item 1.1 (gera DM junto);
  2. Ler pares/p-valores em `data/results/report_8020/mcs.json`;
  3. Inserir: 1 parágrafo no §Avaliação (definição H₀ + por que
     HAC/blocos — parágrafo sugerido no doc de revisão §3.8/[METODO-A1]
     do `03`) + tabela de p-valores dos pares TCN×LSTM,
     TCN-Bi×LSTM-Bi, melhor DL×ARIMA;
  4. Corrigir os números do par. 3 do Resultados: as diferenças de R²
     são **0,080** e **0,042** (o texto diz "0,8" e "0,4") — pendente
     também `V-6ace9991`.
- **Estudo:** §4.2 deste relatório.

### 1.3 Tabelas de hiperparâmetros e do dataset — `V-6ace8ce9`

- **Dados exatos (verificados):**
  - hiperparâmetros: `data/results/optuna/<modelo>/trials.csv` (best em
    `data/results/wfo_optuna/<modelo>/metrics.json`);
  - nº de parâmetros / loss / tamanho: `evaluation.json` por modelo;
  - tempo de treino: `data/results/*/summary.json`
    (`training_time_sec`), `training_history.json`;
  - dataset: colunas de `data/dataset.csv` — coletados `ws{40..260}` (20
    níveis), `dir{...}`, `v{...}`, `press`, `humid`, `temp`;
    computados `cis1..cis19`, `disp{...}`/`vdisp{...}` (confirmar
    dicionário exato).
- **Passo:** montar as duas tabelas (posso gerar dos JSONs — pedir).
- **Decisão embutida:** listar só os 4 modelos canônicos e citar
  ablações em texto (ver doc de revisão §3.6).

### 1.4 Contagem de amostras 7632 × 7561 — `V-6ace7f52`

- **Verificado:** `data/dataset.csv` = 7561 linhas, 16/09 18:00 →
  08/11/2021; série **não completa** (linhas inteiras ausentes, sem
  NaN); grade completa do intervalo ≈ 7606–7776 conforme a contagem.
- **Pendente (mecânico):** confirmar se `dataset.csv` é pós-imputação
  (estágio `impute`, `src/impute_wind_data.py`, lacunas ≤ 12 passos = 2 h)
  ou bruto da campanha — comparar contra o CSV bruto da campanha
  (`data/wind_data.csv` cobre período diferente; localizar o export da
  campanha EOSOLAR). Depois: declarar no texto N ausentes + tratamento
  (frase pronta no doc de revisão §3.4).

### 1.5 Figuras legíveis — `V-6ace71fd` (P1) + `V-6ace992b` (fonte)

- **Heatmap:** regenerar `06_heatmap_mse_horizonte.png` com fontsize
  ~13–16 (ou figsize menor) em `tests/relatorio_resultados.py`;
  re-sincronizar com `python scripts/articles_sync.py sync`.
- **Figura 1 (P1):** escolher abordagem (ampliar para `\linewidth` /
  recorte com P1 centralizado / regerar com marcador destacado);
  conferir se a figura do Overleaf difere da local. Regerar imagem é
  mecânico; a escolha da abordagem está no §2.

---

## 2. Atuados — falta intervenção humana (escrever / dissertar / escolher)

Respostas já verificadas com dados/código; o que falta é decisão ou
redação sua. Textos prontos no doc de revisão, seções indicadas.

| # | Item (thread) | O que já está feito | O que falta de você |
|---|---|---|---|
| 1 | **Escolha da rodada canônica** (afeta 1.1–1.3, `V-6ace8d78`) | As duas candidatas medidas: `report_8020` (ponderada: TCN 0,514/TCN-Bi 0,403/LSTM 0,279/LSTM-Bi 0,225) × `report_8020_mseloss` (mse: 0,490/0,489/0,053/0,330); ablação ITILDE-Q existe (`reports/lidar/8020-itildeq`) | Escolher qual rodada alimenta o artigo (estude §4.1–4.3 antes) |
| 2 | "Tá correto?" — loss ( `V-6ace8d78`) | Pesos 0,7→1,3 corretos (`src/models/losses.py:15-25`); divisão "LSTM ponderada/TCN MSE" **não bate** com nenhuma rodada | Reescrever o parágrafo conforme a escolha 1 (sugestões prontas: doc §3.7 e [METODO-LOSS] no `03`) |
| 3 | Entendimento do negócio (`V-6ace64b6`) | Diagnóstico: subseção descreve dados, não negócio; esqueleto de 4 pontos pronto (doc §3.1) | Dissertar o negócio (geração eólica, Delta Maranhão, por que prever importa) |
| 4 | Ferramentas/versões (`V-6ace820a`) | Versões verificadas (Python 3.10, TF 2.21, PyWavelets 1.8 com algoritmos prontos, sklearn 1.7, statsmodels 0.15, Optuna 4.9); parágrafo pronto (doc §3.5) | Revisar e colar |
| 5 | Verbos pessoais (`F-6ace2c7f`, resolvido no Overleaf mas texto mantém) | Reescrita impessoal pronta (doc §1.1) | Aplicar (ou decidir manter) |
| 6 | "Arquitetura" (`F-6ace2e9e`) | Regra de troca definida (arquitetura = combinação estrutura+rede; rede neural = mecanismo) | Varrer o texto e aplicar o critério |
| 7 | Legenda "Fonte:" + EOSOLAR (`V-6ace641f`) | Bib local `arcilanVentoSul` **já é** o artigo do projeto EOSOLAR (Assireu et al. 2022, Energies 15(7):2319) | Conferir o bib no **Overleaf**; escolher formato da legenda (doc §3.3) |
| 8 | Conclusões: recapitular objetivo / "área de estudo" / validação cruzada (`F-6ace2e79`, `F-6ace2f14`, `F-6ace2fa9`) | 3 reescritas prontas (doc §5.1–5.3); diagnóstico correto: dado é só do P1; CV temporal existe (Roberts 2016, já no bib) | Revisar e aplicar |
| 9 | Citações de baselines (Apêndice D) | Referências identificadas: Box & Jenkins (ARIMA), Breiman 2001 (RF), Hyndman & Athanasopoulos (persistência) | Criar entradas com dados verificados (skill `bibtex-articles`) |

---

## 3. Resolvidos (simples)

| Item | Thread | Resolução |
|---|---|---|
| Typos das Conclusões (Contanto, limtação, codificador/codificador, se demonstrou) | rodada `151ac29` | Corrigidos nos `.tex` locais (doc §5.4) |
| Typos de Material e métodos (tem, foi encontrada, implementados) | rodada `151ac29` | Corrigidos (doc §5.4) |
| Tabela renderizando nas referências | `F-6ace2fd7` | `\usepackage{placeins}` + `\FloatBarrier` no preâmbulo |
| **ANEEL desatualizado** | `V-6ace4915` | **Esta sessão** — ver tabela abaixo; frase reescrita + `@misc{aneelSiga}` no bib |
| Float `[h]` / Placeins validado | — | Pendente só validação visual no PDF do Overleaf |
| `arcilanVentoSul` = projeto EOSOLAR | `V-6ace641f` (parte) | Verificado no bib **local** (resta conferir Overleaf — item 7 do §2) |

### ANEEL — números desta sessão (SIGA, snapshot 2026-10-20)

Fonte: `data/siga-empreendimentos-geracao-diario.csv`
(dadosabertos.aneel.gov.br; autor SCE/ANEEL; mantenedor CEGDI/ANEEL;
versão 1.1; atualização 20/10/2026; 25.058 empreendimentos). Filtro:
`DscFaseUsina = "Operação"`, potência **outorgada** (MdaPotenciaOutorgadaKw).

| Posição | Fonte | GW | Participação |
|---|---|---|---|
| 1º | Hídrica | 110,3 | 49,8% |
| **2º** | **Eólica** | **35,0** | **15,8%** |
| 3º | Fóssil | 32,1 | 14,5% |
| 4º | Solar | 23,8 | 10,8% |
| 5º | Biomassa | 18,3 | 8,3% |
| 6º | Nuclear | 2,0 | 0,9% |
| — | Total | 221,6 | 100% |

Detalhes eólica: 1.138 empreendimentos em operação; UFs: BA 381, RN 319,
PI 126, CE 103, RS 80, PE 46; **MA 16 emp. / 0,4 GW**; pipeline futuro:
2,0 GW em construção (62) + 14,2 GW não iniciada (348). Matriz
renovável (E+S+H+B): 84,7% (fiscalizada). Fiscalizada eólica: 34,9 GW.

Correção conceitual que a troca resolve: o texto antigo dizia "22 GW na
**geração**, terceiro lugar" — misturava capacidade com geração e fonte
errada (EOSOLAR não é fonte de matriz elétrica). Agora: capacidade
(SIGA/ANEEL) com cite próprio; o cite EOSOLAR ficou só para o ranking
global de 2023.

---

## 4. O que estudar (para resolver o que falta — em ordem)

Cada item: por quê → conceito mínimo → material exato → exercício
determinístico de fixação.

### 4.1 R² × variância implícita (30 min — destrava 1.1)

- **Por quê:** é a chave do [CRÍTICO]; explica por que a Tabela 1 não
  pode estar certa.
- **Conceito:** pela própria eq. (1) do artigo,
  `R² = 1 − MSE/Var(y)` ⇒ `Var(y) = MSE/(1−R²)`. Se todos os modelos
  foram avaliados sobre os mesmos valores reais, **todas as linhas
  produzem a mesma variância implícita**. Valores diferentes ⇒ conjuntos
  de avaliação diferentes.
- **Exercício:** abrir `data/results/report_8020/evaluation.json` e
  `data/results/baselines/arima_rf/metrics.json`; calcular MSE/(1−R²)
  por linha e comparar. Repetir com a Tabela 1 do artigo e constatar os
  grupos 2,70 × 2,78.
- **Material:** doc de revisão §4.2 (conta completa feita).

### 4.2 Diebold–Mariano + bootstrap em blocos (o pedido explícito da Vanessa)

- **Por quê:** threads `6ace9381`/`6ace9991`; sem isso, "TCN > LSTM" é
  alegação sem significância.
- **Conceito mínimo:**
  - por origem do rolling forecast, `d_t = L(e_A,t) − L(e_B,t)`
    (perda, ex. MSE). H₀: `E[d_t] = 0`;
  - a estatística DM normaliza a média de `d_t` por um desvio-padrão
    estimado com autocovariâncias (**HAC/Newey–West**) porque `d_t` é
    serialmente correlacionado;
  - correção de amostras pequenas: **Harvey–Leybourne–Newbold (1997)**;
  - **bootstrap em blocos (moving-block):** reamostrar blocos contíguos
    de `d_t` (preserva a dependência temporal) para o p-valor/IC sem
    assumir normalidade.
- **Onde já está:** `tests/relatorio_resultados.py`
  (`diebold_mariano_loss_diff`, agregado e por horizonte) →
  `data/results/report_8020/mcs.json`.
- **Material:** doc de revisão Apêndice C; precedentes da sua coleção:
  `docs/revisions/sbr/kbscout/result-1.md` (gomez2026predicting: Harvey +
  Newey–West; bendes2026bio: Friedman→Wilcoxon+Holm) e `result-4.md`
  (Giacomini–White condicional).
- **Exercício:** rodar o comando do 1.1 e ler `mcs.json`: localizar o
  par TCN×LSTM e interpretar estatística + p-valor.

### 4.3 Métricas relativas: skill score e Theil's U (complemento barato)

- **Por quê:** formaliza "bateu o naive?" por horizonte; você já tem
  `skill_por_horizonte.csv`.
- **Conceito:**
  - **Skill score:** `SS = 1 − MSE_modelo/MSE_naive` (0 = empata com o
    naive; <0 = pior);
  - **Theil U2 (1966):** `RMSE_modelo/RMSE_naive = √(1−SS)` — U2 < 1 ⇔
    SS > 0. Mesma informação, escala diferente;
  - **U1/TIC** (o que parte da literatura de vento chama "Theil
    coefficient") é outro animal: índice em [0,1] sem benchmark —
    descritivo, não diz contra o quê você é bom;
  - Nenhuma das duas **testa significância** — complementam o DM, não
    substituem.
- **Material:** precedentes na coleção — `lin2022deep` (Applied Energy
  2022, cita Theil U como métrica consagrada em vento) e `chen2024research`
  (MSFSC, usa TIC + acurácia direcional); consolidado em
  `docs/revisions/sbr/kbscout/result-5.md`.
- **Exercício:** de `skill_por_horizonte.csv`, calcular U2 = √(1−SS) por
  horizonte e verificar que a curva cruza 1,0 onde a skill cruza 0.

### 4.4 Testes globais e por regime (camada seguinte ao DM)

- **Friedman** (não-paramétrico, múltiplos modelos × múltiplos blocos —
  ex.: 36 horizontes como blocos) → se significativo, **Wilcoxon
  pareado + correção de Holm** nos pares. Precedente:
  `bendes2026bio` (result-1).
- **Giacomini–White**: habilidade preditiva **condicional** (por regime
  de vento, ex. ≥13,9 m/s vs <) — o DM é incondicional. Precedente:
  `jie2026designing` (result-4). Encaixa com seus estratos
  regime/banda em `metodos.json`.
- **MCS (Model Confidence Set)**: você **já tem** (`mcs.json`) e
  **nenhum** artigo da coleção usa — diferencial se reportado.
- **Material:** `docs/revisions/sbr/kbscout/result-1.md` a `result-4.md`.

### 4.5 Validação cruzada temporal (contexto da limitação)

- **Por quê:** thread `F-6ace2fa9` — "não foi possível" está errado;
  existe CV com estrutura temporal.
- **Conceito:** k-fold clássico embaralha e viola a ordem temporal;
  *time series split / blocked CV / rolling-origin* respeita (treino em
  blocos anteriores, teste em posteriores disjuntos). Seu 80/20 é uma
  única dobra temporal; a limitação é não repetir em múltiplas dobras.
- **Material:** Roberts et al. 2016 (chave `crossvalidation`, **já no
  bib**); doc de revisão §5.3.

### 4.6 CRISP-DM: negócio ≠ dados (para o item 3 do §2)

- **Conceito:** "Entendimento do negócio" = contexto/objetivos sob a
  ótica do negócio (o que está em jogo: comercialização, despacho,
  O&M); "Entendimento dos dados" = instrumentação, período, atributos.
  A subseção atual tem o conteúdo da segunda.
- **Material:** `wirth2000crisp` (já no bib); esqueleto pronto no doc
  §3.1.

---

## 5. Ordem de execução sugerida (determinística)

1. Estudar §4.1 e §4.2 (1–2 h) → decidir a rodada canônica (item 1 do §2);
2. Item 1.1: recalcular tudo numa rodada única + conferir variância
   implícita única (critério objetivo de pronto);
3. Item 1.2: DM no artigo (parágrafo no §Avaliação + p-valores +
   corrigir "0,8"/"0,4");
4. Itens 1.3–1.5 (tabelas, contagem, figuras) — posso gerar as tabelas
   e as contagens quando pedido;
5. Decisões de prosa do §2 (uma sessão de escrita sua, com os textos
   prontos do doc de revisão à mão);
6. Antes de submeter: remover todos os blocos `% COMENTÁRIOS DE
   REVISÃO` dos `.tex` + `consideracoes_extras.tex`, e validar o PDF no
   Overleaf.
