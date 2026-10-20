# Revisão SBSR — comentários do Overleaf (rodada 13–14/10/2026)

Análise comentário a comentário da rodada de revisão do artigo SBSR
("Comparação de Modelos de Aprendizado Profundo com Estrutura Seq2Seq
para a Predição da Velocidade do Vento com Dados LiDAR").

**Fonte dos comentários:** export JSON do Overleaf (colado na sessão de
14/10/2026). O export **não traz âncoras de posição** (não indica em qual
linha/trecho cada thread foi aberta), então a localização de cada
comentário foi **inferida** a partir do conteúdo + conferência nos
arquivos `.tex`. Toda localização inferida está marcada com
**(inferido)**; se houver threads no Overleaf que não estão neste export
(o JSON foi truncado na colagem), estender este documento.

**Arquivo do artigo (clone git do Overleaf):**
`articles/sbr/LaTeX_Template_SBSR_Portugues/`

- Autores dos comentários: **Flávio Belizario Mota** (coautor) e
  **Profª. Vanessa Cristina Oliveira de Souza** (orientadora). Alguns
  threads foram resolvidos pela própria Vanessa (`resolved_at`
  2026-10-13).

---

## Contexto da submissão SBSR

- **SBSR** = Simpósio Brasileiro de Sensoriamento Remoto (INPE,
  bienal; <http://sbsr.com.br>). Artigo curto em **português ou
  espanhol**, template oficial two-column (`sbsr.sty`, modelo do
  repositório adaptado de SBSR 2019/2021), referências `unsrt` em
  `07_SBSR.bib`. Estrutura exigida: Resumo/Abstract, Introdução,
  Material e métodos, Resultados, Conclusões
  (`00_Template_SBSR.tex` com `\input` de `01`–`06`).
- **Norma citada pelos revisores:** SBSR **não aceita verbos pessoais**
  (redação impessoal) — thread do Flávio resolvido por Vanessa.
- **Prazos/formatos finais:** conferir o site do evento a cada edição;
  os blocos `% COMENTÁRIOS DE REVISÃO` nos `.tex` e a seção
  `consideracoes_extras.tex` devem ser removidos antes de submeter.
- **Compilação:** o PDF oficial é compilado **no Overleaf**. Localmente
  o `pdflatex` falha por falta do pacote `babel-spanish` no TeX
  instalado (erro pré-existente, não causado pelas edições recentes).

---

## Legenda de status

| Marca | Significado |
|---|---|
| **[CORRIGIDO]** | já aplicado nos `.tex` desta máquina (não commitado/pushado) |
| **[RESPOSTA]** | pergunta dos revisores — respondida aqui com fato verificado |
| **[LACUNA]** | informação não disponível localmente — não inventada; onde buscar |
| **[DECISÃO DO AUTOR]** | escolha subjetiva/estilística — aplicar você mesmo (texto pronto quando possível) |
| **[AÇÃO]** | trabalho a fazer no artigo/código |
| **[GUIA]** | material de estudo + passos + ferramentas |
| **[CRÍTICO]** | risco de rejeição/observação forte de revisor externo |

---

## Checklist consolidado

- [x] **[CORRIGIDO]** Typos das Conclusões (Contanto, limtação,
      codificador/codificador, se demonstrou).
- [x] **[CORRIGIDO]** Typos de Material e métodos (tem, foi encontrada,
      implementados).
- [x] **[CORRIGIDO]** Tabela escorrendo para as referências
      (`placeins` + `\FloatBarrier`).
- [ ] **[CRÍTICO]** Tabela 1 do artigo não bate com nenhuma rodada
      local; **a figura do heatmap vem de outra rodada que a tabela**.
      Recalcular tabela+figuras de UMA rodada canônica.
- [ ] **[AÇÃO]** Teste estatístico (DM + bootstrap em blocos) no
      Resultados — a ferramenta já existe (`tests/relatorio_resultados.py`).
- [ ] **[AÇÃO]** Descrever a função de perda conforme a rodada canônica
      (texto atual "LSTM ponderada, TCN MSE" não bate com nenhuma rodada).
- [x] **[CORRIGIDO]** Atualizar dado ANEEL na Introdução com fonte
      própria (SIGA/ANEEL, snapshot 20-10-2026; cite `aneelSiga`).
- [ ] **[AÇÃO]** Parágrafo de ferramentas (linguagem/bibliotecas/versões).
- [ ] **[AÇÃO]** Tabela de hiperparâmetros + tempo de treino e tabela
      descritiva do dataset.
- [ ] **[AÇÃO]** Justificar contagem de amostras (7632 esperadas × 7561).
- [ ] **[AÇÃO]** Figura 1: P1 visível + aumentar tamanho; legenda no
      formato "Fonte: ..."; conferir cite EOSOLAR no Overleaf.
- [ ] **[AÇÃO]** Heatmap: aumentar fonte.
- [ ] **[DECISÃO DO AUTOR]** Reescrever "Entendimento do negócio" (falar
      do negócio, não só dos dados).
- [ ] **[DECISÃO DO AUTOR]** Recapitular objetivo na 1ª frase das
      Conclusões.
- [ ] **[DECISÃO DO AUTOR]** Reescrever frases sobre "área de estudo"
      (Conclusões par. 2) e validação cruzada (par. 3) — textos prontos
      abaixo.
- [ ] **[DECISÃO DO AUTOR]** Verbos pessoais na Introdução (texto pronto).
- [ ] **[DECISÃO DO AUTOR]** Terminologia "rede" → "arquitetura" onde couber.
- [ ] **[GUIA]** Diebold–Mariano + bootstrap em blocos (conceitos e uso).
- [ ] Remover blocos `% COMENTÁRIOS DE REVISÃO` antes de submeter.

---

# 1. Global (documento inteiro)

## 1.1 Verbos pessoais — thread `6ace2c7f`

- **Onde (inferido):** `02_Introducao.tex`, par. 2 — "Atualmente, para
  **realizarmos** a previsão da velocidade do vento, **utilizamos**
  métodos físicos...". Único trecho com 1ª pessoa do plural no corpo
  (conferir também Resumo e Conclusões ao revisar).
- **Afeta:** norma de redação do SBSR (impessoalidade).
- **Comentário (Flávio):** "Eu particularmente acho que o SBSR ainda não
  aceita artigos que utilizem verbos pessoais." — **RESOLVIDO por
  Vanessa** em 13/10/2026.
- **Análise:** o thread foi resolvido, mas **o texto continua com verbos
  pessoais** — provavelmente ficou acordado corrigir e a correção não
  foi feita (ou foi feita só no Overleaf; conferir).
- **Ação [DECISÃO DO AUTOR]** — sugestão pronta:
  > "Atualmente, a previsão da velocidade do vento em médio e longo
  > prazo é realizada por métodos físicos, que simulam a área local a
  > partir de parâmetros meteorológicos e geográficos coletados
  > \cite{nwpclassic}."
- Alternativa mínima: "para a previsão da velocidade do vento, **são
  utilizados** métodos físicos".

## 1.2 Tabela renderizando no meio das referências — thread `6ace2fd7`

- **Onde:** `00_Template_SBSR.tex` (floats de `04_Resultados.tex` com
  `[h]` escorrendo para depois de `\bibliography`).
- **Comentário (Flávio):** "Sem a ver com o texto aqui, mas o PDF está
  renderizando uma tabela no meio das referências."
- **Análise:** em documento two-column, floats `[h]` que não cabem na
  página são postergados e caem no fim do documento (dentro/after das
  referências). Bug de layout clássico, correção padrão.
- **[CORRIGIDO]** Adicionado `\usepackage{placeins}` no preâmbulo e
  `\FloatBarrier` imediatamente antes de `\renewcommand{\refname}` —
  força figuras/tabelas pendentes a saírem antes das referências.
  Alternativa se preferir: trocar `[h]` por `[!htb]` nas tabelas do
  `04_Resultados.tex` (reverter o FloatBarrier se conflitar com o
  estilo).
- **Nota:** validar visualmente no PDF do Overleaf após o push.

---

# 2. Introdução (`02_Introducao.tex`)

## 2.1 Par. 1 — dado da ANEEL desatualizado — thread `6ace4915`

- **Onde:** par. 1 — "Em 2023, a energia eólica participou com 22 GW na
  geração brasileira, ficando em terceiro lugar no ranking nacional de
  energias, e em sexto lugar globalmente `\cite{arcilanVentoSul}`."
- **Comentário (Vanessa):** "Atualizar essa informação. Pega a fonte da
  aneel mais recente."
- **Análise:** dois problemas: (a) o número é de 2023; (b) o cite que
  sustenta a frase é o artigo do projeto EOSOLAR (`arcilanVentoSul` =
  Assireu et al. 2022, *Energies* 15(7):2319), que **não é** fonte de
  dado de capacidade instalada — a frase precisa de fonte própria
  (ANEEL).
- **[CORRIGIDO — 20/10/2026]** Atualizado com o dataset **SIGA/ANEEL —
  empreendimentos de geração (base diária)**
  (`data/siga-empreendimentos-geracao-diario.csv`; dadosabertos.aneel.gov.br;
  autor SCE/ANEEL; mantenedor CEGDI/ANEEL; versão 1.1; snapshot
  20-10-2026; 25.058 empreendimentos):
  - **Eólica em operação: 35,0 GW (outorgada; 34,9 fiscalizada), 2ª
    maior fonte nacional, 15,8% da matriz** (hídrica 110,3 GW/49,8%;
    fóssil 32,1; solar 23,8; biomassa 18,3; nuclear 2,0; total 221,6);
  - 1.138 empreendimentos eólicos em operação (BA 381, RN 319, PI 126,
    CE 103, RS 80, PE 46; **MA 16 emp./0,4 GW**); pipeline eólico
    futuro: 2,0 GW em construção + 14,2 GW não iniciada;
  - Frase reescrita: ranking global 2023 (6º) permanece com
    `arcilanVentoSul`; capacidade nacional de out/2026 (35 GW, 2ª
    fonte, 15,8%) com cite próprio **`aneelSiga`** (entrada `@misc`
    criada no `07_SBSR.bib`);
  - Correção conceitual: o texto antigo dizia "22 GW na **geração**"
    (misturava capacidade com geração) e usava o EOSOLAR como fonte de
    matriz elétrica.
- **Nota:** relatório da sessão em
  `reports/protocolo/sessao-comentarios-sbsr-1/` (favorito no hub).

---

# 3. Material e métodos (`03_Material_e_metodos.tex`)

## 3.1 §Entendimento do negócio — "falar do negócio" — thread `6ace64b6`

- **Onde:** §3.1 "Entendimento do negócio" (pars. 1–2: base LiDAR da
  campanha EOSOLAR, Complexo Eólico Delta Maranhão, P1 na fig. 1,
  temporada seca set–nov/2021; resolução 10 min, 20 níveis).
- **Comentário (Vanessa):** "aqui é preciso falar do negócio. Achei que
  vc falou muito dos dados."
- **Análise:** na CRISP-DM, "Entendimento do negócio" descreve o
  contexto de negócio e o objetivo do projeto sob a ótica do negócio; a
  subseção atual descreve instrumentação e período de coleta (conteúdo
  de "Entendimento dos dados"). O detalhe de coleta (pars. atuais)
  pertence à subseção seguinte.
- **Ação [DECISÃO DO AUTOR]** — esqueleto sugerido para a subseção:
  1. O negócio: geração eólica no Brasil e no Maranhão (Delta
     Maranhão); o que está em jogo (comercialização de energia,
     confiabilidade do grid, O&M dos aerogeradores).
  2. Por que prever a velocidade do vento importa para esse negócio
     (despacho, preço, manutenção) — pode reaproveitar a motivação da
     Introdução com cites já existentes.
  3. Objetivo do projeto sob a ótica do negócio: comparar arquiteturas
     para estimar a disponibilidade do recurso eólico no sítio.
  4. Mover os detalhes de coleta (LiDAR, alturas, estações,
     temporada) para o início de "Entendimento dos dados".

## 3.2 Figura 1 — "P1 não visível" — threads `6ace62a3` (resolvido) e `6ace71fd` (aberto)

- **Onde:** `03_Material_e_metodos.tex`, figura `\ref{fig:local_estudo}`
  (`fig1_local_estudo.png`, incluída a `0.5\linewidth`), par. "O ponto
  de instalação do LiDAR está localizado no P1 na figura ...".
- **Comentário (Vanessa, 2× texto idêntico):** "Não dá pra ver P1 nessa
  figura." (um thread resolvido por ela, outro permanece aberto).
- **Análise (verificado na imagem):** a figura tem os marcadores
  (pinos) e legenda, mas renderizada a `0.5\linewidth` em artigo
  two-column ela fica minúscula; o rótulo P1 é quase ilegível. O thread
  duplicado sugere que mesmo após a primeira resposta o problema
  persistiu (ou o segundo se refere à mesma figura em outra passada de
  revisão).
- **Ação:** uma de:
  - Aumentar a figura para `\linewidth` (ou gerar a imagem já recortada
    no trecho do mapa com P1 centralizado);
  - Regenerar a figura com marcador/rtulo P1 maior e em destaque
    (cores contrastantes, seta).
  - Conferir no Overleaf se a versão da figura difere da local.

## 3.3 Figura 1 — formato da fonte e referência EOSOLAR — thread `6ace641f`

- **Onde:** legenda da fig. 1 — "Localização do sítio de estudo, no
  litoral maranhense`\cite{arcilanVentoSul}`."
- **Comentário (Vanessa, 2 mensagens):** "Aqui vc coloca a referência e
  depois o `\cite`. Fonte: Assireu et al. (2024) [1] Não sei se é assim.
  É só um exemplo." / "usar a referência do projeto EOSOLAR pq ela é
  anterior ao artigo do arcilan de vento sul".
- **Análise (verificado):** no bib local, a chave `arcilanVentoSul` **já
  é o artigo do projeto EOSOLAR** (Assireu, A. T. et al., "EOSOLAR
  Project: Assessment of Wind Resources of a Coastal Equatorial Region
  of Brazil—Overview and Preliminary Results", *Energies* 15(7):2319,
  **2022**). Ou seja: ou isso já foi resolvido nesta cópia local, ou a
  cópia do Overleaf ainda aponta para outro artigo do Arcilan —
  **conferir o `07_SBSR.bib` no Overleaf**.
- **Ação [DECISÃO DO AUTOR]:** formato da legenda — se adotar o estilo
  "Fonte:", a legenda fica:
  > `\caption{Localização do sítio de estudo, no litoral maranhense.
  > Fonte: Assireu et al. (2022)\cite{arcilanVentoSul}.}`
  (conferir o formato exigido pelo template/manual do SBSR — o exemplo
  da própria Vanessa diz "não sei se é assim").

## 3.4 §Entendimento dos dados, par. 1 — contagem de amostras — thread `6ace7f52`

- **Onde:** par. 1 — "A série temporal coletada possui **7561
  amostras**, de 16/09/2021 até 08/11/2021..."
- **Comentário (Vanessa):** "se a série for completa, essa conta deveria
  dar 7632 amostras. Justificar o pq da diferença."
- **Análise [RESPOSTA] — verificado em `data/dataset.csv`:**
  - O arquivo tem **7561 linhas**, cobrindo **16/09/2021 18:00 →
    08/11/2021** (o intervalo não começa/termina em dia cheio).
  - Série completa de 16/09 a 08/11 (54 dias × 144 amostras/dia a 10
    min) daria **7776**; os 7632 da Vanessa = 53 dias cheios. Pela
    grade do próprio intervalo do arquivo seriam ~7611. **Em qualquer
    contagem, 7561 < esperado → a série NÃO é completa.**
  - Há **timestamps ausentes e alguns fora da grade** (ex.: minutos
    :29) no arquivo; **não há células NaN** — as lacunas são linhas
    inteiras ausentes (indisponibilidade do LiDAR/estações), não
    valores vazios.
  - A divisão 6048 (80%) / 1513 (20%) é consistente com 7561, e as 1406
    predições do rolling forecast = 1513 − 72 − 36 + 1 (fecha a conta
    do Resultados).
- **[LACUNA]** Não sei afirmar se `dataset.csv` já é pós-imputação
  (pipeline tem estágio `impute` que preenche lacunas ≤ 12 passos = 2 h
  por reconstrução wavelet, `src/impute_wind_data.py`) ou o bruto da
  campanha. Verificar com quem exportou o CSV.
- **Ação:** declarar no texto a discrepância com número exato, ex.:
  > "A série apresenta N amostras ausentes (≈P% do intervalo) devido à
  > indisponibilidade pontual do instrumento durante a campanha;
  > lacunas curtas (≤ 2 h) foram preenchidas por imputação via
  > reconstrução wavelet." — com N confirmado (ver LACUNA acima).

## 3.5 §Preparação dos dados — linguagem/bibliotecas/versões — thread `6ace820a`

- **Onde (inferido):** §3.3 "Preparação dos dados" (sugestão: parágrafo
  próprio no fim da subseção, ou subseção "Ferramentas" antes da
  Modelagem).
- **Comentário (Vanessa):** "incluir aqui a linguagem, principais
  bibliotecas e versões. Principalmente se usou os algoritmos prontos
  da biblioteca, como o de wavelet e normalização."
- **Análise [RESPOSTA] — verificado no ambiente (`pip freeze`) e no
  código:**
  - **Python 3.10**; **TensorFlow 2.21.0** (+ Keras 3.12.1) para as
    redes; **PyWavelets 1.8.0** para a wavelet — **sim, algoritmos
    prontos**: `pywt.wavedec`/`pywt.waverec`/`pywt.threshold`
    (`src/utils.py:6`), wavelet-mãe `sym18`, nível 2, limiar **soft**
    de Donoho-Johnstone, modo `periodization`;
  - **scikit-learn 1.7.2** — normalização `MinMaxScaler` e o baseline
    Random Forest; **statsmodels 0.15.0** — ARIMA;
  - **Optuna 4.9.0** (busca de hiperparâmetros), **keras-tcn 3.5.6**
    (blocos TCN), pandas 2.3.3 / numpy 1.26.4 / matplotlib 3.10.8.
- **Ação** — parágrafo pronto para colar (ajuste se alguma versão
  mudar na rodada final):
  > "Os experimentos foram implementados em Python 3.10, utilizando
  > TensorFlow 2.21 (Keras 3) para as arquiteturas seq2seq, com o
  > pacote keras-tcn 3.5.6 para os blocos convolucionais temporais. A
  > suavização por transformada wavelet discreta utilizou a biblioteca
  > PyWavelets 1.8 (wavelet-mãe sym18, limiar soft de
  > Donoho-Johnstone), e a normalização min-máx e o baseline Random
  > Forest utilizaram scikit-learn 1.7. O ARIMA foi ajustado com
  > statsmodels 0.15, e a busca de hiperparâmetros empregou Optuna
  > 4.9."

## 3.6 §Modelagem / §Entendimento dos dados — tabelas de hiperparâmetros e do dataset — thread `6ace8ce9`

- **Onde (inferido):** §3.4 "Modelagem" (hiperparâmetros + tempo de
  treino) e §3.2 "Entendimento dos dados" (tabela descritiva do
  dataset + amostra).
- **Comentário (Vanessa, 2 mensagens):** "João, fazer aqui uma tabela
  com os testes que foram feitos e os hiperparâmetros deles. Se tiver
  também informação de tempo de treinamento, vc inclui aqui." /
  "Também é importante uma tabela com o dataset completo. O atributo,
  se coletado ou computado e seu significado. Faltou aqui você mostrar
  pro leitor o que de fato entreu nos modelos..."
- **Análise:** ferramenta de geração de tabelas por modelo já existe no
  `tests/relatorio_resultados.py` ("Tabela MAE/MSE/R2/loss por modelo +
  baselines"), e os dados vivem em:
  - `data/results/report_8020/evaluation.json` — por modelo: `params`
    (nº de parâmetros), `loss`, `size_mb`, tempos (`timing`);
  - hiperparâmetros ótimos: `data/results/optuna/<modelo>/trials.csv`
    (best_params em `data/results/wfo_optuna/<modelo>/metrics.json`);
  - tempos de treino: ex. `data/results/lstm_bihp_8020/summary.json`
    (`training_time_sec`), `training_history.json` nos relatórios;
  - atributos do dataset: colunas de `data/dataset.csv` — coletados:
    `ws{40..260}` (20 níveis), `dir{...}`, `v{...}` (componente
    vertical), `press`, `humid`, `temp`; computados: `cis1..cis19`
    (cisalhamento), `disp{...}`/`vdisp{...}` (dispersão) — confirmar
    o dicionário exato com quem gerou o CSV.
  - Além dos 4 modelos do artigo, existem ablações registradas
    (`context_mode_ablation.md`, `gate_test_8020`,
    `report_8020_mseloss`, `report_duel_loss`) — a tabela "testes
    feitos" pode listar só os 4 canônicos e citar as ablações em
    texto.
- **Ação:** montar as duas tabelas (posso gerar a partir dos JSONs
  acima — pedir quando for fazer).

## 3.7 §Modelagem — "tá correto?" — thread `6ace8d78`

- **Onde (inferido):** par. da função de perda — "...uma função de
  perda específica foi desenvolvida para os modelos LSTM. A função de
  perda desenvolvida executa uma média ponderada sobre o erro
  quadrático, com pesos (entre 0.7 e 1.3) proporcionais ao horizonte.
  Para os modelos TCN, a função de perda é o erro quadrático médio
  (MSE)." (alvo alternativo: a expressão do campo receptivo da TCN no
  par. anterior).
- **Comentário (Vanessa):** "tá correto?"
- **Análise [RESPOSTA] — verificado no código e nos artefatos:**
  - Os **pesos 0.7 → 1.3 estão corretos**: rampa linear com o horizonte
    normalizada pela média (`src/models/losses.py:15-25`,
    `horizon_weighted_mse`).
  - **MAS** a divisão "LSTM ponderada / TCN com MSE" **não bate com
    nenhuma rodada 80/20 local**:
    - `data/results/report_8020/evaluation.json`: os **4 modelos com
      `horizon_weighted_mse`**;
    - `data/results/report_8020_mseloss/evaluation.json`: os **4 com
      `mse`** (ablação).
  - O script de treino lê o loss dos hiperparâmetros
    (`scripts/lidar_7525_train.py:122`: `wrapper.loss =
    hyperparameters.get("loss", "mse")`).
- **Ação [CRÍTICO]:** corrigir o parágrafo conforme a rodada canônica
  escolhida para o artigo (ver §4.2 abaixo — a escolha da rodada é o
  mesmo dilema das tabelas). Se a canônica for a `report_8020`, o
  texto vira: "...a função de perda utilizada em todos os modelos
  executa uma média ponderada sobre o erro quadrático, com pesos
  (entre 0,7 e 1,3) proporcionais ao horizonte". Se for a `mseloss`:
  "...os modelos foram treinados com o erro quadrático médio (MSE);
  uma variante ponderada pelo horizonte foi avaliada em ablação".

## 3.8 Terminologia — "Melhor chamar de arquitetura" — thread `6ace2e9e`

- **Onde (inferido):** sem âncora no export; candidatos: §3.4
  "Modelagem" ("quatro variações de modelos foram implementados") /
  Resumo ("redes neurais TCN e LSTM") / Resultados par. 3.
- **Comentário (Flávio):** "Melhor chamar de arquitetura."
- **Ação [DECISÃO DO AUTOR]:** varrer o texto trocando "redes" /
  "modelos" por "arquiteturas" nos pontos que se referem à
  combinação estrutura+rede (ex.: "quatro arquiteturas foram
  implementadas", "as arquiteturas TCN e LSTM"). Manter "rede neural"
  quando o referente for a célula/mecanismo interno (LSTM como RNN
  etc.).

---

# 4. Resultados e Discussão (`04_Resultados.tex`)

## 4.1 Teste estatístico Diebold–Mariano + bootstrap em blocos — thread `6ace9381`

- **Onde:** §Avaliação (métodos) e §Resultados (aplicação).
- **Comentário (Vanessa):** "João, eu pedi a IA para me sugerir algum
  teste estatístico... teste de Diebold–Mariano (DM), complementado por
  bootstrap em blocos..."
- **Análise [RESPOSTA]:** **já está implementado no repositório** —
  `tests/relatorio_resultados.py` (`diebold_mariano_loss_diff`,
  ~linha 898; comparação em pares com α, agregada e por horizonte) e
  resultado salvo em `data/results/report_8020/mcs.json`. Falta
  **usar** no artigo: parágrafo no §Avaliação descrevendo o teste +
  tabela/valores-p no §Resultados.
- **Ação [GUIA]:**
  1. Rodar (sem `--no-baselines` para incluir ARIMA/RF/Persistência):
     `python tests/relatorio_resultados.py --models-dir
     pipeline/tmp/trained_8020 --out-dir data/results/report_8020`
  2. Ler os pares/p-valores em `data/results/report_8020/mcs.json` e
     reportar no artigo (ex.: "diferença significativa a 5% entre X e
     Y pelo teste DM com variância HAC").
  3. Estudo (conceitos):
     - **DM test** (Diebold & Mariano, 1995): diferença de perdas
       `d_t = L(e_A,t) − L(e_B,t)` por origem do rolling forecast;
       H₀: E[d_t]=0 (perda esperada igual); estatística sobre a média
       de d_t com variância estimada por HAC/Newey–West (corrige
       autocorrelação).
     - **Bootstrap em blocos** (moving-block): reamostrar blocos
       contíguos de d_t preservando a dependência temporal para
       construir o intervalo de confiança / p-valor — necessário pois
       as perdas do rolling forecast são serialmente correlacionadas.
     - Implementações: o próprio `relatorio_resultados.py` (recomendado,
       já acoplado aos dados); referências: Diebold & Mariano (1995,
       *JBES*); Harvey, Leybourne & Newbold (1997) para a correção de
       amostras pequenas; Politis & White (2009) para largura de bloco
       automática.

## 4.2 Inconsistência R² × MSE (variância implícita ~2,70 vs ~2,78) — thread `6ace944b`

- **Onde:** Tabela `tab:error_all` (Tabela 1 do artigo).
- **Comentário (Vanessa):** "conferir se isso é verdade... inconsistência
  sutil entre R² e MSE (variância implícita ~2,70 vs ~2,78)... verificar
  agregação."
- **Análise [RESPOSTA] — confirmada e explicada com os artefatos
  locais:**
  - Pela eq. (1) do artigo, R² = 1 − MSE/Var(y) → **Var(y) implícita =
    MSE/(1−R²)**:
    - Baselines da tabela (ARIMA 0,508/1,329; RF 0,418/1,572;
      Persistência 0,388/1,653): Var implícita **≈ 2,702** — batem
      **exatamente** com `data/results/baselines/arima_rf/metrics.json`
      (config: `train_ratio 0.8`, `val_end 6048`, **n_origins 1478**,
      **samples 53208** = 1478 × 36);
    - Modelos DL da tabela (TCN-Bi 0,502/1,384; TCN 0,486/1,428;
      LSTM-Bi 0,460/1,501; LSTM 0,406/1,651): Var implícita **≈
      2,779**.
  - **Causa estrutural:** conjuntos de origens diferentes. O rolling
    forecast dos DL consome a janela de entrada (72) + saída (36) →
    **1406 origens** (1513−108+1) = 50 616 amostras; os baselines
    avaliam **1478 origens** (1513−36+1) = 53 208 amostras. Janelas de
    valores reais diferentes → Var(y) diferente → **os R² das duas
    metades da tabela não são diretamente comparáveis** (além de
    possível diferença de agregação).
  - **Agravante [CRÍTICO]:** **nenhum artefato local reproduz
    exatamente a tabela DL do artigo.** O mais próximo é a ablação com
    MSE (`report_8020_mseloss`: TCN 0,490/1,378; TCN-Bi 0,489/1,382 —
    var implícita 2,703, consistente com os baselines!); os valores de
    LSTM (0,406) e LSTM-Bi (0,460) do artigo **não batem com nenhuma
    rodada encontrada** (mseloss: LSTM 0,053 / LSTM-Bi 0,330;
    report_8020 ponderada: LSTM 0,279 / LSTM-Bi 0,225; lstm_bihp:
    0,403).
  - **[CRÍTICO] Figura × tabela:** `fig_mse_heatmap.png` veio da rodada
    **ponderada** (`scripts/articles_sync.py:13` →
    `data/results/report_8020/images/06_heatmap_mse_horizonte.png`),
    cuja ordenação (TCN 0,514 melhor DL; LSTM-Bi pior) **contradiz a
    Tabela 1** (TCN-Bi melhor DL). Figura e tabela no PDF atual vêm de
    **rodadas de treino diferentes** — revisor externo pega.
- **Ação:** escolher **uma** rodada canônica e recalcular **tudo** com
  um único script (`tests/relatorio_resultados.py` já calcula DL +
  baselines nas mesmas origens e gera exatamente as duas tabelas do
  artigo — agregado e h=36 — além do DM): atualizar Tabela 1, Tabela 2
  (h=36), heatmap, claims do texto e Resumo (R² 0,502/0,416). Se for
  preciso comparar com baselines nas mesmas origens, restringir os
  baselines a origens ≥ 72.

## 4.3 Fonte da figura pequena — thread `6ace992b`

- **Onde (inferido):** provável `fig:mse_heatmap` (muitas células +
  rótulos; também vale checar `fig_model_struct`).
- **Comentário (Vanessa):** "Consegue aumentar a fonte dessa figura? Tá
  muito pequeno. Difícil de ler."
- **Ação:** regenerar o heatmap com fontes maiores — no gerador
  (`tests/relatorio_resultados.py`, imagem
  `06_heatmap_mse_horizonte.png`): aumentar `fontsize` dos ticks/labels
  (~13–16) ou reduzir o `figsize` para que a fonte cresça
  relativa à figura; depois re-sincronizar
  (`python scripts/articles_sync.py sync`) e recompilar no Overleaf.

## 4.4 "precisa validar isso estatisticamente" + "vantagem de R² de 0,8" — thread `6ace9991`

- **Onde:** par. 3 — "Nos testes realizados, as TCNs apresentaram
  superioridade às LSTMs, com uma vantagem de R² de **0,8** entre as
  unidirecionais, e 0,4 nas bidirecionais."
- **Comentário (Vanessa):** "João precisa validar isso estatisticamente."
- **Análise [RESPOSTA]:**
  - Números: pela Tabela 1, TCN − LSTM = 0,486 − 0,406 = **0,080** e
    TCN-Bi − LSTM-Bi = 0,502 − 0,460 = **0,042** → "0,8" e "0,4" não
    batem; provável intenção "0,08" e "0,04" (ou em pontos
    percentuais). **Verificar/corrigir.**
  - Validação: é o mesmo pedido do teste DM (§4.1) — o par
    TCN×LSTM (e TCN-Bi×LSTM-Bi) precisa de p-valor para a alegação de
    superioridade.
  - Nota: na rodada `report_8020` (ponderada) a ordem também muda
    (TCN 0,514 vs TCN-Bi 0,403; LSTM 0,279 vs LSTM-Bi 0,225) — outro
    motivo para fixar a rodada canônica antes de reescrever o par.
- **Ação:** corrigir os números conforme a tabela final + inserir os
  p-valores DM.

---

# 5. Conclusões (`06_Conclusoes.tex`)

## 5.1 Par. 1 — recapitular o objetivo — thread `6ace2e79`

- **Comentário (Flávio):** "Acho que aqui é melhor recapitular qual era
  o objetivo do trabalho e apresentar as conclusões obtidas (o que vem
  na sequência)."
- **Análise:** o par. 1 atual ("O objetivo do trabalho foi concluído
  com êxito, contrastando...") afirma que o objetivo foi cumprido sem
  dizer qual era. O formato pedido: objetivo → método (1 frase) →
  conclusões.
- **Ação [DECISÃO DO AUTOR]** — esqueleto:
  > "Este trabalho comparou arquiteturas seq2seq com atenção (TCN e
  > LSTM, com e sem bidirecionalidade no codificador) para a predição
  > da velocidade do vento a 100 m a partir de dados LiDAR coletados no
  > litoral maranhense, com horizonte de 6 h e entrada de 12 h. Os
  > resultados mostram que [conclusão 1 — melhor arquitetura + R²] ...
  > [conclusão 2 — efeito da bidirecionalidade] ..."
  (as sugestões pontuais já registradas — "concluído→alcançado",
  "contrastando o funcionamento→comparando o desempenho" — ficam
  absorvidas se esta reescrita for adotada).

## 5.2 Par. 2 — "variação por área de estudo" — thread `6ace2f14`

- **Comentário (Flávio):** "Isso aqui me deixou confuso. Lá no começo
  do texto é dito que os dados vém da fonte P1. Porque haveria variação,
  neste trabalho, de resultados em diferentes áreas de estudo?"
- **Análise [RESPOSTA]:** Flávio tem razão — todo o dado é do sítio P1;
  a frase "os resultados podem variar de acordo com a área de estudo, e
  período medido" sugere que múltiplas áreas foram testadas (não
  foram). O que se quer dizer é: como só há um sítio/período, a
  generalização para outros contextos é hipótese para trabalho futuro,
  não resultado observado.
- **Ação [DECISÃO DO AUTOR]** — reescrita pronta:
  > "Contudo, este resultado não elimina as variantes LSTM: como os
  > dados são oriundos de um único sítio e período, a superioridade
  > observada não garante generalização para outras áreas e campanhas
  > de medição, o que motiva avaliações em bases adicionais."

## 5.3 Par. 3 — validação cruzada — thread `6ace2fa9`

- **Comentário (Flávio):** "Não é que não foi possível, é que o dado
  exige uma estratégia diferente para que seja possível realizar esse
  tipo de validação. A não ser que você esteja associando validação
  cruzada a outro conceito..." (mensagem truncada no export).
- **Análise [RESPOSTA]:** Flávio está correto. Validação cruzada
  padrão (k-fold com embaralhamento) viola a ordem temporal; mas existe
  **validação cruzada com estrutura temporal** — *time series split /
  blocked CV / rolling-origin* (Roberts et al. 2016 — chave
  `crossvalidation`, **já no bib**). Com ~2 meses de dados, é viável
  p.ex. treinar em blocos anteriores e testar em blocos posteriores
  disjuntos. Ou seja: não é "não foi possível", é "exige estratégia
  própria para dados dependentes no tempo" — que aliás já é o
  procedimento de treino/teste 80/20 adotado (uma única dobra temporal);
  o que falta é repeti-la em múltiplas dobras.
- **Ação [DECISÃO DO AUTOR]** — reescrita pronta:
  > "A principal limitação do trabalho executado foi o conjunto de
  > dados, limitado em tempo. Embora o conjunto se demonstrasse capaz
  > de treinar os modelos, a avaliação restringiu-se a uma única
  > dobra temporal (80%/20%); a validação cruzada com estrutura
  > temporal \cite{crossvalidation} — múltiplas dobras treino/teste
  > respeitando a ordem cronológica — permanece como passo necessário
  > para fortalecer a confiabilidade dos resultados."
  (a sugestão "depreciando→comprometendo" fica absorvida se esta
  reescrita for adotada; e o par. 4 — "A validação cruzada também pode
  ser efetuada..." — já cobre a proposta futura.)

## 5.4 Typos do artigo (par. 2–4) — **[CORRIGIDO]**

Corrigidos nos `.tex` locais (não commitados):

| Antes | Depois |
|---|---|
| "**Contanto**, este resultado..." | "**Contudo**, este resultado..." |
| "A principal **limtação**" | "A principal **limitação**" |
| "Embora o conjunto **se demonstrou** capaz" | "se **demonstrasse** capaz" (concordância modal com "embora") |
| "redes diferentes no **codificador e codificador**" | "no **codificador e no decodificador**" |
| `03:` "Os dados coletados **tem**" | "**têm**" |
| `03:` "**foi encontrada** padrões diários no vento" | "**foram encontrados** padrões diários no vento" |
| `03:` "O ARIMA é um modelo estatístico ... **implementados** através de componentes" | "**implementado**" (sujeito singular) |

Pendências de estilo do mesmo parágrafo (subjetivas, do doc anterior
`docs/comentarios_vanessa.md`): "concluído→alcançado",
"contrastando o funcionamento→comparando o desempenho",
"depreciando→comprometendo", concordância "a TCN ... mostrou/mostraram"
(ambas defensáveis — "a TCN, tanto unidirecional quanto bidirecional"
é apositivo singular).

## 5.5 Par. 4 — referências de GRU e Transformers (herdado)

- Do doc anterior: citar `vaswani2017attention` ao mencionar
  Transformers (**já no bib**) e conferir a entrada da GRU (Cho et al.
  2014, EMNLP — "**dados não conferidos no bib local; verificar antes
  de adicionar**", usar a skill `bibtex-articles`).

---

# Apêndice A — Correções já aplicadas nos arquivos

- `06_Conclusoes.tex` — 4 typos (tabela §5.4).
- `03_Material_e_metodos.tex` — 3 typos (tabela §5.4).
- `00_Template_SBSR.tex` — `\usepackage{placeins}` + `\FloatBarrier`
  antes das referências (§1.2).
- **Sessão 20/10/2026:**
  - `02_Introducao.tex` — frase ANEEL reescrita (SIGA, 35 GW, 2ª fonte,
    15,8%; cite `aneelSiga`) + blocos de comentários dos threads
    `6ace4915` (atendido) e `6ace2c7f`;
  - `07_SBSR.bib` — entrada `@misc{aneelSiga}` criada (SIGA/ANEEL, v1.1,
    snapshot 20-10-2026);
  - `03_Material_e_metodos.tex`, `04_Resultados.tex`, `06_Conclusoes.tex`
    — comentários da rodada Overleaf inseridos como blocos
    `% COMENTÁRIOS DE REVISÃO` junto às localizações inferidas
    (`[V-…]/[F-…]` + ID do thread + status);
  - consolidação dos testes estatísticos da coleção em
    `docs/revisions/sbr/kbscout/result-5.md` (Theil U, DM, GW, Friedman);
  - relatório da sessão: `reports/protocolo/sessao-comentarios-sbsr-1/`
    (favorito no hub).
- **Não commitado/pushado** — o clone `articles/sbr` está com os
  arquivos modificados; sincronizar via `scripts/articles_sync.py` /
  fluxo Overleaf quando aprovado. Compilação local falha por falta de
  `babel-spanish` no TeX da máquina (pré-existente; Overleaf compila).

# Apêndice B — Fontes de dados para as pendências numéricas

| Pendência | Fonte |
|---|---|
| Tabelas de métricas (agregado e h=36) consistentes | `tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020 --out-dir data/results/report_8020` (+baselines sem `--no-baselines`) |
| Hiperparâmetros por modelo | `data/results/optuna/<modelo>/trials.csv`; `evaluation.json` (params/loss/size) |
| Tempo de treino | `data/results/*/summary.json` (`training_time_sec`), `training_history.json` |
| Teste DM / pares | `data/results/report_8020/mcs.json` |
| Dataset (contagem/atributos) | `data/dataset.csv` (7561 linhas, 16/09→08/11/2021); imputação: `src/impute_wind_data.py` |
| Figuras do artigo | `scripts/articles_sync.py` (mapa fig_mse_heatmap → `data/results/report_8020/images/`) |

**Nota sobre rodadas:** `report_8020` = todos os 4 modelos com
`horizon_weighted_mse` (TCN 0,514 / TCN-Bi 0,403 / LSTM 0,279 /
LSTM-Bi 0,225); `report_8020_mseloss` = todos com `mse` (TCN 0,490 /
TCN-Bi 0,489 / LSTM 0,053 / LSTM-Bi 0,330). A Tabela 1 do artigo não
reproduz nenhuma das duas exatamente — escolher a canônica e recalcular.

# Apêndice C — Guia de estudo: DM + bootstrap em blocos

1. **Conceito:** comparação par a par de modelos por origem do rolling
   forecast com `d_t = L_A,t − L_B,t` (perda p.ex. MSE). H₀: E[d_t]=0.
   A estatística DM divide a média de d_t pelo desvio-padrão de d_t
   estimado com autocovariâncias (HAC/Newey–West) porque d_t é
   autocorrelacionado. Correção de amostras pequenas: Harvey–Leybourne–
   Newbold (1997).
2. **Bootstrap em blocos (moving-block):** em vez de assumir
   normalidade, reamostrar blocos contíguos de d_t (comprimento de
   bloco ≈ raiz cúbica de N, ou Politis–White) para construir a
   distribuição nula da média e o p-valor.
3. **Prática neste repo:** tudo já implementado em
   `tests/relatorio_resultados.py` (`diebold_mariano_loss_diff` —
   agregado e por horizonte, com α configurável); saída em `mcs.json`.
   Para o artigo: 1 parágrafo no §Avaliação (definição + por que blocos)
   + tabela de p-valores dos pares relevantes (TCN×LSTM, TCN-Bi×LSTM-Bi,
   melhor DL×ARIMA, etc.).
4. **Referências:** Diebold & Mariano (1995), *J. Business & Economic
   Statistics* 13(3); Harvey, Leybourne & Newbold (1997), *Int. J.
   Forecasting* 13(2); Roberts et al. 2016 (`crossvalidation`, no bib)
   para o contexto temporal.

# Apêndice D — Pendências herdadas de `docs/comentarios_vanessa.md` (rodada commit `151ac29`)

Ainda abertas (além das cobertas acima):

- [ ] Adicionar citações dos baselines no bib: ARIMA (Box & Jenkins,
      *Time Series Analysis: Forecasting and Control*), Random Forest
      (Breiman, 2001, *Machine Learning* 45(1):5–32), Persistência
      (Hyndman & Athanasopoulos, *Forecasting: Principles and
      Practice*, OTexts) — criar entradas com dados verificados
      (skill `bibtex-articles`).
- [ ] Remover os blocos `% COMENTÁRIOS DE REVISÃO` dos `.tex` antes de
      submeter.
- [ ] Conferir entrada GRU (Cho et al. 2014) antes de adicionar (§5.5).
