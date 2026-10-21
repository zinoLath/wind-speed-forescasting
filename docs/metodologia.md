# Metodologia

## 1. Modelagem do problema

O problema central é a **previsão multi-horizonte da velocidade do vento à altura de 100 m (ws100)**, medida em passos de 10 minutos. A tarefa é formulada como um problema de **mapeamento seq2seq (sequência-para-sequência)**: dado um histórico de **72 amostras (12 horas)** de variáveis meteorológicas multivariadas, o modelo deve produzir, em uma única passada (decodificação *direct multi-step*, não recursiva), a trajetória completa de **36 amostras (6 horas)** futuras da variável-alvo.

A modelagem adota três decisões de projeto que definem o comportamento operacional dos modelos:

1. **Convenção de decodificação dual (treino vs. inferência).** Durante o treinamento, o decoder opera em modo *teacher forcing*: recebe como entrada o último valor observado do alvo seguido dos valores reais defasados (`X_decoder[:, 0] = último observado; X_decoder[:, 1:] = y[t+1 ... t+35]`). Em inferência — e, crucialmente, também na **validação** — o decoder nunca vê valores futuros: sua entrada é apenas a repetição do último valor observado do alvo (nos modelos recorrentes) acrescida de um termo de posição normalizado (nos modelos com decoder *direct*). Essa reescrita explícita dos insumos do decoder no conjunto de validação garante que o `val_loss`, o *early stopping* e a função-objetivo do ajuste de hiperparâmetros meçam **o comportamento de deployed**, e não o comportamento artificialmente favorecido do *teacher forcing*.

2. **Alvo suavizado, avaliação honesta.** O modelo é treinado para prever a série **denoizada por wavelet** (`ws100_wavelet`), que remove ruído de alta frequência do sensor. Contudo, as métricas principais (*headline*) são sempre calculadas contra o **ws100 bruto**: as versões contra a série denoizada são mantidas apenas como colunas secundárias (`*_denoised`), para que a suavização não influe artificialmente os resultados reportados.

3. **Partição temporal estrita.** O dataset é dividido de forma cronológica em 75% treino, 20% validação e 5% teste, sem qualquer embaralhamento, eliminando vazamento temporal. Os escalonadores `MinMaxScaler` (um para o alvo, outro para as covariáveis) são ajustados **exclusivamente no treino** e reutilizados em validação, teste e inferência. O baseline de referência é a **persistência** (prever o último valor observado para todos os horizontes).

Cada janela de dados é transformada em tripletos `(X_encoder, X_decoder, y_decoder)` via janelas deslizantes vetorizadas: o encoder recebe as 72 amostras de todas as features; o decoder, seus insumos; e `y_decoder`, os 36 valores futuros do alvo. Há ainda suporte a um modo de alvo **residual** (`target_mode="residual"`), em que a rede prevê o desvio em relação à persistência, somada de volta na inversão de escala.

## 2. Pré-processamento

**Denoising wavelet.** A série do alvo é suavizada por decomposição wavelet (`pywt.wavedec`) com a wavelet **Symlet 18 (`sym18`)**, modo `periodization`, **nível 2**. Os coeficientes de detalhe são limiarizados por **soft-thresholding** com limiar universal de VisuShrink: `σ` estimado pela mediana absoluta dos coeficientes (`MAD/0.6745`) e limiar `σ·√(2·ln N)`. A coluna `ws100_wavelet` é gerada e somada às features, preservando a série bruta.

**Imputação (série longa).** Para o protocolo walk-forward, a série longa com lacunas é reconstruída por imputação iterativa (`IterativeImputer`, `max_iter=10`) com estimadores **Random Forest**, **LightGBM** (ambos ajustados em amostra de 3.000 linhas), **KNN** e **Kalman**. Colunas de direção (0–360°) são convertidas para componentes **seno/cosseno** antes da imputação e convertidas de volta ao final, respeitando a circularidade angular. A qualidade é validada pela **injeção de lacunas artificiais** (comprimentos de 1 a 36 amostras, 30 repetições por comprimento, semente fixa) num segmento completo do dado, medindo MAE, RMSE, viés e R² por coluna e por comprimento de lacuna. O dataset imputado carrega uma coluna flag `imputed`, que permite reportar métricas apenas sobre amostras realmente observadas.

## 3. Arquiteturas dos modelos

Todos os modelos herdam de uma classe-base comum (`Seq2SeqWrapper`), que padroniza preparação de dados, construção, treinamento, predição pontual e a previsão em lote de todos os horizontes — garantindo comparabilidade justa entre arquiteturas. Seis arquiteturas foram implementadas, todas encerrando em uma cabeça `TimeDistributed(Dense(1, linear))` de regressão e treináveis com perda configurável (MSE por padrão):

- **Seq2Seq-LSTM + atenção:** encoder LSTM (`return_sequences=True`, `return_state=True`) cujos estados inicializam um decoder LSTM; sobre as saídas do decoder aplica-se **atenção aditiva (estilo Luong)** sobre as saídas do encoder, o contexto é concatenado às saídas do decoder e projetado para o valor previsto. Dropout independente em encoder e decoder.

- **Seq2Seq-LSTM bidirecional:** encoder BiLSTM com estados *forward/backward* **concatenados** para inicializar um decoder LSTM com o dobro de unidades, seguido do mesmo bloco de atenção.

- **Seq2Seq-TCN:** encoder composto por blocos convolucionais causais dilatados (biblioteca `tcn`, com LayerNorm), cujo resumo é obtido por **média temporal (mean-pooling)**, repetido ao longo dos 36 passos e concatenado às entradas do decoder TCN; as saídas do decoder são projetadas linearmente para a dimensão do encoder antes da atenção, garantindo compatibilidade do produto interno.

- **Seq2Seq-TCN bidirecional:** variação com encoder TCN bidirecional (dimensão dobrada).

- **Híbrido TCN-LSTM:** encoder TCN; o **último passo temporal** é projetado por duas camadas `Dense(tanh)` nos estados `h`/`c` do decoder LSTM, seguido do bloco de atenção.

- **Transformer Pre-LN:** projeção linear para `d_model`, **positional encoding aprendido** (embedding treinável por posição) e blocos *pre-LayerNorm* — no encoder, auto-atenção multi-cabeça + FFN; no decoder, auto-atenção **com máscara causal**, **cross-attention** sobre o encoder e FFN, todos com conexões residuais e normalização final. Opcionalmente usa **AdamW** (com *weight decay*) e um agendador de taxa de aprendizado com **aquecimento linear de 10% dos passos seguido de decaimento cossenoidal**. A variante Post-LN com codificação posicional senoidal foi mantida como referência comparativa.

## 4. Ajuste de hiperparâmetros

A busca de hiperparâmetros é feita com **Optuna**, um estudo por modelo, com **100 ensaios** e no máximo 60 épocas por ensaio. Os elementos metodológicos relevantes são:

- **Amostrador TPE multivariado com agrupamento** (`TPESampler(multivariate=True, group=True)`), 20 ensaios iniciais aleatórios (metade do orçamento, quando menor) e 64 candidatos por iteração de esperança de melhoria;
- **Poda antecipada** com `MedianPruner` (10 ensaios de aquecimento; poda possível a partir da época 8): um callback do Keras reporta o `val_loss` de cada época ao Optuna e interrompe ensaios mediano-inferiores;
- **Função-objetivo: RMSE em validação sob a convenção de inferência.** Ao final de cada ensaio, o modelo prediz todo o conjunto de validação com o decoder em modo inferência, as previsões são inverse-transformadas e o RMSE (não o `val_loss` *teacher-forced*, que é um proxy fraco) orienta a seleção; o RMSE é preferido ao MAE para penalizar mais fortemente erros grandes;
- **Espaços de busca unificados:** taxa de aprendizado log-uniforme em [10⁻⁵, 10⁻²] (LSTMs) ou [10⁻⁴, 10⁻²] (demais); unidades LSTM 64–512 (passo 32); dropouts 0–0,5 (passo 0,05); para os TCNs, espaço compartilhado por encoder/decoder — filtros 32–256 (passo 16), kernel 2–3, 1–2 stacks, dropout 0–0,5 e taxa de dilatação 1–5 (a lista de dilatações é derivada como potências de 2); para o Transformer — `d_model` 32–256 (passo 16), cabeças 1–8, camadas 1–4, `ff_dim` 64–512, dropout 0–0,3;
- **Reprodutibilidade e eficiência:** semente global fixa (42), `K.clear_session()` entre ensaios, estudo persistido em SQLite (retomável com `load_if_exists=True`), e preparação de dados (denoising, escala, janelas) executada **uma única vez** e reutilizada por todos os ensaios — apenas a rede é reconstruída a cada ensaio. O mesmo contrato de construção (`hp.Float/Int/Choice`) é abstraído por adaptadores (`OptunaHyperParameters` para busca; `FixedHyperParameters` para reproduzir o melhor ensaio), desacoplando arquitetura do framework de busca.

## 5. Treinamento dos modelos

O treinamento final usa os **melhores hiperparâmetros do Optuna** e segue um protocolo único para todas as arquiteturas:

- **Otimizador Adam** (ou AdamW/schedule no Transformer), perda **MSE** com MAE monitorada como métrica auxiliar; lote de 32; até 100 épocas;
- **Callbacks:** `EarlyStopping` monitorando `val_loss` (paciência 8, `restore_best_weights=True`) e `ReduceLROnPlateau` (fator 0,5, paciência 4, mínimo 10⁻⁶). Como a validação usa o decoder em convenção de inferência, o ponto de parada corresponde ao melhor desempenho de previsão real, não ao melhor ajuste *teacher-forced*;
- **Rastreabilidade:** o modelo treinado é serializado (`.keras`) junto de um `model.json` contendo classe, hiperparâmetros e sua origem (Optuna/config/extrapolados), modo do decoder, modo do alvo, perda, histórico de épocas e dimensões das partições — permitindo recarregar e reavaliar o modelo bit a bit;
- **Retreino periódico (walk-forward):** reconhecendo o *domain shift* entre períodos/instrumentos (LIDAR vs. SODAR), a avaliação operacional simula janelas deslizantes de **60 dias de treino / 30 dias de teste com avanço de 30 dias**, retreinando o modelo do zero em cada janela (partição interna treino/validação de 80/20, mesmos callbacks e hiperparâmetros da busca), o que mensura a degradação e a recuperação do modelo sob não-estacionariedade.

## 6. Protocolo de avaliação

A avaliação é dupla: (i) **todas as origens** do conjunto de teste geram uma previsão de 36 passos (`predict_all_horizons`), produzindo métricas agregadas e **por horizonte** (1–36), o que expõe a degradação com o horizonte; e (ii) **previsão contínua no horizonte máximo** (*rolling forecast*), em que cada janela é função apenas do histórico conhecido e executada em uma única chamada batched — sem vazamento de valores futuros em nenhum ponto. São reportados MAE, MSE, RMSE, NMSE, NRMSE, NMAE e R², sempre contra o ws100 bruto e com a persistência como linha de base; nos experimentos imputados, as métricas são decompostas em amostras observadas versus imputadas.
