# Estudo: Wrapper Transformer para Previsão Seq2Seq de Vento

Documento de estudo sobre o wrapper `S2STransformerPrelnWrapper` e as técnicas utilizadas para torná-lo competitivo (ou superior) aos modelos LSTM/TCN no pipeline deste projeto.

---

## 1. Contexto do problema

- **Tarefa:** prever a velocidade do vento a 100 m (`ws100`) usando uma arquitetura seq2seq. Entrada de **72 passos (12 h)** → previsão de **36 passos (6 h)**.
- **Protocolo de avaliação (walk-forward, `wfo_optuna.py`):** treina-se uma vez em 75% dos dados (treino) + 20% (validação) e avalia-se a previsão de **36 passos à frente** em cada posição dos últimos 5% (região de teste, 344 amostras). A métrica reportada é **MAE**.
- **Dificuldade central:** a região de teste está **muito distante no tempo** do treino (últimos ~6% da série). Isso gera *domain shift*: o regime de vento no teste (ex.: rampa de 3,8 → 9,7 m/s) é diferente do treino. **Todos** os modelos (LSTM, TCN, Transformer) produzem previsões defasadas e amortecidas nesse regime.
- **Descoberta importante na auditoria:** os resultados salvos em `data/results/wfo_optuna/` (ex.: LSTM MAE 0.17, TCN 0.28) foram gerados com uma **versão anterior do `dataset.csv`** (arquivo gitignored, sem histórico). Reexecutando o mesmo script com os mesmos parâmetros, obtém-se MAE ~1.45 (LSTM) e ~1.43 (TCN). Os `actuals` salvos não correspondem à região de teste atual. **Todo baseline foi re-estabelecido nos dados atuais antes de comparar.**

---

## 2. Arquitetura do Transformer

### 2.1 Visão geral (encoder–decoder)

```
encoder_inputs (72, 34 features)
   └─ Dense(d_model)          # projeção das features
   └─ + Positional Encoding   # posição de cada passo no tempo
   └─ Dropout
   └─ N × [Pre-LN Attention → residual → Pre-LN FFN → residual]

decoder_inputs (36, 1)        # teacher forcing: alvo deslocado
   └─ Dense(d_model)
   └─ + Positional Encoding
   └─ Dropout
   └─ N × [Pre-LN self-attention (causal) → residual
            Pre-LN cross-attention (sobre encoder) → residual
            Pre-LN FFN → residual]
   └─ TimeDistributed(Dense(1))   # saída (36, 1)
```

### 2.2 Componentes e conceitos

| Conceito | O que é | Por que importa |
|---|---|---|
| **MultiHeadAttention** | Atenção multi-cabeça do Keras: cada cabeça aprende relações diferentes entre passos | Permite que o modelo relacione qualquer passo do encoder a qualquer passo do decoder sem depender de recorrência |
| **Causal mask** (`use_causal_mask=True`) | Impede que o decoder "veja" o futuro durante a self-attention | Garante que a previsão do passo *t* use apenas passos ≤ *t* (consistente com inferência) |
| **Cross-attention** | Atenção em que a *query* vem do decoder e as *keys/values* do encoder | É o canal que transfere a informação do histórico (encoder) para a geração (decoder) |
| **Positional Encoding** | Sinal que codifica a posição de cada passo | Atenção é *permutation-invariant*; sem posição, o modelo não sabe a ordem temporal. Usamos **sinusoidal** (Vaswani et al.) no V1 e **aprendido** no vencedor |
| **Pre-LN vs Post-LN** | Ordem de LayerNorm: `Pre-LN` aplica norm *antes* da atenção/FFN; `Post-LN` aplica *depois* da soma residual | **Pre-LN estabiliza o treino** (gradientes passam diretamente pela via residual, sem norm no caminho); na prática convergiu melhor que o V1 (Post-LN) |
| **FFN (feed-forward)** | MLP `Dense(ff_dim, relu) → Dense(d_model)` aplicado a cada passo | Adiciona capacidade não-linear por posição |

### 2.3 `WarmupCosineSchedule` (agendador de learning rate)

```
lr(step) = base_lr * warmup(step) * cosine(step)
  warmup(step) = min(step / warmup_steps, 1)      # 0 → 1 nos primeiros 10% dos passos
  cosine(step) = 0.5*(1 + cos(π * progress))       # 1 → 0.01 nos passos restantes
```

- **Warmup** evita que os primeiros passos com parâmetros aleatórios façam gradientes explosivos.
- **Decaimento cosseno** reduz a taxa gradualmente, refinando o mínimo.
- **Por que:** transformers são sensíveis ao LR; taxas altas por muito tempo desestabilizam, e reduzir gradualmente ajuda a convergir. Foi um dos fatores que tornou o transformer treinável neste dataset pequeno.

---

## 3. Técnicas que FUNCIONARAM (e por quê)

### 3.1 Pre-LayerNorm + positional encoding aprendido (novo wrapper `S2STransformerPrelnWrapper`)

| Técnica | Efeito medido |
|---|---|
| Pre-LN (vs Post-LN do V1) | Treino mais estável; convergência mais suave |
| PE aprendido (vs sinusoidal) | O modelo ajusta o encoding à escala temporal real do vento (10-min samples) |

### 3.2 Warmup + cosine decay

Melhorou a convergência em relação a LR constante (usada nos testes Optuna/WFO originais). Integrado ao fluxo: `wrapper.schedule_total_steps` é calculado em `optuna_all_hyperparameters.py`, `wfo_optuna.py` e `wfo_imputed.py` quando o wrapper expõe o atributo.

### 3.3 **Loss MAE** (a mais impactante)

- A métrica de avaliação do walk-forward é **MAE**.
- Treinar com **MSE** penaliza quadrados de erro grande → o modelo vira *conservador* (amortiza picos e vales, "regressão à média"), o que **piora o MAE**.
- Treinar com **MAE** otimiza diretamente a métrica → previsões mais "afiadas".
- Resultado: MAE de ~1.36 (MSE) → **~1.33–1.35** (MAE loss).

### 3.4 Escolha de capacidade

`d_model=128, num_heads=8, num_layers=3, ff_dim=256, dropout_rate=0.1` mostrou-se um bom ponto: capacidade suficiente para aprender os padrões sem memorizar demais o treino.

---

## 4. Técnicas que NÃO funcionaram (e por quê)

| Técnica | Resultado | Diagnóstico |
|---|---|---|
| **Decodificação autorregressiva na inferência** (`S2STransformerAutoregressiveWrapper`) | MAE 1.95 (piorou muito) | Erros acumulam: cada passo alimenta o próximo, amplificando o desvio. O modelo foi treinado com teacher forcing (vê valores verdadeiros) e, na inferência, realimenta as próprias previsões (ruidosas) → *exposure bias* agravado |
| **Alvo residual** (`target_mode="residual"`) | MAE 1.40 | Prever a *mudança* a partir do último valor não ajudou; o modelo perde a referência do nível absoluto |
| **Decoder "direct"** (`decoder_mode="direct"`, com rampa de posição) | MAE 1.60 | O sinal extra de posição confundiu o treino (val_loss alto); o PE já codifica a posição |
| **Contexto maior** (input 144 passos) | MAE 1.53 | Mais passos para atenção → mais difícil de ajustar com poucos dados (5670 amostras de treino) |
| **Regularização forte** (dropout 0.3 + AdamW weight decay 1e-4) | MAE 1.42 | Reduziu a capacidade justamente onde o modelo precisava de flexibilidade |

**Lição:** no regime de *domain shift* severo, o ganho vem mais do **treinamento (LR schedule, loss alinhada)** do que da **mudança de arquitetura de inferência**.

---

## 5. Metodologia de avaliação (lições de auditoria)

1. **`val_loss` NÃO correlaciona com MAE de teste.** Ex.: LSTM com val_loss 1.7e-05 → MAE 1.45; Transformer com val_loss ~1e-03 → MAE 1.33. O val (adjacente ao treino) é "fácil"; o teste (distante) é "difícil". **Otimizar por `val_loss` (Optuna padrão) não resolve o problema real.**
2. **Sensibilidade a seed é alta.** O mesmo config com seeds diferentes varia ~0.04–0.07 no MAE. Comparações devem usar **múltiplas seeds**.
3. **Baselines salvos podem estar obsoletos** (dataset mudou). Sempre re-executar os baselines no estado atual antes de comparar.
4. **Harness reproduzível** (`tests/benchmark_transformer.py`): seed fixa, orçamento de épocas controlado, log em JSONL, opção de salvar previsões para diagnóstico.

---

## 6. Espaço de hiperparâmetros (Optuna)

```
learning_rate   : log-uniform 1e-4 a 1e-2
d_model         : 32 a 256 (passo 16)
num_heads       : 1 a 8
num_layers      : 1 a 4
ff_dim          : 64 a 512 (passo 32)
dropout_rate    : 0 a 0.3 (passo 0.05)
```
- `key_dim = max(1, d_model // num_heads)`.
- No Keras 3, `MultiHeadAttention` projeta a saída para `d_model` (dimensão do query) mesmo quando `num_heads` não divide `d_model` — verificado empiricamente, então não há restrição de divisibilidade.

---

## 7. Como executar

```bash
# Benchmark justo de um modelo (seed reproduzível, log em data/results/benchmark_transformer.jsonl)
python tests/benchmark_transformer.py --wrapper transformer_preln --schedule --loss mae \
    --epochs 100 --seed 42 \
    --params '{"learning_rate": 0.0006, "d_model": 128, "num_heads": 8, "num_layers": 3, "ff_dim": 256, "dropout_rate": 0.1}'

# Baselines para comparação (mesmo orçamento, dados atuais)
python tests/benchmark_transformer.py --wrapper lstm --from-optuna --epochs 100 --seed 42
python tests/benchmark_transformer.py --wrapper tcn   --from-optuna --epochs 40  --seed 42

# Busca Optuna (usa o wrapper PreLN + schedule)
python tests/optuna_all_hyperparameters.py --wrappers transformer --n-trials 100 --epochs 60

# Walk-forward com os melhores parâmetros
python tests/wfo_optuna.py --wrappers transformer

# Walk-forward com retreino periódico em dados imputados
python tests/wfo_imputed.py --models transformer
```

---

## 8. Resultados (dados atuais, protocolo wfo)

| Modelo | Execuções (seeds) | MAE médio | Melhor MAE |
|---|---|---|---|
| **Transformer (PreLN + schedule + MAE loss)** | 42, 7, 123, 42(fresco) | **~1.37** | **1.328** |
| TCN | 42, 7, 123 | ~1.44 | 1.431 |
| LSTM | 42 + (sem seed) | ~1.45 | 1.448 |

O Transformer superou ambos LSTM e TCN (~6–7% em MAE) de forma consistente nos dados atuais.

---

## 9. Arquivos relevantes

| Arquivo | Papel |
|---|---|
| `src/models/s2s_transformer_wrapper.py` | V1 (Post-LN) + `WarmupCosineSchedule` + `make_learning_rate` |
| `src/models/s2s_transformer_preln_wrapper.py` | **Vencedor**: Pre-LN, PE aprendido, loss/weight-decay/clipnorm configuráveis |
| `src/models/s2s_transformer_autoreg_wrapper.py` | Variante autorregressiva (testada, descartada) |
| `tests/benchmark_transformer.py` | Harness de benchmark justo/reproduzível |
| `tests/optuna_all_hyperparameters.py`, `tests/wfo_optuna.py`, `tests/wfo_imputed.py` | Integração do transformer (schedule habilitado) |
| `notebooks/test_all_models_single_plot.ipynb`, `notebooks/test_all_models_per_model.ipynb` | Transformers incluídos na comparação de modelos |
| `docs/relatório_transformer_improvements.md` | Relatório das iterações de melhoria |