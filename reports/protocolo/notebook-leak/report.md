# Comparativo sob o protocolo do notebook original (com leak)

Data: 2026-09-18. Reprodução fiel do `rolling_forecasting_real_time_improved`
do notebook `OLD/notebooks/wind_speed_forecasting-original.ipynb` para TODOS
os wrappers, com o objetivo de responder: o erro baixo do LSTM original vinha
da arquitetura ou do protocolo?

## Protocolo reproduzido

- dataset.csv lido cru (nomes canônicos), denoise wavelet (sym18, level 2) nas
  colunas do notebook: ws100, disp40, vdisp40, dir40, cis1, humid, temp.
- Scaling MinMax global (target separado dos demais), teacher forcing (decoder
  = último observado), split 77% / 18% / 5% (por sequências).
- Treino: MSE, Adam lr=1e-3, batch 32, early stopping (patience 10),
  reduce-lr (patience 5), até 100 épocas. Sem persistence gate.
- Rolling forecast idêntico ao notebook: previsão = último decoder step;
  actuals de `test_data[i + output_steps - 1]`; e **a janela do encoder recebe
  o valor real futuro** (`window = vstack([window, test_data[i+35]])`) — o leak
  que torna o erro artificialmente baixo.

## Resultados

| modelo | MAE denoised | RMSE denoised | MAE raw(últ. N) | RMSE raw(últ. N) |
|---|---|---|---|---|
| Seq2Seq_Transformer_PreLN | **0.1708** | **0.2283** | **0.2893** | **0.3820** |
| Seq2Seq_LSTM_Original | 0.1724 | 0.2263 | 0.2953 | 0.3810 |
| Seq2Seq_GRU | 0.1731 | 0.2255 | 0.2930 | 0.3807 |
| Seq2Seq_LSTM | 0.1759 | 0.2262 | 0.2946 | 0.3809 |
| Seq2Seq_LSTM_Bidirectional | 0.1771 | 0.2294 | 0.2969 | 0.3826 |
| Seq2Seq_TCN_LSTM | 0.1784 | 0.2308 | 0.2938 | 0.3833 |
| Seq2Seq_GRU_Bidirectional | 0.1812 | 0.2343 | 0.2999 | 0.3852 |
| Seq2Seq_LSTM_CNN | 0.2028 | 0.2617 | 0.3093 | 0.4018 |
| Seq2Seq_TCN_Bidirectional | 0.6536 | 0.8173 | 0.6916 | 0.8722 |
| Seq2Seq_TCN | 1.5213 | 1.8802 | 1.5248 | 1.8976 |

## Conclusão

**O erro baixo do notebook (~0.16) era do protocolo, não da arquitetura.**

Sob o MESMO protocolo (com leak), **todos** os modelos recorrentes atingem
MAE ≈ 0.17–0.18 — LSTM original, LSTM, LSTM-bi, GRU, GRU-bi, TCN-LSTM e até o
Transformer ficam num intervalo praticamente indistinguível (~0.17). O leak
(real futuro injetado na janela) fornece a informação que ancorar a previsão,
achatando a diferença entre arquiteturas.

- O LSTM original **não é especial**: em protocolo honesto (sem leak) seu MAE
  era 0.959 (all-horizons); aqui todos os recorrentes empatam em ~0.17.
- O **Transformer** é o melhor sob o leak (0.1708), mas a diferença para o
  LSTM original (0.1724) é irrelevante (0.002).
- **TCN e TCN-bi colapsam** sob esta receita (MSE + lr 1e-3 + lr constante +
  early stopping): o TCN foi cortado em 27 épocas com val_loss 0.0147 (precisa
  de lr menor e warmup-cosine, como no Optuna). TCN-bi (0.65) também sofre.

## Implicação

A comparação justa entre modelos deve usar o protocolo honesto do pipeline
(step_evaluate): janela de história conhecida, sem futuro. Sob esse protocolo:

- LSTM (Optuna) é o melhor em all-horizons (MAE 0.903).
- TCN é o melhor em rolling / h=36 (MAE 1.15).
- O resultado ~0.16 do notebook é um artefato do leak, não um desempenho real.

Artefatos: `pipeline/tmp/notebook_protocol/{Seq2Seq_*}/predictions.csv` +
`summary.json`/`summary.md`.