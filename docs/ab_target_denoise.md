# A/B: alvo cru (ws100) vs alvo suavizado (ws100_wavelet)

Experimento: `python tests/ab_raw_vs_wavelet.py --epochs 15`

Treina o mesmo wrapper (`lstm_bi`, hiperparâmetros padrão, seed 42) com dois
alvos diferentes e compara as métricas **contra o ws100 cru** no split de
teste (5% final), para que a suavização não mascare o erro.

## Resultados

| Alvo | MAE (cru) | RMSE (cru) | R² (cru) | Épocas |
|---|---|---|---|---|
| `ws100` (cru) | 0.9606 | 1.2495 | 0.1654 | 7 |
| `ws100_wavelet` | **0.9295** | **1.2279** | **0.1940** | 7 |

Execução anterior (com perda MAE, protocolo idêntico): 0.9623 vs 0.9436 —
mesma conclusão.

## Conclusão

O alvo suavizado por wavelet permanece **melhor mesmo medido contra o dado
cru** (~3% de MAE, ~2% de RMSE). O denoising age como regularizador e não
como máscara do erro. O default do pipeline segue `ws100_wavelet` em
`optuna`/`train`/`walkforward`.

Observações de protocolo:

- Métricas principais sempre contra o dado cru; métricas contra a série
  denoised ficam como colunas secundárias (`*_denoised`).
- Early stopping parou ambos os treinos em 7 épocas; a comparação é
  apples-to-apples (mesmo wrapper, seed, split e orçamento).
