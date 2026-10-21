# Imputed-data WFO benchmarks

The benchmark objective is MAE averaged over all 36 forecast horizons. Metrics
exclude targets that were missing in the original data. Persistence repeats the
last available target, which may itself be imputed, across every horizon.

Unless noted otherwise, runs use two seven-day test windows, a 60-day rolling
training window, a 30-day step, 20 maximum epochs, patience 4, seed 42, raw
`ws100`, and Optuna parameters.

```bash
python tests/wfo_imputed.py --models lstm_bi tcn_bi \
  --train-window-days 60 --test-window-days 7 --step-days 30 \
  --epochs 20 --patience 4 --max-windows 2
```

## Architecture choices

| Change | Bi-LSTM MAE | TCN-Bi MAE | Decision |
|---|---:|---:|---|
| Original teacher-forced decoder | 1.5648 | 1.5632 | TCN-Bi retained |
| Direct decoder with horizon input | 1.4829 | 1.5997 | Bi-LSTM retained |
| Circular direction features | 1.5245 | 1.5747 | Rejected |
| Exclude imputed training targets | 1.6244 | 1.5834 | Rejected |
| Residual over persistence | 1.4537 | 1.9194 | Bi-LSTM retained |
| Wider TCN receptive field | - | 1.6223* | Rejected |
| Inference-style validation | - | 1.4721 | TCN-Bi retained |

`*` The receptive-field comparison uses the same February test window; the
baseline TCN-Bi MAE for that window is 1.5985.

The persistence MAE over the two-window benchmark is 1.5921. The retained
Bi-LSTM configuration improves on persistence by 8.7%; retained TCN-Bi improves
by 7.5%.

## Training choices

The following values compare the same February test interval.

| Training history | Bi-LSTM MAE | TCN-Bi MAE | Two-model mean |
|---|---:|---:|---:|
| 30 days | 1.2917 | 1.6367 | 1.4642 |
| 60 days | 1.2995 | 1.5985 | 1.4490 |
| 90 days | 1.3617 | 1.5919 | 1.4768 |

Sixty days is retained as the shared default. MAE loss was also rejected: it
helped the February Bi-LSTM window but hurt January, increasing two-window MAE
from 1.4537 to 1.5159.

## Retained defaults

- Bi-LSTM: direct decoder, normalized horizon input, residual target, MSE loss.
- TCN-Bi: teacher-forced training, inference-style validation, absolute target,
  MSE loss.
- Raw target without noncausal wavelet denoising.
- Optuna parameters loaded from each model's own `best_trial.json`.
- Metrics computed on observed targets over all 36 horizons.
- Imputed training rows retained, with observed/imputed evaluation flags saved.
- Sixty-day rolling training window.
