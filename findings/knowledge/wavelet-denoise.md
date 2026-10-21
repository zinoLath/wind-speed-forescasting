---
title: Denoise wavelet do alvo
area: protocolo
updated: 2026-10-19
---

# Denoise wavelet do alvo (sym18, nível 2)

O treino usa o alvo suavizado `ws100_wavelet`; a avaliação headline usa o
bruto — a suavização não pode inflar os resultados.

- Decomposição `pywt.wavedec` com **Symlet 18 (`sym18`)**, modo
  `periodization`, **nível 2**; coeficientes de detalhe limiarizados por
  **soft-thresholding** com limiar universal de VisuShrink
  (`σ = MAD/0.6745`, limiar `σ·√(2·ln N)`).
- Implementação: `src/utils.py:wavelet_denoising()`.
- A escolha foi validada por A/B alvo cru vs denoised
  (`tests/ab_raw_vs_wavelet.py`; resultado em `docs/ab_target_denoise.md`).

## Fontes

- `docs/metodologia.md` (seção 2); `README.md`; `agents/architecture.md`.
