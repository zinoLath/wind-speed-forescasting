#!/usr/bin/env bash
# Empacota o projeto para rodar o estudo Optuna no Kaggle.
#
# Gera dist/kaggle_optuna.zip com: código-fonte, configs, dataset.csv e o
# progresso atual do estudo optuna_nogate (sqlite + trials.csv). O zip é
# anexado como Dataset do Kaggle e descompactado pelo notebook
# notebooks/kaggle_optuna_nogate.ipynb dentro de /kaggle/working.
#
# Usage:
#   bash scripts/kaggle_package.sh

set -eu

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAMP="$(date '+%Y%m%d')"
OUT="dist/kaggle_optuna_${STAMP}.zip"

mkdir -p dist
rm -f dist/kaggle_optuna_*.zip

zip -qr "$OUT" \
    src \
    pipeline/*.py \
    pipeline/*.json \
    tests \
    scripts \
    requirements.txt \
    data/dataset.csv \
    -x "*__pycache__*" -x "*.pyc"

if [ -d pipeline/tmp/optuna_nogate ]; then
    zip -qr "$OUT" pipeline/tmp/optuna_nogate -x "*__pycache__*" -x "*.pyc"
fi

echo "Pacote gerado: $OUT ($(du -h "$OUT" | cut -f1))"
echo "Anexe-o como Dataset do Kaggle (ex.: 'wind-speed-forecasting') e use"
echo "o notebook notebooks/kaggle_optuna_nogate.ipynb com accelerator GPU."
