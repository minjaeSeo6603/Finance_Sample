#!/usr/bin/env bash
set -euo pipefail

# Run from project root regardless of current directory
cd "$(dirname "$0")"

echo "[1/4] Cleaning raw data..."
python code/python/data_cleaning.py

echo "[2/4] Executing analysis notebook..."
IPYTHONDIR=/tmp/ipython MPLCONFIGDIR=/tmp/mpl \
jupyter nbconvert --to notebook --execute code/notebooks/analysis.ipynb \
  --output analysis_exec --output-dir /tmp \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=python3

echo "[3/4] Generating publication figures..."
python code/python/generate_paper_figures.py

echo "[4/4] Rendering writing sample PDF..."
quarto render code/r/SeoMinjae_Writing_Sample.qmd --to pdf
mkdir -p output/reports
mv -f code/r/SeoMinjae_Writing_Sample.pdf output/reports/SeoMinjae_Writing_Sample.pdf

echo "Done."
echo " - data/clean/cleaned_scf_data.csv"
echo " - output/reports/data_quality_report.txt"
echo " - output/reports/SeoMinjae_Writing_Sample.pdf"
