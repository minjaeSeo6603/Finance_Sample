# Debt Attitudes and Household Borrowing Behavior (SCF 2010-2022)

This project analyzes how household debt attitudes relate to observed borrowing behavior using the U.S. Survey of Consumer Finances (2010, 2016, 2022).

## Environment Setup

Run commands from the project root (the folder containing this `README.md`):

```bash
cd /path/to/your/project-folder
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

System tools required for PDF rendering:

- Quarto CLI
- XeLaTeX (TeX Live or MacTeX)

## One-Command Run

`cleaned_scf_data.csv` is intentionally not committed to GitHub (file is too large).
After cloning, run the command below once to generate all outputs locally.

```bash
bash make.sh
```

`make.sh` runs: data cleaning -> notebook execution -> figure generation -> PDF rendering.

## Pipeline

```bash
# 1) Clean raw data
python code/python/data_cleaning.py

# 2) Run analysis notebook
IPYTHONDIR=/tmp/ipython MPLCONFIGDIR=/tmp/mpl \
jupyter nbconvert --to notebook --execute code/notebooks/analysis.ipynb \
  --output analysis_exec --output-dir /tmp \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=python3

# 3) Generate publication figures
python code/python/generate_paper_figures.py

# 4) Render paper (final PDF only in output/reports)
quarto render code/r/SeoMinjae_Writing_Sample.qmd --to pdf
mv code/r/SeoMinjae_Writing_Sample.pdf output/reports/SeoMinjae_Writing_Sample.pdf
```

## Project Structure

```text
Finance_Sample/
├── README.md
├── requirements.txt
├── make.sh
├── code/
│   ├── python/
│   │   ├── data_cleaning.py
│   │   └── generate_paper_figures.py
│   ├── notebooks/
│   │   └── analysis.ipynb
│   └── r/
│       └── SeoMinjae_Writing_Sample.qmd
├── data/
│   ├── raw/
│   └── clean/
│       └── cleaned_scf_data.csv  (generated locally, not tracked in git)
└── output/
    ├── figures/
    ├── tables/
    └── reports/
        ├── SeoMinjae_Writing_Sample.pdf
        └── data_quality_report.txt
```

## Required Raw Inputs

Place the following files in `data/raw/`:

- `SCFP2010.csv`
- `SCFP2016.csv`
- `SCFP2022.csv`
- `scf_sup_2010.csv`
- `scf_sup_2016.csv`
- `scf_sup_2022.csv`

## Notes

- `data/clean/cleaned_scf_data.csv` is generated and excluded in `.gitignore` because it exceeds GitHub's file-size limit.
- If you cloned this repository and `data/clean/cleaned_scf_data.csv` is missing, run: `python code/python/data_cleaning.py`.
- Monetary variables are inflation-adjusted to 2022 dollars.
- SCF survey weights are used for population-representative estimates.
- Paths are project-relative, so folder name does not need to be `Finance_Sample`.
