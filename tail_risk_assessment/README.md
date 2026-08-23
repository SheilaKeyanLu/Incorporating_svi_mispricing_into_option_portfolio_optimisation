# Tail Risk Assessment

This folder is a self-contained code package for the dissertation tail-risk
assessment.

## Folder structure

```text
tail_risk_assessment/
  tail_risk_assessment.py
  input/
    BL Long-Short.csv
    BL Long-Only.csv
    Delta-Gamma Baseline.csv
    Markowitz Baseline.csv
  output/
    tail_risk_results.csv
    tail_risk_table.tex
```

## Run

From this folder:

```powershell
python tail_risk_assessment.py
```

From `input/` or `output/`:

```powershell
python ..\tail_risk_assessment.py
```

The script always reads CSV files from this package's `input/` folder and writes
results to this package's `output/` folder, regardless of the current working
directory.

## Outputs

- `output/tail_risk_results.csv`: full numerical VaR/CVaR results.
- `output/tail_risk_table.tex`: LaTeX table rows for the dissertation.
