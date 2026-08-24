# Residual Persistence Analysis

This folder is the cleaned, final version of the residual half-life workflow.

## Structure

- `input/`: all raw inputs used by the workflow.
- `output/`: generated regression results, half-life reports, ANOVA output, and the three 15-bucket ACF/PACF figure sets used for Appendix D.
- `run_residual_persistence_analysis.py`: one-click script for all three underlyings.
- `regression_original_reference.py`: original single-underlying regression script kept as a reference document.

## Run

```bash
python run_residual_persistence_analysis.py
```

## Main Outputs

- `output/regression_all_underlyings.csv`
- `output/median_half_life_by_bucket.csv`
- `output/median_half_life_table.txt`
- `output/median_half_life_table_latex.tex`
- `output/anova_half_life.csv`
- `output/half_life_report.md`
- `output/000016_acf_grid.png`
- `output/000016_pacf_grid.png`
- `output/000300_acf_grid.png`
- `output/000300_pacf_grid.png`
- `output/000852_acf_grid.png`
- `output/000852_pacf_grid.png`
