# 04_mispricing_signal_construction

## Purpose

This project constructs option mispricing signals from SVI residuals. It standardizes quote-level residuals within each intraday cross-section, aggregates them into daily contract-level signals, and then plots the OLS persistence relationship between one-day-ahead daily signals.

## Folder Structure

```text
04_mispricing_signal_construction/
data/
  raw/
  intermediate/
  processed/
scripts/
plot/
output/
  report/
  figures/
README.md
```

## Required Python Packages

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`

## Required Input Files

Place the following files in `data/raw/`:

- `option_quotes_with_svi_based_iv.csv`
- `trading_dates.json`

The quote file is the processed output of the Vega-weighted SVI project. The trading calendar is retained as a raw project input for downstream date-aware diagnostics, although the OLS persistence plot does not require manual date selection.

## Run Order

Run the main analysis script first:

```bash
python scripts/01_construct_mispricing_signal.py
```

Then run the plotting script:

```bash
python plot/01_ols_regression_plot.py
```

## Outputs

Intermediate data are saved in `data/intermediate/`:

- `_option_quotes_with_standardised_residual.csv`
- `_daily_mispricing_signal.csv`

The processed dataset is saved in `data/processed/`:

- `option_quotes_with_mispricing_signal.csv`

Reports are saved in `output/report/`:

- `mispricing_signal_summary.txt`
- `ols_regression_plot_summary.txt`

Figures are saved in `output/figures/`:

- `fig_signal_persistence.pdf`
- `fig_signal_persistence.png`

## Plot Scripts

`plot/01_ols_regression_plot.py` reads:

- `data/processed/option_quotes_with_mispricing_signal.csv`

It constructs adjacent daily signal pairs directly from the processed data and generates the OLS persistence plot. No manually entered regression coefficient, date range, or contract symbol is required.

