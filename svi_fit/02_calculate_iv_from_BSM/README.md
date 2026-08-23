# 02_calculate_iv_from_BSM

## Purpose

This project calculates Black-Scholes implied volatility from option quotes that already contain PCP-based discount-factor estimates. It first cleans the quote sample, then computes the forward price and implied volatility, and finally generates representative implied-volatility shape plots from the processed dataset.

## Folder Structure

```text
02_calculate_iv_from_BSM/
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

Place the following file in `data/raw/`:

- `option_quotes_with_discount_factor.csv`

This file is the processed output of the PCP discount-factor estimation project.

## Run Order

Run the main analysis scripts first:

```bash
python scripts/01_prepare_data_for_iv.py
python scripts/02_calculate_implied_volatility.py
```

Then run the plotting script:

```bash
python plot/01_plot_iv_shapes.py
```

## Outputs

Intermediate data are saved in `data/intermediate/`:

- `_option_quotes_clean_for_iv.csv`

The processed dataset is saved in `data/processed/`:

- `option_quotes_with_iv.csv`

The text report is saved in `output/report/`:

- `iv_calculation_summary.txt`

Figures are saved in `output/figures/`:

- `iv_shapes_<underlying>_<maturity>_<first_date>_<last_date>.png`
- `iv_shape_<underlying>_<maturity>_<trade_date>.png`

## Plot Scripts

`plot/01_plot_iv_shapes.py` reads:

- `data/processed/option_quotes_with_iv.csv`

The script selects a liquid underlying/maturity pair with at least four available trade dates directly from the processed data, then saves one combined four-panel plot and one daily plot for each selected date.

