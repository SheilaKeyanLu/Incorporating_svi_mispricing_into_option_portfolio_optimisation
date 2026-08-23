# 03__fit_in_SVI_W_vega

## Purpose

This project fits SVI volatility slices using Vega weighting, evaluates calibration quality, compares it with equal weighting, maps fitted SVI volatility back to quote rows, and generates the dissertation SVI figures.

## Folder Structure

```text
03__fit_in_SVI_W_vega/
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

The scripts also import shared SVI utilities from the repository-level `common/` folder.

## Required Input Files

Place the following file in `data/raw/`:

- `option_quotes_with_iv.csv`

This file is the processed output of the Black-Scholes implied-volatility project.

The comparison script also requires the sibling project output:

- `../03__fit_in_SVI_W_I/data/processed/svi_fit_equal_weighting.csv`

## Run Order

First run the equal-weighting project. Then run the Vega-weighting analysis scripts from this project root:

```bash
python scripts/01_calculate_vega.py
python scripts/02_fit_svi_vega_weighting.py
python scripts/03_summarize_svi_margin.py
python scripts/04_build_svi_based_iv_dataset.py
python scripts/05_write_calibration_comparison.py
```

After the main analysis scripts finish, run the plotting scripts:

```bash
python plot/01_plot_representative_vega_svi_fits.py
python plot/02_plot_single_svi_slice.py
python plot/03_plot_normal_anomalous_comparison.py
```

## Outputs

Intermediate data are saved in `data/intermediate/`:

- `_option_quotes_with_iv_with_vega.csv`
- `_svi_fit.csv`
- `_svi_fit_margin.csv`
- `_svi_margin_summary.csv`
- `_svi_margin_by_maturity.csv`
- `_svi_margin_by_underlying.csv`

Processed data are saved in `data/processed/`:

- `svi_fit_vega_weighting.csv`
- `option_quotes_with_svi_based_iv.csv`

Reports are saved in `output/report/`:

- `calibration_quality_vega_weighting.txt`
- `calibration_quality_comparison.csv`
- `calibration_quality_comparison.txt`
- `calibration_quality_comparison.tex`

Figures are saved in `output/figures/`.

## Plot Scripts

`plot/01_plot_representative_vega_svi_fits.py` reads:

- `data/raw/option_quotes_with_iv.csv`
- `data/intermediate/_svi_fit.csv`

It writes the representative 3 by 3 Vega-weighted SVI fit grid.

`plot/02_plot_single_svi_slice.py` reads:

- `data/raw/option_quotes_with_iv.csv`
- `data/intermediate/_svi_fit.csv`

It writes one selected total-variance SVI slice.

`plot/03_plot_normal_anomalous_comparison.py` reads:

- `data/raw/option_quotes_with_iv.csv`
- `data/intermediate/_svi_fit.csv`

It writes the normal-versus-anomalous SVI fit comparison figure.

