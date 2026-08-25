# Incorporating SVI Mispricing into Option Portfolio Optimisation

Code and empirical workflow accompanying the MSc dissertation *"Incorporating
Implied Volatility Surface Mispricing into Option Portfolio Optimisation"*
(University of Edinburgh, MSc Operational Research with Data Science).

This repository contains the full empirical pipeline: option data preparation,
SVI surface calibration, mispricing-signal construction, return-predictability
validation, Delta-Gamma / Black-Litterman portfolio optimisation, backtesting,
parameter sensitivity, and tail-risk assessment.

---

## 1. Project purpose

The empirical pipeline links four stages described in the dissertation:

1. Recover forward prices and risk-free rates from put-call parity, and infer
   implied volatility from the BSM model.
2. Calibrate SVI volatility surfaces (equal-weighted and Vega-weighted) and
   construct daily SVI-based mispricing signals from the fitted residuals.
3. Test whether the mispricing signals predict next-day option returns, and use
   the retained forecasts as Black-Litterman investor views.
4. Optimise and backtest Delta-Gamma / BL long-short and long-only option
   portfolios, then evaluate parameter sensitivity and tail risk.

---

## 2. Repository structure

```text
svi_mispricing_signal_construction/   # Stages 1-4 in the signal pipeline
calc_greeks/                          # Greeks for portfolio-input construction
halflife_analysis/                    # Signal persistence / half-life (Sec. 5.4, App. D)
signal_return_prediction/             # Return-predictability regressions (Sec. 6.2.3)
model_backtest_comparison/            # Portfolio optimisation + 4-model backtest (Sec. 6.2, 6.4.1)
parameter_sensitivity/                # lambda / delta sensitivity (Sec. 6.4.2)
tail_risk_assessment/                 # VaR / CVaR / Cornish-Fisher (Sec. 6.4.3)
.gitignore
```

Each folder is self-contained and has its own `README.md` with run commands and
detailed input/output listings; this top-level document explains how the
folders relate to each other and to the dissertation.

---

## 3. Dissertation-to-code mapping

| Dissertation section | Content | Code folder |
|---|---|---|
| 5.1 Market Input Construction & IV Extraction | PCP regression for `F`, `r_f`; BSM implied volatility | `svi_mispricing_signal_construction/01_fit_q_r_from_PCP/`, `02_calculate_iv_from_BSM/` |
| 5.2-5.3 Direct-Fit SVI Calibration & Weighting Schemes | Equal-weighted and Vega-weighted SVI calibration, stability-margin diagnostics | `svi_mispricing_signal_construction/03__fit_in_SVI_W_I/`, `03__fit_in_SVI_W_vega/` |
| 5.4 Residual-Based Mispricing Signal & Persistence | Standardised residuals to daily signal; ACF/PACF, half-life, ANOVA | `svi_mispricing_signal_construction/04_mispricing_signal_construction/`, `halflife_analysis/` |
| Appendix B / 6.2.2 Delta-Gamma Return & Risk Modelling | Option Greeks; portfolio-level `u`, `A = V Sigma V' + D` | `calc_greeks/`, `model_backtest_comparison/preprocessing.py` |
| 6.2.3 Return-Prediction Models & BL View Construction | Baseline/Model 1-3 regressions, OOS validation, `P`, `Q`, `Omega` construction | `signal_return_prediction/`, `model_backtest_comparison/preprocessing.py` |
| 6.2.4-6.2.5 Transaction Cost, Margin, MIQP Formulation | Cost/margin vectors, Gurobi MIQP (long-short and long-only) | `model_backtest_comparison/preprocessing.py`, `optimizer.py`, `portfolio_optimisation/` |
| 6.3-6.4.1 Backtesting & Portfolio Comparison | Four fixed-parameter backtests, cumulative-return comparison | `model_backtest_comparison/` (all four `0X_*_backtest.py` entrypoints, `comparison_visualisation/`) |
| 6.4.2 Parameter Sensitivity | `lambda`, `delta` sweeps on the BL Long-Short strategy | `parameter_sensitivity/` |
| 6.4.3 Tail Risk Assessment | Historical and Cornish-Fisher VaR/CVaR | `tail_risk_assessment/` |

Note: PCP/discount-factor estimation, BL posterior construction (`u_BL`,
`A_BL`), and margin/transaction-cost vectors are implemented as shared
preprocessing logic within `svi_mispricing_signal_construction/` and
`model_backtest_comparison/preprocessing.py` respectively, rather than as
separate top-level folders. See each folder's own README for the exact
function-level breakdown.

---

## 4. Suggested reproduction order

The folders are designed to be run in the following order, since later stages
consume the processed outputs of earlier ones:

1. `svi_mispricing_signal_construction/` - run subfolders `01` to `04` in
   sequence to produce `option_quotes_with_mispricing_signal.csv`.
2. `calc_greeks/` - compute Greeks on the IV dataset.
3. `halflife_analysis/` - persistence diagnostics (uses the mispricing-signal
   output; does not feed forward into later stages).
4. `signal_return_prediction/` - validate signal predictability and produce
   rolling-regression coefficients used for BL views.
5. `model_backtest_comparison/` - run the four backtest entrypoints
   (`01_bl_long_short_backtest.py` to `04_historical_mean_variance_backtest.py`)
   to produce the main comparison results (Fig. 6.2).
6. `parameter_sensitivity/` - sweep `lambda`/`delta` on the BL Long-Short
   strategy (Fig. 6.3, 6.4).
7. `tail_risk_assessment/` - compute VaR/CVaR on the return series exported by
   `model_backtest_comparison/comparison_visualisation/`.

Steps 3, 6, and 7 are downstream diagnostics and can be run independently once
their required inputs exist; they do not need to be re-run for one another.

---

## 5. Data availability and large files

Large cleaned/preprocessed CSV files required to run the code are available in
the OneDrive data folder:

<https://uoe-my.sharepoint.com/:f:/r/personal/s2293742_ed_ac_uk/Documents/dissertation_data?d=w657142e6c42245a1a7e32e45261a4e58&csf=1&web=1&e=LEVp8o>

The OneDrive folder contains:

- `option_quotes_with_basic_and_underlying.csv`
- `option_quotes_with_discount_factor.csv`
- `option_quotes_with_iv.csv`
- `option_quotes_with_svi_based_iv.csv`
- `option_quotes_with_mispricing_signal.csv`
- `option_quotes_with_greeks.csv`

These files are excluded from GitHub because they exceed the 100 MB file-size
limit. Each subproject README specifies which files are needed and the exact
repository path where they should be placed before running that subproject.

Python cache files (`__pycache__/`, `*.py[cod]`) are also ignored and are not
needed for reproduction.

---

## 6. Dependencies

- Python 3.x
- Standard scientific stack: `pandas`, `numpy`, `scipy`, `statsmodels`,
  `matplotlib`
- `gurobipy`, with a valid Gurobi license - required by
  `model_backtest_comparison/` and `parameter_sensitivity/` for the MIQP
  portfolio optimisation

Install project-specific requirements from each folder's own README where
provided; a consolidated `requirements.txt` is recommended if not already
present.

---

## 7. Folder-level documentation

For run commands, exact script arguments, and full input/output file listings,
see the README in each folder:

- [`svi_mispricing_signal_construction/01_fit_q_r_from_PCP/README.md`](svi_mispricing_signal_construction/01_fit_q_r_from_PCP/README.md)
- [`svi_mispricing_signal_construction/02_calculate_iv_from_BSM/README.md`](svi_mispricing_signal_construction/02_calculate_iv_from_BSM/README.md)
- [`svi_mispricing_signal_construction/03__fit_in_SVI_W_I/README.md`](svi_mispricing_signal_construction/03__fit_in_SVI_W_I/README.md)
- [`svi_mispricing_signal_construction/03__fit_in_SVI_W_vega/README.md`](svi_mispricing_signal_construction/03__fit_in_SVI_W_vega/README.md)
- [`svi_mispricing_signal_construction/04_mispricing_signal_construction/README.md`](svi_mispricing_signal_construction/04_mispricing_signal_construction/README.md)
- [`calc_greeks/README.md`](calc_greeks/README.md)
- [`halflife_analysis/README.md`](halflife_analysis/README.md)
- [`signal_return_prediction/README.md`](signal_return_prediction/README.md)
- [`model_backtest_comparison/README.md`](model_backtest_comparison/README.md)
- [`parameter_sensitivity/README.md`](parameter_sensitivity/README.md)
- [`tail_risk_assessment/README.md`](tail_risk_assessment/README.md)
