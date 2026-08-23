# Model Backtest Comparison

This folder contains the unified fixed-parameter backtest project for the four empirical comparison models used in the dissertation. It keeps one shared input folder, one shared backtest runner, shared preprocessing/optimizer code, and four model-specific entry scripts.

## Model Mapping

| Run order | Dissertation model | Entry script | Original source folder |
| --- | --- | --- | --- |
| 01 | BL Long-Short | `01_bl_long_short_backtest.py` | `A_short_test_no_tunning` |
| 02 | BL Long-Only | `02_bl_long_only_backtest.py` | `B_long_only_test_no_tunning` |
| 03 | Delta-Gamma Approximation | `03_delta_gamma_approximation_backtest.py` | `C_long_only_no_BL` |
| 04 | Historical Mean-Variance | `04_historical_mean_variance_backtest.py` | `D_Mean_Variance` |

## Code Structure

- `config.py`: central path settings, dissertation fixed parameters, trading-cost settings, and model output paths.
- `preprocessing.py`: shared data filtering, contract indexing, forward-return/covariance construction, BL posterior construction, margin vector, and transaction-cost vector logic.
- `optimizer.py`: shared Gurobi optimisation models for long-short and long-only portfolio weights.
- `backtest_runner.py`: shared fixed-parameter backtest loop, daily result collection, log writing, CSV export, runtime record, and cumulative-return figure generation.
- `portfolio_optimisation/`: model-specific daily optimisation wrappers.
- `comparison_visualisation/`: strategy-level comparison CSVs and the plotting script for the thesis comparison figure.
- `01_bl_long_short_backtest.py` to `04_historical_mean_variance_backtest.py`: four runnable backtest entrypoints.
- `input/`: shared input data for all four models.
- `output/`: model-specific output folders created by each backtest.

## Input

All four models read from the same input folder:

- `input/data_prepared_for_markowitz.csv`: main prepared option/forward dataset used by the backtests.
- `input/option_daily_return_long_table.csv`: historical contract return table used by the Historical Mean-Variance benchmark.
- `input/trading_dates.json`: ordered trading calendar used to select backtest dates and next-day returns.

The active input path is defined in `config.py`:

```python
DATA_FILE = INPUT_DIR / "data_prepared_for_markowitz.csv"
OPTION_RETURN_FILE = INPUT_DIR / "option_daily_return_long_table.csv"
TRADING_DATES_FILE = INPUT_DIR / "trading_dates.json"
```

## Output

Each entry script writes only to its own output folder:

- `output/01_bl_long_short_output/`
- `output/02_bl_long_only_output/`
- `output/03_delta_gamma_approximation_output/`
- `output/04_historical_mean_variance_output/`

Inside each model output folder, a run creates:

- `logs/backtest(<begin>-<end>).log`: daily positions, daily returns, cumulative value, and summary statistics.
- `return_curve/<timestamp>/<begin>-<end>_daily_results.csv`: structured daily backtest results.
- `return_curve/<timestamp>/<begin>-<end>_return_curve.png`: cumulative return curve.
- `return_curve/<timestamp>/runtime.txt`: elapsed runtime in seconds and minutes.

The `comparison_visualisation/` folder contains comparison-ready strategy CSV files and writes the visual comparison outputs next to the plotting script by default:

- `comparison_visualisation/strategy_comparison_thesis.png`
- `comparison_visualisation/strategy_comparison_thesis.pdf`
- `comparison_visualisation/strategy_comparison_thesis_summary.csv`

## How To Run

Open a terminal in the `model_backtest_comparison` folder:

```powershell
cd model_backtest_comparison
```

Run one model at a time:

```powershell
python 01_bl_long_short_backtest.py
python 02_bl_long_only_backtest.py
python 03_delta_gamma_approximation_backtest.py
python 04_historical_mean_variance_backtest.py
```

These scripts run the full fixed-parameter backtest and may take time because they solve a Gurobi optimisation problem for each trading date.

To generate the comparison figure from the CSV files in `comparison_visualisation/`:

```powershell
python comparison_visualisation\plot_strategy_comparison.py
```

Optional examples:

```powershell
python comparison_visualisation\plot_strategy_comparison.py . -o strategy_comparison_thesis.png
python comparison_visualisation\plot_strategy_comparison.py . --exclude scratch,readme
```

For the plotting script, relative input and output paths are resolved relative to the `comparison_visualisation/` folder. The Adaptive BL Model is intentionally excluded from this comparison script.

## Solver Requirement

This project requires Gurobi through the Python package `gurobipy`.

Required setup:

- A working Python environment with the project dependencies installed, including `pandas`, `numpy`, `matplotlib`, and `gurobipy`.
- A valid Gurobi installation and license available to Python.
- The command `python -c "import gurobipy"` should run without errors before launching the backtests.

If Gurobi is missing or the license is unavailable, the scripts will fail when importing `optimizer.py` or when calling the solver.

## Parameter Settings

The dissertation fixed parameters are defined in `config.py`.

Common fixed parameters:

| Parameter | Code name | Value |
| --- | --- | --- |
| Backtest start date | `BEGIN_DATE` | `2025-01-02` |
| Backtest end date | `END_DATE` | `2026-06-30` |
| BL covariance scaling | `TAU_FIXED` | `0.05` |
| Transaction-cost weight | `XI_FIXED` | `1.0` |
| Max single-contract weight | `MAX_WEIGHT` | `0.3` |
| Momentum / mean-covariance window | `TRAIN_DAYS` | `20` |
| Regression training window | `REGRESSION_TRAIN_DAYS` | `250` |
| OOS validation window | `REGRESSION_TEST_DAYS` | `20` |
| Trading fee | `FEE` | `15` |
| Budget | `BUDGET` | `10_000_000` |

Model-specific risk parameters:

| Model | `lambda_risk` | `delta` |
| --- | --- | --- |
| BL Long-Short | `10000` | `5` |
| BL Long-Only | `10000` | `5` |
| Delta-Gamma Approximation | `100` | not used |
| Historical Mean-Variance | `100` | not used |

The non-BL benchmark models do not incorporate BL views, so they do not require the BL view-uncertainty parameter `delta`.

## Notes

- Sensitivity tests and parameter-tuning scripts are intentionally excluded from this folder.
- The four backtests share the same input data and output layout.
- The only model-specific code is the daily construction of expected return/risk inputs and the choice between the long-short or long-only solver.
- `comparison_visualisation/` is for presentation/figure generation only; it does not run portfolio optimisation.
