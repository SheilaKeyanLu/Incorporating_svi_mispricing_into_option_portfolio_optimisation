# Input Document

This input folder contains the files and settings used by `../src/parameter_sensitivity_backtest.py`.

## Files

- `data_prepared_for_markowitz.csv`: prepared option panel used by the BL posterior, margin-aware optimisation, and one-day return calculation.
- `trading_dates.json`: trading calendar used to select the backtest dates and next-day return dates.
- `INPUT_DOCUMENT.md`: input-setting description for the reported parameter-sensitivity experiment.

## Experiment Settings

- Backtest evaluation window: 2 January 2025 to 30 June 2026.
- Lambda sweep: `10000, 12000, 14000, 16000, 18000, 20000`, holding `delta = 5`.
- Delta sweep: `3, 4, 5, 6, 7, 8, 9`, holding `lambda = 10000`.
- Fixed parameters in `../src/parameter_sensitivity_backtest.py`:
  - `tau = 0.05`
  - `xi = 1.0`
  - `max_weight = 0.3`
  - covariance training window: `20` trading days
  - regression training window: `240` trading days
  - regression test window: `20` trading days

## Path Check

The code resolves this folder through `Path(__file__).resolve().parent.parent`, so the default input paths are:

- `../input/data_prepared_for_markowitz.csv`
- `../input/trading_dates.json`
