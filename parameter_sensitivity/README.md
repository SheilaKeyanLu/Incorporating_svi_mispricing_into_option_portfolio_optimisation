# Parameter Sensitivity Section

This folder contains the code combination, input files, and output files used in the thesis subsection `Parameter Sensitivity`.

## Scope

The folder covers only the BL Long-Short parameter-sensitivity experiment described in the subsection:

- Lambda sweep: `lambda = [10000, 12000, 14000, 16000, 18000, 20000]` with `delta = 5`.
- Delta sweep: `delta = [3, 4, 5, 6, 7, 8, 9]` with `lambda = 10000`.
- Evaluation period: 2 January 2025 to 30 June 2026.
- Main figures:
  - Figure 6.3: `output/THESIS_FIGURES/figure_combined_cumulative_return_sensitivity.pdf`
  - Figure 6.4: `output/THESIS_FIGURES/figure_lambda_delta_total_return_sharpe_stacked.pdf`

## Folder Layout

- `input/`: input data and input-setting document used by the code.
- `src/`: minimal code combination for the reported parameter-sensitivity run.
- `output/`: the two thesis figures and their supporting summary tables.

## Code Combination

- `src/parameter_sensitivity_backtest.py`: runs the lambda and delta sweeps and writes:
  - `output/parameter_sensitivity_summary.csv`
  - `output/all_runs_return_changes.csv`
- `src/make_reported_figures.py`: reads the two CSV files above and writes only the two figures used in the text:
  - `output/THESIS_FIGURES/figure_combined_cumulative_return_sensitivity.pdf`
  - `output/THESIS_FIGURES/figure_combined_cumulative_return_sensitivity.png`
  - `output/THESIS_FIGURES/figure_lambda_delta_total_return_sharpe_stacked.pdf`
  - `output/THESIS_FIGURES/figure_lambda_delta_total_return_sharpe_stacked.png`
- `src/run_all.py`: runs the backtest and then regenerates the two reported figures.

No unrelated single-run plots, logs, or extra thesis figures are included in this整理 folder.
