# Prediction Validation Workflow

This folder contains the reproducible workflow for the prediction significance and out-of-sample validation section. The workflow has two steps:

1. Run the main regression script to generate processed CSV files and the text report.
2. Run the plotting script to generate the heatmap figure from the processed coefficient CSV.

The plotting script does not use manually entered significance values. It reads the processed regression output and builds the figure from the estimated signal coefficients and p-value stars.

## Folder Structure

```text
prediction/
  data/
    raw/
      option_quotes_for_ruturn_prediction_model.csv
    processed/
      rolling_split_aggregate_performance.csv
      rolling_split_model_coefficients.csv
      rolling_split_model_performance.csv
      rolling_split_predictions.csv
  output/
    report/
      rolling_split_regression_report.txt
    figures/
      bucket_significance_heatmap.pdf
      bucket_significance_heatmap.png
  plot/
    plot_bucket_significance_heatmap.py
  regression.py
  README.md
```

## Requirements

Use Python 3.10 or later. Install the required packages if they are not already available:

```bash
pip install numpy pandas scipy matplotlib
```

## Step 1: Run the Regression Script

From the project root, run:

```bash
python prediction/regression.py
```

Default input:

```text
prediction/data/raw/option_quotes_for_ruturn_prediction_model.csv
```

Processed CSV outputs:

```text
prediction/data/processed/rolling_split_model_coefficients.csv
prediction/data/processed/rolling_split_model_performance.csv
prediction/data/processed/rolling_split_aggregate_performance.csv
prediction/data/processed/rolling_split_predictions.csv
```

Text report output:

```text
prediction/output/report/rolling_split_regression_report.txt
```

To skip the row-level prediction file, which is relatively large, run:

```bash
python prediction/regression.py --no-predictions
```

To use custom paths:

```bash
python prediction/regression.py --input path/to/input.csv --output-dir path/to/processed --report-dir path/to/report
```

## Step 2: Run the Plotting Script

After Step 1 has produced `rolling_split_model_coefficients.csv`, run:

```bash
python prediction/plot/plot_bucket_significance_heatmap.py
```

Default plot input:

```text
prediction/data/processed/rolling_split_model_coefficients.csv
```

Figure outputs:

```text
prediction/output/figures/bucket_significance_heatmap.pdf
prediction/output/figures/bucket_significance_heatmap.png
```

To use custom paths:

```bash
python prediction/plot/plot_bucket_significance_heatmap.py --input path/to/rolling_split_model_coefficients.csv --output-dir path/to/figures
```

## Input Fields

The raw CSV must contain:

```text
trade_date
target_time
bucket_label
daily_mispricing_signal
vega
daily_return
return_1d
```

Only observations with `target_time = 14:30` are used.

## Model Definitions

```text
baseline: pooled OLS using return_1d on daily_mispricing_signal
model1: bucket-level OLS using return_1d on daily_mispricing_signal
model2: bucket-level OLS using return_1d on daily_mispricing_signal and vega
model3: bucket-level OLS using return_1d on daily_mispricing_signal and current daily_return
```

The rolling splits are:

```text
Split A: train 2023-03-28 to 2023-06-30; test 2023-07-03 to 2023-08-31
Split B: train 2023-05-04 to 2023-08-31; test 2023-09-01 to 2023-10-31
Split C: train 2023-07-03 to 2023-10-31; test 2023-11-01 to 2023-11-30
```

## Notes

The heatmap uses `beta_signal` and `sig_beta_signal` from `rolling_split_model_coefficients.csv`. Positive significant coefficients are shown with `+` labels, negative significant coefficients are shown with `-` labels, and the number of stars follows the p-value thresholds used in the regression script.

Significance stars follow:

```text
*** p < 0.01
** p < 0.05
* p < 0.10
```

The out-of-sample R-squared in the performance files is computed as:

```text
oos_r_squared = 1 - SSE_model / SSE_benchmark
```
