import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(__file__).parent
RAW_DATA_DIR = BASE / "data" / "raw"
PROCESSED_DATA_DIR = BASE / "data" / "processed"
REPORT_DIR = BASE / "output" / "report"
OUTPUT_DIR = PROCESSED_DATA_DIR
INPUT_PATH = RAW_DATA_DIR / "option_quotes_for_ruturn_prediction_model.csv"

COEFFICIENT_OUTPUT_PATH = OUTPUT_DIR / "rolling_split_model_coefficients.csv"
PERFORMANCE_OUTPUT_PATH = OUTPUT_DIR / "rolling_split_model_performance.csv"
AGGREGATE_OUTPUT_PATH = OUTPUT_DIR / "rolling_split_aggregate_performance.csv"
PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "rolling_split_predictions.csv"
REPORT_OUTPUT_PATH = REPORT_DIR / "rolling_split_regression_report.txt"

DATE_COL = "trade_date"
TARGET_TIME_COL = "target_time"
TARGET_TIME = "14:30"
BUCKET_COL = "bucket_label"
SIGNAL_COL = "daily_mispricing_signal"
VEGA_COL = "vega"
CURRENT_RETURN_COL = "daily_return"

HORIZONS = {
    1: "return_1d",
}

MODELS = {
    "baseline": {
        "description": "Pooled OLS baseline: return_1d on daily_mispricing_signal",
        "x_cols": [SIGNAL_COL],
        "coef_names": ["alpha", "beta_signal"],
        "group_by_bucket": False,
    },
    "model1": {
        "description": "Bucket OLS: return_1d on daily_mispricing_signal",
        "x_cols": [SIGNAL_COL],
        "coef_names": ["alpha", "beta_signal"],
        "group_by_bucket": True,
    },
    "model2": {
        "description": "Bucket OLS: return_1d on daily_mispricing_signal and vega",
        "x_cols": [SIGNAL_COL, VEGA_COL],
        "coef_names": ["alpha", "beta_signal", "gamma_vega"],
        "group_by_bucket": True,
    },
    "model3": {
        "description": "Bucket OLS: return_1d on daily_mispricing_signal and current daily_return",
        "x_cols": [SIGNAL_COL, CURRENT_RETURN_COL],
        "coef_names": ["alpha", "beta_signal", "delta_current_return"],
        "group_by_bucket": True,
    },
}

SPLITS = [
    {
        "split": "A",
        "train_start": "2023-03-28",
        "train_end": "2023-06-30",
        "test_start": "2023-07-03",
        "test_end": "2023-08-31",
    },
    {
        "split": "B",
        "train_start": "2023-05-04",
        "train_end": "2023-08-31",
        "test_start": "2023-09-01",
        "test_end": "2023-10-31",
    },
    {
        "split": "C",
        "train_start": "2023-07-03",
        "train_end": "2023-10-31",
        "test_start": "2023-11-01",
        "test_end": "2023-11-30",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit Model 1/2/3 for all predefined rolling out-of-sample splits."
    )
    parser.add_argument(
        "--input",
        default=str(INPUT_PATH),
        help="Input CSV path. Default: ./data/raw/option_quotes_for_ruturn_prediction_model.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Processed CSV output directory. Default: ./data/processed",
    )
    parser.add_argument(
        "--report-dir",
        default=str(REPORT_DIR),
        help="Text report output directory. Default: ./output/report",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip row-level prediction CSV. Performance outputs are still written.",
    )
    return parser.parse_args()


def normalize_splits():
    normalized = []
    for split in SPLITS:
        item = {
            **split,
            "train_start_ts": pd.Timestamp(split["train_start"]),
            "train_end_ts": pd.Timestamp(split["train_end"]),
            "test_start_ts": pd.Timestamp(split["test_start"]),
            "test_end_ts": pd.Timestamp(split["test_end"]),
        }
        if not item["train_start_ts"] <= item["train_end_ts"] < item["test_start_ts"] <= item["test_end_ts"]:
            raise ValueError(f"Invalid split dates for split {item['split']}.")
        normalized.append(item)
    return normalized


def fmt_num(value, digits=6):
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 0.0001:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def sig_mark(p_value):
    if p_value is None or pd.isna(p_value):
        return ""
    p_value = float(p_value)
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def text_table(rows, columns, headers=None):
    headers = headers or columns
    rendered = []
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(fmt_num(value) if isinstance(value, float) else str(value))
        rendered.append(vals)

    widths = [len(header) for header in headers]
    for vals in rendered:
        widths = [max(width, len(value)) for width, value in zip(widths, vals)]

    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    sep_line = "-+-".join("-" * width for width in widths)
    body = [" | ".join(value.ljust(width) for value, width in zip(vals, widths)) for vals in rendered]
    return "\n".join([header_line, sep_line] + body)


def empty_train_acc(k):
    return {
        "n": 0,
        "xtx": np.zeros((k + 1, k + 1), dtype=float),
        "xty": np.zeros(k + 1, dtype=float),
        "yty": 0.0,
        "sum_y": 0.0,
        "sum_y2": 0.0,
    }


def empty_test_acc():
    return {
        "n_test": 0,
        "sum_actual": 0.0,
        "sum_actual2": 0.0,
        "sse_model": 0.0,
        "sse_benchmark": 0.0,
        "sae_model": 0.0,
        "sae_benchmark": 0.0,
        "sum_pred": 0.0,
        "sum_pred2": 0.0,
        "sum_actual_pred": 0.0,
    }


def fit_model(model_name, split, bucket, horizon, return_col, acc):
    model_spec = MODELS[model_name]
    coef_names = model_spec["coef_names"]
    k = len(model_spec["x_cols"])
    n = acc["n"]
    result = {
        "split": split["split"],
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "test_start": split["test_start"],
        "test_end": split["test_end"],
        "model": model_name,
        "bucket_label": bucket,
        "horizon": horizon,
        "return_column": return_col,
        "n_train": n,
        "df_resid": n - k - 1,
        "r_squared_train": math.nan,
        "adjusted_r_squared_train": math.nan,
        "train_mean_return": math.nan,
        "fit_status": "ok",
    }
    for name in coef_names:
        result[name] = math.nan
    for name in coef_names[1:]:
        result[f"std_error_{name}"] = math.nan
        result[f"t_stat_{name}"] = math.nan
        result[f"p_value_{name}"] = math.nan
        result[f"sig_{name}"] = ""

    if n <= k + 1:
        result["fit_status"] = "insufficient_train_observations"
        return result

    try:
        xtx_inv = np.linalg.inv(acc["xtx"])
    except np.linalg.LinAlgError:
        result["fit_status"] = "singular_design_matrix"
        return result

    params = xtx_inv @ acc["xty"]
    for name, value in zip(coef_names, params):
        result[name] = float(value)

    yty = acc["yty"]
    sse = yty - float(params @ acc["xty"])
    if sse < 0 and abs(sse) < 1e-8:
        sse = 0.0

    y_mean = acc["sum_y"] / n
    tss = yty - n * y_mean * y_mean
    r_squared = 1.0 - sse / tss if tss > 0 else math.nan
    df_resid = n - k - 1
    adjusted_r_squared = (
        1.0 - (1.0 - r_squared) * (n - 1) / df_resid
        if df_resid > 0 and not math.isnan(r_squared)
        else math.nan
    )
    sigma2 = sse / df_resid if df_resid > 0 else math.nan
    cov = sigma2 * xtx_inv if sigma2 >= 0 else np.full_like(xtx_inv, math.nan)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    for idx, name in enumerate(coef_names[1:], start=1):
        std_error = float(se[idx])
        t_stat = float(params[idx] / std_error) if std_error > 0 else math.nan
        p_value = (
            float(2.0 * stats.t.sf(abs(t_stat), df=df_resid))
            if not math.isnan(t_stat)
            else math.nan
        )
        result[f"std_error_{name}"] = std_error
        result[f"t_stat_{name}"] = t_stat
        result[f"p_value_{name}"] = p_value
        result[f"sig_{name}"] = sig_mark(p_value)

    result["r_squared_train"] = float(r_squared)
    result["adjusted_r_squared_train"] = float(adjusted_r_squared)
    result["train_mean_return"] = float(y_mean)
    return result


def coefficient_vector(fit):
    model_spec = MODELS[fit["model"]]
    values = [fit.get(name, math.nan) for name in model_spec["coef_names"]]
    if any(pd.isna(value) for value in values):
        return None
    return np.array(values, dtype=float)


def required_columns():
    return {
        DATE_COL,
        TARGET_TIME_COL,
        BUCKET_COL,
        SIGNAL_COL,
        VEGA_COL,
        CURRENT_RETURN_COL,
        *HORIZONS.values(),
    }


def model_groups(temp, spec):
    if spec.get("group_by_bucket", True):
        return temp.groupby(BUCKET_COL, sort=True)
    return [("ALL", temp)]


def build_train_stats(input_path, splits):
    train_stats = {
        split["split"]: {
            model_name: defaultdict(lambda k=len(spec["x_cols"]): empty_train_acc(k))
            for model_name, spec in MODELS.items()
        }
        for split in splits
    }
    counters = {
        "rows_read": 0,
        **{f"rows_train_window_{split['split']}": 0 for split in splits},
    }

    for chunk in pd.read_csv(
        input_path,
        usecols=list(required_columns()),
        chunksize=200_000,
        low_memory=False,
    ):
        counters["rows_read"] += len(chunk)
        trade_dates = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        time_mask = chunk[TARGET_TIME_COL].astype(str) == TARGET_TIME
        if not time_mask.any():
            continue

        for col in [SIGNAL_COL, VEGA_COL, CURRENT_RETURN_COL, *HORIZONS.values()]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        for split in splits:
            split_label = split["split"]
            mask = (
                time_mask
                & (trade_dates >= split["train_start_ts"])
                & (trade_dates <= split["train_end_ts"])
            )
            train_chunk = chunk.loc[mask].copy()
            counters[f"rows_train_window_{split_label}"] += len(train_chunk)
            if train_chunk.empty:
                continue

            for model_name, spec in MODELS.items():
                x_cols = spec["x_cols"]
                for horizon, y_col in HORIZONS.items():
                    cols = [*x_cols, y_col]
                    if spec.get("group_by_bucket", True):
                        cols = [BUCKET_COL, *cols]
                    temp = train_chunk[cols].dropna(subset=cols)
                    if temp.empty:
                        continue

                    for bucket, sub in model_groups(temp, spec):
                        x = sub[x_cols].to_numpy(dtype=float)
                        y = sub[y_col].to_numpy(dtype=float)
                        design = np.column_stack([np.ones(len(sub)), x])
                        acc = train_stats[split_label][model_name][(str(bucket), horizon)]
                        acc["n"] += len(sub)
                        acc["xtx"] += design.T @ design
                        acc["xty"] += design.T @ y
                        acc["yty"] += float(y @ y)
                        acc["sum_y"] += float(y.sum())
                        acc["sum_y2"] += float(y @ y)

    return train_stats, counters


def fit_all_models(train_stats, splits):
    fits = []
    fit_lookup = {}
    split_by_label = {split["split"]: split for split in splits}
    for split_label, stats_for_split in train_stats.items():
        split = split_by_label[split_label]
        for model_name, stats_by_key in stats_for_split.items():
            for (bucket, horizon), acc in sorted(
                stats_by_key.items(), key=lambda item: (item[0][0], item[0][1])
            ):
                fit = fit_model(model_name, split, bucket, horizon, HORIZONS[horizon], acc)
                fits.append(fit)
                fit_lookup[(split_label, model_name, bucket, horizon)] = fit
    return fits, fit_lookup


def write_prediction_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "model",
                "trade_date",
                "target_time",
                "bucket_label",
                "horizon",
                "return_column",
                "actual_return",
                "predicted_return",
                "benchmark_prediction",
                "prediction_error",
                "benchmark_error",
            ],
            lineterminator="\n",
        )
        writer.writeheader()


def append_predictions(path, rows):
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writerows(rows)


def evaluate_out_of_sample(input_path, prediction_path, splits, fit_lookup, write_predictions):
    test_stats = defaultdict(empty_test_acc)
    counters = {
        **{f"rows_test_window_{split['split']}": 0 for split in splits},
        **{f"prediction_rows_{split['split']}": 0 for split in splits},
    }
    if write_predictions:
        write_prediction_header(prediction_path)

    for chunk in pd.read_csv(
        input_path,
        usecols=list(required_columns()),
        chunksize=200_000,
        low_memory=False,
    ):
        trade_dates = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        time_mask = chunk[TARGET_TIME_COL].astype(str) == TARGET_TIME
        if not time_mask.any():
            continue

        chunk["_trade_date_string"] = trade_dates.dt.strftime("%Y-%m-%d")
        for col in [SIGNAL_COL, VEGA_COL, CURRENT_RETURN_COL, *HORIZONS.values()]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        prediction_rows = []
        for split in splits:
            split_label = split["split"]
            mask = (
                time_mask
                & (trade_dates >= split["test_start_ts"])
                & (trade_dates <= split["test_end_ts"])
            )
            test_chunk = chunk.loc[mask].copy()
            counters[f"rows_test_window_{split_label}"] += len(test_chunk)
            if test_chunk.empty:
                continue

            for model_name, spec in MODELS.items():
                x_cols = spec["x_cols"]
                for horizon, y_col in HORIZONS.items():
                    cols = [DATE_COL, "_trade_date_string", TARGET_TIME_COL, *x_cols, y_col]
                    dropna_cols = [*x_cols, y_col]
                    if spec.get("group_by_bucket", True):
                        cols.insert(3, BUCKET_COL)
                        dropna_cols.insert(0, BUCKET_COL)
                    temp = test_chunk[cols].dropna(subset=dropna_cols)
                    if temp.empty:
                        continue

                    for bucket, sub in model_groups(temp, spec):
                        fit = fit_lookup.get((split_label, model_name, str(bucket), horizon))
                        if fit is None or fit.get("fit_status") != "ok":
                            continue
                        params = coefficient_vector(fit)
                        if params is None:
                            continue

                        x = sub[x_cols].to_numpy(dtype=float)
                        y = sub[y_col].to_numpy(dtype=float)
                        design = np.column_stack([np.ones(len(sub)), x])
                        pred = design @ params
                        benchmark = np.full(len(sub), fit["train_mean_return"], dtype=float)

                        err = y - pred
                        bench_err = y - benchmark
                        key = (split_label, model_name, str(bucket), horizon)
                        acc = test_stats[key]
                        acc["n_test"] += len(sub)
                        acc["sum_actual"] += float(y.sum())
                        acc["sum_actual2"] += float(y @ y)
                        acc["sse_model"] += float(err @ err)
                        acc["sse_benchmark"] += float(bench_err @ bench_err)
                        acc["sae_model"] += float(np.abs(err).sum())
                        acc["sae_benchmark"] += float(np.abs(bench_err).sum())
                        acc["sum_pred"] += float(pred.sum())
                        acc["sum_pred2"] += float(pred @ pred)
                        acc["sum_actual_pred"] += float(y @ pred)
                        counters[f"prediction_rows_{split_label}"] += len(sub)

                        if not write_predictions:
                            continue
                        for i, (_, row) in enumerate(sub.iterrows()):
                            prediction_rows.append(
                                {
                                    "split": split_label,
                                    "train_start": split["train_start"],
                                    "train_end": split["train_end"],
                                    "test_start": split["test_start"],
                                    "test_end": split["test_end"],
                                    "model": model_name,
                                    "trade_date": row["_trade_date_string"],
                                    "target_time": row[TARGET_TIME_COL],
                                    "bucket_label": str(bucket),
                                    "horizon": horizon,
                                    "return_column": y_col,
                                    "actual_return": y[i],
                                    "predicted_return": pred[i],
                                    "benchmark_prediction": benchmark[i],
                                    "prediction_error": err[i],
                                    "benchmark_error": bench_err[i],
                                }
                            )
        if write_predictions:
            append_predictions(prediction_path, prediction_rows)

    return test_stats, counters


def build_performance_rows(fits, test_stats):
    rows = []
    for fit in fits:
        key = (fit["split"], fit["model"], fit["bucket_label"], fit["horizon"])
        acc = test_stats.get(key, empty_test_acc())
        n = acc["n_test"]
        rmse = math.sqrt(acc["sse_model"] / n) if n > 0 else math.nan
        benchmark_rmse = math.sqrt(acc["sse_benchmark"] / n) if n > 0 else math.nan
        mae = acc["sae_model"] / n if n > 0 else math.nan
        benchmark_mae = acc["sae_benchmark"] / n if n > 0 else math.nan
        oos_r2 = (
            1.0 - acc["sse_model"] / acc["sse_benchmark"]
            if acc["sse_benchmark"] > 0
            else math.nan
        )
        actual_mean = acc["sum_actual"] / n if n > 0 else math.nan
        pred_mean = acc["sum_pred"] / n if n > 0 else math.nan
        actual_tss = acc["sum_actual2"] - acc["sum_actual"] * acc["sum_actual"] / n if n > 0 else math.nan
        pred_tss = acc["sum_pred2"] - acc["sum_pred"] * acc["sum_pred"] / n if n > 0 else math.nan
        actual_pred_cov = (
            acc["sum_actual_pred"] - acc["sum_actual"] * acc["sum_pred"] / n if n > 0 else math.nan
        )
        pred_corr = (
            actual_pred_cov / math.sqrt(actual_tss * pred_tss)
            if n > 1 and actual_tss > 0 and pred_tss > 0
            else math.nan
        )
        rows.append(
            {
                "split": fit["split"],
                "train_start": fit["train_start"],
                "train_end": fit["train_end"],
                "test_start": fit["test_start"],
                "test_end": fit["test_end"],
                "model": fit["model"],
                "bucket_label": fit["bucket_label"],
                "horizon": fit["horizon"],
                "return_column": fit["return_column"],
                "fit_status": fit["fit_status"],
                "n_train": fit["n_train"],
                "n_test": n,
                "train_adj_r2": fit["adjusted_r_squared_train"],
                "rmse": rmse,
                "benchmark_rmse": benchmark_rmse,
                "mae": mae,
                "benchmark_mae": benchmark_mae,
                "oos_r_squared": oos_r2,
                "actual_mean": actual_mean,
                "predicted_mean": pred_mean,
                "prediction_correlation": pred_corr,
                "sse_model": acc["sse_model"],
                "sse_benchmark": acc["sse_benchmark"],
            }
        )
    return rows


def aggregate_performance(performance_rows):
    aggregate_rows = []
    df = pd.DataFrame(performance_rows)
    if df.empty:
        return aggregate_rows

    for (split_label, model_name), sub in df.groupby(["split", "model"], sort=True):
        valid = sub.loc[sub["n_test"] > 0].copy()
        row_base = {
            "split": split_label,
            "train_start": sub["train_start"].iloc[0],
            "train_end": sub["train_end"].iloc[0],
            "test_start": sub["test_start"].iloc[0],
            "test_end": sub["test_end"].iloc[0],
            "model": model_name,
        }
        if valid.empty:
            aggregate_rows.append(
                {
                    **row_base,
                    "buckets": 0,
                    "total_n_train": 0,
                    "total_n_test": 0,
                    "rmse": math.nan,
                    "benchmark_rmse": math.nan,
                    "mae": math.nan,
                    "benchmark_mae": math.nan,
                    "oos_r_squared": math.nan,
                    "mean_train_adj_r2": math.nan,
                }
            )
            continue

        total_n = int(valid["n_test"].sum())
        sse_model = float(valid["sse_model"].sum())
        sse_benchmark = float(valid["sse_benchmark"].sum())
        sae_model = float((valid["mae"] * valid["n_test"]).sum())
        sae_benchmark = float((valid["benchmark_mae"] * valid["n_test"]).sum())
        aggregate_rows.append(
            {
                **row_base,
                "buckets": len(valid),
                "total_n_train": int(valid["n_train"].sum()),
                "total_n_test": total_n,
                "rmse": math.sqrt(sse_model / total_n) if total_n > 0 else math.nan,
                "benchmark_rmse": math.sqrt(sse_benchmark / total_n) if total_n > 0 else math.nan,
                "mae": sae_model / total_n if total_n > 0 else math.nan,
                "benchmark_mae": sae_benchmark / total_n if total_n > 0 else math.nan,
                "oos_r_squared": (
                    1.0 - sse_model / sse_benchmark if sse_benchmark > 0 else math.nan
                ),
                "mean_train_adj_r2": float(valid["train_adj_r2"].mean()),
            }
        )
    return aggregate_rows


def build_training_coefficient_rows(fits):
    rows = []
    for fit in fits:
        rows.append(
            {
                "split": fit["split"],
                "model": fit["model"],
                "bucket_label": fit["bucket_label"],
                "n_train": fit["n_train"],
                "fit_status": fit["fit_status"],
                "beta_signal": fit.get("beta_signal", math.nan),
                "t_stat_beta_signal": fit.get("t_stat_beta_signal", math.nan),
                "p_value_beta_signal": fit.get("p_value_beta_signal", math.nan),
                "sig_beta_signal": fit.get("sig_beta_signal", ""),
                "gamma_vega": fit.get("gamma_vega", math.nan),
                "t_stat_gamma_vega": fit.get("t_stat_gamma_vega", math.nan),
                "delta_current_return": fit.get("delta_current_return", math.nan),
                "t_stat_delta_current_return": fit.get("t_stat_delta_current_return", math.nan),
                "train_adj_r2": fit["adjusted_r_squared_train"],
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    paths,
    splits,
    counters,
    performance_rows,
    aggregate_rows,
    training_coefficient_rows,
    predictions_written,
):
    lines = [
        "Rolling Split Out-of-Sample Regression Report",
        "=" * 45,
        "",
        "1. Sample Splits",
        text_table(
            splits,
            ["split", "train_start", "train_end", "test_start", "test_end"],
        ),
        "",
        "2. Models",
    ]
    for model_name, spec in MODELS.items():
        lines.append(f"{model_name}: {spec['description']}")

    count_rows = []
    for split in splits:
        label = split["split"]
        count_rows.append(
            {
                "split": label,
                "rows_train_window": counters[f"rows_train_window_{label}"],
                "rows_test_window": counters[f"rows_test_window_{label}"],
                "prediction_rows": counters[f"prediction_rows_{label}"],
            }
        )

    lines.extend(
        [
            "",
            "3. Row Counts",
            f"Input rows read for training pass: {counters['rows_read']}",
            text_table(count_rows, ["split", "rows_train_window", "rows_test_window", "prediction_rows"]),
            "",
            "4. Aggregate Out-of-Sample Performance",
            text_table(
                aggregate_rows,
                [
                    "split",
                    "model",
                    "buckets",
                    "total_n_train",
                    "total_n_test",
                    "rmse",
                    "benchmark_rmse",
                    "mae",
                    "benchmark_mae",
                    "oos_r_squared",
                    "mean_train_adj_r2",
                ],
            ),
            "",
            "5. Training-Window Coefficients",
            "Focus variable: beta_signal is the coefficient on daily_mispricing_signal.",
            text_table(
                training_coefficient_rows,
                [
                    "split",
                    "model",
                    "bucket_label",
                    "n_train",
                    "fit_status",
                    "beta_signal",
                    "t_stat_beta_signal",
                    "p_value_beta_signal",
                    "sig_beta_signal",
                    "gamma_vega",
                    "t_stat_gamma_vega",
                    "delta_current_return",
                    "t_stat_delta_current_return",
                    "train_adj_r2",
                ],
            ),
            "",
            "6. Bucket-Level Out-of-Sample Performance",
            text_table(
                performance_rows,
                [
                    "split",
                    "model",
                    "bucket_label",
                    "n_train",
                    "n_test",
                    "train_adj_r2",
                    "rmse",
                    "benchmark_rmse",
                    "mae",
                    "benchmark_mae",
                    "oos_r_squared",
                    "prediction_correlation",
                ],
            ),
            "",
            "7. Output Files",
            f"Coefficient file: {paths['coefficients']}",
            f"Performance file: {paths['performance']}",
            f"Aggregate performance file: {paths['aggregate']}",
            f"Prediction file: {paths['predictions'] if predictions_written else 'not written (--no-predictions)'}",
            f"Report file: {paths['report']}",
            "",
            "8. Notes",
            "The benchmark prediction is the in-sample mean return for the same split, model, and bucket.",
            "oos_r_squared = 1 - SSE_model / SSE_benchmark.",
            "A positive oos_r_squared means the fitted model improves on the in-sample-mean benchmark out of sample.",
        ]
    )
    paths["report"].write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    paths = {
        "coefficients": output_dir / COEFFICIENT_OUTPUT_PATH.name,
        "performance": output_dir / PERFORMANCE_OUTPUT_PATH.name,
        "aggregate": output_dir / AGGREGATE_OUTPUT_PATH.name,
        "predictions": output_dir / PREDICTION_OUTPUT_PATH.name,
        "report": report_dir / REPORT_OUTPUT_PATH.name,
    }
    splits = normalize_splits()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_stats, train_counters = build_train_stats(input_path, splits)
    fits, fit_lookup = fit_all_models(train_stats, splits)
    test_stats, test_counters = evaluate_out_of_sample(
        input_path,
        paths["predictions"],
        splits,
        fit_lookup,
        write_predictions=not args.no_predictions,
    )
    performance_rows = build_performance_rows(fits, test_stats)
    aggregate_rows = aggregate_performance(performance_rows)
    training_coefficient_rows = build_training_coefficient_rows(fits)

    coefficient_fieldnames = [
        "split",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        *sorted(
            {
                key
                for row in fits
                for key in row.keys()
                if key not in {"split", "train_start", "train_end", "test_start", "test_end"}
            }
        ),
    ]
    performance_fieldnames = [
        "split",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "model",
        "bucket_label",
        "horizon",
        "return_column",
        "fit_status",
        "n_train",
        "n_test",
        "train_adj_r2",
        "rmse",
        "benchmark_rmse",
        "mae",
        "benchmark_mae",
        "oos_r_squared",
        "actual_mean",
        "predicted_mean",
        "prediction_correlation",
        "sse_model",
        "sse_benchmark",
    ]
    aggregate_fieldnames = [
        "split",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "model",
        "buckets",
        "total_n_train",
        "total_n_test",
        "rmse",
        "benchmark_rmse",
        "mae",
        "benchmark_mae",
        "oos_r_squared",
        "mean_train_adj_r2",
    ]

    write_csv(paths["coefficients"], fits, coefficient_fieldnames)
    write_csv(paths["performance"], performance_rows, performance_fieldnames)
    write_csv(paths["aggregate"], aggregate_rows, aggregate_fieldnames)

    counters = {
        **train_counters,
        **test_counters,
    }
    write_report(
        paths,
        splits,
        counters,
        performance_rows,
        aggregate_rows,
        training_coefficient_rows,
        not args.no_predictions,
    )

    print(f"splits={','.join(split['split'] for split in splits)}")
    print(f"rows_read_training_pass={counters['rows_read']}")
    for split in splits:
        label = split["split"]
        print(
            f"split={label} "
            f"train={split['train_start']}..{split['train_end']} "
            f"test={split['test_start']}..{split['test_end']} "
            f"rows_train={counters[f'rows_train_window_{label}']} "
            f"rows_test={counters[f'rows_test_window_{label}']} "
            f"prediction_rows={counters[f'prediction_rows_{label}']}"
        )
    print(f"models_fit={len(fits)}")
    print(f"output_dir={output_dir}")
    print(f"coefficients={paths['coefficients']}")
    print(f"performance={paths['performance']}")
    print(f"aggregate={paths['aggregate']}")
    if not args.no_predictions:
        print(f"predictions={paths['predictions']}")
    print(f"report={paths['report']}")


if __name__ == "__main__":
    main()
