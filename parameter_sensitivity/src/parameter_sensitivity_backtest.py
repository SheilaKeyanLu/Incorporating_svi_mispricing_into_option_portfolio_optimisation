import argparse
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from config import BUDGET, FEE, TRADING_DATES
from portofolio_optimisaiton import calculate_next_day_return, optimize_daily_portfolio


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_PATH = INPUT_DIR / "data_prepared_for_markowitz.csv"

BEGIN_DATE = "2025-01-01"
END_DATE = "2026-06-30"
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0

TAU_FIXED = 0.05
XI_FIXED = 1.0
MAX_WEIGHT = 0.3
TRAIN_DAYS = 20
REGRESSION_TRAIN_DAYS = 240
REGRESSION_TEST_DAYS = 20

LAMBDA_SWEEP = [10000, 12000, 14000, 16000, 18000, 20000]
DELTA_SWEEP = [3, 4, 5, 6, 7, 8, 9]


def build_run_plan():
    runs = [
        {
            "group": "lambda_sweep_delta_5",
            "run_id": f"lambda_{lambda_risk}_delta_5",
            "lambda_risk": lambda_risk,
            "delta": 5,
        }
        for lambda_risk in LAMBDA_SWEEP
    ]

    for delta in DELTA_SWEEP:
        run_id = (
            "lambda_10000_delta_5_delta_sweep"
            if delta == 5
            else f"lambda_10000_delta_{delta}"
        )
        runs.append(
            {
                "group": "delta_sweep_lambda_10000",
                "run_id": run_id,
                "lambda_risk": 10000,
                "delta": delta,
            }
        )
    return runs


def available_backtest_dates(begin_date, end_date):
    dates = [date for date in TRADING_DATES if begin_date <= date <= end_date]
    return [
        date
        for date in dates
        if bisect_left(TRADING_DATES, date) + 1 < len(TRADING_DATES)
    ]


def run_backtest(info_df, params):
    trading_dates = available_backtest_dates(BEGIN_DATE, END_DATE)
    previous_positions = {}
    cumulative_value = 1.0
    records = []

    for date in trading_dates:
        result = optimize_daily_portfolio(
            info_df=info_df,
            date=date,
            previous_positions=previous_positions,
            lambda_risk=params["lambda_risk"],
            xi=XI_FIXED,
            tau=TAU_FIXED,
            delta=params["delta"],
            fee=FEE,
            budget=BUDGET,
            max_weight=MAX_WEIGHT,
            training_days=TRAIN_DAYS,
            regression_train_days=REGRESSION_TRAIN_DAYS,
            regression_test_days=REGRESSION_TEST_DAYS,
        )
        if result["w"] is None:
            raise RuntimeError(f"Optimization failed on {date}.")

        current_positions = result["positions"].copy()
        daily_return, next_date = calculate_next_day_return(
            info_df=info_df,
            date=date,
            positions=current_positions,
        )
        cumulative_value *= 1.0 + daily_return
        records.append(
            {
                "date": date,
                "next_date": next_date,
                "daily_return": daily_return,
                "cumulative_value": cumulative_value,
                "cumulative_return": cumulative_value - 1.0,
                "objective": result["obj"],
                "positions": json.dumps(
                    {
                        symbol: float(weight)
                        for symbol, weight in current_positions.items()
                        if abs(weight) > 1e-8
                    },
                    ensure_ascii=False,
                ),
            }
        )
        previous_positions = current_positions.copy()

    return pd.DataFrame(records)


def compute_metrics(daily_results):
    returns = daily_results["daily_return"].astype(float)
    cumulative_value = daily_results["cumulative_value"].astype(float)
    n_days = len(returns)
    total_return = cumulative_value.iloc[-1] - 1.0
    annual_return = cumulative_value.iloc[-1] ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
    annual_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf
    sharpe_ratio = (
        excess_returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if returns.std(ddof=1) > 0
        else np.nan
    )
    downside_returns = np.minimum(excess_returns, 0.0)
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    sortino_ratio = (
        (returns.mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE) / downside_deviation
        if downside_deviation > 0
        else np.nan
    )
    drawdown = cumulative_value / cumulative_value.cummax() - 1.0
    max_drawdown = drawdown.min()
    return {
        "trading_days": n_days,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "win_rate": (returns > 0).mean(),
        "final_value": cumulative_value.iloc[-1],
        "mean_daily_return": returns.mean(),
        "min_daily_return": returns.min(),
        "max_daily_return": returns.max(),
    }


def save_combined_return_table(all_daily_results, output_dir):
    combined = None
    for run_id, daily in all_daily_results.items():
        part = daily[["date", "next_date", "daily_return", "cumulative_return"]].copy()
        part = part.rename(
            columns={
                "daily_return": f"{run_id}_daily_return",
                "cumulative_return": f"{run_id}_cumulative_return",
            }
        )
        combined = part if combined is None else combined.merge(
            part, on=["date", "next_date"], how="outer"
        )

    output_path = output_dir / "all_runs_return_changes.csv"
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    data_path = args.input_dir / "data_prepared_for_markowitz.csv"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    info_df = pd.read_csv(data_path)
    records = []
    all_daily_results = {}

    for params in build_run_plan():
        daily_results = run_backtest(info_df, params)
        all_daily_results[params["run_id"]] = daily_results
        records.append({**params, **compute_metrics(daily_results)})

    summary = pd.DataFrame(records)
    summary.to_csv(
        output_dir / "parameter_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_combined_return_table(all_daily_results, output_dir)


if __name__ == "__main__":
    main()
