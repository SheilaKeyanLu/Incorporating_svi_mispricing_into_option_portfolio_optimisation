from bisect import bisect_left
from datetime import datetime
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import TRADING_DATES


def load_input_data(path):
    return pd.read_csv(path)


def run_fixed_parameter_backtest(
    info_df,
    begin_date,
    end_date,
    parameters,
    optimisation_module,
    output_dir,
):
    trading_dates = [
        date for date in TRADING_DATES
        if begin_date <= date <= end_date
    ]

    if not trading_dates:
        raise ValueError(f"No trading dates between {begin_date} and {end_date}.")

    trading_dates = [
        date for date in trading_dates
        if bisect_left(TRADING_DATES, date) + 1 < len(TRADING_DATES)
    ]

    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    curve_dir = output_dir / "return_curve" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"backtest({begin_date}-{end_date}).log"
    previous_positions = {}
    cumulative_value = 1.0
    records = []

    start_time = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"{parameters['model_name']} fixed-parameter backtest\n")
        log.write(f"Period: {begin_date} - {end_date}\n")
        for key, value in parameters.items():
            if key != "model_name":
                log.write(f"{key} = {value}\n")
        log.write("=" * 80 + "\n\n")

        for date in trading_dates:
            print(
                f"[{parameters['model_name']}] Processing {date}, "
                f"current value: {cumulative_value:.4f}"
            )

            result = optimisation_module.optimize_daily_portfolio(
                info_df=info_df,
                date=date,
                previous_positions=previous_positions,
                lambda_risk=parameters["lambda_risk"],
                xi=parameters["xi"],
                tau=parameters["tau"],
                delta=parameters.get("delta", 0.0),
                fee=parameters["fee"],
                budget=parameters["budget"],
                max_weight=parameters["max_weight"],
                training_days=parameters["training_days"],
                regression_train_days=parameters["regression_train_days"],
                regression_test_days=parameters["regression_test_days"],
            )

            if result["w"] is None:
                raise RuntimeError(f"Optimization failed on {date}.")

            current_positions = result["positions"].copy()
            daily_return, next_date = optimisation_module.calculate_next_day_return(
                info_df=info_df,
                date=date,
                positions=current_positions,
            )

            cumulative_value *= 1 + daily_return
            nonzero_positions = {
                symbol: float(weight)
                for symbol, weight in current_positions.items()
                if abs(weight) > 1e-8
            }

            records.append({
                "date": date,
                "next_date": next_date,
                "daily_return": daily_return,
                "cumulative_value": cumulative_value,
                "objective": result["obj"],
                "positions": nonzero_positions,
            })

            log.write(f"Date: {date}\n")
            log.write(f"Next date: {next_date}\n")
            log.write("Positions:\n")
            for symbol, weight in nonzero_positions.items():
                log.write(f"  {symbol:<25}{weight:>12.6f}\n")
            log.write(f"Daily return: {daily_return:.6f} ({daily_return:.2%})\n")
            log.write(f"Cumulative value: {cumulative_value:.6f}\n")
            log.write(f"Cumulative return: {cumulative_value - 1:.6f} ({cumulative_value - 1:.2%})\n")
            log.write("-" * 80 + "\n\n")

            previous_positions = current_positions.copy()

        daily_results = pd.DataFrame(records)
        mean_return = daily_results["daily_return"].mean()
        volatility = daily_results["daily_return"].std(ddof=1)
        sharpe = mean_return / volatility if volatility > 0 else np.nan

        log.write("=" * 80 + "\n")
        log.write("BACKTEST SUMMARY\n")
        log.write("=" * 80 + "\n")
        log.write(f"Trading days: {len(records)}\n")
        log.write(f"Mean daily return: {mean_return:.6f}\n")
        log.write(f"Daily volatility: {volatility:.6f}\n")
        log.write(f"Sharpe: {sharpe:.6f}\n")
        log.write(f"Final value: {cumulative_value:.6f}\n")
        log.write(f"Final return: {cumulative_value - 1:.6f} ({cumulative_value - 1:.2%})\n")

    daily_results["date"] = pd.to_datetime(daily_results["date"])
    daily_results["next_date"] = pd.to_datetime(daily_results["next_date"])
    daily_results["cumulative_return"] = daily_results["cumulative_value"] - 1.0

    filename = f"{begin_date}-{end_date}"
    csv_path = curve_dir / f"{filename}_daily_results.csv"
    curve_path = curve_dir / f"{filename}_return_curve.png"
    daily_results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(daily_results["next_date"], daily_results["cumulative_return"], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{parameters['model_name']} Fixed Parameter Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    elapsed_seconds = time.perf_counter() - start_time
    runtime_path = curve_dir / "runtime.txt"
    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write(f"Elapsed seconds: {elapsed_seconds:.2f}\n")
        f.write(f"Elapsed minutes: {elapsed_seconds / 60:.2f}\n")

    return {
        "daily_results": daily_results,
        "final_value": cumulative_value,
        "final_return": cumulative_value - 1,
        "mean_daily_return": mean_return,
        "daily_volatility": volatility,
        "sharpe": sharpe,
        "final_positions": previous_positions,
        "log_path": log_path,
        "csv_path": csv_path,
        "curve_path": curve_path,
        "runtime_path": runtime_path,
    }
