import pandas as pd
from bisect import bisect_left
from itertools import product
from pathlib import Path
from config import TRADING_DATES, TAU, TRAIN_DAYS, FEE, BUDGET, MAX_WEIGHT, XI_GRID
from config import REGRESSION_TRAIN_DAYS, REGRESSION_TEST_DAYS, LAMBDA_RISK_GRID, TAU_GRID, DELTA_GRID
from config import TUNING_DAYS
from optimizer import solve_portfolio
from preprocessing import build_bl_posterior, build_margin_vector, build_eta, get_daily_contract_index
import numpy as np




def previous_trading_days(date, days, trading_dates=TRADING_DATES):
    if days < 0:
        raise ValueError("days must be non-negative.")
    trading_dates = trading_dates
    index = bisect_left(trading_dates, date)
    if index < days:
        raise ValueError(f"Not enough trading dates before {date}.")

    return trading_dates[index - days:index]



def optimize_daily_portfolio(
    info_df,
    date,
    previous_positions,
    lambda_risk,
    xi,
    tau,
    delta,
    fee,
    budget,
    max_weight,
    training_days=TRAIN_DAYS,
    regression_train_days=REGRESSION_TRAIN_DAYS,
    regression_test_days=REGRESSION_TEST_DAYS,
):
    bl = build_bl_posterior(info_df=info_df, date=date, training_days=training_days, regression_train_days=regression_train_days,
        regression_test_days=regression_test_days, tau=tau, delta=delta,)

    margin, index_margin = build_margin_vector(info_df, date)
    eta, index_eta = build_eta(info_df, date)

    contract_index = bl["contract_index"]

    # 检验是否 contract_index, index_margin, index_eta 对齐
    if not (contract_index == index_margin == index_eta):
        raise ValueError("Contract indices are not aligned.")

    w_prev = np.array([previous_positions.get(symbol, 0.0) for symbol in contract_index],dtype=float,)
    
    result = solve_portfolio(
        u_BL=bl["u_BL"],
        A_BL=bl["A_BL"],
        w_prev=w_prev,
        transaction_eta=eta,
        margin=margin,
        lambda_risk=lambda_risk,
        xi=xi,
        fee=fee,
        budget=budget,
        max_weight=max_weight,)

    result["contract_index"] = contract_index
    result["positions"] = (dict(zip(contract_index, result["w"]))
        if result["w"] is not None else None)
    result["bl"] = bl

    return result


def calculate_next_day_return(info_df, date, positions):
    # return_1d 已经是 "当日 -> 次日" 的收益，直接取调仓日 date 当天的行
    today_df, _ = get_daily_contract_index(info_df, date)
    today_return = dict(
        zip(today_df["symbol"],
            today_df["return_1d"].to_numpy(dtype=float),))

    portfolio_return = sum(
        weight * today_return.get(symbol, 0.0)
        for symbol, weight in positions.items()
    )

    # next_date 仅用于记录展示，不再用于取收益
    next_idx = bisect_left(TRADING_DATES, date) + 1
    if next_idx >= len(TRADING_DATES):
        raise ValueError(f"No next trading date after {date}.")
    next_date = TRADING_DATES[next_idx]

    return portfolio_return, next_date

    
def run_backtest_window(info_df,end_date,tuning_days,lambda_risk,xi,
    fee,budget,max_weight,training_days=TRAIN_DAYS,regression_train_days=REGRESSION_TRAIN_DAYS,
    regression_test_days=REGRESSION_TEST_DAYS,tau=TAU,delta=1.0,):

    trade_dates = previous_trading_days(end_date, tuning_days)
    previous_positions = {}
    daily_results = []
    cumulative_value = 1.0

    for date in trade_dates:
        result = optimize_daily_portfolio(
            info_df=info_df,
            date=date,
            previous_positions=previous_positions,
            lambda_risk=lambda_risk,
            xi=xi,
            fee=fee,
            budget=budget,
            max_weight=max_weight,
            training_days=training_days,
            regression_train_days=regression_train_days,
            regression_test_days=regression_test_days,
            tau=tau,
            delta=delta,)

        if result["w"] is None:
            raise RuntimeError(f"Optimization failed on {date}.")

        previous_positions = result["positions"]
        daily_return, next_date = calculate_next_day_return(info_df, date, previous_positions,)
        cumulative_value *= 1 + daily_return

        daily_results.append({
            "date": date,
            "next_date": next_date,
            "daily_return": daily_return,
            "cumulative_value": cumulative_value,
            "objective": result["obj"],
            "positions": previous_positions.copy(),
        })

    return {
        "daily_results": pd.DataFrame(daily_results),
        "final_return": cumulative_value - 1,
        "final_value": cumulative_value,
        "final_positions": previous_positions,
    }


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
info_df = pd.read_csv(PROJECT_DIR / "input" / "data_prepared_for_markowitz.csv")
# result = run_backtest_window(
#     info_df=info_df,
#     end_date="2024-06-05",
#     tuning_days=4,
#     lambda_risk=1.0,
#     xi=1,
#     fee=15,
#     budget=10_000_000,
#     max_weight=2.0,
#     tau=0.05,
#     delta=1.0,)

# print(result["final_return"])


def grid_search_backtest(
    info_df,
    end_date,
    tuning_days,
    lambda_risk_grid,
    tau_grid,
    delta_grid,
    xi_grid,
    fee=FEE,
    budget=BUDGET,
    max_weight=MAX_WEIGHT,
    training_days=TRAIN_DAYS,
    regression_train_days=REGRESSION_TRAIN_DAYS,
    regression_test_days=REGRESSION_TEST_DAYS,
):
    records = []

    for lambda_risk, tau, delta, xi in product(
        lambda_risk_grid,
        tau_grid,
        delta_grid,
        xi_grid,
    ):
        try:
            result = run_backtest_window(
                info_df=info_df,
                end_date=end_date,
                tuning_days=tuning_days,
                lambda_risk=lambda_risk,
                xi=xi,
                fee=fee,
                budget=budget,
                max_weight=max_weight,
                tau=tau,
                training_days=training_days,
                regression_train_days=regression_train_days,
                regression_test_days=regression_test_days,
                delta=delta,
            )

            daily = result["daily_results"]
            mean_return = daily["daily_return"].mean()
            volatility = daily["daily_return"].std(ddof=1)

            sharpe = (
                mean_return / volatility
                if volatility > 0 else np.nan
            )

            records.append({
                "lambda_risk": lambda_risk,
                "tau": tau,
                "delta": delta,
                "xi": xi,
                "final_return": result["final_return"],
                "final_value": result["final_value"],
                "mean_daily_return": mean_return,
                "daily_volatility": volatility,
                "min_daily_return": daily["daily_return"].min(),
                "max_daily_return": daily["daily_return"].max(),
                "sharpe": sharpe,
                "status": "success",
            })

        except Exception as error:
            records.append({
                "lambda_risk": lambda_risk,
                "tau": tau,
                "delta": delta,
                "xi": xi,
                "final_return": np.nan,
                "final_value": np.nan,
                "mean_daily_return": np.nan,
                "daily_volatility": np.nan,
                "min_daily_return": np.nan,
                "max_daily_return": np.nan,
                "sharpe": np.nan,
                "status": str(error),
            })

    grid_result = (
        pd.DataFrame(records)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )

    # 保存结果
    output_dir = Path(__file__).resolve().parent / "TUNNING_PARAMETER"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{pd.to_datetime(end_date).strftime('%Y-%m-%d')}.csv"
    output_path = output_dir / filename

    grid_result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Sharpe 最高的一组参数
    successful = grid_result.loc[
        (grid_result["status"] == "success")
        & grid_result["sharpe"].notna()
    ]

    if successful.empty:
        best_params = None
    else:
        best_row = successful.iloc[0]

        best_params = {
            "lambda_risk": best_row["lambda_risk"],
            "tau": best_row["tau"],
            "delta": best_row["delta"],
            "xi": best_row["xi"],
        }

    return grid_result, best_params


# grid_result, best_params = grid_search_backtest(
#     info_df=info_df,
#     end_date="2024-06-03",
#     tuning_days=TUNING_DAYS,
#     # lambda_risk_grid=LAMBDA_RISK_GRID,
#     # tau_grid=TAU_GRID,
#     # delta_grid=DELTA_GRID,
#     # xi_grid=XI_GRID,
#     lambda_risk_grid=[500, 1000],
#     tau_grid=[0.05],
#     delta_grid=[1.0],
#     xi_grid=[1.0],)
# # 

def optimize_rebalance_day(
    info_df,
    rebalance_date,
    previous_positions,
    tuning_days=TUNING_DAYS,
    lambda_risk_grid=LAMBDA_RISK_GRID,
    tau_grid=TAU_GRID,
    delta_grid=DELTA_GRID,
    xi_grid=XI_GRID,
    fee=FEE,
    budget=BUDGET,
    max_weight=MAX_WEIGHT,
    training_days=TRAIN_DAYS,
    regression_train_days=REGRESSION_TRAIN_DAYS,
    regression_test_days=REGRESSION_TEST_DAYS,
):
    # 1. 用调仓日前 tuning_days 个交易日选择最优参数
    grid_result, best_params = grid_search_backtest(
        info_df=info_df,
        end_date=rebalance_date,
        tuning_days=tuning_days,
        lambda_risk_grid=lambda_risk_grid,
        tau_grid=tau_grid,
        delta_grid=delta_grid,
        xi_grid=xi_grid,
        fee=fee,
        budget=budget,
        max_weight=max_weight,
        training_days=training_days,
        regression_train_days=regression_train_days,
        regression_test_days=regression_test_days,
    )

    if best_params is None:
        raise RuntimeError(
            f"No valid parameter combination found before {rebalance_date}."
        )

    # 2. 使用最优参数，在调仓日当天进行组合优化
    result = optimize_daily_portfolio(
        info_df=info_df,
        date=rebalance_date,
        previous_positions=previous_positions,
        lambda_risk=best_params["lambda_risk"],
        xi=best_params["xi"],
        tau=best_params["tau"],
        delta=best_params["delta"],
        fee=fee,
        budget=budget,
        max_weight=max_weight,
        training_days=training_days,
        regression_train_days=regression_train_days,
        regression_test_days=regression_test_days,
    )

    if result["w"] is None:
        raise RuntimeError(
            f"Portfolio optimization failed on {rebalance_date}."
        )

    positions = result["positions"]

    # 3. 计算本次调仓到下一交易日的实现收益
    realized_return, next_date = calculate_next_day_return(
        info_df=info_df,
        date=rebalance_date,
        positions=positions,
    )

    return {
        "date": rebalance_date,
        "next_date": next_date,

        # tuning 得到的最优参数
        "best_params": best_params,

        # 当日优化结果
        "weights": result["w"],
        "contract_index": result["contract_index"],
        "positions": positions,
        "objective": result["obj"],

        # 本次调仓实现收益
        "return": realized_return,

        # 如果之后需要查看 tuning 结果
        "grid_result": grid_result,
    }

# result = optimize_rebalance_day(
#     info_df=info_df,
#     rebalance_date="2024-06-12",
#     previous_positions={},
#     lambda_risk_grid=[500, 1000],
#     tau_grid=[0.05],
#     delta_grid=[1.0],
#     xi_grid=[1],
#     )


def summarize_positions(info_df, date, positions, threshold=1e-8):
    position_df = pd.DataFrame(
        [
            {"symbol": symbol, "weight": float(weight)}
            for symbol, weight in positions.items()
            if abs(weight) > threshold
        ]
    )

    return_df = info_df.loc[
        pd.to_datetime(info_df["date"]) == pd.to_datetime(date),
        ["symbol", "return_1d"],
    ]

    return (position_df
        .merge(return_df, on="symbol", how="left")
        .assign(abs_weight=lambda x: x["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .drop(columns="abs_weight")
        .reset_index(drop=True))


# print(result["best_params"])
# print(summarize_positions(info_df, result["date"], result["positions"]))
# print(result["return"])
# previous_positions = result["positions"]


def run_full_backtest(
    info_df,
    begin_date,
    end_date,
    tuning_days=TUNING_DAYS,
    lambda_risk_grid=LAMBDA_RISK_GRID,
    tau_grid=TAU_GRID,
    delta_grid=DELTA_GRID,
    xi_grid=XI_GRID,
    fee=FEE,
    budget=BUDGET,
    max_weight=MAX_WEIGHT,
    training_days=TRAIN_DAYS,
    regression_train_days=REGRESSION_TRAIN_DAYS,
    regression_test_days=REGRESSION_TEST_DAYS,
):
    trading_dates = [
        date for date in TRADING_DATES
        if begin_date <= date <= end_date
    ]

    if not trading_dates:
        raise ValueError(
            f"No trading dates between {begin_date} and {end_date}."
        )

    # 最后一天必须有下一交易日才能计算收益
    trading_dates = [
        date for date in trading_dates
        if bisect_left(TRADING_DATES, date) + 1 < len(TRADING_DATES)
    ]

    previous_positions = {}
    cumulative_value = 1.0
    records = []

    # log 文件
    output_dir = Path(__file__).resolve().parent / "BACKTEST_LOG"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"backtest({begin_date}-{end_date}).log"

    with open(log_path, "w", encoding="utf-8") as log:

        log.write(
            f"Backtest period: {begin_date} - {end_date}\n"
        )
        log.write("=" * 80 + "\n\n")

        for date in trading_dates:

            result = optimize_rebalance_day(
                info_df=info_df,
                rebalance_date=date,
                previous_positions=previous_positions,
                tuning_days=tuning_days,
                lambda_risk_grid=lambda_risk_grid,
                tau_grid=tau_grid,
                delta_grid=delta_grid,
                xi_grid=xi_grid,
                fee=fee,
                budget=budget,
                max_weight=max_weight,
                training_days=training_days,
                regression_train_days=regression_train_days,
                regression_test_days=regression_test_days,
            )

            # 更新持仓，下一交易日作为 previous_positions
            previous_positions = result["positions"].copy()

            daily_return = result["return"]
            cumulative_value *= 1 + daily_return

            # 只保留非零权重
            nonzero_positions = {
                symbol: float(weight)
                for symbol, weight in result["positions"].items()
                if abs(weight) > 1e-8
            }
            print(
                f"Next date: {result['next_date']}, "
                f"cumulative value: {cumulative_value:.2f}, "
                f"positions: { {k: round(v, 4) for k, v in nonzero_positions.items()} }"
            )
            # 保存结构化结果
            records.append({
                "date": date,
                "next_date": result["next_date"],
                "lambda_risk": result["best_params"]["lambda_risk"],
                "tau": result["best_params"]["tau"],
                "delta": result["best_params"]["delta"],
                "xi": result["best_params"]["xi"],
                "daily_return": daily_return,
                "cumulative_value": cumulative_value,
                "positions": nonzero_positions,
            })

            # ---------- log ----------
            log.write(f"Date: {date}\n")
            log.write(f"Next date: {result['next_date']}\n")

            log.write(
                "Best parameters: "
                f"lambda_risk={result['best_params']['lambda_risk']}, "

                f"tau={result['best_params']['tau']}, "
                f"delta={result['best_params']['delta']}, "
                f"xi={result['best_params']['xi']}\n"
            )

            log.write("Positions:\n")

            for symbol, weight in nonzero_positions.items():
                log.write(
                    f"  {symbol:<25} "
                    f"{weight:>12.6f}\n"
                )

            log.write(
                f"Daily return:      {daily_return:.6f} "
                f"({daily_return:.2%})\n"
            )

            log.write(f"Cumulative value:  {cumulative_value:.6f}\n")

            log.write(
                f"Cumulative return: "
                f"{cumulative_value - 1:.6f} "
                f"({cumulative_value - 1:.2%})\n"
            )

            log.write("-" * 80 + "\n\n")

        # 最终总结
        log.write("=" * 80 + "\n")
        log.write("BACKTEST SUMMARY\n")
        log.write("=" * 80 + "\n")

        log.write(f"Begin date: {begin_date}\n")
        log.write(f"End date:   {end_date}\n")
        log.write(f"Trading days: {len(records)}\n")

        log.write(
            f"Final value:  {cumulative_value:.6f}\n"
        )

        log.write(
            f"Final return: {cumulative_value - 1:.6f} "
            f"({cumulative_value - 1:.2%})\n"
        )

    backtest_result = pd.DataFrame(records)

    return {
        "daily_results": backtest_result,
        "final_value": cumulative_value,
        "final_return": cumulative_value - 1,
        "final_positions": previous_positions,
        "log_path": log_path,
    }

# result = run_full_backtest(
#     info_df=info_df,
#     begin_date="2024-07-03",
#     end_date="2025-06-28",
#     lambda_risk_grid=[500],
#     tau_grid=[0.05],
#     delta_grid=[0.05],
# )

# print(result["final_return"])
# print(result["log_path"])
