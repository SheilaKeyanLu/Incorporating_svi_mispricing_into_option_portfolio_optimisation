from bisect import bisect_left

import numpy as np

from config import REGRESSION_TEST_DAYS, REGRESSION_TRAIN_DAYS, TRAIN_DAYS, TRADING_DATES
from optimizer import solve_portfolio
from preprocessing import build_bl_posterior, build_eta, build_margin_vector, get_daily_contract_index


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
    bl = build_bl_posterior(
        info_df=info_df,
        date=date,
        training_days=training_days,
        regression_train_days=regression_train_days,
        regression_test_days=regression_test_days,
        tau=tau,
        delta=delta,
    )
    margin, index_margin = build_margin_vector(info_df, date)
    eta, index_eta = build_eta(info_df, date)
    contract_index = bl["contract_index"]

    if not (contract_index == index_margin == index_eta):
        raise ValueError("Contract indices are not aligned.")

    w_prev = np.array([previous_positions.get(symbol, 0.0) for symbol in contract_index], dtype=float)
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
        max_weight=max_weight,
    )
    result["contract_index"] = contract_index
    result["positions"] = dict(zip(contract_index, result["w"])) if result["w"] is not None else None
    result["bl"] = bl
    return result


def calculate_next_day_return(info_df, date, positions):
    today_df, _ = get_daily_contract_index(info_df, date)
    today_return = dict(zip(today_df["symbol"], today_df["return_1d"].to_numpy(dtype=float)))
    portfolio_return = sum(weight * today_return.get(symbol, 0.0) for symbol, weight in positions.items())
    next_idx = bisect_left(TRADING_DATES, date) + 1
    if next_idx >= len(TRADING_DATES):
        raise ValueError(f"No next trading date after {date}.")
    return portfolio_return, TRADING_DATES[next_idx]
