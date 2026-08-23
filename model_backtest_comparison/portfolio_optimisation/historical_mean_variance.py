from bisect import bisect_left

import numpy as np

from config import REGRESSION_TEST_DAYS, REGRESSION_TRAIN_DAYS, TRAIN_DAYS, TRADING_DATES
from optimizer import solve_portfolio_long_only
from preprocessing import (
    build_contract_return_mean,
    build_contract_return_variance,
    build_eta,
    build_margin_vector,
    get_daily_contract_index,
)


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
    u_vector, index_u = build_contract_return_mean(info_df, date, training_days=training_days)
    a_matrix, index_a = build_contract_return_variance(info_df, date, training_days=training_days)
    eta, index_eta = build_eta(info_df, date)
    _, index_margin = build_margin_vector(info_df, date)

    if not (index_u == index_a == index_margin == index_eta):
        raise ValueError("Contract indices are not aligned.")

    w_prev = np.array([previous_positions.get(symbol, 0.0) for symbol in index_u], dtype=float)
    result = solve_portfolio_long_only(
        u_BL=u_vector,
        A_BL=a_matrix,
        w_prev=w_prev,
        transaction_eta=eta,
        lambda_risk=lambda_risk,
        xi=xi,
        fee=fee,
        max_weight=max_weight,
    )
    result["contract_index"] = index_u
    result["positions"] = dict(zip(index_u, result["w"])) if result["w"] is not None else None
    return result


def calculate_next_day_return(info_df, date, positions):
    today_df, _ = get_daily_contract_index(info_df, date)
    today_return = dict(zip(today_df["symbol"], today_df["return_1d"].to_numpy(dtype=float)))
    portfolio_return = sum(weight * today_return.get(symbol, 0.0) for symbol, weight in positions.items())
    next_idx = bisect_left(TRADING_DATES, date) + 1
    if next_idx >= len(TRADING_DATES):
        raise ValueError(f"No next trading date after {date}.")
    return portfolio_return, TRADING_DATES[next_idx]
