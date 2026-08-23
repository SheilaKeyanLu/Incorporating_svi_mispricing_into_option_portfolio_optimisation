from bisect import bisect_left

import numpy as np

from config import REGRESSION_TEST_DAYS, REGRESSION_TRAIN_DAYS, TRAIN_DAYS, TRADING_DATES
from optimizer import solve_portfolio_long_only
from preprocessing import calculate_A, calculate_u_base, build_eta, build_margin_vector, get_daily_contract_index


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
    a_summary = calculate_A(info_df, date, training_days=training_days, tau=tau)
    a_matrix = a_summary["A"]
    a_index = a_summary["contract_index"]
    u_vector, u_index = calculate_u_base(info_df, date, training_days=training_days)
    eta, index_eta = build_eta(info_df, date)
    _, index_margin = build_margin_vector(info_df, date)

    if not (a_index == u_index == index_margin == index_eta):
        raise ValueError("Contract indices are not aligned.")

    w_prev = np.array([previous_positions.get(symbol, 0.0) for symbol in a_index], dtype=float)
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
    result["contract_index"] = a_index
    result["positions"] = dict(zip(a_index, result["w"])) if result["w"] is not None else None
    return result


def calculate_next_day_return(info_df, date, positions):
    today_df, _ = get_daily_contract_index(info_df, date)
    today_return = dict(zip(today_df["symbol"], today_df["return_1d"].to_numpy(dtype=float)))
    portfolio_return = sum(weight * today_return.get(symbol, 0.0) for symbol, weight in positions.items())
    next_idx = bisect_left(TRADING_DATES, date) + 1
    if next_idx >= len(TRADING_DATES):
        raise ValueError(f"No next trading date after {date}.")
    return portfolio_return, TRADING_DATES[next_idx]
