"""Estimate risk-free rates and dividend yields from put-call parity."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_DIR / "data" / "intermediate"

INPUT_PATH = INTERMEDIATE_DIR / "_option_quotes_drop.csv"
OUTPUT_PATH = INTERMEDIATE_DIR / "_option_groups_pcp_regression.csv"

GROUP_COLS = [
    "ths_underlying_code_option",
    "trade_date",
    "target_time",
    "ths_maturity_date_option",
]


def add_pcp_regression(df: pd.DataFrame, min_strikes: int = 3) -> pd.DataFrame:
    """Run one PCP regression per underlying/date/time/maturity group."""
    results = []

    for _, group in df.groupby(GROUP_COLS):
        group = group.dropna(
            subset=["call_price", "put_price", "strike_price", "underlying_price", "T"]
        ).copy()
        if group.empty:
            continue

        spot = group["underlying_price"].iloc[0]
        tenor = group["T"].iloc[0]
        if tenor <= 0 or spot <= 0:
            continue
        if group["strike_price"].nunique() < min_strikes:
            continue

        y = (group["call_price"] - group["put_price"]).to_numpy()
        x = np.column_stack(
            [group["underlying_price"].to_numpy(), group["strike_price"].to_numpy()]
        )

        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        beta_s, beta_k = beta
        if not (0 < beta_s and beta_k < 0):
            continue

        fitted = x @ beta
        residual = y - fitted
        dividend_yield = -np.log(beta_s) / tenor
        risk_free_rate = -np.log(-beta_k) / tenor

        result = group[GROUP_COLS + ["strike_price", "call_price", "put_price"]].copy()
        result["T"] = tenor
        result["S"] = spot
        result["n_strikes"] = group["strike_price"].nunique()
        result["n_obs"] = len(group)
        result["beta_S"] = beta_s
        result["beta_K"] = beta_k
        result["r_reg"] = risk_free_rate
        result["delta_reg"] = dividend_yield
        result["C_minus_P"] = y
        result["C_minus_P_hat"] = fitted
        result["pcp_residual"] = residual
        results.append(result)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing required intermediate file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    group_stats = add_pcp_regression(df)
    group_stats.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    valid_groups = group_stats[GROUP_COLS].drop_duplicates().shape[0]
    print(f"Valid PCP regression groups: {valid_groups:,}")
    print(f"Saved regression data: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

