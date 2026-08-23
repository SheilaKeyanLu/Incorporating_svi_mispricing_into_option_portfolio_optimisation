"""Calculate Black-Scholes implied volatility for cleaned option quotes."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


PROJECT_DIR = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_DIR / "data" / "intermediate"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "output" / "report"

INPUT_PATH = INTERMEDIATE_DIR / "_option_quotes_clean_for_iv.csv"
OUTPUT_PATH = PROCESSED_DIR / "option_quotes_with_iv.csv"
REPORT_PATH = REPORT_DIR / "iv_calculation_summary.txt"

GROUP_COLS = [
    "ths_underlying_code_option",
    "target_datetime",
    "strike_price",
    "ths_maturity_date_option",
]
SEARCH_INTERVALS = [(1e-6, 5), (1e-6, 10), (1e-6, 20), (1e-6, 50), (0.01, 100)]


def bs_call_price(forward: float, strike: float, discount_factor: float, tenor: float, sigma: float) -> float:
    """Return the Black-Scholes call price using forward price notation."""
    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * tenor) / (sigma * np.sqrt(tenor))
    d2 = d1 - sigma * np.sqrt(tenor)
    return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def bs_put_price(forward: float, strike: float, discount_factor: float, tenor: float, sigma: float) -> float:
    """Return the Black-Scholes put price using forward price notation."""
    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * tenor) / (sigma * np.sqrt(tenor))
    d2 = d1 - sigma * np.sqrt(tenor)
    return discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_volatility(price_function, forward, strike, discount_factor, tenor, market_price) -> float:
    """Invert a Black-Scholes price with robust fallback search intervals."""
    def objective(sigma: float) -> float:
        return price_function(forward, strike, discount_factor, tenor, sigma) - market_price

    for lower, upper in SEARCH_INTERVALS:
        try:
            f_lower = objective(lower)
            f_upper = objective(upper)
            if np.sign(f_lower) * np.sign(f_upper) < 0:
                return brentq(objective, lower, upper, maxiter=100)
            if abs(f_lower) < 1e-8:
                return lower
            if abs(f_upper) < 1e-8:
                return upper
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            continue

    raise ValueError("No valid implied volatility root was found.")


def calc_iv_once(group: pd.DataFrame) -> pd.DataFrame:
    """Assign one implied-volatility value to a call-put pair at the same strike."""
    iv = np.nan

    call_row = group[group["option_type"] == "C"]
    if len(call_row) > 0:
        row = call_row.iloc[0]
        try:
            iv = implied_volatility(
                bs_call_price,
                forward=row["F"],
                strike=row["strike_price"],
                discount_factor=-row["beta_K"],
                tenor=row["T"],
                market_price=row["lastest_price"],
            )
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            iv = np.nan

    if pd.isna(iv):
        put_row = group[group["option_type"] == "P"]
        if len(put_row) > 0:
            row = put_row.iloc[0]
            try:
                iv = implied_volatility(
                    bs_put_price,
                    forward=row["F"],
                    strike=row["strike_price"],
                    discount_factor=-row["beta_K"],
                    tenor=row["T"],
                    market_price=row["lastest_price"],
                )
            except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
                iv = np.nan

    result = group.copy()
    result["implied_volatility"] = iv
    return result


def write_report(input_rows: int, output_rows: int, valid_iv_rows: int, group_count: int) -> None:
    """Write a text report for the implied-volatility calculation."""
    valid_rate = valid_iv_rows / output_rows if output_rows else float("nan")
    lines = [
        "Black-Scholes implied-volatility calculation summary",
        "=====================================================",
        "",
        f"Clean input file: {INPUT_PATH.relative_to(PROJECT_DIR)}",
        f"Processed output: {OUTPUT_PATH.relative_to(PROJECT_DIR)}",
        "",
        f"Clean input rows: {input_rows:,}",
        f"Strike groups processed: {group_count:,}",
        f"Output rows written: {output_rows:,}",
        f"Rows with finite implied volatility: {valid_iv_rows:,}",
        f"Finite implied-volatility rate: {valid_rate:.2%}",
        "",
        "Added columns: F, implied_volatility.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(chunk_size: int = 5_000) -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing required intermediate file: {INPUT_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["F"] = df["strike_price"] + (
        df["beta_S"] * df["underlying_price"] + df["beta_K"] * df["strike_price"]
    ) / (-df["beta_K"])

    groups = list(df.groupby(GROUP_COLS, sort=False))
    OUTPUT_PATH.unlink(missing_ok=True)

    results = []
    first_write = True
    output_rows = 0
    valid_iv_rows = 0

    for index, (_, group) in enumerate(groups, start=1):
        results.append(calc_iv_once(group))

        if len(results) >= chunk_size:
            batch = pd.concat(results, ignore_index=True)
            batch.to_csv(
                OUTPUT_PATH,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig",
            )
            output_rows += len(batch)
            valid_iv_rows += batch["implied_volatility"].notna().sum()
            results.clear()
            first_write = False

        if index % 500 == 0:
            print(f"Progress: {index:,}/{len(groups):,} groups")

    if results:
        batch = pd.concat(results, ignore_index=True)
        batch.to_csv(
            OUTPUT_PATH,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig",
        )
        output_rows += len(batch)
        valid_iv_rows += batch["implied_volatility"].notna().sum()

    write_report(
        input_rows=len(df),
        output_rows=output_rows,
        valid_iv_rows=int(valid_iv_rows),
        group_count=len(groups),
    )
    print(f"Saved processed data: {OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

