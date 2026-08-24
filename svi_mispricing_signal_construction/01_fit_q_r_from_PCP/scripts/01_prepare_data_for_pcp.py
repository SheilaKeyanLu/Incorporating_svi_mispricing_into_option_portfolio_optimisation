"""Prepare option quotes for put-call-parity regressions."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_DIR / "data" / "intermediate"

INPUT_PATH = RAW_DIR / "option_quotes_with_basic_and_underlying.csv"
OUTPUT_PATH = INTERMEDIATE_DIR / "_option_quotes_drop.csv"

SPREAD_THRESHOLD = 0.1
QUOTE_TIME_GAP_THRESHOLD_SECONDS = 5
MAX_TRADING_DAYS_LEFT = 90
MIN_TRADING_DAYS_LEFT = 7
MONEYNESS_LOWER_BOUND = -0.075
MONEYNESS_UPPER_BOUND = 0.075


def calc_signed_time_diff_series(df: pd.DataFrame) -> pd.Series:
    """Return quote update time minus target snapshot time in seconds."""
    target = pd.to_datetime(
        df["trade_date"].astype(str) + " " + df["target_time"].astype(str),
        errors="coerce",
    )
    update = pd.to_datetime(df["update_time"], errors="coerce")
    result = (update - target).dt.total_seconds().copy()

    missing = result.isna()
    if missing.any():
        time_parts = (
            df.loc[missing, "update_time"]
            .astype(str)
            .str.strip()
            .str.extract(
                r"^(?:(?P<hour>\d{1,2}):)?"
                r"(?P<minute>\d{1,2}):(?P<second>\d{1,2}(?:\.\d+)?)$"
            )
        )
        matched = time_parts.notna().all(axis=1) | (
            time_parts["hour"].isna()
            & time_parts[["minute", "second"]].notna().all(axis=1)
        )
        if matched.any():
            matched_index = time_parts.index[matched]
            matched_target = target.loc[matched_index]
            hour = pd.to_numeric(time_parts.loc[matched_index, "hour"], errors="coerce")
            minute = pd.to_numeric(time_parts.loc[matched_index, "minute"], errors="coerce")
            second = pd.to_numeric(time_parts.loc[matched_index, "second"], errors="coerce")
            hour = hour.fillna(matched_target.dt.hour)
            result.loc[matched_index] = (
                (hour - matched_target.dt.hour) * 3600
                + (minute - matched_target.dt.minute) * 60
                + (second - matched_target.dt.second)
            )

    return result


def clean_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter quotes to a stable sample suitable for PCP regressions."""
    df = df.copy()
    df["log_moneyness"] = np.log(df["underlying_price"] / df["strike_price"])
    df = df.loc[
        df["log_moneyness"].between(MONEYNESS_LOWER_BOUND, MONEYNESS_UPPER_BOUND)
    ].copy()

    df["bid1"] = pd.to_numeric(df["bid1"], errors="coerce")
    df["ask1"] = pd.to_numeric(df["ask1"], errors="coerce")
    valid_quote_mask = (
        df["bid1"].notna()
        & df["ask1"].notna()
        & (df["bid1"] > 0)
        & (df["ask1"] > 0)
        & (df["ask1"] >= df["bid1"])
    )
    df = df.loc[valid_quote_mask].copy()

    spread = df["ask1"] - df["bid1"]
    df["relative_spread"] = spread / df["lastest_price"]
    df = df.loc[df["relative_spread"] <= SPREAD_THRESHOLD].copy()

    df = df.loc[
        (df["trading_days_left"] > MIN_TRADING_DAYS_LEFT)
        & (df["trading_days_left"] < MAX_TRADING_DAYS_LEFT)
    ].copy()

    group_cols = [
        "ths_underlying_code_option",
        "trade_date",
        "target_time",
        "ths_maturity_date_option",
        "strike_price",
    ]

    df["_update_time_offset_seconds"] = calc_signed_time_diff_series(df)
    quote_time_gap = df.groupby(group_cols)["_update_time_offset_seconds"].transform(
        lambda values: values.max() - values.min()
    )
    df = df.loc[
        quote_time_gap.notna() & (quote_time_gap <= QUOTE_TIME_GAP_THRESHOLD_SECONDS)
    ].copy()
    df = df.drop(columns="_update_time_offset_seconds")

    pair_counts = df.groupby(group_cols).size().reset_index(name="option_count")
    valid_groups = pair_counts.loc[pair_counts["option_count"] != 1, group_cols]
    return df.merge(valid_groups, on=group_cols, how="inner")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing required input file: {INPUT_PATH}")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    quotes = pd.read_csv(INPUT_PATH, low_memory=False)
    clean_df = clean_quotes(quotes)
    clean_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Input rows: {len(quotes):,}")
    print(f"Clean PCP rows: {len(clean_df):,}")
    print(f"Saved intermediate data: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

