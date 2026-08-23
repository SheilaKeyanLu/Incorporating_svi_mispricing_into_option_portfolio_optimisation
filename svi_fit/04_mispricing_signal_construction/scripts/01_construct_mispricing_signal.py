"""Construct intraday and daily option mispricing signals from SVI residuals."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_DIR / "data" / "intermediate"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "output" / "report"

INPUT_PATH = RAW_DIR / "option_quotes_with_svi_based_iv.csv"
INTRADAY_OUTPUT_PATH = INTERMEDIATE_DIR / "_option_quotes_with_standardised_residual.csv"
DAILY_OUTPUT_PATH = INTERMEDIATE_DIR / "_daily_mispricing_signal.csv"
PROCESSED_OUTPUT_PATH = PROCESSED_DIR / "option_quotes_with_mispricing_signal.csv"
REPORT_PATH = REPORT_DIR / "mispricing_signal_summary.txt"

RESIDUAL_COL = "iv_minus_SVI_based_volatility"


def construct_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct snapshot-standardized residuals and daily contract-level signals."""
    result = df.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.dropna(subset=["trade_date", "target_time", "symbol", RESIDUAL_COL])

    snapshot_group = result.groupby(["trade_date", "target_time"], observed=True)[RESIDUAL_COL]
    result["snapshot_residual_mean"] = snapshot_group.transform("mean")
    result["snapshot_residual_std"] = snapshot_group.transform("std")
    result["snapshot_contract_count"] = snapshot_group.transform("count")

    result["standardised_residual"] = (
        result[RESIDUAL_COL] - result["snapshot_residual_mean"]
    ) / result["snapshot_residual_std"]
    result.loc[result["snapshot_residual_std"].eq(0), "standardised_residual"] = np.nan

    daily_signal = (
        result.groupby(["trade_date", "symbol"], as_index=False, observed=True)
        .agg(
            daily_mispricing_signal=("standardised_residual", "mean"),
            intraday_signal_std=("standardised_residual", "std"),
            valid_snapshot_count=("standardised_residual", "count"),
            raw_residual_mean=(RESIDUAL_COL, "mean"),
            raw_residual_std=(RESIDUAL_COL, "std"),
            SVI_based_volatility_mean=("SVI_based_volatility", "mean"),
        )
    )
    daily_signal = daily_signal[daily_signal["valid_snapshot_count"] >= 3].reset_index(drop=True)

    result = result.merge(
        daily_signal[
            [
                "trade_date",
                "symbol",
                "daily_mispricing_signal",
                "intraday_signal_std",
                "valid_snapshot_count",
            ]
        ],
        on=["trade_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    return result, daily_signal


def write_report(raw_rows: int, result: pd.DataFrame, daily_signal: pd.DataFrame) -> None:
    """Write a text summary of the mispricing-signal construction."""
    finite_standardised = result["standardised_residual"].notna().sum()
    finite_daily = result["daily_mispricing_signal"].notna().sum()
    date_min = result["trade_date"].min()
    date_max = result["trade_date"].max()
    lines = [
        "Mispricing signal construction summary",
        "======================================",
        "",
        f"Raw input file: {INPUT_PATH.relative_to(PROJECT_DIR)}",
        f"Processed output: {PROCESSED_OUTPUT_PATH.relative_to(PROJECT_DIR)}",
        "",
        f"Raw input rows: {raw_rows:,}",
        f"Rows retained for signal construction: {len(result):,}",
        f"Rows with finite standardized residuals: {finite_standardised:,}",
        f"Rows with matched daily signal: {finite_daily:,}",
        f"Daily contract-level signals: {len(daily_signal):,}",
        f"Unique symbols with daily signals: {daily_signal['symbol'].nunique():,}",
        f"Date range: {date_min.date()} to {date_max.date()}",
        "",
        "Added columns include snapshot_residual_mean, snapshot_residual_std,",
        "snapshot_contract_count, standardised_residual, daily_mispricing_signal,",
        "intraday_signal_std, and valid_snapshot_count.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing required input file: {INPUT_PATH}")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    result, daily_signal = construct_signals(df)

    result.to_csv(INTRADAY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    daily_signal.to_csv(DAILY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    result.to_csv(PROCESSED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    write_report(len(df), result, daily_signal)

    print(f"Saved intermediate quote-level residuals: {INTRADAY_OUTPUT_PATH}")
    print(f"Saved intermediate daily signals: {DAILY_OUTPUT_PATH}")
    print(f"Saved processed data: {PROCESSED_OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

