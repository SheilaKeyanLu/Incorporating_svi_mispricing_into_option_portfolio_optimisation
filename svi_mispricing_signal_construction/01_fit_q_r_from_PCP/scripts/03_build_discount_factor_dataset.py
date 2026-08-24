"""Map PCP regression estimates back to the full quote file."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_DIR / "data" / "intermediate"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "output" / "report"

SOURCE_PATH = RAW_DIR / "option_quotes_with_basic_and_underlying.csv"
GROUP_PATH = INTERMEDIATE_DIR / "_option_groups_pcp_regression.csv"
OUTPUT_PATH = PROCESSED_DIR / "option_quotes_with_discount_factor.csv"
REPORT_PATH = REPORT_DIR / "pcp_discount_factor_summary.txt"

GROUP_COLS = [
    "ths_underlying_code_option",
    "trade_date",
    "target_time",
    "ths_maturity_date_option",
]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize merge keys before joining quotes with PCP estimates."""
    result = df.copy()
    result["ths_underlying_code_option"] = (
        result["ths_underlying_code_option"].astype("string").str.strip()
    )
    result["trade_date"] = result["trade_date"].astype("string").str.strip()
    result["target_time"] = result["target_time"].astype("string").str.strip()
    result["ths_maturity_date_option"] = pd.to_numeric(
        result["ths_maturity_date_option"], errors="coerce"
    ).astype("Int64")
    return result


def write_report(
    total_rows: int,
    matched_rows: int,
    group_count: int,
    output_path: Path,
    report_path: Path,
) -> None:
    """Write a text summary of the completed PCP mapping step."""
    match_rate = matched_rows / total_rows if total_rows else float("nan")
    lines = [
        "PCP discount-factor estimation summary",
        "======================================",
        "",
        f"Source quote file: {SOURCE_PATH.relative_to(PROJECT_DIR)}",
        f"Regression file: {GROUP_PATH.relative_to(PROJECT_DIR)}",
        f"Processed output: {output_path.relative_to(PROJECT_DIR)}",
        "",
        f"Quote rows written: {total_rows:,}",
        f"Rows matched with PCP estimates: {matched_rows:,}",
        f"Row match rate: {match_rate:.2%}",
        f"Distinct regression groups used: {group_count:,}",
        "",
        "Added columns: beta_K, beta_S, r_reg, delta_reg.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(chunk_size: int = 200_000) -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing required raw file: {SOURCE_PATH}")
    if not GROUP_PATH.exists():
        raise FileNotFoundError(f"Missing required regression file: {GROUP_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    group_df = pd.read_csv(GROUP_PATH, low_memory=False)
    group_df = normalize_keys(group_df)
    group_df = group_df[
        GROUP_COLS + ["beta_K", "beta_S", "r_reg", "delta_reg"]
    ].drop_duplicates(subset=GROUP_COLS, keep="last")

    OUTPUT_PATH.unlink(missing_ok=True)

    total_rows = 0
    matched_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(SOURCE_PATH, chunksize=chunk_size, low_memory=False), start=1
    ):
        chunk = normalize_keys(chunk)
        result = chunk.merge(group_df, on=GROUP_COLS, how="left", validate="many_to_one")
        total_rows += len(result)
        matched_rows += result["r_reg"].notna().sum()
        result.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=(chunk_number == 1),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"Chunk {chunk_number}: wrote {len(result):,} rows, "
            f"matched {result['r_reg'].notna().sum():,}"
        )

    write_report(
        total_rows=total_rows,
        matched_rows=matched_rows,
        group_count=len(group_df),
        output_path=OUTPUT_PATH,
        report_path=REPORT_PATH,
    )
    print(f"Saved processed data: {OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

