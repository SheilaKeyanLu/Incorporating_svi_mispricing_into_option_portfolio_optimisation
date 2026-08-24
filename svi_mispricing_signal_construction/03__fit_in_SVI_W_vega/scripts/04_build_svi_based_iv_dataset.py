"""Map Vega-weighted SVI fitted volatility back to option quote rows."""

import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common.svi_core import svi_based_volatility_frame


RAW_PATH = PROJECT_DIR / "data" / "raw" / "option_quotes_with_iv.csv"
SVI_MARGIN_PATH = PROJECT_DIR / "data" / "intermediate" / "_svi_fit_margin.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "option_quotes_with_svi_based_iv.csv"

KEY_COLUMNS = [
    "target_datetime",
    "ths_underlying_code_option",
    "ths_maturity_date_option",
]
Z_COLUMNS = [f"z{i}" for i in range(6)]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing required raw file: {RAW_PATH}")
    if not SVI_MARGIN_PATH.exists():
        raise FileNotFoundError(f"Missing required SVI margin file: {SVI_MARGIN_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    svi_df = pd.read_csv(SVI_MARGIN_PATH, low_memory=False)
    svi_df = svi_df[svi_df["margin"] <= 1][KEY_COLUMNS + Z_COLUMNS + ["margin", "fit_status"]]

    quote_df = pd.read_csv(RAW_PATH, low_memory=False)
    df = quote_df.merge(svi_df, on=KEY_COLUMNS, how="left")

    numeric_columns = ["strike_price", "F", "T", "implied_volatility"] + Z_COLUMNS
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["SVI_based_volatility"] = svi_based_volatility_frame(df)
    df["iv_minus_SVI_based_volatility"] = (
        df["implied_volatility"] - df["SVI_based_volatility"]
    )
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved processed SVI-based IV data: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

