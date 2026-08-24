"""Add Black-Scholes Vega to the implied-volatility quote file."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "option_quotes_with_iv.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "intermediate" / "_option_quotes_with_iv_with_vega.csv"


def calculate_vega(frame: pd.DataFrame) -> pd.Series:
    """Calculate option Vega using forward-price Black-Scholes notation."""
    forward = pd.to_numeric(frame["F"], errors="coerce")
    strike = pd.to_numeric(frame["strike_price"], errors="coerce")
    discount_factor = -pd.to_numeric(frame["beta_K"], errors="coerce")
    tenor = pd.to_numeric(frame["T"], errors="coerce")
    sigma = pd.to_numeric(frame["implied_volatility"], errors="coerce")

    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * tenor) / (sigma * np.sqrt(tenor))
    return forward * discount_factor * np.sqrt(tenor) * norm.pdf(d1)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing required raw file: {RAW_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df["vega"] = calculate_vega(df)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved Vega-enriched data: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

