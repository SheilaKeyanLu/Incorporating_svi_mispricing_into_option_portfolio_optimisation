from pathlib import Path

import numpy as np
import pandas as pd

from calc_greeks import calc_delta, calc_gamma, calc_theta, calc_vega


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "option_quotes_with_iv.csv"
OUTPUT_PATH = BASE_DIR / "option_quotes_with_greeks.csv"
CHUNKSIZE = 500_000

REQUIRED_COLUMNS = [
    "F",
    "T",
    "implied_volatility",
    "strike_price",
    "option_type",
    "r_reg",
    "beta_K",
]


def _prepare_numeric_columns(chunk):
    for column in ["F", "T", "implied_volatility", "strike_price", "r_reg", "beta_K"]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    chunk["option_type"] = chunk["option_type"].astype("string").str.strip().str.upper()
    return chunk


def _add_greeks(chunk):
    chunk = _prepare_numeric_columns(chunk)

    F = chunk["F"]
    T = chunk["T"]
    sigma = chunk["implied_volatility"]
    K = chunk["strike_price"]
    option_type = chunk["option_type"]
    r = chunk["r_reg"]
    discount_factor = -chunk["beta_K"]

    valid = (
        F.gt(0)
        & T.gt(0)
        & sigma.gt(0)
        & K.gt(0)
        & discount_factor.gt(0)
        & option_type.isin(["C", "P"])
    )

    chunk["discount_factor"] = discount_factor
    chunk["delta"] = np.nan
    chunk["gamma"] = np.nan
    chunk["theta"] = np.nan
    chunk["vega"] = np.nan

    if valid.any():
        idx = valid.to_numpy()
        chunk.loc[valid, "delta"] = calc_delta(
            F.to_numpy()[idx],
            discount_factor.to_numpy()[idx],
            T.to_numpy()[idx],
            sigma.to_numpy()[idx],
            K.to_numpy()[idx],
            option_type.to_numpy()[idx],
        )
        chunk.loc[valid, "gamma"] = calc_gamma(
            F.to_numpy()[idx],
            discount_factor.to_numpy()[idx],
            T.to_numpy()[idx],
            sigma.to_numpy()[idx],
            K.to_numpy()[idx],
        )
        chunk.loc[valid, "theta"] = calc_theta(
            F.to_numpy()[idx],
            discount_factor.to_numpy()[idx],
            T.to_numpy()[idx],
            sigma.to_numpy()[idx],
            K.to_numpy()[idx],
            option_type.to_numpy()[idx],
            r.to_numpy()[idx],
        )
        chunk.loc[valid, "vega"] = calc_vega(
            F.to_numpy()[idx],
            discount_factor.to_numpy()[idx],
            T.to_numpy()[idx],
            sigma.to_numpy()[idx],
            K.to_numpy()[idx],
        )

    return chunk


def main():
    first_chunk = True
    rows = 0

    for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE):
        missing = [column for column in REQUIRED_COLUMNS if column not in chunk.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        result = _add_greeks(chunk)
        result.to_csv(OUTPUT_PATH, mode="w" if first_chunk else "a", index=False, header=first_chunk)
        first_chunk = False
        rows += len(result)
        print(f"processed {rows:,} rows")

    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
