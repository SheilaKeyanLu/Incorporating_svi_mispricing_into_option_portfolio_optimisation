"""Shared helpers for Vega-weighted SVI figures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common.svi_core import svi_implied_vol, svi_total_variance


RAW_PATH = PROJECT_DIR / "data" / "raw" / "option_quotes_with_iv.csv"
SVI_PATH = PROJECT_DIR / "data" / "intermediate" / "_svi_fit.csv"
FIGURE_DIR = PROJECT_DIR / "output" / "figures"

UNDERLYINGS = {
    "000016.SH": "SSE 50",
    "000300.SH": "CSI 300",
    "000852.SH": "CSI 1000",
}
INTRADAY_TIMES = ["10:00", "10:40", "11:20", "13:10", "13:50", "14:30"]
TARGET_TIME = "14:30"
TENOR_LABELS = ["Short", "Medium", "Long"]
TENOR_QUANTILES = {"Short": 0.0, "Medium": 0.5, "Long": 1.0}

QUOTE_USECOLS = [
    "trade_date",
    "target_time",
    "target_datetime",
    "ths_underlying_code_option",
    "ths_maturity_date_option",
    "option_type",
    "strike_price",
    "underlying_price",
    "F",
    "T",
    "implied_volatility",
]
SVI_USECOLS = [
    "target_datetime",
    "ths_underlying_code_option",
    "ths_maturity_date_option",
    "z0",
    "z1",
    "z2",
    "z3",
    "z4",
    "z5",
    "fit_status",
    "fit_rmse",
]


def normalize_target_time(target_time: str) -> str:
    """Normalize a time-like value to HH:MM."""
    return pd.to_datetime(target_time).strftime("%H:%M")


def load_svi_fits() -> pd.DataFrame:
    """Load successful SVI fit rows."""
    df = pd.read_csv(SVI_PATH, usecols=SVI_USECOLS)
    df["target_datetime"] = df["target_datetime"].astype(str)
    df["ths_maturity_date_option"] = pd.to_numeric(
        df["ths_maturity_date_option"], errors="coerce"
    ).astype("Int64")
    df["target_dt"] = pd.to_datetime(df["target_datetime"])
    df["trade_date"] = df["target_dt"].dt.strftime("%Y-%m-%d")
    df["target_time"] = df["target_dt"].dt.strftime("%H:%M")
    df = df[
        df["ths_underlying_code_option"].isin(UNDERLYINGS)
        & df["fit_status"].astype(str).str.startswith("success")
    ].copy()
    for column in [f"z{i}" for i in range(6)] + ["fit_rmse"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=[f"z{i}" for i in range(6)] + ["ths_maturity_date_option"])


def load_quotes_for_date_times(
    trade_date: str,
    target_times: set[str],
    underlyings: set[str] | None = None,
) -> pd.DataFrame:
    """Load quote rows for selected dates and intraday times."""
    parts = []
    underlyings = underlyings or set(UNDERLYINGS)
    for chunk in pd.read_csv(RAW_PATH, usecols=QUOTE_USECOLS, chunksize=300_000):
        mask = (
            chunk["trade_date"].astype(str).eq(trade_date)
            & chunk["target_time"].astype(str).isin(target_times)
            & chunk["ths_underlying_code_option"].isin(underlyings)
        )
        if mask.any():
            parts.append(chunk.loc[mask].copy())
    if not parts:
        return pd.DataFrame(columns=QUOTE_USECOLS)

    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df["target_time"] = df["target_time"].astype(str)
    df["target_datetime"] = df["target_datetime"].astype(str)
    df["ths_maturity_date_option"] = pd.to_numeric(
        df["ths_maturity_date_option"], errors="coerce"
    ).astype("Int64")
    for column in ["strike_price", "underlying_price", "F", "T", "implied_volatility"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(
        subset=["strike_price", "F", "T", "implied_volatility", "ths_maturity_date_option"]
    )
    df = df[
        (df["strike_price"] > 0)
        & (df["F"] > 0)
        & (df["T"] > 0)
        & (df["implied_volatility"] > 0)
    ].copy()
    df["log_moneyness"] = np.log(df["strike_price"] / df["F"])
    return df


def select_representative_date(svi_fits: pd.DataFrame) -> str:
    """Select the middle eligible date with three maturities for every underlying."""
    snapshot = svi_fits[svi_fits["target_time"].eq(TARGET_TIME)]
    maturity_counts = (
        snapshot.groupby(["trade_date", "ths_underlying_code_option"])[
            "ths_maturity_date_option"
        ]
        .nunique()
        .unstack()
    )
    common_dates = maturity_counts.dropna()
    common_dates = common_dates[(common_dates[list(UNDERLYINGS)] >= 3).all(axis=1)]
    if common_dates.empty:
        raise RuntimeError("No eligible date has at least three maturities for all underlyings.")
    dates = sorted(common_dates.index)
    return dates[len(dates) // 2]


def select_representative_maturities(
    quotes: pd.DataFrame,
    svi_fits: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    """Select short, medium, and long maturities from the available tenor distribution."""
    snapshot_fits = svi_fits[
        svi_fits["trade_date"].eq(trade_date) & svi_fits["target_time"].eq(TARGET_TIME)
    ]
    rows = []
    for code, name in UNDERLYINGS.items():
        fit_maturities = set(
            snapshot_fits.loc[
                snapshot_fits["ths_underlying_code_option"].eq(code),
                "ths_maturity_date_option",
            ].astype(int)
        )
        q = quotes[
            quotes["ths_underlying_code_option"].eq(code)
            & quotes["target_time"].eq(TARGET_TIME)
            & quotes["ths_maturity_date_option"].astype(int).isin(fit_maturities)
        ]
        tenors = q.groupby("ths_maturity_date_option")["T"].median().dropna().sort_values()
        if len(tenors) < 3:
            raise RuntimeError(f"{name} has fewer than three usable maturities on {trade_date}.")
        for label in TENOR_LABELS:
            target_tenor = tenors.quantile(TENOR_QUANTILES[label])
            maturity = (tenors - target_tenor).abs().idxmin()
            rows.append(
                {
                    "underlying_code": code,
                    "underlying_name": name,
                    "tenor_label": label,
                    "maturity": int(maturity),
                    "T": float(tenors.loc[maturity]),
                }
            )
    return pd.DataFrame(rows)


def get_fit_row(
    svi_fits: pd.DataFrame,
    target_datetime: str,
    underlying_code: str,
    maturity: int,
) -> pd.Series | None:
    """Return the matching fit row for one slice."""
    fit = svi_fits[
        svi_fits["target_datetime"].eq(target_datetime)
        & svi_fits["ths_underlying_code_option"].eq(underlying_code)
        & svi_fits["ths_maturity_date_option"].astype(int).eq(int(maturity))
    ]
    if fit.empty:
        return None
    return fit.iloc[-1]


def panel_data(
    quotes: pd.DataFrame,
    svi_fits: pd.DataFrame,
    trade_date: str,
    target_time: str,
    underlying_code: str,
    maturity: int,
) -> tuple[pd.DataFrame, pd.Series | None, np.ndarray, np.ndarray]:
    """Return observed points and the fitted SVI IV curve for one panel."""
    q = quotes[
        quotes["trade_date"].eq(trade_date)
        & quotes["target_time"].eq(target_time)
        & quotes["ths_underlying_code_option"].eq(underlying_code)
        & quotes["ths_maturity_date_option"].astype(int).eq(int(maturity))
    ].copy()
    if q.empty:
        return q, None, np.array([]), np.array([])
    fit_row = get_fit_row(svi_fits, q["target_datetime"].iloc[0], underlying_code, maturity)
    if fit_row is None:
        return q, None, np.array([]), np.array([])

    x_min = q["log_moneyness"].min()
    x_max = q["log_moneyness"].max()
    x_grid = np.linspace(x_min, x_max, 240)
    z = fit_row[[f"z{i}" for i in range(6)]].to_numpy(dtype=float)
    y_grid = svi_implied_vol(x_grid, z, q["T"].median())
    if not np.isfinite(y_grid).any():
        return q, fit_row, np.array([]), np.array([])
    return q, fit_row, x_grid, y_grid


def load_slice(
    underlying_code: str,
    trade_date: str,
    maturity_date: int,
    target_time: str,
) -> pd.DataFrame:
    """Load one quote slice for a single-slice or comparison figure."""
    quotes = load_quotes_for_date_times(trade_date, {target_time}, {underlying_code})
    quotes = quotes[quotes["ths_maturity_date_option"].astype(int).eq(int(maturity_date))]
    if quotes.empty:
        raise RuntimeError("No quote rows found for the requested slice.")
    return quotes.sort_values(["option_type", "strike_price"])


def fitted_total_variance(x: np.ndarray, fit: pd.Series) -> np.ndarray:
    """Return fitted total variance for one fit row."""
    z = fit[[f"z{i}" for i in range(6)]].to_numpy(dtype=float)
    return svi_total_variance(x, z)

