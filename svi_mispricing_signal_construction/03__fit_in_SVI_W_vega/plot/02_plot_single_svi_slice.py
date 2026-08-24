"""Plot one Vega-weighted SVI fitted total-variance slice."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from svi_plot_helpers import FIGURE_DIR, fitted_total_variance, get_fit_row, load_slice, load_svi_fits


DEFAULT_UNDERLYING = "000016.SH"
DEFAULT_TRADE_DATE = "2023-01-06"
DEFAULT_MATURITY = 20230317
DEFAULT_TARGET_TIME = "10:40"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selecting one SVI slice."""
    parser = argparse.ArgumentParser(description="Plot a single Vega-weighted SVI fitted slice.")
    parser.add_argument("--underlying", default=DEFAULT_UNDERLYING)
    parser.add_argument("--trade-date", default=DEFAULT_TRADE_DATE)
    parser.add_argument("--maturity", type=int, default=DEFAULT_MATURITY)
    parser.add_argument("--target-time", default=DEFAULT_TARGET_TIME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_time = pd.to_datetime(args.target_time).strftime("%H:%M")
    target_datetime = f"{args.trade_date} {target_time}:00"
    slice_df = load_slice(args.underlying, args.trade_date, args.maturity, target_time)
    fit = get_fit_row(load_svi_fits(), target_datetime, args.underlying, args.maturity)
    if fit is None:
        raise RuntimeError("No successful SVI fit found for the requested slice.")

    forward = slice_df["F"].median()
    tenor = slice_df["T"].median()
    slice_df = slice_df.assign(
        future_log_moneyness=np.log(slice_df["strike_price"] / slice_df["F"]),
        total_variance=slice_df["implied_volatility"] ** 2 * slice_df["T"],
    )
    x = np.linspace(slice_df["future_log_moneyness"].min(), slice_df["future_log_moneyness"].max(), 320)
    y = fitted_total_variance(x, fit)
    if not np.isfinite(y).any():
        raise RuntimeError("SVI curve produced no finite fitted total variance values.")

    observed = (
        slice_df[["future_log_moneyness", "total_variance"]]
        .drop_duplicates()
        .sort_values("future_log_moneyness")
    )
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(
        observed["future_log_moneyness"],
        observed["total_variance"],
        s=46,
        marker="o",
        color="#1f77b4",
        alpha=0.86,
        label="observed total variance",
    )
    ax.plot(x, y, color="#111111", linewidth=2.2, label="SVI fitted total variance")
    spot_x = np.log(slice_df["underlying_price"].median() / forward)
    ax.axvline(spot_x, color="#666666", linestyle=":", linewidth=1.5, label="Spot price")
    rmse = fit.get("fit_rmse")
    rmse_text = f", RMSE={rmse:.6f}" if pd.notna(rmse) else ""
    ax.set_title(f"SVI fit: {args.underlying}, maturity {args.maturity}, {target_datetime}{rmse_text}")
    ax.set_xlabel("future_log_moneyness")
    ax.set_ylabel("total variance")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / (
        f"svi_fit_slice_{args.underlying.replace('.', '')}_{args.maturity}_"
        f"{args.trade_date}_{target_time.replace(':', '')}.png"
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Filtered rows: {len(slice_df)}")
    print(f"Unique strikes: {slice_df['strike_price'].nunique()}")
    print(f"T median: {tenor:.8f}")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()

