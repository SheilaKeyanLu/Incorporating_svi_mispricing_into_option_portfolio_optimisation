"""Plot the OLS persistence regression for daily mispricing signals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_DIR / "data" / "processed" / "option_quotes_with_mispricing_signal.csv"
FIGURE_DIR = PROJECT_DIR / "output" / "figures"
REPORT_DIR = PROJECT_DIR / "output" / "report"
PLOT_REPORT_PATH = REPORT_DIR / "ols_regression_plot_summary.txt"


def build_signal_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Construct adjacent trading-day signal pairs by contract symbol."""
    daily_signal = (
        df[["symbol", "trade_date", "daily_mispricing_signal"]]
        .dropna()
        .drop_duplicates(["symbol", "trade_date"])
        .copy()
    )
    daily_signal["trade_date"] = pd.to_datetime(daily_signal["trade_date"])
    daily_signal = daily_signal.sort_values(["symbol", "trade_date"])
    daily_signal["next_day_signal"] = daily_signal.groupby("symbol")[
        "daily_mispricing_signal"
    ].shift(-1)
    return daily_signal.dropna(subset=["daily_mispricing_signal", "next_day_signal"])


def configure_plot_style() -> None:
    """Apply publication-style settings for the OLS scatter plot."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 300,
        }
    )


def write_plot_report(pair_count: int, slope: float, intercept: float, r_squared: float, p_value: float) -> None:
    """Write the OLS statistics used in the plot."""
    lines = [
        "OLS persistence plot summary",
        "============================",
        "",
        f"Processed input file: {PROCESSED_PATH.relative_to(PROJECT_DIR)}",
        f"Adjacent contract-day pairs: {pair_count:,}",
        f"OLS slope: {slope:.6f}",
        f"OLS intercept: {intercept:.6f}",
        f"R-squared: {r_squared:.6f}",
        f"p-value: {p_value:.6e}",
    ]
    PLOT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Missing processed input file: {PROCESSED_PATH}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.read_csv(PROCESSED_PATH, low_memory=False)
    pairs = build_signal_pairs(result)
    if pairs.empty:
        raise RuntimeError("No adjacent daily signal pairs are available for OLS plotting.")

    x = pairs["daily_mispricing_signal"].to_numpy()
    y = pairs["next_day_signal"].to_numpy()
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(
        x,
        y,
        s=10,
        facecolors="#3B6FA0",
        edgecolors="none",
        alpha=0.35,
        rasterized=True,
        label="Contract-day pairs",
    )
    ax.plot(
        x_line,
        y_line,
        color="#B33018",
        linewidth=1.4,
        label=rf"OLS fit ($\beta$={slope:.3f}, $R^2$={r_value**2:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"$z_{i,d}^{\mathrm{daily}}$")
    ax.set_ylabel(r"$z_{i,d+1}^{\mathrm{daily}}$")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    pdf_path = FIGURE_DIR / "fig_signal_persistence.pdf"
    png_path = FIGURE_DIR / "fig_signal_persistence.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    write_plot_report(len(pairs), slope, intercept, r_value**2, p_value)
    print(f"Saved OLS persistence PDF: {pdf_path}")
    print(f"Saved OLS persistence PNG: {png_path}")
    print(f"Saved plot report: {PLOT_REPORT_PATH}")


if __name__ == "__main__":
    main()

