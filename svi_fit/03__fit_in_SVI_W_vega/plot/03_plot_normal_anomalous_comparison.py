"""Plot normal and anomalous Vega-weighted SVI fits side by side."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from svi_plot_helpers import FIGURE_DIR, fitted_total_variance, get_fit_row, load_slice, load_svi_fits


UNDERLYING_CODE = "000016.SH"
MATURITY_DATE = 20230317
CASES = [
    {"label": "(a) Normal SVI Fit", "trade_date": "2023-01-04", "target_time": "10:00"},
    {"label": "(b) Anomalous SVI Fit", "trade_date": "2023-03-01", "target_time": "13:50"},
]
PAPER_COLORS = {
    "blue": "#4C72B0",
    "black": "#2A2A2A",
    "grey": "#7F7F7F",
}


def configure_paper_style() -> None:
    """Apply publication-style figure settings."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "Times New Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.color": PAPER_COLORS["grey"],
            "grid.alpha": 0.30,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax) -> None:
    """Style one comparison axis."""
    ax.set_facecolor("#F2F2F2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, prune="both"))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, prune="both"))


def padded_limits(values: list[np.ndarray], pad_fraction: float) -> tuple[float, float]:
    """Return padded lower and upper bounds for finite values."""
    combined = np.concatenate([value[np.isfinite(value)] for value in values])
    lower = combined.min()
    upper = combined.max()
    padding = (upper - lower) * pad_fraction
    return lower - padding, upper + padding


def prepare_case(case: dict, svi_fits: pd.DataFrame) -> dict:
    """Prepare observed and fitted data for one comparison panel."""
    target_time = pd.to_datetime(case["target_time"]).strftime("%H:%M")
    target_datetime = f"{case['trade_date']} {target_time}:00"
    quote_df = load_slice(UNDERLYING_CODE, case["trade_date"], MATURITY_DATE, target_time)
    fit = get_fit_row(svi_fits, target_datetime, UNDERLYING_CODE, MATURITY_DATE)
    if fit is None:
        raise RuntimeError(f"No successful SVI fit found for {target_datetime}.")

    quote_df = quote_df.assign(
        future_log_moneyness=np.log(quote_df["strike_price"] / quote_df["F"])
    )
    observed = (
        quote_df[["future_log_moneyness", "implied_volatility"]]
        .drop_duplicates()
        .sort_values("future_log_moneyness")
    )
    x = np.linspace(quote_df["future_log_moneyness"].min(), quote_df["future_log_moneyness"].max(), 360)
    total_variance = fitted_total_variance(x, fit)
    fitted_iv = np.full_like(total_variance, np.nan, dtype=float)
    valid_variance = total_variance >= 0
    fitted_iv[valid_variance] = np.sqrt(total_variance[valid_variance] / quote_df["T"].median())
    valid = np.isfinite(fitted_iv)
    if not valid.any():
        raise RuntimeError(f"No finite SVI implied volatility values for {target_datetime}.")
    spot_x = np.log(quote_df["underlying_price"].median() / quote_df["F"].median())
    return {
        "label": case["label"],
        "observed": observed,
        "x": x,
        "fitted_iv": fitted_iv,
        "valid": valid,
        "spot_x": spot_x,
        "rows": len(quote_df),
        "points": len(observed),
        "rmse": fit.get("fit_rmse"),
    }


def main() -> None:
    configure_paper_style()
    svi_fits = load_svi_fits()
    cases = [prepare_case(case, svi_fits) for case in CASES]
    xlim = padded_limits([case["x"] for case in cases], 0.04)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=False)
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.09, right=0.99, wspace=0.12)
    for ax, case in zip(axes, cases):
        ax.scatter(
            case["observed"]["future_log_moneyness"],
            case["observed"]["implied_volatility"],
            s=9,
            marker="o",
            color=PAPER_COLORS["black"],
            edgecolors="none",
            alpha=0.55,
            label="Obs.",
            zorder=3,
        )
        ax.plot(case["x"], case["fitted_iv"], color=PAPER_COLORS["blue"], linewidth=1.6, label="SVI fit")
        ax.axvline(
            case["spot_x"],
            color=PAPER_COLORS["grey"],
            linestyle=":",
            linewidth=0.9,
            alpha=0.75,
            label="Spot",
        )
        ax.set_title(case["label"], fontsize=10, fontweight="bold", pad=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(
            *padded_limits(
                [case["observed"]["implied_volatility"].to_numpy(), case["fitted_iv"][case["valid"]]],
                0.07,
            )
        )
        ax.set_xlabel(r"Log-moneyness $k = \log(K/F)$")
        style_axis(ax)
        ax.legend(loc="upper right", ncol=1, handletextpad=0.5, borderaxespad=0.3)
    axes[0].set_ylabel("Implied volatility")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "svi_normal_special_comparison_000016SH_20230317.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    for case in cases:
        rmse = case["rmse"]
        rmse_text = f"{rmse:.10f}" if pd.notna(rmse) else "NA"
        print(f"{case['label']}: rows={case['rows']}, observed_points={case['points']}, RMSE={rmse_text}")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()

