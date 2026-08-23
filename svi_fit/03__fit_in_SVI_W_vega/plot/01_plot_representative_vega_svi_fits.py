"""Plot representative Vega-weighted SVI fits for the dissertation figure."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from svi_plot_helpers import (
    FIGURE_DIR,
    INTRADAY_TIMES,
    TARGET_TIME,
    TENOR_LABELS,
    UNDERLYINGS,
    load_quotes_for_date_times,
    load_svi_fits,
    panel_data,
    select_representative_date,
    select_representative_maturities,
)


PAPER_COLORS = {
    "blue": "#4C72B0",
    "black": "#2A2A2A",
    "grey": "#7F7F7F",
}


def configure_paper_style() -> None:
    """Apply a compact publication style for SVI figures."""
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
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
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


def collect_axis_limits(payloads, include_curve_x: bool = True):
    """Collect padded axis limits from panel payloads."""
    x_values = []
    y_values = []
    for quotes, x_curve, y_curve in payloads:
        x_values.append(quotes["log_moneyness"].to_numpy(dtype=float))
        y_values.append(quotes["implied_volatility"].to_numpy(dtype=float))
        if include_curve_x:
            x_values.append(x_curve)
        y_values.append(y_curve)

    x = np.concatenate([values[np.isfinite(values)] for values in x_values if len(values)])
    y = np.concatenate([values[np.isfinite(values)] for values in y_values if len(values)])
    x_pad = max((x.max() - x.min()) * 0.08, 0.01)
    y_pad = max((y.max() - y.min()) * 0.10, 0.01)
    return (x.min() - x_pad, x.max() + x_pad), (max(0.0, y.min() - y_pad), y.max() + y_pad)


def style_axis(ax) -> None:
    """Style one plot axis."""
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, prune="both"))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, prune="both"))


def plot_fit_panel(ax, panel_df, x_curve, y_curve, title, show_ylabel, show_xlabel) -> None:
    """Plot one SVI panel."""
    dedup = panel_df.drop_duplicates(subset="log_moneyness", keep="first")
    ax.scatter(
        dedup["log_moneyness"],
        dedup["implied_volatility"],
        s=9,
        marker="o",
        color=PAPER_COLORS["black"],
        alpha=0.55,
        edgecolors="none",
    )
    ax.plot(x_curve, y_curve, color=PAPER_COLORS["blue"], linewidth=1.6)
    ax.set_title(title, fontsize=9, pad=4)
    if show_ylabel:
        ax.set_ylabel("Implied volatility", fontsize=9)
    if show_xlabel:
        ax.set_xlabel(r"Log-moneyness $k=\log(K/F)$", fontsize=9)
    style_axis(ax)


def save_main_grid(quotes, svi_fits, trade_date, maturity_selection):
    """Save the 3 by 3 representative SVI fit grid."""
    panel_xlims = {}
    panel_ylims = {}
    for code in UNDERLYINGS:
        for tenor_label in TENOR_LABELS:
            selected = maturity_selection[
                maturity_selection["underlying_code"].eq(code)
                & maturity_selection["tenor_label"].eq(tenor_label)
            ].iloc[0]
            q, _, x_curve, y_curve = panel_data(
                quotes, svi_fits, trade_date, TARGET_TIME, code, selected["maturity"]
            )
            panel_xlims[(code, tenor_label)], panel_ylims[(code, tenor_label)] = collect_axis_limits(
                [(q, x_curve, y_curve)], include_curve_x=False
            )

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=False, sharey=False)
    for row_idx, (code, name) in enumerate(UNDERLYINGS.items()):
        for col_idx, tenor_label in enumerate(TENOR_LABELS):
            selected = maturity_selection[
                maturity_selection["underlying_code"].eq(code)
                & maturity_selection["tenor_label"].eq(tenor_label)
            ].iloc[0]
            q, fit_row, x_curve, y_curve = panel_data(
                quotes, svi_fits, trade_date, TARGET_TIME, code, selected["maturity"]
            )
            if q.empty or fit_row is None or len(x_curve) == 0:
                raise RuntimeError(f"Missing panel data for {name}, {tenor_label}.")
            tenor_days = int(round(float(selected["T"]) * 242))
            plot_fit_panel(
                axes[row_idx, col_idx],
                q,
                x_curve,
                y_curve,
                f"{tenor_label} ({tenor_days}d)",
                show_ylabel=(col_idx == 0),
                show_xlabel=(row_idx == 2),
            )
            axes[row_idx, col_idx].set_xlim(panel_xlims[(code, tenor_label)])
            axes[row_idx, col_idx].set_ylim(panel_ylims[(code, tenor_label)])
        axes[row_idx, 0].annotate(
            name,
            xy=(-0.32, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout(rect=(0.05, 0.0, 1.0, 0.96), w_pad=0.6, h_pad=0.85)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"svi_fit_grid_vega_{trade_date.replace('-', '')}_{TARGET_TIME.replace(':', '')}"
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    configure_paper_style()
    svi_fits = load_svi_fits()
    trade_date = select_representative_date(svi_fits)
    quotes = load_quotes_for_date_times(trade_date, set(INTRADAY_TIMES))
    maturity_selection = select_representative_maturities(quotes, svi_fits, trade_date)
    main_path = save_main_grid(quotes, svi_fits, trade_date, maturity_selection)
    print(f"Representative date: {trade_date}")
    print(f"Saved main grid: {main_path}")
    print(f"Saved main grid PDF: {main_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
