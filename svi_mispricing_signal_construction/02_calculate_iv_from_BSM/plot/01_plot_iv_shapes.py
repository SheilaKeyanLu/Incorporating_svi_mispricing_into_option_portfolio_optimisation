"""Plot representative implied-volatility shapes from the processed dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_DIR / "data" / "processed" / "option_quotes_with_iv.csv"
FIGURE_DIR = PROJECT_DIR / "output" / "figures"

USECOLS = [
    "trade_date",
    "target_time",
    "ths_underlying_code_option",
    "ths_maturity_date_option",
    "option_type",
    "strike_price",
    "implied_volatility",
]


def select_representative_slice(chunk_size: int = 300_000) -> tuple[str, int, list[str]]:
    """Choose a liquid underlying/maturity pair and the first four trade dates."""
    counts = []
    for chunk in pd.read_csv(PROCESSED_PATH, usecols=USECOLS, chunksize=chunk_size):
        chunk = chunk.dropna(subset=["implied_volatility"])
        grouped = (
            chunk.groupby(["ths_underlying_code_option", "ths_maturity_date_option", "trade_date"])
            .size()
            .reset_index(name="row_count")
        )
        counts.append(grouped)

    if not counts:
        raise RuntimeError("No finite implied-volatility rows are available for plotting.")

    count_df = pd.concat(counts, ignore_index=True)
    count_df = (
        count_df.groupby(["ths_underlying_code_option", "ths_maturity_date_option", "trade_date"], as_index=False)[
            "row_count"
        ]
        .sum()
    )

    pair_summary = (
        count_df.groupby(["ths_underlying_code_option", "ths_maturity_date_option"])
        .agg(date_count=("trade_date", "nunique"), row_count=("row_count", "sum"))
        .reset_index()
    )
    pair_summary = pair_summary[pair_summary["date_count"] >= 4]
    if pair_summary.empty:
        raise RuntimeError("No underlying/maturity pair has at least four trade dates.")

    selected = pair_summary.sort_values(
        ["date_count", "row_count", "ths_underlying_code_option", "ths_maturity_date_option"],
        ascending=[False, False, True, True],
    ).iloc[0]
    underlying_code = selected["ths_underlying_code_option"]
    maturity_date = int(selected["ths_maturity_date_option"])

    trade_dates = (
        count_df[
            count_df["ths_underlying_code_option"].eq(underlying_code)
            & count_df["ths_maturity_date_option"].eq(maturity_date)
        ]["trade_date"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .head(4)
        .tolist()
    )
    return underlying_code, maturity_date, trade_dates


def load_target_quotes(underlying_code: str, maturity_date: int, trade_dates: list[str]) -> pd.DataFrame:
    """Load only the selected rows needed by the figure."""
    parts = []
    wanted_dates = set(trade_dates)

    for chunk in pd.read_csv(PROCESSED_PATH, usecols=USECOLS, chunksize=300_000):
        mask = (
            chunk["ths_underlying_code_option"].eq(underlying_code)
            & chunk["ths_maturity_date_option"].eq(maturity_date)
            & chunk["trade_date"].astype(str).isin(wanted_dates)
        )
        if mask.any():
            parts.append(chunk.loc[mask].copy())

    if not parts:
        return pd.DataFrame(columns=USECOLS)

    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df["target_time"] = df["target_time"].astype(str)
    df = df.dropna(subset=["strike_price", "implied_volatility"])
    return df.sort_values(["trade_date", "target_time", "option_type", "strike_price"])


def plot_one_day(ax, day_df: pd.DataFrame, trade_date: str, show_legend: bool) -> None:
    """Plot intraday IV curves for one trade date."""
    times = sorted(day_df["target_time"].unique())
    cmap = plt.get_cmap("tab10")
    markers = {"C": "o", "P": "s"}

    for time_idx, target_time in enumerate(times):
        time_df = day_df[day_df["target_time"].eq(target_time)]
        color = cmap(time_idx % 10)

        for option_type in ["C", "P"]:
            curve = time_df[time_df["option_type"].eq(option_type)].sort_values("strike_price")
            if curve.empty:
                continue

            label = f"{target_time} {option_type}" if show_legend else None
            ax.plot(
                curve["strike_price"],
                curve["implied_volatility"],
                color=color,
                marker=markers.get(option_type, "o"),
                markersize=4.5,
                linewidth=1.0,
                alpha=0.82,
                label=label,
            )

    ax.set_title(trade_date)
    ax.set_xlabel("Strike price")
    ax.set_ylabel("Implied volatility")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)


def save_combined_plot(df: pd.DataFrame, underlying_code: str, maturity_date: int, trade_dates: list[str]) -> Path:
    """Save a four-panel IV-shape figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for index, trade_date in enumerate(trade_dates):
        day_df = df[df["trade_date"].eq(trade_date)]
        plot_one_day(axes[index], day_df, trade_date, show_legend=(index == 0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Snapshot / type", fontsize=8)
    fig.suptitle(
        f"IV shapes for {underlying_code}, maturity {maturity_date}",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    safe_underlying = underlying_code.replace(".", "")
    date_span = f"{trade_dates[0].replace('-', '')}_{trade_dates[-1].replace('-', '')}"
    out_path = FIGURE_DIR / f"iv_shapes_{safe_underlying}_{maturity_date}_{date_span}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_daily_plots(df: pd.DataFrame, underlying_code: str, maturity_date: int, trade_dates: list[str]) -> list[Path]:
    """Save one IV-shape figure per selected trade date."""
    out_paths = []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    safe_underlying = underlying_code.replace(".", "")

    for trade_date in trade_dates:
        day_df = df[df["trade_date"].eq(trade_date)]
        fig, ax = plt.subplots(figsize=(11, 7))
        plot_one_day(ax, day_df, trade_date, show_legend=True)
        ax.legend(title="Snapshot / type", fontsize=8, ncol=2)
        fig.suptitle(f"IV shape for {underlying_code}, maturity {maturity_date}", fontsize=13)
        fig.tight_layout()

        out_path = FIGURE_DIR / f"iv_shape_{safe_underlying}_{maturity_date}_{trade_date}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Missing processed file: {PROCESSED_PATH}")

    underlying_code, maturity_date, trade_dates = select_representative_slice()
    df = load_target_quotes(underlying_code, maturity_date, trade_dates)
    if df.empty:
        raise RuntimeError("No matching rows found for the selected plotting slice.")

    combined = save_combined_plot(df, underlying_code, maturity_date, trade_dates)
    daily = save_daily_plots(df, underlying_code, maturity_date, trade_dates)

    print(f"Selected underlying: {underlying_code}")
    print(f"Selected maturity: {maturity_date}")
    print(f"Selected trade dates: {', '.join(trade_dates)}")
    print(f"Filtered rows: {len(df):,}")
    print(f"Saved combined plot: {combined}")
    for path in daily:
        print(f"Saved daily plot: {path}")


if __name__ == "__main__":
    main()

