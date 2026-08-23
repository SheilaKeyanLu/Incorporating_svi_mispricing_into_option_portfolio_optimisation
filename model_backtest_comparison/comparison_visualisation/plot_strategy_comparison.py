#!/usr/bin/env python3
"""
Plot strategy CSV files as a publication-style comparison figure.

The script reads every CSV in a folder, constructs cumulative net value
and drawdown series, and saves a thesis-ready figure with a compact
summary table.

Usage
-----
    python plot_strategy_comparison_thesis.py
    python plot_strategy_comparison_thesis.py path/to/folder
    python plot_strategy_comparison_thesis.py path/to/folder -o strategy_comparison.png
    python plot_strategy_comparison_thesis.py path/to/folder --exclude scratch,readme

Expected CSV columns, case-insensitive
--------------------------------------
    date              required
    cumulative_value  used directly if present
    daily_return      used if cumulative_value is absent
    positions         optional stringified {ticker: weight} dictionary
"""

import argparse
import ast
import colorsys
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent

RENAME = {
    "BL CVAR": "BL Long-Short (CVaR)",
    "BL Long Short": "BL Long-Short",
    "BL Long-Only": "BL Long-Only",
    "Delta-Gamma Baseline": "Delta-Gamma Approximation",
    "Markowitz Baseline": "Historical Mean-Variance",
}

PLOT_ORDER = [
    "BL Long-Short",
    "BL Long-Short (CVaR)",
    "BL Long-Only",
    "Delta-Gamma Approximation",
    "Historical Mean-Variance",
]

SKIP_LABELS = {
    "adaptive bl",
    "adaptive bl model",
}

BASE_COLORS = [
    "#1f2937",
    "#2563eb",
    "#b45309",
    "#047857",
    "#7c3aed",
    "#be123c",
    "#4b5563",
    "#0f766e",
]

LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]


def boost_saturation(hex_color, factor=1.15):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, l, min(1.0, s * factor))
    return "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in (r, g, b)))


COLOR_CYCLE = [boost_saturation(color) for color in BASE_COLORS]


def set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "SimSun"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def pct_axis(x, _):
    return f"{x:.0%}"


def dd_axis(x, _):
    return f"{x:.0%}"


def label_from_file(path):
    raw = path.stem.replace("_", " ")
    return RENAME.get(path.stem, RENAME.get(raw, raw))


def should_skip_label(label):
    return label.strip().lower() in SKIP_LABELS


def relative_to_script(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def load_strategy(path):
    try:
        df = pd.read_csv(path)
    except Exception as error:
        print(f"  skip {path.name}: could not read ({error})")
        return None, None

    cols = {column.lower(): column for column in df.columns}
    if "date" not in cols:
        print(f"  skip {path.name}: no date column")
        return None, None

    df["__date"] = pd.to_datetime(df[cols["date"]])
    df = df.sort_values("__date").set_index("__date")

    has_cum = "cumulative_value" in cols
    has_ret = "daily_return" in cols
    if not has_cum and not has_ret:
        print(f"  skip {path.name}: no cumulative_value or daily_return column")
        return None, None

    if has_cum:
        cumulative_value = df[cols["cumulative_value"]].astype(float)
    else:
        cumulative_value = (1.0 + df[cols["daily_return"]].astype(float)).cumprod()

    cumulative_value = cumulative_value / cumulative_value.iloc[0]
    return cumulative_value, df


def position_stats(df):
    cols = {column.lower(): column for column in df.columns}
    if "positions" not in cols:
        return None, None

    gross_exposure = []
    number_positions = []
    for value in df[cols["positions"]].dropna():
        try:
            positions = ast.literal_eval(value)
        except Exception:
            continue
        gross_exposure.append(sum(abs(weight) for weight in positions.values()))
        number_positions.append(sum(1 for weight in positions.values() if abs(weight) > 1e-8))

    if not gross_exposure:
        return None, None
    return float(np.mean(gross_exposure)), float(np.mean(number_positions))


def stats_from_cumulative_value(cumulative_value):
    returns = cumulative_value.pct_change().dropna()
    if returns.empty:
        keys = [
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
        ]
        return dict.fromkeys(keys, np.nan)

    total_return = cumulative_value.iloc[-1] - 1.0
    n_days = len(returns)
    final_value = cumulative_value.iloc[-1]
    annual_return = final_value ** (252 / n_days) - 1.0
    annual_volatility = returns.std(ddof=1) * np.sqrt(252)
    sharpe_ratio = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        if returns.std(ddof=1) > 0
        else np.nan
    )

    downside_returns = np.minimum(returns, 0.0)
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    sortino_ratio = (
        returns.mean() * 252 / downside_deviation
        if downside_deviation > 0
        else np.nan
    )

    drawdown = cumulative_value / cumulative_value.cummax() - 1.0
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    win_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
    }


def format_axes(ax):
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.9)
    ax.grid(True, axis="x", color="#efefef", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def make_summary_table(stats, pos_stats):
    rows = []
    for strategy, values in stats.items():
        gross_exposure, average_positions = pos_stats[strategy]
        rows.append(
            {
                "Strategy": strategy,
                "R_tot": values["total_return"],
                "R_ann": values["annual_return"],
                "sigma_ann": values["annual_volatility"],
                "SR": values["sharpe_ratio"],
                "SoR": values["sortino_ratio"],
                "MDD": values["max_drawdown"],
                "CR": values["calmar_ratio"],
                "WR": values["win_rate"],
                "GrossExp": gross_exposure,
                "AvgPos": average_positions,
            }
        )

    table = pd.DataFrame(rows)
    display = table.copy()
    pct_cols = ["R_tot", "R_ann", "sigma_ann", "MDD", "WR", "GrossExp"]
    num_cols = ["SR", "SoR", "CR", "AvgPos"]
    for column in pct_cols:
        if column in display.columns:
            display[column] = display[column].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.1%}"
            )
    for column in num_cols:
        if column in display.columns:
            display[column] = display[column].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
            )
    return table, display


def plot_comparison(series, styles, pos_stats, output_path, title=None):
    drawdowns = {
        name: cumulative_value / cumulative_value.cummax() - 1.0
        for name, cumulative_value in series.items()
    }
    stats = {
        name: stats_from_cumulative_value(cumulative_value)
        for name, cumulative_value in series.items()
    }
    stats_table, display_table = make_summary_table(stats, pos_stats)

    n = len(series)
    table_height = max(1.35, 0.28 * (n + 1) + 0.35)
    fig_height = 6.2 + table_height
    fig = plt.figure(figsize=(8.2, fig_height))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[2.35, 1.0, table_height],
        hspace=0.16,
    )

    ax_net = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_net)
    ax_table = fig.add_subplot(gs[2])

    for name, cumulative_value in series.items():
        style = styles[name]
        ax_net.plot(
            cumulative_value.index,
            cumulative_value - 1.0,
            label=name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    ax_net.axhline(0.0, color="black", linewidth=0.8)
    ax_net.set_ylabel("Cumulative return")
    ax_net.yaxis.set_major_formatter(FuncFormatter(pct_axis))
    if title:
        ax_net.set_title(title, loc="left", pad=6)
    format_axes(ax_net)
    ax_net.legend(loc="upper left", ncols=2, frameon=False, handlelength=2.4)
    plt.setp(ax_net.get_xticklabels(), visible=False)

    for name, drawdown in drawdowns.items():
        style = styles[name]
        ax_dd.fill_between(
            drawdown.index,
            drawdown,
            0,
            color=style["color"],
            alpha=0.13,
            linewidth=0,
        )
        ax_dd.plot(
            drawdown.index,
            drawdown,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.0,
        )

    ax_dd.axhline(0.0, color="black", linewidth=0.7)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(dd_axis))
    ax_dd.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_dd.tick_params(axis="x", labelbottom=True, pad=3)
    plt.setp(ax_dd.get_xticklabels(), rotation=30, ha="right")
    format_axes(ax_dd)

    ax_table.axis("off")
    table_columns = [
        "Strategy",
        "R_tot",
        "R_ann",
        "sigma_ann",
        "SR",
        "SoR",
        "MDD",
        "CR",
        "WR",
    ]
    if display_table["GrossExp"].astype(str).str.len().gt(0).any():
        table_columns.extend(["GrossExp", "AvgPos"])

    table_column_labels = [
        {
            "R_tot": r"$R_{\mathrm{tot}}$",
            "R_ann": r"$R_{\mathrm{ann}}$",
            "sigma_ann": r"$\sigma_{\mathrm{ann}}$",
        }.get(column, column)
        for column in table_columns
    ]

    strategy_col_width = 0.25
    other_col_width = (1.0 - strategy_col_width) / (len(table_columns) - 1)
    col_widths = [strategy_col_width] + [other_col_width] * (len(table_columns) - 1)

    mpl_table = ax_table.table(
        cellText=display_table[table_columns].values,
        colLabels=table_column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[-0.055, 0.09, 1.055, 0.6],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(7.6)
    mpl_table.scale(1.0, 1.2)

    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#4b5563")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f3f4f6")
        elif row % 2 == 0:
            cell.set_facecolor("#fafafa")
        if col == 0:
            cell.set_text_props(ha="left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    stats_table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return pdf_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder containing strategy CSV files, relative to this script.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="strategy_comparison_thesis.png",
        help="Output PNG path, relative to this script. A PDF and summary CSV are also saved.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for input files.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated filename substrings to exclude.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional in-figure title. Leave empty for thesis-style captions.",
    )
    args = parser.parse_args()

    set_publication_style()

    folder = relative_to_script(args.folder)
    output_path = relative_to_script(args.output)
    exclude = [item.strip().lower() for item in args.exclude.split(",") if item.strip()]
    files = sorted(
        file
        for file in folder.glob(args.pattern)
        if not any(item in file.name.lower() for item in exclude)
    )

    if not files:
        sys.exit(f"No CSV files matched in {folder.resolve()}")

    print(f"Found {len(files)} file(s) in {folder.resolve()}:")
    series = {}
    styles = {}
    position_summary = {}
    color_index = 0

    for file in files:
        cumulative_value, raw_df = load_strategy(file)
        if cumulative_value is None:
            continue

        label = label_from_file(file)
        if should_skip_label(label):
            print(f"  skip {file.name}: Adaptive BL Model removed from comparison")
            continue

        is_baseline = "baseline" in file.stem.lower()
        styles[label] = {
            "color": COLOR_CYCLE[color_index % len(COLOR_CYCLE)],
            "linestyle": "--" if is_baseline else LINE_STYLES[color_index % len(LINE_STYLES)],
            "linewidth": 1.45 if is_baseline else 1.75,
        }
        color_index += 1
        series[label] = cumulative_value
        position_summary[label] = position_stats(raw_df)
        print(
            f"  loaded {file.name} -> {label} "
            f"({len(cumulative_value)} rows, "
            f"{cumulative_value.index.min().date()} to {cumulative_value.index.max().date()})"
        )

    if not series:
        sys.exit("No usable CSVs found.")

    ordered_names = [name for name in PLOT_ORDER if name in series]
    ordered_names.extend(name for name in series if name not in ordered_names)
    series = {name: series[name] for name in ordered_names}
    styles = {name: styles[name] for name in ordered_names}
    position_summary = {name: position_summary[name] for name in ordered_names}

    pdf_path, summary_path = plot_comparison(
        series,
        styles,
        position_summary,
        output_path,
        title=args.title.strip() or None,
    )

    print(f"\nSaved PNG: {output_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")
    print(f"Saved summary CSV: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
