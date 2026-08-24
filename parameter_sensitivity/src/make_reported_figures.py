from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIG_DIR = OUTPUT_DIR / "THESIS_FIGURES"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "SimSun"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def pct_axis(x, _):
    return f"{x:.0%}"


def format_axes(ax):
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.9)
    ax.grid(True, axis="x", color="#efefef", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def save_figure(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_combined_cumulative_return(summary, returns):
    lambda_df = (
        summary[summary["group"] == "lambda_sweep_delta_5"]
        .sort_values("lambda_risk")
        .reset_index(drop=True)
    )
    delta_df = (
        summary[summary["group"] == "delta_sweep_lambda_10000"]
        .sort_values("delta")
        .reset_index(drop=True)
    )

    lambda_columns = [
        f"lambda_{int(value)}_delta_5_cumulative_return"
        for value in lambda_df["lambda_risk"]
    ]
    lambda_labels = [rf"$\lambda={int(value)}$" for value in lambda_df["lambda_risk"]]

    delta_columns = []
    delta_labels = []
    for value in delta_df["delta"]:
        run_id = (
            "lambda_10000_delta_5_delta_sweep"
            if int(value) == 5
            else f"lambda_10000_delta_{int(value)}"
        )
        delta_columns.append(f"{run_id}_cumulative_return")
        delta_labels.append(rf"$\delta={int(value)}$")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    dates = pd.to_datetime(returns["next_date"])
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    colors = ["#1f2937", "#2563eb", "#b45309", "#047857", "#7c3aed", "#be123c", "#4b5563"]

    panels = [
        (
            axes[0],
            lambda_columns,
            lambda_labels,
            r"(a) Risk aversion $\lambda$ with $\delta=5$",
        ),
        (
            axes[1],
            delta_columns,
            delta_labels,
            r"(b) View uncertainty $\delta$ with $\lambda=10000$",
        ),
    ]

    for ax, columns, labels, title in panels:
        for i, (column, label) in enumerate(zip(columns, labels)):
            ax.plot(
                dates,
                returns[column].astype(float),
                label=label,
                linestyle=styles[i % len(styles)],
                color=colors[i % len(colors)],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.yaxis.set_major_formatter(FuncFormatter(pct_axis))
        format_axes(ax)
        ax.legend(frameon=False, ncol=2, loc="upper left")

    axes[0].set_ylabel("Cumulative return")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, "figure_combined_cumulative_return_sensitivity")


def annotate_points(ax, x, y, is_percent):
    for xi, yi in zip(x, y):
        label = f"{yi:.1%}" if is_percent else f"{yi:.2f}"
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.8,
        )


def plot_metric(ax, x, y, color, y_label, is_percent):
    ax.plot(x, y, marker="o", color=color)
    annotate_points(ax, x, y, is_percent)
    ax.set_ylabel(y_label)
    if is_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(pct_axis))
    format_axes(ax)


def plot_lambda_delta_total_return_sharpe(summary):
    lambda_df = (
        summary[summary["group"] == "lambda_sweep_delta_5"]
        .sort_values("lambda_risk")
        .reset_index(drop=True)
    )
    delta_df = (
        summary[summary["group"] == "delta_sweep_lambda_10000"]
        .sort_values("delta")
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9))
    lambda_x = lambda_df["lambda_risk"].astype(int)
    delta_x = delta_df["delta"].astype(int)

    plot_metric(
        axes[0, 0],
        lambda_x,
        lambda_df["total_return"].astype(float),
        "#1f2937",
        "Total return",
        True,
    )
    plot_metric(
        axes[0, 1],
        lambda_x,
        lambda_df["sharpe_ratio"].astype(float),
        "#2563eb",
        "Sharpe ratio",
        False,
    )
    plot_metric(
        axes[1, 0],
        delta_x,
        delta_df["total_return"].astype(float),
        "#1f2937",
        "Total return",
        True,
    )
    plot_metric(
        axes[1, 1],
        delta_x,
        delta_df["sharpe_ratio"].astype(float),
        "#2563eb",
        "Sharpe ratio",
        False,
    )

    axes[0, 0].set_title("Total return", y=1.02)
    axes[0, 1].set_title("Sharpe ratio", y=1.02)
    axes[0, 0].set_xlabel(r"$\lambda$")
    axes[0, 1].set_xlabel(r"$\lambda$")
    axes[1, 0].set_xlabel(r"$\delta$")
    axes[1, 1].set_xlabel(r"$\delta$")

    fig.text(0.5, 0.982, r"(a) $\lambda$ change", ha="center", va="top", fontsize=10.5)
    fig.text(0.5, 0.472, r"(b) $\delta$ change", ha="center", va="top", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.948], h_pad=3.0, w_pad=2.05)
    save_figure(fig, "figure_lambda_delta_total_return_sharpe_stacked")


def main():
    summary = pd.read_csv(OUTPUT_DIR / "parameter_sensitivity_summary.csv")
    returns = pd.read_csv(OUTPUT_DIR / "all_runs_return_changes.csv")
    plot_combined_cumulative_return(summary, returns)
    plot_lambda_delta_total_return_sharpe(summary)


if __name__ == "__main__":
    main()
