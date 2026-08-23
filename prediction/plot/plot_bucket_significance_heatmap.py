import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


BASE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE / "data" / "processed" / "rolling_split_model_coefficients.csv"
DEFAULT_OUTPUT_DIR = BASE / "output" / "figures"

BUCKET_ORDER = [
    "ATM_LT",
    "ATM_MT",
    "ATM_ST",
    "CW_LT",
    "CW_MT",
    "CW_ST",
    "DCW_LT",
    "DCW_MT",
    "DCW_ST",
    "DPW_LT",
    "DPW_MT",
    "DPW_ST",
    "PW_LT",
    "PW_MT",
    "PW_ST",
]

MODEL_LABELS = {
    "model1": "M1",
    "model2": "M2",
    "model3": "M3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot bucket-level signal-coefficient significance from processed rolling-split outputs."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Processed coefficient CSV. Default: ../data/processed/rolling_split_model_coefficients.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Figure output directory. Default: ../output/figures",
    )
    parser.add_argument(
        "--basename",
        default="bucket_significance_heatmap",
        help="Base filename for the PDF and PNG outputs.",
    )
    return parser.parse_args()


def significance_level(stars):
    if not isinstance(stars, str):
        return 0
    return len(stars.strip())


def ordered_values(values, preferred_order=None):
    preferred_order = preferred_order or []
    present = list(dict.fromkeys(str(value) for value in values))
    ordered = [value for value in preferred_order if value in present]
    ordered.extend(sorted(value for value in present if value not in ordered))
    return ordered


def load_significance_grid(path):
    df = pd.read_csv(path)
    required = {
        "split",
        "model",
        "bucket_label",
        "beta_signal",
        "sig_beta_signal",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df.loc[df["model"].isin(MODEL_LABELS)].copy()
    if df.empty:
        raise ValueError("The coefficient file does not contain model1, model2, or model3 rows.")

    df["bucket_label"] = df["bucket_label"].astype(str)
    df["split"] = df["split"].astype(str)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["beta_signal"] = pd.to_numeric(df["beta_signal"], errors="coerce")
    df["sig_level"] = df["sig_beta_signal"].apply(significance_level)
    df["sign"] = np.sign(df["beta_signal"]).fillna(0).astype(int)
    df["plot_value"] = df["sign"] * df["sig_level"]
    df["plot_label"] = np.where(
        df["sig_level"] > 0,
        np.where(df["sign"] >= 0, "+", "-") + df["sig_beta_signal"].astype(str),
        "",
    )

    splits = ordered_values(df["split"], ["A", "B", "C"])
    models = [MODEL_LABELS[key] for key in ["model1", "model2", "model3"]]
    buckets = ordered_values(df["bucket_label"], BUCKET_ORDER)
    columns = [(split, model) for split in splits for model in models]

    value_matrix = np.zeros((len(buckets), len(columns)))
    label_matrix = np.full((len(buckets), len(columns)), "", dtype=object)

    indexed = df.set_index(["bucket_label", "split", "model_label"])
    for row_idx, bucket in enumerate(buckets):
        for col_idx, (split, model) in enumerate(columns):
            key = (bucket, split, model)
            if key not in indexed.index:
                continue
            record = indexed.loc[key]
            if isinstance(record, pd.DataFrame):
                record = record.iloc[0]
            value_matrix[row_idx, col_idx] = float(record["plot_value"])
            label_matrix[row_idx, col_idx] = str(record["plot_label"])

    return buckets, splits, models, columns, value_matrix, label_matrix


def plot_heatmap(buckets, splits, columns, value_matrix, label_matrix, output_dir, basename):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
        }
    )

    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    cmap = plt.cm.RdBu_r

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    ax.imshow(value_matrix, cmap=cmap, norm=norm, aspect="auto")

    n_rows, n_cols = value_matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            label = label_matrix[i, j]
            if label:
                value = value_matrix[i, j]
                text_color = "white" if abs(value) >= 2 else "black"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([model for _, model in columns], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(buckets, fontsize=9.5)

    for split_index, split in enumerate(splits):
        center = split_index * 3 + 1
        ax.text(center, -1.15, f"Split {split}", ha="center", va="center", fontsize=11, fontweight="bold")

    for split_boundary in range(1, len(splits)):
        ax.axvline(split_boundary * 3 - 0.5, color="black", linewidth=1.2)

    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.0)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.6)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -1.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor=cmap(norm(3)), edgecolor="none", label="+ *** (p<0.01)"),
        mpatches.Patch(facecolor=cmap(norm(2)), edgecolor="none", label="+ ** (p<0.05)"),
        mpatches.Patch(facecolor=cmap(norm(1)), edgecolor="none", label="+ * (p<0.10)"),
        mpatches.Patch(facecolor=cmap(norm(0)), edgecolor="gray", label="not significant"),
        mpatches.Patch(facecolor=cmap(norm(-1)), edgecolor="none", label="- * (p<0.10)"),
        mpatches.Patch(facecolor=cmap(norm(-2)), edgecolor="none", label="- ** (p<0.05)"),
        mpatches.Patch(facecolor=cmap(norm(-3)), edgecolor="none", label="- *** (p<0.01)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        handleheight=1.4,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"

    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    buckets, splits, models, columns, value_matrix, label_matrix = load_significance_grid(input_path)
    pdf_path, png_path = plot_heatmap(
        buckets,
        splits,
        columns,
        value_matrix,
        label_matrix,
        output_dir,
        args.basename,
    )

    print(f"input={input_path}")
    print(f"splits={','.join(splits)}")
    print(f"models={','.join(models)}")
    print(f"buckets={len(buckets)}")
    print(f"pdf={pdf_path}")
    print(f"png={png_path}")


if __name__ == "__main__":
    main()
