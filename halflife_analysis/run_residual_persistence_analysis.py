import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.tsa.stattools import acf, pacf


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL_COL = "symbol"
DATE_COL = "trade_date"
RESID_COL = "standardised_residual"
MONEY_COL = "moneyness_bucket"
TENOR_COL = "tenor_bucket"

UNDERLYINGS = {
    "SSE 50": {
        "index": "000016.SH",
        "file": "option_quotes_for_residual_study_000016_SH.csv",
    },
    "CSI 300": {
        "index": "000300.SH",
        "file": "option_quotes_for_residual_study_000300_SH.csv",
    },
    "CSI 1000": {
        "index": "000852.SH",
        "file": "option_quotes_for_residual_study_000852_SH.csv",
    },
}

MONEYNESS_ORDER = ["DPW", "PW", "ATM", "CW", "DCW"]
TENOR_ORDER = ["ST", "MT", "LT"]
UNDERLYING_ORDER = ["SSE 50", "CSI 300", "CSI 1000"]
TABLE_UNDERLYING_ORDER = ["SSE 50", "CSI 300", "CSI 1000"]

MIN_OBS_PER_CONTRACT = 10
SIG_LEVEL = 0.10
PHI_MIN = 1e-6
PHI_MAX = 0.999
N_LAGS = 15
ALPHA = 0.05
MIN_OBS_ACF = 20

PAPER_COLORS = {
    "fit": "#1F3B63",
    "grey": "#7F7F7F",
    "black": "#2A2A2A",
}


def clean_output_dir() -> None:
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 8.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_calendar() -> dict:
    with open(INPUT_DIR / "trading_dates.json", encoding="utf-8") as f:
        calendar = pd.to_datetime(json.load(f))
    return pd.Series(np.arange(len(calendar)), index=calendar).sort_index().to_dict()


def load_underlying_quotes(label: str) -> pd.DataFrame:
    path = INPUT_DIR / UNDERLYINGS[label]["file"]
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def build_daily_quotes(df: pd.DataFrame, day_idx_map: dict) -> pd.DataFrame:
    df = df.sort_values([SYMBOL_COL, DATE_COL])
    daily = (
        df.groupby([SYMBOL_COL, DATE_COL])
        .agg(
            r=(RESID_COL, "mean"),
            moneyness_bucket=(MONEY_COL, "last"),
            tenor_bucket=(TENOR_COL, "last"),
        )
        .reset_index()
    )
    daily["day_idx"] = daily[DATE_COL].map(day_idx_map)
    missing_calendar = daily["day_idx"].isna().sum()
    if missing_calendar:
        print(f"Dropped {missing_calendar} rows whose trade_date is not in trading_dates.json")
        daily = daily.dropna(subset=["day_idx"])
    return daily.sort_values([SYMBOL_COL, "day_idx"]).reset_index(drop=True)


def build_consecutive_pairs(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["r_lag"] = daily.groupby(SYMBOL_COL)["r"].shift(1)
    daily["day_idx_lag"] = daily.groupby(SYMBOL_COL)["day_idx"].shift(1)
    daily["is_consecutive"] = (daily["day_idx"] - daily["day_idx_lag"]) == 1
    return daily[daily["is_consecutive"] & daily["r_lag"].notna()].copy()


def fit_ar1(group: pd.DataFrame) -> pd.Series:
    keys = ["c_j", "phi_j", "se_phi", "t_stat_phi", "p_value_phi", "r2", "n_obs"]
    if len(group) < MIN_OBS_PER_CONTRACT:
        return pd.Series({key: np.nan for key in keys})

    x = sm.add_constant(group["r_lag"].to_numpy())
    y = group["r"].to_numpy()
    model = sm.OLS(y, x).fit()
    return pd.Series({
        "c_j": model.params[0],
        "phi_j": model.params[1],
        "se_phi": model.bse[1],
        "t_stat_phi": model.tvalues[1],
        "p_value_phi": model.pvalues[1],
        "r2": model.rsquared,
        "n_obs": int(model.nobs),
    })


def run_regressions(day_idx_map: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    all_results = []
    daily_by_underlying = {}

    for label in UNDERLYING_ORDER:
        print(f"Running AR(1) regressions for {label}...")
        quotes = load_underlying_quotes(label)
        daily = build_daily_quotes(quotes, day_idx_map)
        pairs = build_consecutive_pairs(daily)

        grouped = pairs.groupby(
            [SYMBOL_COL, "moneyness_bucket", "tenor_bucket"],
            observed=True,
        )
        try:
            results = grouped.apply(fit_ar1, include_groups=False)
        except TypeError:
            results = grouped.apply(fit_ar1)

        results = results.dropna(subset=["phi_j"]).reset_index()
        results["underlying"] = label
        results["index"] = UNDERLYINGS[label]["index"]

        all_results.append(results)
        daily_by_underlying[label] = daily

        print(f"  fitted {len(results)} contract-level regressions")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "regression_all_underlyings.csv", index=False)
    return combined, daily_by_underlying


def valid_half_life_sample(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[
        (results["phi_j"] > PHI_MIN)
        & (results["phi_j"] < PHI_MAX)
        & (results["p_value_phi"] < SIG_LEVEL)
    ].copy()
    valid["half_life_days"] = np.log(0.5) / np.log(valid["phi_j"])
    return valid


def make_retention_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in UNDERLYING_ORDER:
        sub = results[results["underlying"] == label]
        retained = sub[
            (sub["phi_j"] > PHI_MIN)
            & (sub["phi_j"] < PHI_MAX)
            & (sub["p_value_phi"] < SIG_LEVEL)
        ]
        rows.append({
            "underlying": label,
            "n_total": len(sub),
            "n_retained": len(retained),
            "retention_rate": len(retained) / len(sub),
        })
    return pd.DataFrame(rows)


def make_bucket_table(valid: pd.DataFrame) -> pd.DataFrame:
    bucket = (
        valid.groupby(["underlying", "moneyness_bucket", "tenor_bucket"], observed=True)
        ["half_life_days"]
        .agg(median_hl="median", mean_hl="mean", n="count")
        .reset_index()
    )
    bucket.to_csv(OUTPUT_DIR / "median_half_life_by_bucket.csv", index=False)
    return bucket


def format_cell(value: float, count: float) -> str:
    if pd.isna(value) or pd.isna(count):
        return ""
    return f"{value:.2f} ({int(count)})"


def build_plain_text_half_life_table(
    valid: pd.DataFrame,
    bucket: pd.DataFrame,
    retention: pd.DataFrame,
) -> str:
    lines = [
        "Median Half-Life (Trading Days) across Indices and Moneyness-Tenor Buckets",
        "",
    ]
    money_width = 14
    cell_width = 14
    group_width = cell_width * len(TENOR_ORDER)

    header = " " * money_width
    for underlying in TABLE_UNDERLYING_ORDER:
        header += underlying.center(group_width)
    lines.append(header.rstrip())

    tenor_header = "Moneyness".ljust(money_width)
    for _ in TABLE_UNDERLYING_ORDER:
        for tenor in TENOR_ORDER:
            tenor_header += tenor.center(cell_width)
    lines.append(tenor_header.rstrip())
    lines.append("-" * len(tenor_header.rstrip()))

    for money in MONEYNESS_ORDER:
        row = money.ljust(money_width)
        for underlying in TABLE_UNDERLYING_ORDER:
            for tenor in TENOR_ORDER:
                sub = bucket[
                    (bucket["underlying"] == underlying)
                    & (bucket["moneyness_bucket"] == money)
                    & (bucket["tenor_bucket"] == tenor)
                ]
                if sub.empty:
                    row += "".center(cell_width)
                else:
                    row += format_cell(sub.iloc[0]["median_hl"], sub.iloc[0]["n"]).center(cell_width)
        lines.append(row.rstrip())

    lines.append("-" * len(tenor_header.rstrip()))
    row = "All moneyness".ljust(money_width)
    for underlying in TABLE_UNDERLYING_ORDER:
        for tenor in TENOR_ORDER:
            sub = valid[
                (valid["underlying"] == underlying)
                & (valid["tenor_bucket"] == tenor)
            ]
            row += ("" if sub.empty else f"{sub['half_life_days'].median():.2f} ({len(sub)})").center(cell_width)
    lines.append(row.rstrip())

    row = "Retention rate".ljust(money_width)
    for underlying in TABLE_UNDERLYING_ORDER:
        sub = retention[retention["underlying"] == underlying].iloc[0]
        value = f"{sub['retention_rate'] * 100:.1f}% ({int(sub['n_retained'])}/{int(sub['n_total'])})"
        row += value.center(group_width)
    lines.append("-" * len(tenor_header.rstrip()))
    lines.append(row.rstrip())
    return "\n".join(lines)


def build_latex_half_life_table(
    valid: pd.DataFrame,
    bucket: pd.DataFrame,
    retention: pd.DataFrame,
) -> str:
    def latex_cell(underlying: str, money: str, tenor: str) -> str:
        sub = bucket[
            (bucket["underlying"] == underlying)
            & (bucket["moneyness_bucket"] == money)
            & (bucket["tenor_bucket"] == tenor)
        ]
        if sub.empty:
            return r"\phicell{}{0}"
        return rf"\phicell{{{sub.iloc[0]['median_hl']:.2f}}}{{{int(sub.iloc[0]['n'])}}}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Median Half-Life (Trading Days) across Indices and Moneyness--Tenor Buckets}",
        r"\label{tab:halflife_dataquality}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular*}{0.98\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{SSE 50} & \multicolumn{3}{c}{CSI 300} & \multicolumn{3}{c}{CSI 1000} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Moneyness & ST & MT & LT & ST & MT & LT & ST & MT & LT \\",
        r"\midrule",
    ]
    for money in MONEYNESS_ORDER:
        cells = [
            latex_cell(underlying, money, tenor)
            for underlying in TABLE_UNDERLYING_ORDER
            for tenor in TENOR_ORDER
        ]
        lines.append(f"{money} & " + " & ".join(cells) + r" \\")

    all_cells = []
    for underlying in TABLE_UNDERLYING_ORDER:
        for tenor in TENOR_ORDER:
            sub = valid[
                (valid["underlying"] == underlying)
                & (valid["tenor_bucket"] == tenor)
            ]
            all_cells.append(r"\phicell{}{0}" if sub.empty else rf"\phicell{{{sub['half_life_days'].median():.2f}}}{{{len(sub)}}}")

    retention_cells = []
    for underlying in TABLE_UNDERLYING_ORDER:
        sub = retention[retention["underlying"] == underlying].iloc[0]
        retention_cells.append(
            rf"\multicolumn{{3}}{{c}}{{{sub['retention_rate'] * 100:.1f}\% ({int(sub['n_retained'])}/{int(sub['n_total'])})}}"
        )

    lines.extend([
        r"\midrule",
        r"All moneyness$^{\dagger}$ & " + " & ".join(all_cells) + r" \\",
        r"\midrule",
        r"Retention rate$^{\ddagger}$ & " + " & ".join(retention_cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def run_anova(valid: pd.DataFrame) -> pd.DataFrame:
    model = smf.ols(
        "half_life_days ~ C(underlying) + C(moneyness_bucket) + C(tenor_bucket) "
        "+ C(underlying):C(moneyness_bucket) + C(underlying):C(tenor_bucket) "
        "+ C(moneyness_bucket):C(tenor_bucket)",
        data=valid,
    ).fit()
    table = anova_lm(model, typ=2)
    table.to_csv(OUTPUT_DIR / "anova_half_life.csv")
    return table


def write_report(
    results: pd.DataFrame,
    valid: pd.DataFrame,
    bucket: pd.DataFrame,
    retention: pd.DataFrame,
    anova_table: pd.DataFrame,
) -> None:
    plain_table = build_plain_text_half_life_table(valid, bucket, retention)
    latex_table = build_latex_half_life_table(valid, bucket, retention)

    (OUTPUT_DIR / "median_half_life_table.txt").write_text(plain_table, encoding="utf-8")
    (OUTPUT_DIR / "median_half_life_table_latex.tex").write_text(latex_table, encoding="utf-8")

    describe = valid["half_life_days"].describe()
    tenor_median = (
        valid.groupby(["underlying", "tenor_bucket"])["half_life_days"]
        .median()
        .unstack()
        .reindex(UNDERLYING_ORDER)[TENOR_ORDER]
    )

    report = [
        "# Residual Persistence and Half-Life Report",
        "",
        "## Data Quality and Retention",
        retention.to_string(index=False),
        "",
        "## Half-Life Distribution",
        describe.to_string(),
        "",
        "## Median Half-Life Table",
        plain_table,
        "",
        "## Tenor Median Half-Life",
        tenor_median.round(4).to_string(),
        "",
        "## Factorial ANOVA",
        anova_table.round(4).to_string(),
        "",
        "## Key Numbers for Thesis Text",
        f"Overall retained sample median half-life: {valid['half_life_days'].median():.2f} trading days.",
        f"Overall retained sample IQR: {valid['half_life_days'].quantile(0.25):.2f}-{valid['half_life_days'].quantile(0.75):.2f} trading days.",
        f"Overall retained sample maximum: {valid['half_life_days'].max():.2f} trading days.",
        f"Total regressions: {len(results)}.",
        f"Retained regressions: {len(valid)}.",
    ]
    (OUTPUT_DIR / "half_life_report.md").write_text("\n".join(report), encoding="utf-8")


def compute_acf_pacf(series: pd.Series, nlags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = series.to_numpy()
    acf_values, acf_ci = acf(values, nlags=nlags, alpha=ALPHA, fft=True)
    pacf_values, pacf_ci = pacf(values, nlags=nlags, alpha=ALPHA, method="ywm")
    return acf_values, acf_ci, pacf_values, pacf_ci


def draw_correlation_panel(ax, values: np.ndarray, ci: np.ndarray, nlags: int, ylim: tuple[float, float]) -> None:
    lags = np.arange(len(values))
    lower = ci[:, 0] - values
    upper = ci[:, 1] - values

    ax.fill_between(lags, lower, upper, color=PAPER_COLORS["grey"], alpha=0.20, zorder=1, linewidth=0)
    ax.vlines(lags, 0, values, color=PAPER_COLORS["fit"], linewidth=1.1, zorder=2)
    ax.scatter(lags, values, color=PAPER_COLORS["fit"], s=10, zorder=3, edgecolors="none")
    ax.axhline(0, color=PAPER_COLORS["black"], linewidth=0.6, zorder=1)
    ax.set_xlim(-0.5, nlags + 0.5)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_bucket_series(daily: pd.DataFrame, money: str, tenor: str) -> pd.Series:
    sub = daily[(daily["moneyness_bucket"] == money) & (daily["tenor_bucket"] == tenor)]
    return sub.groupby("day_idx")["r"].mean().sort_index()


def plot_underlying_grids(label: str, daily: pd.DataFrame) -> None:
    for kind in ["acf", "pacf"]:
        fig, axes = plt.subplots(
            len(MONEYNESS_ORDER),
            len(TENOR_ORDER),
            figsize=(9, 10),
            sharex=True,
            sharey=True,
        )
        for i, money in enumerate(MONEYNESS_ORDER):
            for j, tenor in enumerate(TENOR_ORDER):
                ax = axes[i, j]
                series = make_bucket_series(daily, money, tenor)
                if len(series) < MIN_OBS_ACF:
                    ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", fontsize=7, color="gray", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                nlags = min(N_LAGS, len(series) // 2 - 1)
                acf_values, acf_ci, pacf_values, pacf_ci = compute_acf_pacf(series, nlags)
                values, ci = (acf_values, acf_ci) if kind == "acf" else (pacf_values, pacf_ci)
                draw_correlation_panel(ax, values, ci, nlags, (-1.0, 1.0))
                n_contracts = daily[
                    (daily["moneyness_bucket"] == money)
                    & (daily["tenor_bucket"] == tenor)
                ][SYMBOL_COL].nunique()
                ax.set_title(f"{money}/{tenor} (n={n_contracts})", pad=3)
                if i == len(MONEYNESS_ORDER) - 1:
                    ax.set_xlabel("Lag")
                if j == 0:
                    ax.set_ylabel(money)

        fig.suptitle(f"{kind.upper()} by Moneyness-Tenor Bucket: {label}", y=1.002, fontsize=10)
        fig.tight_layout()
        symbol = UNDERLYINGS[label]["index"].split(".")[0]
        fig.savefig(OUTPUT_DIR / f"{symbol}_{kind}_grid.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    clean_output_dir()
    configure_plot_style()
    day_idx_map = load_calendar()

    results, daily_by_underlying = run_regressions(day_idx_map)
    valid = valid_half_life_sample(results)
    retention = make_retention_table(results)
    bucket = make_bucket_table(valid)
    anova_table = run_anova(valid)
    write_report(results, valid, bucket, retention, anova_table)

    for label in UNDERLYING_ORDER:
        plot_underlying_grids(label, daily_by_underlying[label])

    print(f"Done. All outputs were written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
