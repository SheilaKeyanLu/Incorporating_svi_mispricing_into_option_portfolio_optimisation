"""Write the equal-weighting versus Vega-weighting calibration comparison."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
EQUAL_PROJECT_DIR = ROOT_DIR / "03__fit_in_SVI_W_I"

EQUAL_PATH = EQUAL_PROJECT_DIR / "data" / "processed" / "svi_fit_equal_weighting.csv"
VEGA_PATH = PROJECT_DIR / "data" / "processed" / "svi_fit_vega_weighting.csv"
REPORT_DIR = PROJECT_DIR / "output" / "report"
CSV_OUTPUT_PATH = REPORT_DIR / "calibration_quality_comparison.csv"
TEX_OUTPUT_PATH = REPORT_DIR / "calibration_quality_comparison.tex"
TXT_OUTPUT_PATH = REPORT_DIR / "calibration_quality_comparison.txt"

UNDERLYING_NAMES = {
    "000016.SH": "SSE 50",
    "000300.SH": "CSI 300",
    "000852.SH": "CSI 1000",
}


def summarize_scheme(path: Path) -> pd.DataFrame:
    """Summarize one weighting scheme from generated SVI fit data."""
    if not path.exists():
        raise FileNotFoundError(f"Missing processed SVI fit file: {path}")

    df = pd.read_csv(path, low_memory=False)
    success = df["fit_status"].astype(str).str.startswith("success")
    rows = []
    for code, name in UNDERLYING_NAMES.items():
        group = df[df["ths_underlying_code_option"].eq(code)]
        fitted = group[success.loc[group.index]]
        rows.append(
            {
                "underlying": name,
                "underlying_code": code,
                "N": len(group),
                "pass_rate": (fitted["margin"] <= 0).mean(),
                "rmse_median": fitted["fit_rmse"].median(),
            }
        )

    fitted_all = df[success]
    rows.append(
        {
            "underlying": "Overall",
            "underlying_code": "ALL",
            "N": len(df),
            "pass_rate": (fitted_all["margin"] <= 0).mean(),
            "rmse_median": fitted_all["fit_rmse"].median(),
        }
    )
    return pd.DataFrame(rows)


def format_scientific(value: float) -> str:
    """Format a number as LaTeX scientific notation."""
    text = f"{value:.2e}"
    mantissa, exponent = text.split("e")
    return rf"${float(mantissa):.2f}\times10^{{{int(exponent)}}}$"


def build_comparison_table(equal: pd.DataFrame, vega: pd.DataFrame) -> pd.DataFrame:
    """Join both schemes into one comparison table."""
    return equal.merge(
        vega,
        on=["underlying", "underlying_code", "N"],
        suffixes=("_equal", "_vega"),
    )


def write_text_report(table: pd.DataFrame) -> None:
    """Write a plain-English calibration comparison report."""
    lines = [
        "Calibration quality comparison across weighting schemes",
        "=======================================================",
        "",
        "N counts all fitted slices attempted for each underlying.",
        "Stability margin pass rate and RMSE are calculated over successful fits.",
        "",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['underlying']}: N={int(row['N']):,}, "
            f"equal pass rate={row['pass_rate_equal']:.2%}, "
            f"equal median RMSE={row['rmse_median_equal']:.2e}, "
            f"Vega pass rate={row['pass_rate_vega']:.2%}, "
            f"Vega median RMSE={row['rmse_median_vega']:.2e}"
        )
    TXT_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(table: pd.DataFrame) -> None:
    """Write the LaTeX table used in the dissertation text."""
    rows = []
    for _, row in table.iterrows():
        is_overall = row["underlying"] == "Overall"
        name = rf"\textbf{{{row['underlying']}}}" if is_overall else row["underlying"]
        n_value = rf"\textbf{{{int(row['N']):,}}}" if is_overall else f"{int(row['N']):,}"
        equal_pass = f"{row['pass_rate_equal']:.2%}"
        vega_pass = f"{row['pass_rate_vega']:.2%}"
        if is_overall:
            equal_pass = rf"\textbf{{{equal_pass}}}"
            vega_pass = rf"\textbf{{{vega_pass}}}"
        rows.append(
            "\n".join(
                [
                    name,
                    f"& {n_value}",
                    f"& {equal_pass}",
                    f"& {format_scientific(row['rmse_median_equal'])}",
                    f"& {vega_pass}",
                    f"& {format_scientific(row['rmse_median_vega'])} \\\\",
                ]
            )
        )

    body = "\n\n".join(rows)
    latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Calibration quality comparison across weighting schemes}}
\label{{tab:calibration}}
\renewcommand{{\arraystretch}}{{1.2}}
\resizebox{{0.98\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
& & \multicolumn{{2}}{{c}}{{Equal Weighting ($W=I$)}}
& \multicolumn{{2}}{{c}}{{Vega Weighting}} \\
\cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}}
Underlying & $N$ &Stab. margin pass rate & RMSE (median)
& Stab. margin pass Rate & RMSE (median) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize
\vspace{{0.6em}}
\setstretch{{1.1}}
\footnotetext{{
\textit{{Note:}} Stab. margin pass rate refers to the proportion of normal fitted slices with a nonpositive stability margin, i.e., slices whose fitted surface does not exhibit the branch-intersection distortion described in this section.}}
\vspace{{0.4em}}
\end{{minipage}}
\end{{table}}
"""
    TEX_OUTPUT_PATH.write_text(latex, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    equal = summarize_scheme(EQUAL_PATH)
    vega = summarize_scheme(VEGA_PATH)
    table = build_comparison_table(equal, vega)
    table.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    write_text_report(table)
    write_latex_table(table)
    print(f"Saved comparison CSV: {CSV_OUTPUT_PATH}")
    print(f"Saved comparison text report: {TXT_OUTPUT_PATH}")
    print(f"Saved comparison LaTeX table: {TEX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

