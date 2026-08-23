"""Calculate SVI stability margins and write Vega-weighting reports."""

import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common.svi_margin import write_margin_summary


INTERMEDIATE_DIR = PROJECT_DIR / "data" / "intermediate"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "output" / "report"

INPUT_PATH = INTERMEDIATE_DIR / "_svi_fit.csv"
MARGIN_PATH = INTERMEDIATE_DIR / "_svi_fit_margin.csv"
SUMMARY_PATH = INTERMEDIATE_DIR / "_svi_margin_summary.csv"
BY_MATURITY_PATH = INTERMEDIATE_DIR / "_svi_margin_by_maturity.csv"
BY_UNDERLYING_PATH = INTERMEDIATE_DIR / "_svi_margin_by_underlying.csv"
PROCESSED_PATH = PROCESSED_DIR / "svi_fit_vega_weighting.csv"
REPORT_PATH = REPORT_DIR / "calibration_quality_vega_weighting.txt"

UNDERLYING_NAMES = {
    "000016.SH": "SSE 50",
    "000300.SH": "CSI 300",
    "000852.SH": "CSI 1000",
}


def build_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibration quality by underlying."""
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
                "successful_fits": len(fitted),
                "stability_margin_pass_rate": (fitted["margin"] <= 0).mean(),
                "rmse_median": fitted["fit_rmse"].median(),
            }
        )

    fitted_all = df[success]
    rows.append(
        {
            "underlying": "Overall",
            "underlying_code": "ALL",
            "N": len(df),
            "successful_fits": len(fitted_all),
            "stability_margin_pass_rate": (fitted_all["margin"] <= 0).mean(),
            "rmse_median": fitted_all["fit_rmse"].median(),
        }
    )
    return pd.DataFrame(rows)


def write_quality_report(table: pd.DataFrame) -> None:
    """Write an English summary report for the Vega-weighted calibration."""
    lines = [
        "Vega-weighted SVI calibration quality",
        "=====================================",
        "",
        "N counts all fitted slices attempted for each underlying.",
        "Stability margin pass rate and RMSE are calculated over successful fits.",
        "",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['underlying']}: N={int(row['N']):,}, "
            f"successful fits={int(row['successful_fits']):,}, "
            f"stability margin pass rate={row['stability_margin_pass_rate']:.2%}, "
            f"median RMSE={row['rmse_median']:.6e}"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_margin_summary(
        input_path=INPUT_PATH,
        margin_output_path=MARGIN_PATH,
        summary_output_path=SUMMARY_PATH,
        by_maturity_output_path=BY_MATURITY_PATH,
        by_underlying_output_path=BY_UNDERLYING_PATH,
    )
    df = pd.read_csv(MARGIN_PATH, low_memory=False)
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    table = build_quality_table(df)
    write_quality_report(table)
    print(f"Saved processed data: {PROCESSED_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

