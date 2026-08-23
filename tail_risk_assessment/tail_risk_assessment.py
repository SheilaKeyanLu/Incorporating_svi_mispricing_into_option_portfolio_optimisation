"""
Tail Risk Assessment -- VaR / CVaR computation
==============================================

This script computes historical and Cornish-Fisher tail-risk measures for the
four backtested portfolio strategies used in the dissertation.

Folder layout
-------------
tail_risk_assessment/
  input/    required strategy return CSV files
  output/   generated result CSV and LaTeX table

The script resolves input and output paths relative to its own location, so it
can be run from any working directory, including the project root, input/, or
output/.

Methodology
-----------
Historical measures:
  VaR_alpha       = -Q_{1-alpha}(R_p,1, ..., R_p,Nd)
  D_alpha         = {d : R_p,d <= -VaR_alpha}
  CVaR_alpha      = -(1/|D_alpha|) * sum_{d in D_alpha} R_p,d

Cornish-Fisher measures:
  z_{1-alpha}^CF  = z_{1-alpha} + (1/6)(z_{1-alpha}^2-1)*gamma1
                                  + (1/24)(z_{1-alpha}^3-3z_{1-alpha})*gamma2
                                  - (1/36)(2z_{1-alpha}^3-5z_{1-alpha})*gamma1^2
  VaR_alpha^CF    = -(Rbar_p + z_{1-alpha}^CF * sigma_p)
  CVaR_alpha^CF   = -Rbar_p + sigma_p * phi(z_{1-alpha}^CF)/(1-alpha) *
                    [1 + (gamma1/6)*(z_{1-alpha}^CF)^3
                       + (gamma2/24)*((z_{1-alpha}^CF)^4
                                      - 2*(z_{1-alpha}^CF)^2 - 1)]

Required input files
--------------------
  input/BL Long-Short.csv          -> BL Long-Short
  input/BL Long-Only.csv           -> BL Long-Only
  input/Delta-Gamma Baseline.csv   -> Delta-Gamma Long-Only
  input/Markowitz Baseline.csv     -> Historical Mean-Variance

Outputs
-------
  output/tail_risk_results.csv
  output/tail_risk_table.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

FILES = {
    "BL Long-Short": "BL Long-Short.csv",
    "BL Long-Only": "BL Long-Only.csv",
    "Delta-Gamma Long-Only": "Delta-Gamma Baseline.csv",
    "Historical Mean-Variance": "Markowitz Baseline.csv",
}

STRATEGY_ORDER = [
    "BL Long-Short",
    "BL Long-Only",
    "Delta-Gamma Long-Only",
    "Historical Mean-Variance",
]

CONFIDENCE_LEVELS = [0.95, 0.99]
RETURN_COLUMN = "daily_return"
EXCESS_KURTOSIS_RELIABILITY_THRESHOLD = 10.0


def historical_var_cvar(returns: np.ndarray, alpha: float):
    """Compute empirical lower-tail VaR and CVaR."""
    quantile = np.quantile(returns, 1 - alpha, method="linear")
    var = -quantile

    tail_mask = returns <= -var
    n_tail = int(tail_mask.sum())
    cvar = var if n_tail == 0 else -returns[tail_mask].mean()

    return var, cvar, n_tail


def cornish_fisher_var_cvar(returns: np.ndarray, alpha: float):
    """
    Compute Cornish-Fisher VaR and CVaR from the realised return series.

    gamma1 and gamma2 are the sample skewness and sample excess kurtosis of the
    same realised daily portfolio return series used for historical VaR/CVaR.
    """
    r_bar = returns.mean()
    sigma = returns.std(ddof=1)

    centered = returns - r_bar
    m2 = np.mean(centered ** 2)
    m3 = np.mean(centered ** 3)
    m4 = np.mean(centered ** 4)
    gamma1 = m3 / m2 ** 1.5
    gamma2 = m4 / m2 ** 2 - 3

    z = norm.ppf(1 - alpha)
    z_cf = (
        z
        + (1 / 6) * (z ** 2 - 1) * gamma1
        + (1 / 24) * (z ** 3 - 3 * z) * gamma2
        - (1 / 36) * (2 * z ** 3 - 5 * z) * gamma1 ** 2
    )

    var_cf = -(r_bar + z_cf * sigma)

    phi_zcf = norm.pdf(z_cf)
    gc_tail_correction = (
        1
        + (gamma1 / 6) * z_cf ** 3
        + (gamma2 / 24) * (z_cf ** 4 - 2 * z_cf ** 2 - 1)
    )
    cvar_cf = -r_bar + sigma * (phi_zcf / (1 - alpha)) * gc_tail_correction

    return var_cf, cvar_cf, gamma1, gamma2


def load_returns(filename: str) -> np.ndarray:
    path = INPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    if RETURN_COLUMN not in df.columns:
        raise KeyError(f"{path.name} does not contain column '{RETURN_COLUMN}'")

    return df[RETURN_COLUMN].dropna().to_numpy()


def build_results() -> pd.DataFrame:
    rows = []

    for strategy, filename in FILES.items():
        returns = load_returns(filename)
        record = {"Strategy": strategy, "N_obs": len(returns)}

        for alpha in CONFIDENCE_LEVELS:
            pct = int(alpha * 100)
            var_h, cvar_h, n_tail = historical_var_cvar(returns, alpha)
            var_cf, cvar_cf, gamma1, gamma2 = cornish_fisher_var_cvar(returns, alpha)

            record[f"VaR_{pct}"] = var_h
            record[f"CVaR_{pct}"] = cvar_h
            record[f"VaR_{pct}_CF"] = var_cf
            record[f"CVaR_{pct}_CF"] = cvar_cf
            record[f"D_alpha_size_{pct}"] = n_tail

        _, _, gamma1, gamma2 = cornish_fisher_var_cvar(returns, 0.95)
        record["skewness"] = gamma1
        record["excess_kurtosis"] = gamma2
        record["CF_reliable"] = (
            abs(gamma2) <= EXCESS_KURTOSIS_RELIABILITY_THRESHOLD
        )
        rows.append(record)

    return pd.DataFrame(rows).set_index("Strategy")


def format_latex_table(results: pd.DataFrame) -> str:
    def pct(x):
        return f"{x * 100:.2f}\\%"

    lines = []
    any_unreliable = False

    for strategy in STRATEGY_ORDER:
        row = results.loc[strategy]
        if row["CF_reliable"]:
            cf_var95 = pct(row["VaR_95_CF"])
            cf_cvar95 = pct(row["CVaR_95_CF"])
        else:
            cf_var95 = "--"
            cf_cvar95 = "--"
            any_unreliable = True

        lines.append(
            f"{strategy:<28} & {pct(row['VaR_95'])} & {pct(row['CVaR_95'])} & "
            f"{pct(row['VaR_99'])} & {pct(row['CVaR_99'])} & "
            f"{cf_var95} & {cf_cvar95} \\\\"
        )

    table = "\n".join(lines)
    if any_unreliable:
        table += (
            "\n% Note: \"--\" indicates the Cornish-Fisher estimator is not "
            "reported for this strategy because its sample excess kurtosis "
            f"exceeds {EXCESS_KURTOSIS_RELIABILITY_THRESHOLD:.0f}; the CF "
            "quantile correction is unreliable outside the validity region of "
            "the expansion under such conditions."
        )

    return table


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = build_results()

    results_path = OUTPUT_DIR / "tail_risk_results.csv"
    table_path = OUTPUT_DIR / "tail_risk_table.tex"

    results.to_csv(results_path, float_format="%.6f")
    latex_table = format_latex_table(results)
    table_path.write_text(latex_table + "\n", encoding="utf-8")

    pd.set_option("display.width", 140)
    pd.set_option(
        "display.float_format",
        lambda x: f"{x:.4%}" if abs(x) < 1 else f"{x:.4f}",
    )

    display_cols = [
        "N_obs",
        "VaR_95",
        "CVaR_95",
        "VaR_99",
        "CVaR_99",
        "VaR_95_CF",
        "CVaR_95_CF",
        "VaR_99_CF",
        "CVaR_99_CF",
        "skewness",
        "excess_kurtosis",
    ]

    print("=" * 100)
    print("Tail Risk Metrics")
    print("=" * 100)
    print(results[display_cols].to_string())
    print("=" * 100)
    print(f"\nInput folder:  {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Saved full results to: {results_path}")
    print(f"Saved LaTeX table to:  {table_path}")


if __name__ == "__main__":
    main()
