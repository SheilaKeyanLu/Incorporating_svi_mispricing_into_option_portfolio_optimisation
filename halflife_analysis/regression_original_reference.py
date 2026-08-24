import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
HERE = Path(__file__).parent
OUTER = HERE.parent
# ============ Column names ============
SYMBOL_COL = "symbol"
DATE_COL = "trade_date"
RESID_COL = "standardised_residual"
MONEY_COL = "moneyness_bucket"
TENOR_COL = "tenor_bucket"

TRADING_CALENDAR_PATH = OUTER / "trading_dates.json"
MIN_OBS_PER_CONTRACT = 10   # Minimum number of (r_{t-1}, r_t) pairs required for one contract regression

# ============ 0. Load data ============
df = pd.read_csv(HERE / "option_quotes_for_residual_study_000016_SH.csv") 
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# ============ 1. Trading calendar -> consecutive ordinal index ============
with open(TRADING_CALENDAR_PATH) as f:
    calendar = pd.to_datetime(json.load(f))
calendar = pd.Series(np.arange(len(calendar)), index=calendar).sort_index()  # date -> ordinal
day_idx_map = calendar.to_dict()

# ============ 2. Daily aggregation: one row per symbol-trade_date ============
df = df.sort_values([SYMBOL_COL, DATE_COL])

daily_resid = (
    df.groupby([SYMBOL_COL, DATE_COL])[RESID_COL]
      .mean()
      .rename("r")
      .reset_index()
)
daily_bucket = (
    df.groupby([SYMBOL_COL, DATE_COL])[[MONEY_COL, TENOR_COL]]
      .last()
      .reset_index()
)
daily = daily_resid.merge(daily_bucket, on=[SYMBOL_COL, DATE_COL])
daily["day_idx"] = daily[DATE_COL].map(day_idx_map)

if daily["day_idx"].isna().any():
    bad = daily.loc[daily["day_idx"].isna(), DATE_COL].unique()
    print(f"Warning: {len(bad)} trade_date values are not in the official trading calendar and were dropped: {bad[:5]}...")
    daily = daily.dropna(subset=["day_idx"])

daily = daily.sort_values([SYMBOL_COL, "day_idx"]).reset_index(drop=True)

# ============ 3. Build (r_{t-1}, r_t) pairs, keeping only consecutive trading days ============
daily["r_lag"] = daily.groupby(SYMBOL_COL)["r"].shift(1)
daily["day_idx_lag"] = daily.groupby(SYMBOL_COL)["day_idx"].shift(1)
daily["is_consecutive"] = (daily["day_idx"] - daily["day_idx_lag"]) == 1

pairs = daily[daily["is_consecutive"] & daily["r_lag"].notna()].copy()
# Bucket classification is taken from time t, the current observation.
pairs = pairs.rename(columns={MONEY_COL: "moneyness_bucket", TENOR_COL: "tenor_bucket"})

print(f"Built {len(pairs)} consecutive-trading-day pairs covering {pairs[SYMBOL_COL].nunique()} contracts")

# ============ 4. Contract-level AR(1) regression: r_t = c_j + phi_j * r_{t-1} + eps ============
def fit_ar1(g: pd.DataFrame) -> pd.Series:
    keys = ["c_j", "phi_j", "se_phi", "t_stat_phi", "p_value_phi", "r2", "n_obs"]
    if len(g) < MIN_OBS_PER_CONTRACT:
        return pd.Series({k: np.nan for k in keys})   # Keep the same structure as the normal return path.
    X = sm.add_constant(g["r_lag"].to_numpy())
    y = g["r"].to_numpy()
    model = sm.OLS(y, X).fit()
    return pd.Series({
        "c_j": model.params[0],
        "phi_j": model.params[1],
        "se_phi": model.bse[1],
        "t_stat_phi": model.tvalues[1],
        "p_value_phi": model.pvalues[1],
        "r2": model.rsquared,
        "n_obs": int(model.nobs),
    })

results = (
    pairs.groupby([SYMBOL_COL, "moneyness_bucket", "tenor_bucket"], observed=True)
         .apply(fit_ar1, include_groups=False)   # include_groups=False removes the pandas FutureWarning.
         .dropna(subset=["phi_j"])               # Drop contracts with insufficient samples or failed fits.
         .reset_index()
)

print(f"\nSuccessfully estimated {len(results)} contract-level AR(1) regressions")
results.to_csv(HERE / "regression.csv", index=False)
print(results.head())
# ============ 5. Summarize phi_j distribution across the 15 buckets ============
bucket_summary = (
    results.groupby(["moneyness_bucket", "tenor_bucket"])["phi_j"]
           .agg(mean_phi="mean", median_phi="median", std_phi="std",
                n_contracts="count")
           .reset_index()
)

# Preserve the ordering used in the thesis.
moneyness_order = ["DPW", "PW", "ATM", "CW", "DCW"]
tenor_order = ["ST", "MT", "LT"]
bucket_summary["moneyness_bucket"] = pd.Categorical(
    bucket_summary["moneyness_bucket"], categories=moneyness_order, ordered=True
)
bucket_summary["tenor_bucket"] = pd.Categorical(
    bucket_summary["tenor_bucket"], categories=tenor_order, ordered=True
)
bucket_summary = bucket_summary.sort_values(["tenor_bucket", "moneyness_bucket"])

print("\n=== phi_j summary for the 15 (moneyness x tenor) buckets ===")
print(bucket_summary.to_string(index=False))

# Pivot to a 5x3 table for direct thesis use.
pivot_table = bucket_summary.pivot(index="moneyness_bucket", columns="tenor_bucket",
                                     values="mean_phi").reindex(moneyness_order)[tenor_order]
print("\n=== Bucket mean pivot table for the mean-reversion coefficient phi_j ===")
print(pivot_table.round(4))
