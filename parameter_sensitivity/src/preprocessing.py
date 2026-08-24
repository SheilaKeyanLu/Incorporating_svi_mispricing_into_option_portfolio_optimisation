import numpy as np
from config import TRADING_DATES
from config import TRAIN_DAYS, TUNING_DAYS, TAU
from config import UNDERLYING_ORDER, TEST_DF, TEST_DF2, CONTRACT_MULTIPLIER
from config import REGRESSION_TRAIN_DAYS, REGRESSION_TEST_DAYS
import statsmodels.api as sm
UNDERLYING_ORDER = ["000016.SH", "000300.SH", "000852.SH"]
OPTION_TYPE_ORDER = ["C", "P"]


import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
info_df = pd.read_csv(PROJECT_DIR / "input" / "data_prepared_for_markowitz.csv")
date = "2023-12-28"
# print(build_V(info_df, date))



# Get the contract index for the given date.
def get_daily_contract_index(info_df, date):
    df_today = info_df.loc[
        pd.to_datetime(info_df["date"]) == pd.to_datetime(date)
    ].copy()
    if df_today.empty:
        raise ValueError(f"No option data found for {date}.")
    
    if df_today["symbol"].duplicated().any():
        duplicated = df_today.loc[df_today["symbol"].duplicated(), "symbol"].tolist()
        raise ValueError(f"Duplicated symbols on {date}: {duplicated}")
    
    df_today["ths_underlying_code_option"] = pd.Categorical(
        df_today["ths_underlying_code_option"],
        categories=UNDERLYING_ORDER,
        ordered=True)
    
    df_today["option_type"] = pd.Categorical(
        df_today["option_type"],
        categories=OPTION_TYPE_ORDER,
        ordered=True)
    
    df_today = df_today.sort_values(
        ["ths_underlying_code_option","option_type",  "symbol"],
        kind="stable",
    ).reset_index(drop=True)
    return df_today, df_today["symbol"].tolist()


# Build the V matrix for the given date.
def build_V(info_df, date):
    df_today, contract_index = get_daily_contract_index(info_df, date)
    mapping = {name: i for i, name in enumerate(UNDERLYING_ORDER)}

    V = np.zeros((len(df_today), len(UNDERLYING_ORDER)))
    rows = np.arange(len(df_today))
    cols = df_today["ths_underlying_code_option"].map(mapping).to_numpy()

    V[rows, cols] = (
        df_today["delta"].to_numpy(dtype=float)
        * df_today["F"].to_numpy(dtype=float)
        / df_today["settlement_price"].to_numpy(dtype=float))

    return V, contract_index
# print(get_daily_contract_index(TEST_DF, "2023-03-28"))

def _build_forward_price_matrix(df, date, training_days=TRAIN_DAYS):
    date = pd.to_datetime(date)
    trading_dates = pd.DatetimeIndex(pd.to_datetime(TRADING_DATES)).sort_values()
    history_dates = trading_dates[trading_dates < date][-(training_days + 1):]

    if len(history_dates) < training_days + 1:
        raise ValueError("Insufficient historical trading dates.")

    available_dates = trading_dates[trading_dates <= history_dates[-1]]
    F = (
        df.loc[
            pd.to_datetime(df["date"]).isin(available_dates),
            ["date", "ths_underlying_code_option", "F"],
        ]
        .assign(
            date=lambda x: pd.to_datetime(x["date"]),
            F=lambda x: pd.to_numeric(x["F"], errors="raise"),
        )
        .pivot_table(
            index="date",
            columns="ths_underlying_code_option",
            values="F",
            aggfunc="mean",
        )
        .reindex(index=available_dates, columns=UNDERLYING_ORDER)
        .ffill()
        .reindex(index=history_dates, columns=UNDERLYING_ORDER)
    )

    if F.isna().any().any():
        raise ValueError("Missing forward prices in the estimation window after forward fill.")

    return F


# Build covariance matrix of underlying forward returns.
def build_forward_covariance(df, date, training_days=TRAIN_DAYS):
    F = _build_forward_price_matrix(df, date, training_days)
    return F.pct_change(fill_method=None).dropna().cov().to_numpy()


# Build the forward mean vector for the given date.
def build_forward_mean(df, date, training_days=TRAIN_DAYS):
    F = _build_forward_price_matrix(df, date, training_days)
    return F.pct_change(fill_method=None).dropna().mean().to_numpy()


# Build the A matrix for the given date.
def build_A(V, Sigma_F, tau):
    V = np.asarray(V, dtype=float)
    Sigma_F = np.asarray(Sigma_F, dtype=float)

    if V.ndim != 2:
        raise ValueError("V must be a two-dimensional matrix.")
    if Sigma_F.shape != (V.shape[1], V.shape[1]):
        raise ValueError("Sigma_F must have shape (J, J), where J = V.shape[1].")
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    
    base_risk = V @ Sigma_F @ V.T
    diagonal_risk = np.diag(np.diag(base_risk))
    A = base_risk + tau * diagonal_risk
    return A


# Build the u_base vector for the given date.
def build_u_base(info_df, date, mu_F, Sigma_F):
    df_today, contract_index = get_daily_contract_index(info_df, date)

    mu_F = np.asarray(mu_F, dtype=float)
    Sigma_F = np.asarray(Sigma_F, dtype=float)

    if mu_F.shape != (len(UNDERLYING_ORDER),):
        raise ValueError("mu_F shape must match UNDERLYING_ORDER.")
    if Sigma_F.shape != (len(UNDERLYING_ORDER), len(UNDERLYING_ORDER)):
        raise ValueError("Sigma_F shape must match UNDERLYING_ORDER.")

    mapping = {name: i for i, name in enumerate(UNDERLYING_ORDER)}
    idx = df_today["ths_underlying_code_option"].map(mapping).to_numpy()

    F = df_today["F"].to_numpy(dtype=float)
    price = df_today["settlement_price"].to_numpy(dtype=float)
    delta = df_today["delta"].to_numpy(dtype=float)
    gamma = df_today["gamma"].to_numpy(dtype=float)
    theta = df_today["theta"].to_numpy(dtype=float)

    u_base = (
        delta * mu_F[idx] * F
        + theta
        + 0.5 * gamma * np.diag(Sigma_F)[idx] * F**2
    ) / price

    return u_base, contract_index











# ------------------------------------------------------
# Calculate the u_base vector for the given date.
# ------------------------------------------------------
def calculate_u_base(info_df, date, training_days=TRAIN_DAYS):
    mu_F = build_forward_mean(info_df, date, training_days)
    Sigma_F = build_forward_covariance(info_df, date, training_days)
    return build_u_base(info_df, date, mu_F, Sigma_F)

# ------------------------------------------------------
# Calculate the A matrix for the given date.
# ------------------------------------------------------
def calculate_A(info_df, date, training_days, tau):
    V, contract_index = build_V(info_df, date)
    Sigma_F = build_forward_covariance(info_df, date, training_days)
    A = build_A(V, Sigma_F, tau)

    return {
        "A": A,
        "V": V,
        "Sigma_F": Sigma_F,
        "contract_index": contract_index,
    }


# ------------------------------------------------------
# Build the margin vector for the given date.
# ------------------------------------------------------
def build_margin_vector(info_df, date):
    df_today, contract_index = get_daily_contract_index(info_df, date)
    margin = df_today["deposit"].to_numpy(dtype=float)
    return margin, contract_index


# ------------------------------------------------------
# Calculate the eta vector for the given date.
# ------------------------------------------------------
def build_eta(info_df, date, phi=CONTRACT_MULTIPLIER):
    df_today, contract_index = get_daily_contract_index(info_df, date)
    price = df_today["settlement_price"].to_numpy(dtype=float)
    eta = 1.0 / (price * phi)
    return eta, contract_index






MODEL_X = {
    "M1": ["daily_mispricing_signal"],
    "M2": ["daily_mispricing_signal", "vega"],
    "M3": ["daily_mispricing_signal", "daily_return"],
}


def _fit_nw(data, x_cols):
    X = sm.add_constant(data[x_cols], has_constant="add")
    y = data["return_1d"]

    maxlags = int(np.floor(4 * (len(data) / 100) ** (2 / 9)))
    return sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )


def build_bl_views(info_df,date,regression_train_days=REGRESSION_TRAIN_DAYS,
                   regression_test_days=REGRESSION_TEST_DAYS,delta=1.0):
    date = pd.to_datetime(date)
    trading_dates = pd.DatetimeIndex(pd.to_datetime(TRADING_DATES)).sort_values()

    if date not in trading_dates:
        raise ValueError(f"{date.date()} is not in TRADING_DATES.")

    t = trading_dates.get_loc(date)
    if t < regression_train_days + regression_test_days:
        raise ValueError("Insufficient trading dates before rebalance date.")

    train_dates = trading_dates[t - regression_train_days - regression_test_days:t - regression_test_days]
    test_dates = trading_dates[t - regression_test_days:t]

    data = info_df.copy()
    data["_date"] = pd.to_datetime(data["date"])

    train = data[data["_date"].isin(train_dates)]
    test = data[data["_date"].isin(test_dates)]

    bucket_results = {}

    for bucket, train_b in train.groupby("bucket_label"):
        test_b = test[test["bucket_label"] == bucket]

        if test_b.empty:
            continue

        models = {
            name: _fit_nw(train_b, x_cols)
            for name, x_cols in MODEL_X.items()
        }

        betas = np.array([
            models[name].params["daily_mispricing_signal"]
            for name in MODEL_X
        ])
        tvalues = np.array([
            models[name].tvalues["daily_mispricing_signal"]
            for name in MODEL_X
        ])

        m1 = models["M1"]
        X_test = sm.add_constant(
            test_b[MODEL_X["M1"]],
            has_constant="add",
        )
        y_test = test_b["return_1d"].to_numpy(dtype=float)
        y_pred = m1.predict(X_test).to_numpy(dtype=float)

        denominator = np.sum(
            (y_test - train_b["return_1d"].mean()) ** 2
        )
        oos_r2 = (
            1 - np.sum((y_test - y_pred) ** 2) / denominator
            if denominator > 0 else -np.inf
        )


        same_beta_sign = np.all(betas > 0) or np.all(betas < 0)
        selected = same_beta_sign and np.all(np.abs(tvalues) > 1.645) and oos_r2 > 0

        bucket_results[bucket] = {
            "selected": selected,
            "alpha": m1.params["const"],
            "beta": m1.params["daily_mispricing_signal"],
            "error_variance": np.var(y_test - y_pred, ddof=1),
            "oos_r2": oos_r2,
            "betas": betas,
            "tvalues": tvalues,
        }

    df_today, contract_index = get_daily_contract_index(info_df, date)

    selected_buckets = {
        bucket
        for bucket, result in bucket_results.items()
        if result["selected"]
    }

    view_mask = df_today["bucket_label"].isin(selected_buckets)
    view_rows = np.flatnonzero(view_mask.to_numpy())

    N, M = len(df_today), len(view_rows)
    P = np.zeros((M, N))
    P[np.arange(M), view_rows] = 1.0

    view_df = df_today.iloc[view_rows]
    nu = np.array([
        bucket_results[row.bucket_label]["alpha"]
        + bucket_results[row.bucket_label]["beta"]
        * row.daily_mispricing_signal
        for row in view_df.itertuples()
    ])

    omega_diag = np.array([
        delta * bucket_results[bucket]["error_variance"]
        for bucket in view_df["bucket_label"]
    ])
    Omega = np.diag(omega_diag)

    return {
        "P": P,
        "Omega": Omega,
        "nu": nu,
        "contract_index": contract_index,
        "view_contracts": view_df["symbol"].tolist(),
        "selected_buckets": sorted(selected_buckets),
        "bucket_results": bucket_results,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }



def build_bl_posterior(info_df,date,training_days=TRAIN_DAYS,regression_train_days=REGRESSION_TRAIN_DAYS,
                       regression_test_days=REGRESSION_TEST_DAYS,tau=TAU,delta=1.0,):
    u_base, index_u = calculate_u_base(info_df, date, training_days)
    A_result = calculate_A(info_df, date, training_days, tau)
    A = A_result["A"]
    index_A = A_result["contract_index"]

    views = build_bl_views(info_df,date,regression_train_days=regression_train_days,regression_test_days=regression_test_days,delta=delta)

    P = views["P"]
    Omega = views["Omega"]
    nu = views["nu"]
    index_P = views["contract_index"]

    if not (index_u == index_A == index_P):
        raise ValueError("Contract indices are not aligned.")

    if P.shape[0] == 0:
        return {"u_BL": u_base, "A_BL": A, "u_base": u_base, "A": A, "P": P,
            "Omega": Omega, "nu": nu, "contract_index": index_u, "view_contracts": [],}

    middle = P @ A @ P.T + Omega
    correction = np.linalg.solve(middle, nu - P @ u_base,)

    u_BL = u_base + A @ P.T @ correction
    A_BL = A - A @ P.T @ np.linalg.solve(middle,P @ A,)

    return {
        "u_BL": u_BL,
        "A_BL": A_BL,
        "u_base": u_base,
        "A": A,
        "P": P,
        "Omega": Omega,
        "nu": nu,
        "contract_index": index_u,
        "view_contracts": views["view_contracts"],
        "selected_buckets": views["selected_buckets"],}



# TEST FOR bl_posterior
# result = build_bl_posterior(info_df, "2024-06-03")
# print("u_BL:", result["u_BL"])
# print("A_BL:", result["A_BL"])  
# print("P:", result["P"])
# print("Omega:", result["Omega"])
# print("nu:", result["nu"])
# print("contract_index:", result["contract_index"])
# print(len(result["contract_index"]))
# print(len(result["view_contracts"]))
# print("view_contracts:", result["view_contracts"])
# print("selected_buckets:", result["selected_buckets"])


# TEST FOR BASIC FUNCTION
# print(build_V(TEST_DF, "2023-03-28"))
# print(build_forward_covariance(TEST_DF2, "2023-04-03", training_days=3))
# print(build_forward_mean(TEST_DF2, "2023-04-03", training_days=3))
# print(build_margin_vector(TEST_DF, "2023-03-28"))
# print(build_eta(TEST_DF, "2023-03-28"))


# TEST FOR BL EXTENSION
# print(build_forward_covariance(info_df, "2023-10-10", training_days=16))

# result = build_bl_views(
#     info_df,
#     date="2024-06-25",
#     regression_train_days=200,
#     regression_test_days=40,
#     delta=1.0,
# )


# P = result["P"]
# Omega = result["Omega"]
# nu = result["nu"]
# print("P:", P)
# print("Omega:", Omega)      
# print("nu:", nu)
# np.savetxt(
#     "P_matrix.txt",
#     result["P"],
#     fmt="%.0f",
# )
# print(result["P"].shape)
# print(result["Omega"].shape)
# print(result["nu"].shape)



# def summarize_bucket_results(result):
#     rows = []

#     for bucket, r in result["bucket_results"].items():
#         rows.append({
#             "bucket": bucket,
#             "beta_M1": r["betas"][0],
#             "beta_M2": r["betas"][1],
#             "beta_M3": r["betas"][2],
#             "t_M1": r["tvalues"][0],
#             "t_M2": r["tvalues"][1],
#             "t_M3": r["tvalues"][2],
#             "oos_r2": r["oos_r2"],
#             "selected": r["selected"],
#         })

#     return pd.DataFrame(rows)
# bucket_results_df = summarize_bucket_results(result)
# print(bucket_results_df)

# P = result["P"]
# Omega = result["Omega"]
# nu = result["nu"]
# print("P:", P)
# print("Omega:", Omega)      
# print("nu:", nu)
# np.savetxt(
#     "P_matrix.txt",
#     result["P"],
#     fmt="%.0f",
# )
# print(result["P"].shape)
# print(result["Omega"].shape)
# print(result["nu"].shape)
