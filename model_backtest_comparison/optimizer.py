"""
Portfolio optimization module: Black-Litterman + margin-aware capital
constraint + L1 transaction cost.

Objective:
    min_{w,y} -w^T u_BL
             + lambda_risk * w^T A_BL w
             + xi * fee * budget * ||transaction_eta * (w - w_prev)||_1

Constraints:
    (1-y)^T w + (transaction_eta * margin)^T (y * w) <= 1
    -max_weight * y <= w <= max_weight * (1-y)
    w in R^N, y in {0,1}^N
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_portfolio(
    u_BL,
    A_BL,
    w_prev,
    transaction_eta,
    margin,
    lambda_risk,
    xi,
    fee,
    budget,
    max_weight,
    verbose=False,
):
    N = len(u_BL)
    u_BL = np.asarray(u_BL, dtype=float)
    A_BL = np.asarray(A_BL, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)
    transaction_eta = np.asarray(transaction_eta, dtype=float)
    margin = np.asarray(margin, dtype=float)

    model = gp.Model("BL_margin_transaction_cost")

    w = model.addMVar(N, lb=-max_weight, ub=max_weight, vtype=GRB.CONTINUOUS, name="w")
    y = model.addMVar(N, vtype=GRB.BINARY, name="y")
    z = model.addMVar(N, lb=-max_weight, ub=max_weight, vtype=GRB.CONTINUOUS, name="z")
    t = model.addMVar(N, lb=0.0, name="t")

    model.addConstr(w >= -max_weight * y, name="short_bound")
    model.addConstr(w <= max_weight * (1 - y), name="long_bound")

    model.addConstr(z <= max_weight * y, name="z_ub1")
    model.addConstr(z >= -max_weight * y, name="z_lb1")
    model.addConstr(z <= w + max_weight * (1 - y), name="z_ub2")
    model.addConstr(z >= w - max_weight * (1 - y), name="z_lb2")

    model.addConstr(
        w.sum() - z.sum() - (transaction_eta * margin) @ z <= 1,
        name="capital_margin",
    )

    diff = w - w_prev
    model.addConstr(t >= transaction_eta * diff, name="tc_pos")
    model.addConstr(t >= -transaction_eta * diff, name="tc_neg")

    obj = -u_BL @ w + lambda_risk * (w @ A_BL @ w) + xi * fee * t.sum()
    model.setObjective(obj, GRB.MINIMIZE)
    model.Params.OutputFlag = 1 if verbose else 0
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "w": w.X,
            "y": y.X,
            "obj": model.ObjVal,
        }

    return {"status": model.Status, "w": None, "y": None, "obj": None}


if __name__ == "__main__":
    fee = 15
    budget = 10_000_000
    max_weight = 2
    lambda_risk = None
    xi = None

    date_list = []
    w_prev = None
    daily_results = {}

    for date in date_list:
        u_BL_t = None
        A_BL_t = None
        transaction_eta_t = None
        margin_t = None

        res = solve_portfolio(
            u_BL=u_BL_t,
            A_BL=A_BL_t,
            w_prev=w_prev,
            transaction_eta=transaction_eta_t,
            margin=margin_t,
            lambda_risk=lambda_risk,
            xi=xi,
            fee=fee,
            budget=budget,
            max_weight=max_weight,
        )

        daily_results[date] = res

        if res["status"] == "optimal":
            w_prev = res["w"]


def solve_portfolio(
    u_BL,
    A_BL,
    w_prev,
    transaction_eta,
    margin,
    lambda_risk,
    xi,
    fee,
    budget,
    max_weight,
    verbose=False,
):
    N = len(u_BL)
    u_BL = np.asarray(u_BL, dtype=float)
    A_BL = np.asarray(A_BL, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)
    transaction_eta = np.asarray(transaction_eta, dtype=float)
    margin = np.asarray(margin, dtype=float)

    model = gp.Model("BL_margin_transaction_cost")

    w = model.addMVar(N, lb=-max_weight, ub=max_weight, vtype=GRB.CONTINUOUS, name="w")
    y = model.addMVar(N, vtype=GRB.BINARY, name="y")
    z = model.addMVar(N, lb=-max_weight, ub=max_weight, vtype=GRB.CONTINUOUS, name="z")
    t = model.addMVar(N, lb=0.0, name="t")

    model.addConstr(w >= -max_weight * y, name="short_bound")
    model.addConstr(w <= max_weight * (1 - y), name="long_bound")

    model.addConstr(z <= max_weight * y, name="z_ub1")
    model.addConstr(z >= -max_weight * y, name="z_lb1")
    model.addConstr(z <= w + max_weight * (1 - y), name="z_ub2")
    model.addConstr(z >= w - max_weight * (1 - y), name="z_lb2")

    model.addConstr(
        w.sum() - z.sum() - (transaction_eta * margin) @ z <= 1,
        name="capital_margin",
    )

    diff = w - w_prev
    model.addConstr(t >= transaction_eta * diff, name="tc_pos")
    model.addConstr(t >= -transaction_eta * diff, name="tc_neg")

    obj = -u_BL @ w + lambda_risk * (w @ A_BL @ w) + xi * fee * t.sum()
    model.setObjective(obj, GRB.MINIMIZE)
    model.Params.OutputFlag = 1 if verbose else 0
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "w": w.X,
            "y": y.X,
            "obj": model.ObjVal,
        }

    return {"status": model.Status, "w": None, "y": None, "obj": None}


if __name__ == "__main__":
    fee = 15
    budget = 10_000_000
    max_weight = 2
    lambda_risk = None
    xi = None

    date_list = []
    w_prev = None
    daily_results = {}

    for date in date_list:
        u_BL_t = None
        A_BL_t = None
        transaction_eta_t = None
        margin_t = None

        res = solve_portfolio(
            u_BL=u_BL_t,
            A_BL=A_BL_t,
            w_prev=w_prev,
            transaction_eta=transaction_eta_t,
            margin=margin_t,
            lambda_risk=lambda_risk,
            xi=xi,
            fee=fee,
            budget=budget,
            max_weight=max_weight,
        )

        daily_results[date] = res

        if res["status"] == "optimal":
            w_prev = res["w"]
def solve_portfolio_long_only(
    u_BL,
    A_BL,
    w_prev,
    transaction_eta,
    lambda_risk,
    xi,
    fee,
    max_weight,
    verbose=False,
):
    N = len(u_BL)

    u_BL = np.asarray(u_BL, dtype=float)
    A_BL = np.asarray(A_BL, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)
    transaction_eta = np.asarray(transaction_eta, dtype=float)

    model = gp.Model("BL_long_only_transaction_cost")

    # Long only
    w = model.addMVar(
        N,
        lb=0.0,
        ub=max_weight,
        vtype=GRB.CONTINUOUS,
        name="w",
    )

    # Absolute transaction-cost variable
    t = model.addMVar(
        N,
        lb=0.0,
        name="t",
    )

    # Total capital constraint
    model.addConstr(
        w.sum() == 1.0,
        name="capital",
    )

    # Transaction cost
    diff = w - w_prev

    model.addConstr(
        t >= transaction_eta * diff,
        name="tc_pos",
    )

    model.addConstr(
        t >= -transaction_eta * diff,
        name="tc_neg",
    )

    # Objective
    obj = (
        -u_BL @ w
        + lambda_risk * (w @ A_BL @ w)
        + xi * fee * t.sum()
    )

    model.setObjective(
        obj,
        GRB.MINIMIZE,
    )

    model.Params.OutputFlag = 1 if verbose else 0
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "w": w.X,
            "obj": model.ObjVal,
        }

    return {
        "status": model.Status,
        "w": None,
        "obj": None,
    }