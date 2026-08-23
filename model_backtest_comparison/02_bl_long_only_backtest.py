from backtest_runner import load_input_data, run_fixed_parameter_backtest
from config import (
    BEGIN_DATE,
    BUDGET,
    DATA_FILE,
    END_DATE,
    FEE,
    MAX_WEIGHT,
    MODEL_OUTPUT_DIRS,
    REGRESSION_TEST_DAYS,
    REGRESSION_TRAIN_DAYS,
    TAU_FIXED,
    TRAIN_DAYS,
    XI_FIXED,
)
from portfolio_optimisation import bl_long_only


PARAMETERS = {
    "model_name": "BL Long-Only",
    "lambda_risk": 10000,
    "tau": TAU_FIXED,
    "delta": 5,
    "xi": XI_FIXED,
    "fee": FEE,
    "budget": BUDGET,
    "max_weight": MAX_WEIGHT,
    "training_days": TRAIN_DAYS,
    "regression_train_days": REGRESSION_TRAIN_DAYS,
    "regression_test_days": REGRESSION_TEST_DAYS,
}


if __name__ == "__main__":
    info_df = load_input_data(DATA_FILE)
    result = run_fixed_parameter_backtest(
        info_df=info_df,
        begin_date=BEGIN_DATE,
        end_date=END_DATE,
        parameters=PARAMETERS,
        optimisation_module=bl_long_only,
        output_dir=MODEL_OUTPUT_DIRS["bl_long_only"],
    )
    print(f"Backtest completed: {PARAMETERS['model_name']}")
    print(f"Final return: {result['final_return']:.2%}")
    print(f"Sharpe: {result['sharpe']:.4f}")
    print(f"Log: {result['log_path']}")
    print(f"Daily results CSV: {result['csv_path']}")
    print(f"Return curve PNG: {result['curve_path']}")
