import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
DATA_FILE = INPUT_DIR / "data_prepared_for_markowitz.csv"
OPTION_RETURN_FILE = INPUT_DIR / "option_daily_return_long_table.csv"
TRADING_DATES_FILE = INPUT_DIR / "trading_dates.json"

with open(TRADING_DATES_FILE, "r", encoding="utf-8") as f:
    TRADING_DATES = json.load(f)

# Dissertation fixed sample window.
BEGIN_DATE = "2025-01-02"
END_DATE = "2026-06-30"

# Dissertation fixed common parameters.
TAU_FIXED = 0.05
XI_FIXED = 1.0
MAX_WEIGHT = 0.3
TRAIN_DAYS = 20
REGRESSION_TRAIN_DAYS = 250
REGRESSION_TEST_DAYS = 20

# Capital and trading.
BUDGET = 10_000_000
FEE = 15
CONTRACT_MULTIPLIER = 100

# Legacy aliases required by imported preprocessing/optimisation modules.
TAU = TAU_FIXED
TUNING_DAYS = 4
LAMBDA_RISK_GRID = []
TAU_GRID = []
DELTA_GRID = []
XI_GRID = [XI_FIXED]
ALLOW_SHORT = True
ROUND_POSITION = True
UNDERLYING_ORDER = ["000016.SH", "000300.SH", "000852.SH"]


MODEL_OUTPUT_DIRS = {
    "bl_long_short": OUTPUT_DIR / "01_bl_long_short_output",
    "bl_long_only": OUTPUT_DIR / "02_bl_long_only_output",
    "delta_gamma_approximation": OUTPUT_DIR / "03_delta_gamma_approximation_output",
    "historical_mean_variance": OUTPUT_DIR / "04_historical_mean_variance_output",
}
