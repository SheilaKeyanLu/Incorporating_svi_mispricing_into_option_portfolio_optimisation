"""Fit SVI slices with Vega weighting."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common.svi_core import fit_svi_file


INPUT_PATH = PROJECT_DIR / "data" / "intermediate" / "_option_quotes_with_iv_with_vega.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "intermediate" / "_svi_fit.csv"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fit_svi_file(INPUT_PATH, OUTPUT_PATH, weight_column="vega")


if __name__ == "__main__":
    main()

