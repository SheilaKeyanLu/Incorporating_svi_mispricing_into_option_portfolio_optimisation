from make_reported_figures import main as make_reported_figures
from parameter_sensitivity_backtest import main as run_parameter_sensitivity_backtest


def main():
    run_parameter_sensitivity_backtest()
    make_reported_figures()


if __name__ == "__main__":
    main()
