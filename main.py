from config import *
from optimizer import optimize_hyperparameters
from utils import print_results
from backtest import run_backtest

import pandas as pd

data = pd.read_csv('aapl_5m_train.csv')

def main():
    # Define configurations
    backtest_config = BacktestConfig(
        initial_capital = 1_000_000,
        commission = 0.125/100
    )

    optimization_config = OptimizationConfig(
        n_trials = 10,
        direction = 'maximize',
        n_jobs = -1,
        show_progress_bar = True,
    )

    # Optimize hyperparameters
    study = optimize_hyperparameters(
        data = data,
        backtest_config = backtest_config,
        optimization_config = optimization_config,
        metric = 'sharpe'
    )

    best_params = study.best_params
    max_sharpe = study.best_value

    # Run backtest with best hyperparameters to get all metrics
    metrics, n_long_trades, n_short_trades, portfolio_value, capital = run_backtest(
        data = data,
        config = backtest_config,
        params = best_params
    )

    # Print results
    print_results(best_params, metrics, capital)


if __name__ == '__main__':
    main()