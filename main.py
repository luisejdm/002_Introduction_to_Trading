from config import BacktestConfig, OptimizationConfig
from optimizer import optimize_hyperparameters
from utils import clean_split_data
from prints import print_best_params, print_metrics
from backtest import run_backtest
from visualization import plot_portfolio_value

import pandas as pd

data = pd.read_csv('Binance_BTCUSDT_1h.csv')
train_data, test_data, validation_data = clean_split_data(data, 0.6, 0.2, 0.2)

initial_capital = 1_000_000

def main():
    # Backtest and optimization configurations
    backtest_config = BacktestConfig(
        initial_capital = initial_capital,
        commission = 0.125/100
    )

    optimization_config = OptimizationConfig(
        n_trials = 100,
        direction = 'maximize',
        n_jobs = -1,
        show_progress_bar = True,
    )

    # Optimize hyperparameters
    study = optimize_hyperparameters(
        data = train_data,
        backtest_config = backtest_config,
        optimization_config = optimization_config,
        metric = 'sharpe'
    )

    best_params = study.best_params
    max_sharpe = study.best_value
    print_best_params(best_params, 'Test')

    # Evalation on test set
    test_metrics, test_n_long_trades, test_n_short_trades, test_portfolio_value, test_capital = run_backtest(
        data = test_data, # Just test data
        config = backtest_config,
        params = best_params
    )
    print_metrics(test_metrics, initial_capital, test_capital, 'test') # Initial capital from the backtest config

    # Validation configuration
    valid_backtest_config = BacktestConfig(
        initial_capital = test_capital, # Start with the capital from the test set
        commission = 0.125 / 100
    )

    # Evaluation on validation set
    valid_metrics, valid_n_long_trades, valid_n_short_trades, valid_portfolio_value, valid_capital = run_backtest(
        data = validation_data, # Just validation data
        config = valid_backtest_config,
        params = best_params
    )
    print_metrics(valid_metrics, test_capital, valid_capital, 'validation') # Initial capital from the test set

    # Plot portfolio value
    plot_portfolio_value(test_portfolio_value, valid_portfolio_value)


if __name__ == '__main__':
    main()