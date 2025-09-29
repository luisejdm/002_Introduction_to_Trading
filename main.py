from config import BacktestConfig, OptimizationConfig
from optimizer import optimize_hyperparameters
from utils import clean_split_data
from prints import print_best_params, print_metrics
from backtest import run_backtest
from visualization import plot_portfolio_value, plot_training_portfolio_value

import pandas as pd

data = pd.read_csv('Binance_BTCUSDT_1h.csv')
train_data, test_data, validation_data = clean_split_data(data, 0.6, 0.2, 0.2)

initial_capital = 1_000_000
optimization_metric = 'Calmar' # 'Sharpe', 'Sortino', 'Calmar'
n_trials = 50

def main():
    # ---- Backtest and optimization configurations
    backtest_config = BacktestConfig(
        initial_capital = initial_capital,
        commission = 0.125/100
    )
    optimization_config = OptimizationConfig(
        n_trials=n_trials,
        direction='maximize',
        n_jobs=-1,
        show_progress_bar=True,
    )

    # ---- Optimize hyperparameters
    study = optimize_hyperparameters(
        data=train_data,
        backtest_config=backtest_config,
        optimization_config=optimization_config,
        metric=optimization_metric
    )
    best_params = study.best_params
    best_value = study.best_value
    print(f'\n {'=' * 50}\nBest {optimization_metric}: {best_value:.4f}')
    print_best_params(best_params)

    # ---- Run backtest on training data with best hyperparameters
    train_backtest_config = BacktestConfig(
        initial_capital = initial_capital,
        commission = 0.125 / 100
    )
    train_metrics, train_n_long_trades, train_n_short_trades, train_portfolio_value, train_capital = run_backtest(
        data=train_data,
        config=train_backtest_config,
        params=best_params
    )
    print_metrics(
        train_metrics, initial_capital, train_capital, 'train',
        train_n_long_trades, train_n_short_trades
    )
    plot_training_portfolio_value(train_portfolio_value)

    # ---- Evaluation on test set
    test_backtest_config = BacktestConfig(
        initial_capital = initial_capital, # Start with the initial capital
        commission = 0.125 / 100
    )
    test_metrics, test_n_long_trades, test_n_short_trades, test_portfolio_value, test_capital = run_backtest(
        data=test_data, # Just test data
        config=test_backtest_config,
        params=best_params
    )
    print_metrics(
        test_metrics, initial_capital, test_capital, 'test',
        test_n_long_trades, test_n_short_trades
    )

    # ---- Evaluation on validation set
    valid_backtest_config = BacktestConfig(
        initial_capital = test_capital, # Start with the capital from the test set
        commission = 0.125 / 100
    )
    valid_metrics, valid_n_long_trades, valid_n_short_trades, valid_portfolio_value, valid_capital = run_backtest(
        data = validation_data, # Just validation data
        config = valid_backtest_config,
        params = best_params
    )
    print_metrics(
        valid_metrics, test_capital, valid_capital, 'validation',
        valid_n_long_trades, valid_n_short_trades
    ) # Initial capital from the test set

    # ---- Plot portfolio value over test and validation sets
    plot_portfolio_value(test_portfolio_value, valid_portfolio_value)


if __name__ == '__main__':
    main()