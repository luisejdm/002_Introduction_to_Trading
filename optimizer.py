import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from config import BacktestConfig, OptimizationConfig
from backtest import run_backtest


# Optimization without cross-validation
"""def create_objective(
        data: pd.DataFrame, backtest_config: BacktestConfig, metric: str
):
    
    Create an objective function for Optuna hyperparameter optimization.
    Args:
        data (pd.DataFrame): The historical price data for backtesting.
        backtest_config (BacktestConfig): Configuration for the backtest.
        metric (str): The performance metric to optimize ('Sharpe', 'Sortino', 'Calmar').
    Returns:
        function: An objective function that Optuna can use for optimization.
    
    def objective(trial):
        params = {
            'rsi_window': trial.suggest_int('rsi_window', 8, 50),
            'rsi_lower': trial.suggest_int('rsi_lower', 5, 35),
            'rsi_upper': trial.suggest_int('rsi_upper', 65, 95),

            'ema_short_window': trial.suggest_int('ema_short_window', 5, 20),
            'ema_long_window': trial.suggest_int('ema_long_window', 21, 100),

            'macd_short_window': trial.suggest_int('macd_short_window', 8, 20),
            'macd_long_window': trial.suggest_int('macd_long_window', 21, 50),
            'macd_signal_window': trial.suggest_int('macd_signal_window', 5, 20),

            'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.15),
            'take_profit': trial.suggest_float('take_profit', 0.01, 0.15),
            'capital_fraction': trial.suggest_float('capital_fraction', 0.01, 0.02)
            #'n_shares': trial.suggest_int('n_shares', 1, 50)
        }
        metrics, _, _, _, _ = run_backtest(data, backtest_config, params)
        return metrics[metric]

    return objective"""


def cross_validated_objective(
        trial, data: pd.DataFrame, backtest_config: BacktestConfig,
        n_splits: int, metric: str
) -> float:
    """
    Objective function for Optuna hyperparameter optimization with time series cross-validation.
    Args:
        trial (optuna.trial.Trial): The trial object for suggesting hyperparameters.
        data (pd.DataFrame): The historical price data for backtesting.
        backtest_config (BacktestConfig): Configuration for the backtest.
        n_splits (int): The number of splits for time series cross-validation.
        metric (str): The performance metric to optimize ('Sharpe', 'Sortino', 'Calmar').
    Returns:
        float: The average performance metric across all cross-validation splits.
    """
    params = {
        'rsi_window': trial.suggest_int('rsi_window', 8, 50),
        'rsi_lower': trial.suggest_int('rsi_lower', 5, 35),
        'rsi_upper': trial.suggest_int('rsi_upper', 65, 95),

        'ema_short_window': trial.suggest_int('ema_short_window', 5, 20),
        'ema_long_window': trial.suggest_int('ema_long_window', 21, 100),

        'macd_short_window': trial.suggest_int('macd_short_window', 8, 20),
        'macd_long_window': trial.suggest_int('macd_long_window', 21, 50),
        'macd_signal_window': trial.suggest_int('macd_signal_window', 5, 20),

        'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.15),
        'take_profit': trial.suggest_float('take_profit', 0.01, 0.15),
        'capital_fraction': trial.suggest_float('capital_fraction', 0.01, 0.02)
        #'n_shares': trial.suggest_int('n_shares', 1, 50) # Uncomment if needed
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for _, test_idx in tscv.split(data):
        test_data = data.iloc[test_idx].reset_index(drop=True)
        metrics, _, _, _, _ = run_backtest(test_data, backtest_config, params)
        scores.append(metrics[metric])

    return float(np.mean(scores))


def optimize_hyperparameters(
        data: pd.DataFrame, backtest_config: BacktestConfig,
        optimization_config: OptimizationConfig, metric: str
) -> optuna.study.Study:
    """
    Optimize hyperparameters using Optuna.
    Args:
        data (pd.DataFrame): The historical price data for backtesting.
        backtest_config (BacktestConfig): Configuration for the backtest.
        optimization_config (OptimizationConfig): Configuration for the optimization process.
        metric (str): The performance metric to optimize ('Sharpe', 'Sortino', 'Calmar').
    Returns:
        optuna.study.Study: The study object containing optimization results.
    """
    print("\nStarting hyperparameter optimization...\n")

    def objective(trial):  # If no cross-validation, use create_objective
        return cross_validated_objective(
            trial, data, backtest_config,
            optimization_config.n_splits, metric
        )

    study = optuna.create_study(
        direction=optimization_config.direction,
        study_name='Hyperparameter Optimization'
    )
    study.optimize(
        objective,
        n_trials=optimization_config.n_trials,
        n_jobs=optimization_config.n_jobs,
        show_progress_bar=optimization_config.show_progress_bar
    )
    return study