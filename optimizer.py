import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from config import BacktestConfig, OptimizationConfig
from backtest import run_backtest
from trial_params import get_trial_params


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
    data = data.copy()
    params = get_trial_params(trial)

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

    def objective(trial):
        return cross_validated_objective(
            trial, data, backtest_config, optimization_config.n_splits, metric
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