import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
from config import *

from backtest import run_backtest


def create_objective(
        data: pd.DataFrame, backtest_config: BacktestConfig, metric: str
):
    def objective(trial):
        params = {
            'rsi_window': trial.suggest_int('rsi_window', 8, 80),
            'rsi_lower': trial.suggest_int('rsi_lower', 5, 35),
            'rsi_upper': trial.suggest_int('rsi_upper', 65, 95),
            'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.15),
            'take_profit': trial.suggest_float('take_profit', 0.01, 0.15),
            'n_shares': trial.suggest_int('n_shares', 5, 500)
        }
        metrics, _, _, _, _ = run_backtest(data, backtest_config, params)
        return metrics[metric]

    return objective


def optimize_hyperparameters(
        data: pd.DataFrame, backtest_config: BacktestConfig,
        optimization_config: OptimizationConfig, metric: str
):
    print("\nStarting hyperparameter optimization...\n")

    objective = create_objective(data, backtest_config, metric)
    study = optuna.create_study(direction=optimization_config.direction)
    study.optimize(objective,
                   n_trials=optimization_config.n_trials,
                   n_jobs=optimization_config.n_jobs,
                   show_progress_bar=optimization_config.show_progress_bar)
    return study
