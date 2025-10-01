from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """
    Configuration for backtesting a trading strategy.
    Attributes:
        initial_capital (float): The initial capital for the backtest.
        commission (float): The commission rate per trade (as a decimal).
    """
    initial_capital: float = 1_000_000
    commission: float = 0.125 / 100

@dataclass
class OptimizationConfig:
    """
    Configuration for hyperparameter optimization using Optuna.
    Attributes:
        n_trials (int): The number of optimization trials to run.
        direction (str): The optimization direction ('maximize' or 'minimize').
        n_jobs (int): The number of parallel jobs to run. -1 uses all available cores.
        show_progress_bar (bool): Whether to display a progress bar during optimization.
        n_splits (int): The number of splits for time series cross-validation.
    """
    n_trials: int = 50
    direction: str = 'maximize'
    n_jobs: int = -1
    show_progress_bar: bool = True
    n_splits: int = 5