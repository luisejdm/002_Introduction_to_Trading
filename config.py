from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000
    commission: float = 0.125 / 100

@dataclass
class OptimizationConfig:
    n_trials: int = 10
    direction: str = 'maximize'
    n_jobs: int = -1
    show_progress_bar: bool = True