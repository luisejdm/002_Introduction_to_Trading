def print_best_params(best_params: dict) -> None:
    """
    Print the best hyperparameters.

    Args:
        best_params (dict): The best hyperparameters found during optimization.
    """
    print('\n\nBest Hyperparameters on:')
    for param, value in best_params.items():
        print(f'  {param}: {value:.4f}')


def print_metrics(metrics: dict, initial_capital: float, final_capital: float, data_set: str) -> None:
    """
    Print the performance metrics.

    Args:
        metrics (dict): A dictionary containing performance metrics.
        initial_capital (float): The initial capital before backtesting.
        final_capital (float): The final capital after backtesting.
        data_set (str): The dataset on which the metrics were evaluated.
    """
    print(f'\n\nPerformance Metrics on {data_set}:')
    for metric, value in metrics.items():
        print(f'  {metric}: {value:.4f}')

    print(f'  Initial Capital: ${initial_capital:.4f}')
    print(f'  Final Capital: ${final_capital:,.4f}')
    print(f'  Net Profit: ${final_capital - initial_capital:,.4f}')
    print(f'  Return on Investment: {(final_capital - initial_capital) / initial_capital * 100:.4f}%')