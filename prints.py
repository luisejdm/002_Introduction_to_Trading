def print_best_params(best_params: dict) -> None:
    """
    Print the best hyperparameters.
    Args:
        best_params (dict): The best hyperparameters found during optimization.
    """
    print('\n' + '=' * 50)
    print('\nBest Hyperparameters:')
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f'  {param}: {value:.4f}')
        else:
            print(f'  {param}: {value}')


def print_metrics(
        metrics: dict, initial_capital: float, final_capital: float, data_set: str,
        n_long_trades: int, n_short_trades: int
) -> None:
    """
    Print the performance metrics.
    Args:
        metrics (dict): A dictionary containing performance metrics.
        initial_capital (float): The initial capital before backtesting.
        final_capital (float): The final capital after backtesting.
        data_set (str): The dataset on which the metrics were evaluated.
        n_long_trades (int): Number of long trades executed.
        n_short_trades (int): Number of short trades executed.
    """
    print('\n' + '=' * 50)
    print(f'\nPerformance Metrics on {data_set}:')
    for metric, value in metrics.items():
        print(f'  {metric}: {value:.4f}')
    print(f'  Initial Capital: ${initial_capital:,.4f}')
    print(f'  Final Capital: ${final_capital:,.4f}')
    print(f'  Number of Long Trades: {n_long_trades}')
    print(f'  Number of Short Trades: {n_short_trades}')
    print(f'  Net Profit: ${final_capital - initial_capital:,.4f}')
    print(f'  Return on Investment: {(final_capital - initial_capital) / initial_capital * 100:.4f}%')