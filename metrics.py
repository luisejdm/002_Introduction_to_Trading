import pandas as pd
import numpy as np


def get_sharpe(data: pd.DataFrame) -> float:
    """
    Calculate the Sharpe ratio of the portfolio.
    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.

    Returns:
        float: The Sharpe ratio of the portfolio.
    """
    mean = data.rets.mean()
    std = data.rets.std()

    annual_rets = mean * (365*24)
    annual_std = std * np.sqrt(365*24)

    return annual_rets / annual_std if annual_std != 0 else 0


def get_sortino(data: pd.DataFrame) -> float:
    """
    Calculate the Sortino ratio of the portfolio.
    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.

    Returns:
        float: The Sortino ratio of the portfolio.
    """
    mean = data.rets.mean()
    std = data.rets.std()
    down_risk = data.rets[data.rets < 0].fillna(0).std()

    annual_rets = mean * (365*24)
    annual_std = std * np.sqrt(365*24)
    annual_down_risk = down_risk * np.sqrt(365*24)

    return annual_rets / annual_down_risk if annual_std != 0 else 0


def get_maximum_drawdown(data: pd.DataFrame) -> float:
    """
    Calculate the maximum drawdown of the portfolio.
    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.

    Returns:
        float: The maximum drawdown of the portfolio.
    """
    roll_max = data['Value'].cummax()
    drawdown = (data['Value'] - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return abs(max_drawdown)


def get_calmar(data: pd.DataFrame, periods_per_year: int = 365*24) -> float:
    """
    Calculate the Calmar ratio of the portfolio.
    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.
        periods_per_year (int): Number of periods in a year. Default is 365*24 for hourly data.

    Returns:
        calmar_ratio (float): The Calmar ratio of the portfolio.
    """
    total_return = data['Value'].iloc[-1] / data['Value'].iloc[0] - 1
    n_years = len(data) / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1

    roll_max = data['Value'].cummax()
    drawdown = (data['Value'] - roll_max) / roll_max
    max_drawdown = drawdown.min()

    return cagr / abs(max_drawdown) if max_drawdown != 0 else 0


def get_win_rate(closed_positions: list) -> float:
    """
    Calculate the win rate of trades of the portfolio.
    Args:
        closed_positions (list): A list of the closed positions.

    Returns:
        win_rate (float): The win rate of the trades.
    """
    if not closed_positions:
        return 0

    n_wins = sum(1 for pos in closed_positions if pos.is_win)
    return n_wins / len(closed_positions)