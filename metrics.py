import pandas as pd
import numpy as np


def get_sharpe(data: pd.DataFrame) -> float:
    """
    Calculate the Sharpe ratio of a portfolio.

    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.

    Returns:
        float: The Sharpe ratio of the portfolio.
    """
    mean = data.rets.mean()
    std = data.rets.std()

    annual_rets = mean * (60 * 6.5 * 252 / 5)
    annual_std = std * np.sqrt(60 * 6.5 * 252 / 5)

    return annual_rets / annual_std if annual_std != 0 else 0


def get_sortino(data: pd.DataFrame) -> float:
    """
    Calculate the Sortino ratio of a portfolio.

    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.

    Returns:
        float: The Sortino ratio of the portfolio.
    """
    mean = data.rets.mean()
    std = data.rets.std()
    down_risk = data.rets[data.rets < 0].fillna(0).std()

    annual_rets = mean * (60 * 6.5 * 252 / 5)
    annual_std = std * np.sqrt(60 * 6.5 * 252 / 5)
    annual_down_risk = down_risk * np.sqrt(60 * 6.5 * 252 / 5)

    return annual_rets / annual_down_risk if annual_down_risk != 0 else 0