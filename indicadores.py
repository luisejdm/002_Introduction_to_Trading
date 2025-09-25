import ta
import pandas as pd

def get_rsi(data: pd.DataFrame, rsi_window: int) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) for a given DataFrame.

    Args:
        data (pd.DataFrame): The input data containing price information.
        rsi_window (int): The window size for calculating the RSI.

    Returns:
        pd.Series: A pandas Series containing the RSI values.
    """
    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    return rsi_indicator.rsi()