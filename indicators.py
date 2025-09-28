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


def get_ema_signals(
        data: pd.DataFrame, short_window: int, long_window: int
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate buy and sell signals based on EMA crossover strategy.
    Args:
        data (pd.DataFrame): DataFrame containing price data with a 'Close' column.
        short_window (int): The window size for the short-term EMA.
        long_window (int): The window size for the long-term EMA.

    Returns:
        buy_signal (pd.Series): A pandas Series containing the buy signals.
        sell_signal (pd.Series): A pandas Series containing the sell signals.
    """

    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()

    buy_signal = ema_short > ema_long
    sell_signal = ema_short < ema_long
    return buy_signal, sell_signal


def get_macd(
        data: pd.DataFrame, short_window: int, long_window: int, signal_window: int
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) and its signal line.

    Args:
        data (pd.DataFrame): DataFrame containing price data with a 'Close' column.
        short_window (int): The window size for the short-term EMA. Default is 12.
        long_window (int): The window size for the long-term EMA. Default is 26.
        signal_window (int): The window size for the signal line EMA. Default is 9.

    Returns:
        pd.DataFrame: A DataFrame containing the MACD, signal line and histogram.
    """
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    macd_buy = macd > signal
    macd_sell = macd < signal

    return macd_buy, macd_sell
