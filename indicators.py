import ta
import pandas as pd

def get_rsi(
        data: pd.DataFrame, rsi_window: int, rsi_lower, rsi_upper
) -> tuple[pd.Series, pd.Series]:
    """
    Calculates the Relative Strength Index (RSI) for a given DataFrame.
    Args:
        data (pd.DataFrame): The input data containing price information.
        rsi_window (int): The window size for calculating the RSI.
        rsi_lower (float): The lower threshold for generating buy signals.
        rsi_upper (float): The upper threshold for generating sell signals.
    Returns:
        pd.Series: A pandas Series containing the RSI values.
    """
    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)

    buy_signal = rsi_indicator.rsi() < rsi_lower
    sell_signal = rsi_indicator.rsi() > rsi_upper
    return buy_signal, sell_signal


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
        macd_buy (pd.Series): A pandas Series containing the MACD buy signals.
        macd_sell (pd.Series): A pandas Series containing the MACD sell signals.
    """
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    buy_signal = macd > signal
    sell_signal = macd < signal
    return buy_signal, sell_signal


def get_bollinger_bands(
        data: pd.DataFrame, window: int, num_std_dev: float
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate buy and sell signals based on Bollinger Bands strategy.
    Args:
        data (pd.DataFrame): DataFrame containing price data with a 'Close' column.
        window (int): The window size for calculating the moving average.
        num_std_dev (float): The number of standard deviations for the bands.
    Returns:
        buy_signal (pd.Series): A pandas Series containing the buy signals.
        sell_signal (pd.Series): A pandas Series containing the sell signals.
    """
    bollinger_bands = ta.volatility.BollingerBands(data['Close'], window=window, window_dev=num_std_dev)
    lower_band = bollinger_bands.bollinger_lband()
    upper_band = bollinger_bands.bollinger_hband()

    buy_signal = data['Close'] < lower_band
    sell_signal = data['Close'] > upper_band
    return buy_signal, sell_signal


def get_stochastic_oscillator(
        data: pd.DataFrame, k_window: int, smooth_window: int,
        lower_threshold: float, upper_threshold: float
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate buy and sell signals based on Stochastic Oscillator strategy.

    Args:
        data (pd.DataFrame): DataFrame containing price data with 'High', 'Low', and 'Close' columns.
        k_window (int): The lookback window size for %K.
        smooth_window (int): The smoothing window for %K (typically 3).
        lower_threshold (float): The lower threshold for generating buy signals.
        upper_threshold (float): The upper threshold for generating sell signals.

    Returns:
        buy_signal (pd.Series): A pandas Series containing the buy signals (True/False).
        sell_signal (pd.Series): A pandas Series containing the sell signals (True/False).
    """
    stochastic_oscillator = ta.momentum.StochasticOscillator(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        window=k_window,
        smooth_window=smooth_window
    )

    k_percent = stochastic_oscillator.stoch()
    d_percent = stochastic_oscillator.stoch_signal()

    buy_signal = (k_percent < lower_threshold) & (d_percent < lower_threshold)
    sell_signal = (k_percent > upper_threshold) & (d_percent > upper_threshold)

    return buy_signal, sell_signal


def get_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Generate buy and sell signals based on multiple technical indicators.
    Args:
        data (pd.DataFrame): DataFrame containing price data with 'Close', 'High', and 'Low' columns.
        params (dict): A dictionary containing parameters for each technical indicator.
    Returns:
        df (pd.DataFrame): A DataFrame with additional columns for buy and sell signals.

    """
    df = data.copy()
    # Calculate individual indicator signals
    df['rsi_buy'], df['rsi_sell'] = get_rsi(
        df, params['rsi_window'], params['rsi_lower'], params['rsi_upper']
    )
    df['ema_buy'], df['ema_sell'] = get_ema_signals(
        df, params['ema_short_window'], params['ema_long_window']
    )
    df['macd_buy'], df['macd_sell'] = get_macd(
        df, params['macd_short_window'], params['macd_long_window'], params['macd_signal_window']
    )
    df['bollinger_buy'], df['bollinger_sell'] = get_bollinger_bands(
        df, params['bollinger_window'], params['bollinger_num_std_dev']
    )
    df['stochastic_buy'], df['stochastic_sell'] = get_stochastic_oscillator(
        df, params['stoch_k_window'], params['stoch_smooth_window'],
        params['stoch_lower_threshold'], params['stoch_upper_threshold']
    )
    # Combine signals
    df['buy_signal'] = (df[[
        'rsi_buy', 'ema_buy', 'macd_buy', 'bollinger_buy', 'stochastic_buy'
    ]].sum(axis=1) >= 2)

    df['sell_signal'] = (df[[
        'rsi_sell', 'ema_sell', 'macd_sell', 'bollinger_sell', 'stochastic_sell'
    ]].sum(axis=1) >= 2)

    # Drop rows with NaN values in any of the signal columns
    df = df.dropna(
        subset=[
            'rsi_buy', 'rsi_sell', 'ema_buy', 'ema_sell', 'macd_buy', 'macd_sell',
            'bollinger_buy', 'bollinger_sell', 'stochastic_buy', 'stochastic_sell'
        ]
    ).reset_index(drop=True)

    return df