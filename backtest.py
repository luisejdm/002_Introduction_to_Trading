from dataclasses import dataclass
import pandas as pd

from metrics import get_sharpe, get_sortino
from config import BacktestConfig
from utils import get_portfolio_value
from indicators import get_rsi, get_ema_signals, get_macd


@dataclass
class Position:
    """
    Represents a trading position.

    Attributes:
        ticker (str): The ticker symbol of the asset.
        quantity (float): The number of shares in the position.
        price (pd.Series): The entry price of the position.
        sl (float): The stop-loss price.
        tp (float): The take-profit price.
        time (pd.Series): The time the position was opened.
    """
    ticker: str
    quantity: float
    price: float
    sl: float
    tp: float
    time: pd.Series


def run_backtest(data: pd.DataFrame,  config: BacktestConfig, params: dict) -> tuple:
    data = data.copy()
    data['Close'] = data['Close'].astype(float)

    n_long_trades = 0
    n_short_trades = 0

    # Hyperparameters to optimize
    rsi_window = params['rsi_window']
    rsi_lower = params['rsi_lower']
    rsi_upper = params['rsi_upper']

    ema_short_window = params['ema_short_window']
    ema_long_window = params['ema_long_window']

    macd_short_window = params['macd_short_window']
    macd_long_window = params['macd_long_window']
    macd_signal_window = params['macd_signal_window']

    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    capital_fraction = params['capital_fraction']

    # Calculate indicators and signals
    rsi = get_rsi(data, rsi_window)
    ema_buy, ema_sell = get_ema_signals(data,
                                        ema_short_window,
                                        ema_long_window)
    macd_buy, macd_sell = get_macd(data,
                                   macd_short_window,
                                   macd_long_window,
                                   macd_signal_window)

    data['rsi_buy'] = rsi < rsi_lower
    data['rsi_sell'] = rsi > rsi_upper

    data['ema_buy'] = ema_buy
    data['ema_sell'] = ema_sell

    data['macd_buy'] = macd_buy
    data['macd_sell'] = macd_sell

    data['buy_signal'] = (data[['rsi_buy', 'ema_buy', 'macd_buy']].sum(axis=1) >= 2)
    data['sell_signal'] = (data[['rsi_sell', 'ema_sell', 'macd_sell']].sum(axis=1) >= 2)

    data = data.dropna(
        subset=['rsi_buy', 'rsi_sell', 'ema_buy', 'ema_sell', 'macd_buy', 'macd_sell']
    ).reset_index(drop=True)

    capital = float(config.initial_capital)
    commission = float(config.commission)

    portfolio_value = [capital]
    active_long_positions: list[Position] = []
    active_short_positions: list[Position] = []

    # Start backtesting
    for i, row in data.iterrows():
        price = row.Close
        # -- LONG ACTIVE ORDERS -- #
        for position in active_long_positions.copy():
            # Stop Loss or take profit Check
            if price > position.tp or price < position.sl:
                # Add profits / losses to capital
                capital += price * position.quantity * (1-commission)
                #Remove position from active pos
                active_long_positions.remove(position)

        # -- LONG -- #
        # Check Signal
        if row.buy_signal:
            # Cacluate BTC position size based on capital fraction
            quantity = (capital * capital_fraction) / price
            cost = quantity * price * (1+commission)
            # Do we have enough capital cash?
            if capital >= cost:
                # Discount cash
                cost = quantity * price * (1+commission)
                capital -= cost
                # Add position to portfolio
                pos = Position(
                    ticker='BTCUSDT',
                    quantity=quantity,
                    price=price,
                    sl=price * (1-stop_loss),
                    tp=price * (1+take_profit),
                    time=row['Datetime']
                )
                active_long_positions.append(pos)
                n_long_trades += 1

        current_value = get_portfolio_value(capital, active_long_positions, active_short_positions, price)
        portfolio_value.append(current_value)

    #At the end of the backtesting, we should close all active positions
    last_price = data.iloc[-1].Close
    for position in active_long_positions:
        capital += last_price * position.quantity * (1-commission)
    active_long_positions = []

    df = pd.DataFrame({'Value': portfolio_value})
    df['rets'] = df.Value.pct_change()
    df.dropna(inplace=True)

    metrics = {
        'Sharpe': get_sharpe(df),
        'Sortino': get_sortino(df)
    }

    return metrics, n_long_trades, n_short_trades, portfolio_value, capital
