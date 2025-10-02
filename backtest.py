from dataclasses import dataclass
import pandas as pd

from metrics import get_sharpe, get_sortino, get_maximum_drawdown, get_calmar, get_win_rate
from config import BacktestConfig
from utils import get_portfolio_value
from indicators import get_rsi, get_ema_signals, get_macd


@dataclass
class Position:
    """
    Represents a trading position.

    Attributes:
        ticker (str): The ticker symbol of the asset.
        quantity (float): The quantity of the asset held in the position.
        price (pd.Series): The entry price of the position.
        sl (float): The stop-loss price.
        tp (float): The take-profit price.
        time (pd.Series): The time the position was opened.
        cost (float): The cost of the position.
        is_win (bool): Indicates if the position was closed at a profit.
        exit_price (float): The price at which the position was closed.
        type (str): The type of position ('long' or 'short').
    """
    ticker: str
    quantity: float
    price: float
    sl: float
    tp: float
    time: pd.Series
    cost: float
    is_win: bool = None
    exit_price: float = None
    type: str = None


def run_backtest(
        data: pd.DataFrame,  config: BacktestConfig, params: dict
) -> tuple[dict, int, int, list, float]:
    """
    Backtest a trading strategy on historical data.
    Args:
        data (pd.DataFrame): The historical price data for backtesting.
        config (BacktestConfig): Configuration for the backtest.
        params (dict): Hyperparameters for the trading strategy.
    Returns:
        metrics (dict): A dictionary containing performance metrics.
        n_long_trades (int): The number of long trades executed.
        n_short_trades (int): The number of short trades executed.
        portfolio_value (list): The portfolio value over time.
        final_capital (float): The final capital after backtesting.
    """
    data = data.copy()
    data['Close'] = data['Close'].astype(float)

    n_long_trades = 0
    n_short_trades = 0
    closed_long_positions: list[Position] = []
    closed_short_positions: list[Position] = []

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
    rsi_buy, rsi_sell = get_rsi(data,
                                rsi_window,
                                rsi_lower,
                                rsi_upper)
    ema_buy, ema_sell = get_ema_signals(data,
                                        ema_short_window,
                                        ema_long_window)
    macd_buy, macd_sell = get_macd(data,
                                   macd_short_window,
                                   macd_long_window,
                                   macd_signal_window)

    # Add signals to DataFrame
    data['rsi_buy'] = rsi_buy
    data['rsi_sell'] = rsi_sell

    data['ema_buy'] = ema_buy
    data['ema_sell'] = ema_sell

    data['macd_buy'] = macd_buy
    data['macd_sell'] = macd_sell

    data['buy_signal'] = (data[['rsi_buy', 'ema_buy', 'macd_buy']].sum(axis=1) >= 2)
    data['sell_signal'] = (data[['rsi_sell', 'ema_sell', 'macd_sell']].sum(axis=1) >= 2)

    # Drop rows with NaN values in signal columns
    data = data.dropna(
        subset=['rsi_buy', 'rsi_sell', 'ema_buy', 'ema_sell', 'macd_buy', 'macd_sell']
    ).reset_index(drop=True)

    # Initial capital and commission
    capital = float(config.initial_capital)
    commission = float(config.commission)

    # Initialize portfolio
    portfolio_value = [capital]
    active_long_positions: list[Position] = []
    active_short_positions: list[Position] = []

    # Start backtesting
    for i, row in data.iterrows():
        price = row.Close
        # ---- LONG ACTIVE ORDERS
        for position in active_long_positions.copy():
            # Stop Loss or take profit Check
            if price > position.tp or price < position.sl:
                # Add profits / losses to capital
                capital += price * position.quantity * (1-commission)
                # Register exit price and if it was a win
                position.exit_price = price
                position.is_win = price > position.price  # True if we closed with profit
                # Remove position from active positions and add to closed positions
                active_long_positions.remove(position)
                closed_long_positions.append(position)

        # ---- SHORT ACTIVE ORDERS
        for position in active_short_positions.copy():
            # Stop Loss or take profit Check
            if price > position.sl or price < position.tp:
                # Add profits / losses to capital
                pnl = (position.price-price) * position.quantity * (1-commission)
                capital += position.cost + pnl
                # Register exit price and if it was a win
                position.exit_price = price
                position.is_win = price < position.price # True if we closed with profit
                # Remove position from active positions and add to closed positions
                active_short_positions.remove(position)
                closed_short_positions.append(position)

        # ---- CHECK FOR NEW LONG ORDERS
        if row.buy_signal:
            # Calculate BTC position size based on capital fraction
            quantity = (capital * capital_fraction) / price
            cost = quantity * price * (1+commission)
            # Do we have enough capital?
            if capital >= cost:
                # Discount cash
                capital -= cost
                # Add position to portfolio
                pos = Position(
                    ticker='BTCUSDT',
                    quantity=quantity,
                    price=price,
                    sl=price * (1-stop_loss),
                    tp=price * (1+take_profit),
                    time=row['Datetime'],
                    cost=cost,
                    type='long'
                )
                active_long_positions.append(pos)
                n_long_trades += 1

        # ---- CHECK FOR NEW SHORT ORDERS
        if row.sell_signal:
            # Calculate BTC position size based on capital fraction
            quantity = (capital*capital_fraction) / price
            cost = quantity * price * (1+commission)
            # Do we have enough capital?
            if capital >= cost:
                # Discount cash
                capital -= cost
                # Add position to portfolio
                pos = Position(
                    ticker='BTCUSDT',
                    quantity=quantity,
                    price=price,
                    sl=price * (1+stop_loss),
                    tp=price * (1-take_profit),
                    time=row['Datetime'],
                    cost=cost,
                    type='short'
                )
                active_short_positions.append(pos)
                n_short_trades += 1

        current_value = get_portfolio_value(
            capital, active_long_positions, active_short_positions, price, commission
        )
        portfolio_value.append(current_value)

    #At the end of the backtesting, we should close all active positions
    last_price = data.iloc[-1].Close

    for position in active_long_positions:
        position.exit_price = last_price
        position.is_win = last_price > position.price
        capital += last_price * position.quantity * (1-commission)
        closed_long_positions.append(position)
    active_long_positions = []

    for position in active_short_positions:
        position.exit_price = last_price
        position.is_win = last_price < position.price
        pnl = (position.price-last_price) * position.quantity * (1-commission)
        capital += position.cost + pnl
        closed_short_positions.append(position)
    active_short_positions = []

    df = pd.DataFrame({'Value': portfolio_value})
    df['rets'] = df.Value.pct_change()
    df.dropna(inplace=True)

    metrics = {
        'Sharpe': get_sharpe(df),
        'Sortino': get_sortino(df),
        'Maximum Drawdown': get_maximum_drawdown(df),
        'Calmar': get_calmar(df),
        'Win rate on long positions': get_win_rate(closed_long_positions)
    }

    return metrics, n_long_trades, n_short_trades, portfolio_value, capital