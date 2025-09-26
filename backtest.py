from dataclasses import dataclass
from datetime import datetime
from typing import Union

import pandas as pd

from metrics import *
from config import BacktestConfig
from utils import get_portfolio_value
from indicadores import get_rsi


@dataclass
class Position:
    """
    Represents a trading position.

    Attributes:
        ticker (str): The ticker symbol of the asset.
        n_shares (int): The number of shares in the position.
        price (pd.Series): The entry price of the position.
        sl (float): The stop-loss price.
        tp (float): The take-profit price.
        time (pd.Series): The time the position was opened.
    """
    ticker: str
    n_shares: int
    price: pd.Series
    sl: float
    tp: float
    time: pd.Series


def run_backtest(data: pd.DataFrame,  config: BacktestConfig, params: dict) -> tuple:
    data = data.copy()

    n_long_trades = 0
    n_short_trades = 0

    # Hyperparameters to optimize
    rsi_window = params['rsi_window']
    rsi_lower = params['rsi_lower']
    rsi_upper = params['rsi_upper']
    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    n_shares = params['n_shares']

    rsi = get_rsi(data, rsi_window)
    data['buy_signal'] = rsi < rsi_lower
    data['sell_signal'] = rsi > rsi_upper

    data = data.dropna()

    capital = config.initial_capital
    commision = config.commission

    portfolio_value = [capital]
    active_long_positions: list[Position] = []
    active_short_positions: list[Position] = []

    # Start backtesting
    for i, row in data.iterrows():
        # -- LONG ACTIVE ORDERS -- #
        for position in active_long_positions.copy():
            # Stop Loss or take profit Check
            if row.Close > position.tp or row.Close < position.sl:
                # Add profits / losses to capital
                capital += row.Close * position.n_shares * (1-commision)
                #Remove position from active pos
                active_long_positions.remove(position)

        # -- LONG -- #
        # Check Signal
        if row.buy_signal:
            cost = row.Close * n_shares * (1 + commision)
            # Do we have enough capital cash?
            if capital > cost:
                # Discount cash
                capital -= cost
                # Add position to portfolio
                pos = Position(
                    ticker='AAPL',
                    n_shares=n_shares,
                    price=row['Close'],
                    sl=row['Close'] * (1 - stop_loss),
                    tp=row['Close'] * (1 + take_profit),
                    time=row['Datetime']
                )
                active_long_positions.append(pos)
                n_long_trades += 1

        value = get_portfolio_value(capital, active_long_positions, [], row.Close, n_shares)
        portfolio_value.append(value)

    #At the end of the backtesting, we should close all active positions
    capital += row.Close * len(active_long_positions) * n_shares * (1-commision)
    active_long_positions = []

    df = pd.DataFrame()
    df['Value'] = portfolio_value
    df['rets'] = df.Value.pct_change()
    df.dropna(inplace=True)

    metrics = {
        'sharpe': get_sharpe(df),
        'sortino': get_sortino(df)
    }

    return metrics, n_long_trades, n_short_trades, portfolio_value, capital
