import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import optuna
from dataclasses import dataclass

from utils import get_portfolio_value
from indicadores import get_rsi

import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Position:
    """
    Represents a trading position.
    """
    ticker: str
    n_shares: int
    price: float
    sl: float
    tp: float
    time: str

    
def objective(trial, data):
    data = data.copy()

    # Hyperparameters to optimize
    rsi_window = trial.suggest_int('rsi_window', 8, 80)
    rsi_lower = trial.suggest_int('rsi_lower', 5, 35)
    #rsi_upper = trial.suggest_int('rsi_upper', 65, 95)
    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.15)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.15)
    n_shares = trial.suggest_int('n_shares', 5, 500)

    rsi = get_rsi(data, rsi_window)
    data['buy_signal'] = rsi < rsi_lower
    #data['sell_signal'] = rsi > rsi_upper

    data = data.dropna()

    COM = 0.125/100

    STOP_LOSS = stop_loss
    TAKE_PROFIT = take_profit
    N_SHARES = n_shares

    capital = 1_000_000
    portfolio_value = [capital]
    active_long_positions: list[Position] = []
    # active_short_positions: list[Position] = []
    
    # Start backtesting
    for i, row in data.iterrows():
        # -- LONG ACTIVE ORDERS -- #
        for position in active_long_positions.copy():
            # Stop Loss or take profit Check
            if row.Close > position.tp or row.Close < position.sl:
                # Add profits / losses to capital
                capital += row.Close * position.n_shares * (1-COM)
                #Remove position from active pos
                active_long_positions.remove(position)

        # -- LONG -- #
        # Check Signal
        if row.buy_signal:
            cost = row.Close * N_SHARES * (1 + COM)
            # Do we have enough capital cash?
            if capital > cost:
                # Discount cash
                capital -= cost
                # Add position to portfolio
                pos = Position(
                    ticker='AAPL',
                    n_shares=N_SHARES,
                    price=row['Close'],
                    sl=row['Close'] * (1 - STOP_LOSS),
                    tp=row['Close'] * (1 + TAKE_PROFIT),
                    time=row['Datetime']
                )
                active_long_positions.append(pos)

        portfolio_value.append(get_portfolio_value(capital, active_long_positions, row.Close, N_SHARES))

    #At the end of the backtesting, we should close all active positions
    capital += row.Close * len(active_long_positions) * N_SHARES * (1-COM)
    active_long_positions = []

    df = pd.DataFrame()
    df['Value'] = portfolio_value
    df['rets'] = df.Value.pct_change()
    df.dropna(inplace=True)

    mean = df.rets.mean()
    std = df.rets.std()
    down_risk = df.rets[df.rets < 0].fillna(0).std()

    annual_rets = mean * (60 * 6.5 * 252 / 5)
    annual_std = std * np.sqrt(60 * 6.5 * 252 / 5)
    annual_down_risk = down_risk * np.sqrt(60 * 6.5 * 252 / 5)

    #return annual_rets / annual_down_risk if annual_std != 0 else 0 #MAX SORTINO RATIO
    #return annual_rets / annual_std if annual_std != 0 else 0 #MAX SHARPE RATIO
    return (capital / 1_000_000) - 1  # MAX NET PROFIT





