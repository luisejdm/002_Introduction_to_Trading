def get_portfolio_value(capital: float, long_positions: list, short_positions:list, current_price: float, n_shares: int) -> float:
    """Estimate the portfolio value.

    Args:
        capital (float): The current capital available.
        long_positions (list): A list of active long positions.
        current_price (float): The current market price of the asset.
        n_shares (int): The number of shares held in each position.

    Returns:
        float: The total portfolio value.
    """
    value = capital
    value += len(long_positions) * n_shares * current_price
    return value