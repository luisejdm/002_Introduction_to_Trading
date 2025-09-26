import pandas as pd

def clean_split_data(
        data: pd.DataFrame, train: float, test: float, validation: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean the input DataFrame by removing rows with NaN values and fixing Datetime.

    Args:
        data (pd.DataFrame): The input data to be cleaned.
        train (float): Proportion of data to be used for training.
        test (float): Proportion of data to be used for testing.
        validation (float): Proportion of data to be used for validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The cleaned and split data.
    """
    data = data.copy()
    data[['Date', 'Hour']] = data['Date'].str.split(' ', expand=True)
    data['Datetime'] = pd.to_datetime(
        data['Date'] + ' ' + data['Hour'],
        format='%d/%m/%y %H:%M',
        errors='coerce'
    )
    data.drop(columns=['Date', 'Hour', 'Unix'], inplace=True)
    data = data[~data.Datetime.isnull()]
    data = data.sort_values('Datetime').reset_index(drop=True)

    n = len(data)
    train_end = int(n*train)
    test_end = int(n*(1 - validation))

    train_data = data.iloc[:train_end].copy()
    test_data = data.iloc[train_end:test_end].copy()
    validation_data = data.iloc[test_end:].copy()

    return train_data, test_data, validation_data


def get_portfolio_value(
        capital: float, long_positions: list, short_positions:list,
        current_price: float, n_shares: int
) -> float:
    """
    Estimate the portfolio value.

    Args:
        capital (float): The current capital available.
        long_positions (list): A list of active long positions.
        short_positions (list): A list of active short positions.
        current_price (float): The current market price of the asset.
        n_shares (int): The number of shares held in each position.

    Returns:
        float: The total portfolio value.
    """
    value = capital
    value += len(long_positions) * n_shares * current_price
    return value
