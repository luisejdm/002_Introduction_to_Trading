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
        current_price: float, commission: float
) -> float:
    """
    Estimate the portfolio value for graphing purposes.
    Args:
        capital (float): The current capital available.
        long_positions (list): A list of active long positions.
        short_positions (list): A list of active short positions.
        current_price (float): The current market price of the asset.
        commission (float): Commission
    Returns:
        float: The total portfolio value.
    """
    value = capital

    long_val = sum([
        pos.quantity * current_price for pos in long_positions
    ])

    short_val = sum([
        pos.quantity * (pos.price - current_price) + pos.price * pos.quantity
        for pos in short_positions
    ])

    return value + long_val + short_val


def get_returns_table(portfolio_value: list, dates: list) -> dict:
    """
    Calculate monthly, quarterly, and annual returns from portfolio value data.
    Args:
        portfolio_value (pd.DataFrame): A DataFrame containing 'Datetime' and 'Value'
        dates (list): A list of datetime objects corresponding to the portfolio values.
    Returns:
        results (dict): A dictionary containing DataFrames for monthly, quarterly, and annual returns.
    """
    df = pd.DataFrame({
        'Datetime': dates,
        'Value': portfolio_value
    }).set_index('Datetime')
    df['Returns'] = df['Value'].pct_change()

    frequency = {
        'ME': 'Monthly Returns',
        'QE': 'Quarterly Returns',
        'YE': 'Annual Returns'
    }

    results = {}
    for freq, label in frequency.items():
        temp = df['Returns'].resample(freq).apply(lambda x: (1 + x).prod() - 1).to_frame(name=label)
        temp.index = temp.index.strftime('%d-%m-%Y')
        results[label] = temp

    return results
