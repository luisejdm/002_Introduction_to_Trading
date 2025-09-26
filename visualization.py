import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['grid.alpha'] = 0.3

'''def plot_portfolio_value(portfolio_values: list) -> None:
    plt.figure()
    plt.plot(portfolio_values, label='Portfolio Value', color='cadetblue')
    plt.title('Portfolio Value Over Time')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Time Steps')
    plt.show()'''


# Plot where test portfolio value and validation portfolio value are shown in the same plot

def plot_portfolio_value(test_portfolio_value: list, valid_portfolio_value: list) -> None:
    """
    Plot the portfolio value over time for both test and validation sets.

    Args:
        test_portfolio_value:
        valid_portfolio_value:

    Returns:
    """
    plt.figure()
    plt.plot(test_portfolio_value, label='Test Portfolio Value', color='cadetblue')
    plt.plot(valid_portfolio_value, label='Validation Portfolio Value', color='coral')
    plt.title('Portfolio Value Over Time')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Time Steps')
    plt.legend(loc='best')
    plt.show()