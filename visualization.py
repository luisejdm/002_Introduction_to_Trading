import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # For formatting y-axis with commas
import numpy as np
import seaborn as sns
sns.set_theme()

plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fancybox'] = True


# Plot training portfolio value
def plot_training_portfolio_value(portfolio_values: list, dates: list) -> None:
    """
    Plot the portfolio value over time for the training set.
    Args:
        portfolio_values: list: portfolio values from the training set
        dates: list: corresponding dates for the portfolio values
    Returns:
    """
    plt.figure()
    plt.plot(dates, portfolio_values, label='Portfolio Value on Training Set', color='#313131', lw=2)
    plt.title('Portfolio Value (TRAINING SET)')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Time Steps')
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # For formatting y-axis with commas
    plt.legend(loc='best')
    plt.show()


def plot_portfolio_value(test_portfolio_value: list, valid_portfolio_value: list,
                         test_dates: list, valid_dates: list) -> None:
    """
    Plot the portfolio value over time for both test and validation sets.

    Args:
        test_portfolio_value: list: portfolio values from the test set
        valid_portfolio_value: list: portfolio values from the validation set
        test_dates: list: corresponding dates for the test portfolio values
        valid_dates: list: corresponding dates for the validation portfolio values
    Returns:

    """
    test_values = np.array(test_portfolio_value)
    valid_values = np.array(valid_portfolio_value)

    valid_adjusted = valid_values - valid_values[0] + test_values[-1]
    combined_values = np.concatenate([test_values, valid_adjusted[1:]])

    plt.figure()
    plt.plot(test_dates, test_values, label='Test', color='#2E457B', lw=2)
    plt.plot(valid_dates, valid_values, label='Validation', color='#205c2e', lw=2)
    plt.title('Portfolio Value on Test and Validation Sets)')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Time Steps')
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # For formatting y-axis with commas
    plt.legend(loc='best')
    plt.show()
