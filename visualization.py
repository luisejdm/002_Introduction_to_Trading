import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['grid.alpha'] = 0.3


# Plot training portfolio value
def plot_training_portfolio_value(portfolio_values: list) -> None:
    """
    Plot the portfolio value over time for the training set.
    Args:
        portfolio_values: list: portfolio values from the training set
    Returns:

    """
    plt.figure()
    plt.plot(portfolio_values, label='Training Portfolio Value', color='#305556')
    plt.title('Portfolio Value (TRAINING SET)')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Time Steps')
    plt.legend(loc='best')
    plt.show()

def plot_portfolio_value(test_portfolio_value: list, valid_portfolio_value: list,) -> None:
    """
    Plot the portfolio value over time for both test and validation sets.

    Args:
        test_portfolio_value: list: portfolio values from the test set
        valid_portfolio_value: list: portfolio values from the validation set
    Returns:

    """
    test_values = np.array(test_portfolio_value)
    valid_values = np.array(valid_portfolio_value)

    valid_adjusted = valid_values - valid_values[0] + test_values[-1]
    combined_values = np.concatenate([test_values, valid_adjusted[1:]])

    plt.figure()
    plt.plot(np.arange(len(test_values)), test_values, label='Test', color='#23698C')
    plt.plot(np.arange(len(test_values)-1, len(combined_values)), combined_values[len(test_values)-1:], label='Validation', color='#277138')
    plt.title('Portfolio Value (TEST & VALIDATION SETS)')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Time Steps')
    plt.legend(loc='best')
    plt.show()
