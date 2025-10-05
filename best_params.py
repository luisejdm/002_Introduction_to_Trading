def get_best_params() -> tuple[dict, float]:
    """
    Return the best hyperparameters found during optimization.
    Returns:
        best_optimized_params (dict): The best hyperparameters.
        best_optimized_value (float): The best value of the optimization metric.
    """
    best_optimized_params = {
    'rsi_window': 10,
    'rsi_lower': 40,
    'rsi_upper': 90,
    'ema_short_window': 9,
    'ema_long_window': 75,
    'macd_short_window': 8,
    'macd_long_window': 71,
    'macd_signal_window': 15,
    'bollinger_window': 14,
    'bollinger_num_std_dev': 1.7815,
    'stoch_k_window': 11,
    'stoch_smooth_window': 9,
    'stoch_lower_threshold': 25.2229,
    'stoch_upper_threshold': 94.9743,
    'stop_loss': 0.2936,
    'take_profit': 0.2846,
    'capital_fraction': 0.1826
}
    best_optimized_value = 1.7013

    return best_optimized_params, best_optimized_value