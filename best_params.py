def get_best_params() -> tuple[dict, float]:
    """
    Return the best hyperparameters found during optimization.
    Returns:
        best_optimized_params (dict): The best hyperparameters.
        best_optimized_value (float): The best value of the optimization metric.
    """
    best_optimized_params = {
        'rsi_window': 37,
        'rsi_lower': 25,
        'rsi_upper': 88,
        'ema_short_window': 13,
        'ema_long_window': 46,
        'macd_short_window': 16,
        'macd_long_window': 44,
        'macd_signal_window': 8,
        'bollinger_window': 11,
        'bollinger_num_std_dev': 2.9481,
        'stoch_k_window': 7,
        'stoch_smooth_window': 3,
        'stoch_lower_threshold': 29.2803,
        'stoch_upper_threshold': 88.2503,
        'stop_loss': 0.2445,
        'take_profit': 0.2587,
        'capital_fraction': 0.1378
    }
    best_optimized_value = 2.3685

    return best_optimized_params, best_optimized_value