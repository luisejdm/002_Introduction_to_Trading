def get_trial_params(trial) -> dict:
    """
    Suggest hyperparameters for the trading strategy using Optuna.
    Args:
        trial (optuna.trial.Trial): The trial object for suggesting hyperparameters.
    Returns:
        params (dict): A dictionary containing the suggested hyperparameters.
    """
    params = {
        'rsi_window': trial.suggest_int('rsi_window', 8, 50),
        'rsi_lower': trial.suggest_int('rsi_lower', 5, 40),
        'rsi_upper': trial.suggest_int('rsi_upper', 60, 90),

        'ema_short_window': trial.suggest_int('ema_short_window', 5, 20),
        'ema_long_window': trial.suggest_int('ema_long_window', 21, 100),

        'macd_short_window': trial.suggest_int('macd_short_window', 5, 20),
        'macd_long_window': trial.suggest_int('macd_long_window', 21, 100),
        'macd_signal_window': trial.suggest_int('macd_signal_window', 5, 30),

        'bollinger_window': trial.suggest_int('bollinger_window', 10, 60),
        'bollinger_num_std_dev': trial.suggest_float('bollinger_num_std_dev', 0.5, 3.5),

        'stoch_k_window': trial.suggest_int("stoch_k_window", 5, 30),
        'stoch_smooth_window': trial.suggest_int("stoch_smooth_window", 2, 10),
        'stoch_lower_threshold': trial.suggest_float("stoch_lower_threshold", 5, 30),
        'stoch_upper_threshold': trial.suggest_float("stoch_upper_threshold", 70, 95),

        'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.30),
        'take_profit': trial.suggest_float('take_profit', 0.01, 0.30),
        'capital_fraction': trial.suggest_float('capital_fraction', 0.01, 0.25)
    }
    return params