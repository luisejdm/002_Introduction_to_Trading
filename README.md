# 002_Introduction_to_Trading

A project focused on developing systematic trading strategies using technical analysis indicators, designed for hands-on learning in quantitative finance and algorithmic trading.

## Overview

This repository aims to the process of designing, implementing, and analyzing trading strategies. The project includes:

- Building multi-indicator technical strategies
- Backtesting systems with realistic constraints
- Parameter optimization
- Professional-grade reporting and analysis

### Objective

- **Maximize Calmar Ratio**
- **Dataset split:** 60% Train, 20% Test, 20% Validation

## Requirements

- Transaction fees: **0.125%**
- No leverage
- Both long and short positions
- Signal confirmation: **2 out of 3 indicators must agree**
- Walk-forward analysis to prevent overfitting
- Performance metrics:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Maximum Drawdown
  - Win Rate
- Charts & tables:
  - Portfolio value over time
  - Monthly, quarterly, and annual returns

## Technologies

- **Primary language:** Python
- Key libraries (suggested): pandas, numpy, matplotlib, ta-lib, scipy, scikit-learn, jupyter

## Installation

```bash
# Clone the repository
git clone https://github.com/luisejdm/002_Introduction_to_Trading.git
cd 002_Introduction_to_Trading

# Install requirements
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.
