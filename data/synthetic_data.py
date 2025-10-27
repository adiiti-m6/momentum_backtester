"""
Synthetic test data for analytics testing.

This file contains synthetically generated backtest results for testing
the analytics module. Data includes:
- Equity curve with defined CAGR and volatility
- Daily returns with known statistical properties
- Position and trades data for analysis

Generated: October 2025
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


# ===== SYNTHETIC EQUITY CURVE DATA =====
# This data is SYNTHETIC - generated for testing purposes
# Characteristics:
# - 252 trading days (1 year)
# - Initial capital: $1,000,000
# - CAGR target: 15%
# - Target annualized volatility: 12%
# - Sharpe ratio target: 1.25 (with 2% risk-free rate)

np.random.seed(42)

# Generate daily returns with target characteristics
target_annual_return = 0.15
target_annual_vol = 0.12
daily_mean = target_annual_return / 252.0
daily_std = target_annual_vol / np.sqrt(252)

synthetic_daily_returns = np.random.normal(daily_mean, daily_std, 252)
synthetic_daily_returns_series = pd.Series(
    synthetic_daily_returns,
    index=pd.date_range('2024-01-01', periods=252, freq='B')
)

# Build equity curve
initial_capital = 1_000_000.0
equity_values = [initial_capital]
for ret in synthetic_daily_returns:
    equity_values.append(equity_values[-1] * (1 + ret))

synthetic_equity_curve = pd.Series(
    equity_values[1:],  # Skip initial capital
    index=pd.date_range('2024-01-01', periods=252, freq='B')
)

# ===== SYNTHETIC POSITIONS DATA =====
# Create holdings for 5 tickers over time
np.random.seed(42)

dates = pd.date_range('2024-01-01', periods=252, freq='B')
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

positions_data = {}
for ticker in tickers:
    # Generate random holding periods
    holdings = np.zeros(len(dates))
    entry_points = np.random.choice(len(dates), size=5, replace=False)
    
    for entry in entry_points:
        # Random holding period 10-50 days
        holding_length = np.random.randint(10, 50)
        exit_point = min(entry + holding_length, len(dates))
        
        # Random number of shares (10-100)
        shares = np.random.randint(100, 1000)
        holdings[entry:exit_point] = shares
    
    positions_data[ticker] = holdings

synthetic_positions = pd.DataFrame(
    positions_data,
    index=dates
)

# ===== SYNTHETIC TRADES DATA =====
# Create 50 transactions across the period

np.random.seed(42)
trades_list = []

for _ in range(50):
    date_idx = np.random.randint(0, len(dates))
    ticker = np.random.choice(tickers)
    
    # Buy or sell
    shares = np.random.choice([-100, -50, 50, 100, 200])
    price = 100.0 + np.random.normal(0, 10)  # Price around $100
    commission = abs(shares * price) * 0.0005  # 5 bps
    
    trades_list.append({
        'Date': dates[date_idx],
        'Ticker': ticker,
        'Shares': shares,
        'Price': price,
        'Commission': commission,
    })

synthetic_trades = pd.DataFrame(trades_list).sort_values('Date').reset_index(drop=True)

# ===== SYNTHETIC CONFIG =====
synthetic_config = {
    'universe_size': 116,
    'entry_count': 24,
    'exit_rank_threshold': 36,
    'transaction_cost_bps': 5.0,
    'slippage_bps': 2.0,
    'lookback_months': 3,
    'holding_quarters': 2,
    'rebalance_frequency': 'quarterly',
    'min_history_days': 63,
    'cash_rate': 0.02,  # 2% risk-free rate
}

# ===== EXPORT FOR FILE STORAGE =====

def save_synthetic_data(output_dir: str = 'd:\\momentum\\data'):
    """Save synthetic test data to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    synthetic_equity_curve.to_csv(f'{output_dir}/synthetic_equity_curve.csv')
    synthetic_daily_returns_series.to_csv(f'{output_dir}/synthetic_daily_returns.csv')
    synthetic_positions.to_csv(f'{output_dir}/synthetic_positions.csv')
    synthetic_trades.to_csv(f'{output_dir}/synthetic_trades.csv', index=False)
    
    print(f"Synthetic data saved to {output_dir}/")


if __name__ == '__main__':
    save_synthetic_data()
