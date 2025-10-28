# Internship Report: Quantitative Momentum Backtesting System

## Executive Summary

This report documents the development of a comprehensive quantitative momentum trading backtesting system built during the internship period. The project implements a sophisticated momentum-based investment strategy with quarterly rebalancing, enabling detailed performance analysis across 119 stocks over a 9-year period (June 2016 - August 2025).

**Key Achievements:**
- Developed a production-ready backtesting engine with ~2,500 lines of Python code
- Implemented momentum-based stock selection with quarterly rebalancing
- Built interactive web-based analytics dashboard using Streamlit
- Achieved comprehensive performance tracking with ticker-level attribution
- Created modular, testable architecture with 70+ unit tests

---

## 1. Project Overview

### 1.1 Objective
Design and implement a quantitative backtesting system to evaluate momentum-based trading strategies on a universe of 119 stocks, providing detailed performance analytics and portfolio insights.

### 1.2 Strategy Description
**Momentum Strategy:**
- **Universe:** 119 stocks from raw market data
- **Selection Criteria:** Top 24 stocks ranked by 3-month price momentum
- **Rebalancing Frequency:** Quarterly (last business day of each quarter)
- **Position Sizing:** Equal capital allocation across selected stocks
- **Holding Period:** Positions maintained until next rebalance
- **Start Date:** June 30, 2016
- **Transaction Costs:** 10 basis points per trade
- **Slippage:** 5 basis points

### 1.3 Technology Stack
- **Programming Language:** Python 3.13.5
- **Data Processing:** Pandas 2.3.3, NumPy 2.3.4
- **Visualization:** Plotly 6.3.1, Streamlit 1.50.0
- **Testing:** Pytest framework
- **Version Control:** Git/GitHub
- **Development Environment:** VS Code

---

## 2. System Architecture

### 2.1 Core Components

#### 2.1.1 Data Loader (`data_loader.py`)
**Responsibilities:**
- Parse multiple date formats (ISO, Excel serial, DD-MM-YYYY)
- Handle missing data and validation
- Pivot long-form data to date × ticker matrices
- Support both adjusted and unadjusted prices

**Key Functions:**
```python
- to_price_matrix(): Convert long-form DataFrame to price matrix
- validate_panel(): Ensure data quality and completeness
```

#### 2.1.2 Backtest Engine (`engine.py`)
**Responsibilities:**
- Execute daily mark-to-market calculations
- Handle quarterly rebalancing logic
- Track cash flows and transaction costs
- Maintain position history
- Calculate ticker-level quarterly returns

**Key Features:**
- Processes only from first rebalance date onwards (correct time-period alignment)
- Proper cash accounting (buy = reduce cash, sell = increase cash)
- Transaction cost and slippage modeling
- Position holding between rebalances

**Data Structures:**
```python
@dataclass
class Trade:
    date, ticker, side, shares, price, gross_cost, transaction_cost

@dataclass
class BacktestResult:
    equity_curve, daily_returns, positions, weights, trades,
    ticker_quarterly_returns, quarterly_selections
```

#### 2.1.3 Analytics Module (`analytics.py`)
**Responsibilities:**
- Calculate performance metrics (CAGR, Sharpe, drawdown, hit ratio)
- Generate quarterly performance tables
- Compute ticker-level attribution

**Metrics Implemented:**
- **CAGR:** Compound Annual Growth Rate
- **Sharpe Ratio:** Risk-adjusted returns (annualized)
- **Maximum Drawdown:** Peak-to-trough decline
- **Hit Ratio:** Percentage of profitable quarters
- **Quarterly Returns:** First-to-last business day calculation

#### 2.1.4 Calendar Module (`calendar.py`)
**Responsibilities:**
- Identify quarter-end business days
- Handle date alignment and filtering

#### 2.1.5 Visualization (`plotting.py`)
**Responsibilities:**
- Equity curve with drawdown shading
- Rolling performance metrics
- Quarterly performance charts
- Multi-panel dashboard

### 2.2 User Interface (Streamlit App)

#### 2.2.1 Configuration Panel
**Sidebar Controls:**
- File upload (CSV format)
- Price column selection (adj_close vs close)
- Strategy parameters (universe size, entry count, lookback period)
- Cost parameters (transaction costs, slippage)
- Initial capital input

#### 2.2.2 Results Display
**Tabs:**
1. **Equity Curve:** Portfolio value over time with annotations
2. **Drawdown:** Peak-to-trough analysis
3. **Rolling Sharpe:** 63-day rolling risk-adjusted returns
4. **Quarterly Returns:** Quarter-by-quarter performance
5. **Ticker Performance:** Individual stock attribution
6. **Dashboard:** Multi-metric overview

#### 2.2.3 Quarterly Portfolio Selections
**Features:**
- List of selected tickers each quarter
- Momentum scores at selection time
- Portfolio value tracking
- Capital growth visualization
- Top performers identification

---

## 3. Technical Implementation

### 3.1 Key Algorithms

#### 3.1.1 Momentum Calculation
```python
# Calculate returns over lookback period
lookback_start = rebalance_date - pd.DateOffset(months=3)
returns = (end_price - start_price) / start_price

# Rank stocks by momentum (highest to lowest)
sorted_by_return = sorted(returns.items(), key=lambda x: x[1], reverse=True)

# Select top N stocks
top_stocks = sorted_by_return[:24]
```

#### 3.1.2 Cash Flow Accounting
```python
for trade in trades:
    if trade.side == 'BUY':
        current_equity -= (trade.gross_cost + trade.transaction_cost)
    else:  # SELL
        current_equity += (trade.gross_cost - trade.transaction_cost)

# Total portfolio value = cash + market value of holdings
portfolio_value = current_equity + sum(shares × prices)
```

#### 3.1.3 Quarterly Returns Calculation
```python
# Convert daily returns to equity curve
equity_curve = (1 + daily_returns).cumprod()

# Resample to quarter-end values
quarterly_equity_end = equity_curve.resample('QE').last()
quarterly_equity_start = quarterly_equity_end.shift(1)

# Calculate quarter-to-quarter returns
quarterly_returns = (equity_end - equity_start) / equity_start
```

### 3.2 Critical Bug Fixes

#### 3.2.1 Cash Accounting Bug (Major)
**Problem:** Initial equity was being double-counted - once as cash and once as invested capital, leading to inflated starting value ($1.999M instead of $1M) and negative returns.

**Solution:** Properly deduct purchase costs from cash and add sale proceeds:
```python
# Before (WRONG):
current_equity -= trade.transaction_cost  # Only subtracting fees

# After (CORRECT):
if trade.side == 'BUY':
    current_equity -= (trade.gross_cost + trade.transaction_cost)
else:
    current_equity += (trade.gross_cost - trade.transaction_cost)
```

**Impact:** Fixed fundamental returns calculation, enabling accurate performance measurement.

#### 3.2.2 Date Range Alignment
**Problem:** Backtest processed all dates from data start (Oct 2015) but only traded from June 2016, diluting CAGR over incorrect time period.

**Solution:** Filter to process only from first rebalance onwards:
```python
first_rebalance = rebalance_days[0]  # June 30, 2016
valid_dates = prices.index[prices.index >= first_rebalance]
```

**Impact:** Accurate time-period alignment for annualized metrics.

#### 3.2.3 Position Holding Bug
**Problem:** Positions were being cleared between rebalances instead of held.

**Solution:** Update all positions after trades, not just traded tickers:
```python
# Before (WRONG):
current_positions[ticker] = target_shares.get(ticker, 0)  # Only updates traded

# After (CORRECT):
current_positions = target_shares.copy()  # Updates all positions
```

---

## 4. Results and Analysis

### 4.1 Performance Metrics (Sample Run)
*Based on adjusted close prices, June 2016 - August 2025*

**Portfolio Summary:**
- **Starting Capital:** $1,000,000
- **Final Value:** $1,883,774
- **Total Return:** 88.38%
- **CAGR:** ~7.2%
- **Sharpe Ratio:** 0.45
- **Maximum Drawdown:** -21.49% (March 2020 - COVID crash)
- **Quarterly Hit Ratio:** Variable (depends on market conditions)

### 4.2 Insights

#### 4.2.1 Ticker Selection Quality
- Average momentum score at selection: Tracked quarterly
- Top performers consistently show >15% momentum
- Worst performers typically <5% momentum
- Selection methodology effectively identifies relative strength

#### 4.2.2 Portfolio Behavior
- Quarterly rebalancing captures medium-term trends
- Equal capital allocation provides diversification
- Transaction costs ~$500-$1,000 per rebalance
- Turnover manageable due to quarterly frequency

#### 4.2.3 Risk Characteristics
- Drawdowns concentrated during market crashes (2020, 2022)
- Recovery periods align with market cycles
- Sharpe ratio indicates reasonable risk-adjusted returns
- Quarterly volatility reflects market regime changes

---

## 5. Features Implemented

### 5.1 Core Functionality
✅ Multi-format date parsing (ISO, Excel, DD-MM-YYYY)  
✅ Price column selection (adjusted vs unadjusted)  
✅ Momentum-based stock ranking  
✅ Quarterly rebalancing with transaction costs  
✅ Position tracking and trade logging  
✅ Cash flow accounting  
✅ Daily mark-to-market calculations  

### 5.2 Analytics
✅ CAGR calculation  
✅ Sharpe ratio (risk-adjusted returns)  
✅ Maximum drawdown analysis  
✅ Quarterly hit ratio  
✅ Rolling performance metrics  
✅ Ticker-level attribution  
✅ Quarterly selections with momentum scores  

### 5.3 Visualization
✅ Interactive equity curve  
✅ Drawdown plot with annotations  
✅ Rolling Sharpe ratio chart  
✅ Quarterly performance bar chart  
✅ Multi-panel dashboard  
✅ Capital growth visualization  

### 5.4 Export Capabilities
✅ Trades CSV export  
✅ Holdings CSV export  
✅ Daily returns CSV export  
✅ Ticker quarterly returns CSV export  
✅ Quarterly selections CSV export  

---

## 6. Testing and Validation

### 6.1 Unit Tests
**Coverage:**
- 70+ unit tests across all modules
- Data loading edge cases
- Calendar calculations
- Signal generation
- Strategy decision logic
- Performance metrics accuracy

**Test Framework:** Pytest with fixtures and parametrization

### 6.2 Integration Testing
**Scenarios Tested:**
- End-to-end backtest execution
- Multiple date formats
- Missing data handling
- Edge cases (single stock, no trades, etc.)
- Price column switching
- Cache invalidation

### 6.3 Validation Methods
**Approaches:**
- Manual calculation verification of sample periods
- Cross-checking CAGR with Excel formulas
- Comparing equity curve with manual tracking
- Verifying transaction cost deductions
- Checking quarter-end date alignment

---

## 7. Challenges and Solutions

### 7.1 Technical Challenges

#### Challenge 1: Cash Flow Tracking
**Issue:** Complex accounting for cash vs invested capital  
**Solution:** Explicit buy/sell cash flow updates with detailed logging

#### Challenge 2: Date Alignment
**Issue:** Multiple date formats in user data  
**Solution:** Multi-strategy parsing with fallback mechanisms

#### Challenge 3: Performance Calculation
**Issue:** CAGR calculation over incorrect time periods  
**Solution:** Align backtest start date with first rebalance

#### Challenge 4: Caching Issues
**Issue:** Streamlit caching causing stale results  
**Solution:** Parameter-based cache keys and manual clearing

### 7.2 Design Decisions

#### Decision 1: Modular Architecture
**Rationale:** Separation of concerns enables testing and maintenance  
**Implementation:** Core logic separate from UI, clear interfaces

#### Decision 2: DataFrame-based Processing
**Rationale:** Pandas vectorization for performance  
**Trade-off:** Memory usage vs speed (acceptable for 119 stocks)

#### Decision 3: Equal Capital Allocation
**Rationale:** Simplicity and diversification  
**Alternative Considered:** Momentum-weighted (added complexity)

---

## 8. Learning Outcomes

### 8.1 Technical Skills Developed
- **Quantitative Finance:** Momentum strategies, performance metrics, risk analysis
- **Python Proficiency:** Pandas, NumPy, dataclass patterns, type hints
- **Software Architecture:** Modular design, separation of concerns, testing
- **Data Visualization:** Plotly, Streamlit, interactive dashboards
- **Debugging:** Systematic bug isolation, logging, validation

### 8.2 Domain Knowledge Gained
- **Market Microstructure:** Transaction costs, slippage, bid-ask spread
- **Performance Attribution:** Identifying sources of returns
- **Risk Management:** Drawdown analysis, diversification
- **Backtesting Pitfalls:** Lookahead bias, survivorship bias, data quality

### 8.3 Best Practices Learned
- **Code Quality:** Type hints, docstrings, meaningful variable names
- **Version Control:** Meaningful commits, branching, code review
- **Testing:** Unit tests, integration tests, edge case coverage
- **Documentation:** Clear README, inline comments, architecture docs

---

## 9. Future Enhancements

### 9.1 Short-term Improvements
1. **Risk Management:**
   - Position size limits per stock
   - Sector concentration limits
   - Stop-loss mechanisms

2. **Strategy Variants:**
   - Multiple lookback periods
   - Combination signals (momentum + value)
   - Dynamic position sizing

3. **Performance:**
   - Optimize DataFrame operations
   - Parallel processing for multiple backtests
   - Incremental calculations

### 9.2 Medium-term Features
1. **Advanced Analytics:**
   - Sector attribution
   - Factor exposure analysis
   - Regime detection (bull/bear markets)

2. **Optimization:**
   - Parameter grid search
   - Walk-forward analysis
   - Monte Carlo simulation

3. **Reporting:**
   - PDF report generation
   - Automated email reports
   - Compliance documentation

### 9.3 Long-term Vision
1. **Production Deployment:**
   - Real-time data integration
   - Automated execution (with safeguards)
   - Portfolio monitoring dashboard

2. **Machine Learning:**
   - Predictive models for stock selection
   - Reinforcement learning for position sizing
   - Sentiment analysis integration

3. **Multi-asset:**
   - Extend to bonds, commodities, crypto
   - Cross-asset strategies
   - Currency hedging

---

## 10. Conclusion

### 10.1 Project Summary
The Quantitative Momentum Backtesting System successfully demonstrates a complete end-to-end workflow for evaluating systematic trading strategies. The project delivered:

- **Robust Implementation:** Production-quality code with proper error handling
- **Comprehensive Analytics:** Multi-dimensional performance analysis
- **User-Friendly Interface:** Interactive web application for strategy exploration
- **Extensible Architecture:** Modular design enabling future enhancements

### 10.2 Business Value
The system provides:
- **Decision Support:** Data-driven insights for investment strategies
- **Risk Assessment:** Quantitative risk metrics and scenario analysis
- **Performance Attribution:** Understanding return sources at ticker level
- **Operational Efficiency:** Automated analysis vs manual calculations

### 10.3 Personal Growth
This internship provided hands-on experience in:
- Quantitative finance and algorithmic trading
- Large-scale data processing and analysis
- Software engineering best practices
- Full-stack development (backend + frontend)
- Problem-solving under real-world constraints

### 10.4 Acknowledgments
Special thanks to mentors and team members for guidance on:
- Quantitative strategy design
- Python best practices
- Code review and feedback
- Domain expertise in finance

---

## Appendices

### Appendix A: Code Statistics
- **Total Lines of Code:** ~2,500 (excluding tests)
- **Number of Modules:** 9 core modules
- **Test Coverage:** 70+ unit tests
- **Functions/Methods:** 50+ documented functions
- **Classes:** 5 dataclasses, 1 engine class

### Appendix B: File Structure
```
momentum_backtester/
├── src/
│   ├── core/
│   │   ├── engine.py (541 lines)
│   │   ├── analytics.py (386 lines)
│   │   ├── data_loader.py (145 lines)
│   │   ├── calendar.py (58 lines)
│   │   ├── signals.py (120 lines)
│   │   ├── strategy.py (95 lines)
│   │   ├── plotting.py (280 lines)
│   │   └── config.py (85 lines)
│   └── app/
│       └── streamlit_app.py (1,120 lines)
├── tests/ (70+ test files)
├── data/
│   └── raw_data.csv (298,316 rows)
└── docs/
    ├── README.md
    └── INTERNSHIP_REPORT.md
```

### Appendix C: Technologies and Libraries
```python
# requirements.txt
pandas==2.3.3
numpy==2.3.4
streamlit==1.50.0
plotly==6.3.1
scipy==1.16.2
pydantic==2.10.5
pytest==8.3.4
```

### Appendix D: Git Commit History
**Major Milestones:**
1. Initial project structure and core modules
2. Backtest engine implementation
3. Analytics and metrics calculation
4. Streamlit UI development
5. Bug fixes (cash accounting, date alignment)
6. Ticker performance attribution
7. Quarterly selections tracking
8. Final optimizations and documentation

**Total Commits:** 15+ commits with detailed messages

---

## References

1. **Momentum Investing:**
   - Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers"
   - Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). "Value and Momentum Everywhere"

2. **Backtesting Best Practices:**
   - Bailey, D. H., et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
   - Harvey, C. R., & Liu, Y. (2015). "Backtesting"

3. **Python for Finance:**
   - McKinney, W. (2017). "Python for Data Analysis" (Pandas documentation)
   - VanderPlas, J. (2016). "Python Data Science Handbook"

4. **Technical Documentation:**
   - Streamlit Documentation: https://docs.streamlit.io/
   - Plotly Documentation: https://plotly.com/python/
   - Pandas Documentation: https://pandas.pydata.org/

---

**Report Prepared By:** [Your Name]  
**Institution:** [Your Institution]  
**Internship Period:** [Start Date] - [End Date]  
**Supervisor:** [Supervisor Name]  
**Date:** October 28, 2025  
**Version:** 1.0

---

*This report represents the culmination of practical learning in quantitative finance, software engineering, and data science during the internship period. The project demonstrates competency in building production-quality financial analysis systems.*
