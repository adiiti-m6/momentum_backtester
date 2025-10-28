# Edge Case Coverage Audit

## Status: ⚠️ PARTIAL - Some edge cases missing

### ✅ HANDLED Edge Cases

#### Data Loading (`data_loader.py`)
- ✅ Missing `adj_close` column → fallback to `close`
- ✅ Duplicate date/ticker pairs → keep first
- ✅ Invalid dates (NaN after parsing) → raise error
- ✅ Missing required columns → raise error with clear message
- ✅ Non-numeric prices/volume → coerce with `pd.to_numeric(..., errors='coerce')`
- ✅ Minimum history validation → require 63+ days per ticker

#### Engine (`engine.py`)
- ✅ Missing price on a date → drop position with warning
- ✅ Zero equity division → check before dividing
- ✅ Zero price → skip share calculation
- ✅ No positions on day 1 → daily_return = 0.0
- ✅ Empty rebalance (no target weights) → skip trades
- ✅ Weight normalization → re-normalize when filtering unavailable tickers

#### Signals (`signals.py`)
- ✅ Invalid momentum dataframe → raise ValueError
- ✅ NaN momentum values → ranked as NaN (excluded)
- ✅ Tie-breaking in ranks → alphabetical by ticker
- ✅ Invalid n or threshold parameters → raise ValueError with message

---

### ❌ MISSING/INCOMPLETE Edge Cases

#### 1. **Insufficient Historical Data for Momentum**
**Problem**: First rebalance date may not have 3 months of history
**Current**: Strategy returns empty weights if `len(lookback_prices) < 2`
**Impact**: First few quarters may have no positions entered
**Fix**: Should skip first rebalance or use available history

#### 2. **Insufficient Tickers for Entry**
**Problem**: Universe has fewer tickers than `entry_count`
**Current**: Strategy warns but continues with fewer tickers (dynamic count)
**Better**: Could enforce minimum universe size or skip rebalance

#### 3. **All Momentum Returns are NaN**
**Problem**: If all prices are NaN in lookback window
**Current**: `rank_by_momentum()` returns all NaN
**Impact**: Ranks will be all NaN → no positions entered
**Fix**: Should detect and handle gracefully

#### 4. **Zero Price in Momentum Calculation**
**Problem**: If price is 0 or negative (data error)
**Current**: Will cause division by zero in returns
**Fix**: Need to validate prices > 0 before momentum calc

#### 5. **Extreme Volatility / Gap Days**
**Problem**: Stock gaps up/down 100%+ (merger, split)
**Current**: Counts as valid momentum signal
**Fix**: Could add outlier detection (e.g., cap returns at ±100%)

#### 6. **Delisted Stocks (sudden NaN)**
**Problem**: Stock disappears from data mid-backtest
**Current**: Position dropped with warning, but could cause slippage
**Fix**: Could force-close with last known price

#### 7. **Negative Transaction Costs**
**Problem**: Invalid config with negative slippage_bps
**Current**: No validation in engine
**Fix**: Should validate in Config with Pydantic

#### 8. **Division by Very Small Equity**
**Problem**: After many losing trades, equity approaches 0
**Current**: Check `if equity > 0` but uses 0.0 in calculation
**Impact**: Could get division issues with floating point
**Fix**: Add minimum equity threshold (e.g., $1)

#### 9. **NaN in Daily Returns Propagation**
**Problem**: Single NaN in daily_returns breaks Sharpe/other metrics
**Current**: Metrics may fail silently
**Fix**: Should fill NaN returns with 0.0 or drop

#### 10. **Quarter-end Date Edge Cases**
**Problem**: 
  - No quarter-ends found in entire dataset
  - First quarter-end is also the first date (no history)
  - Gap between last data and last quarter-end requested
**Current**: Checks `if len(qe_days) < 1` but not other cases
**Fix**: Need better validation

---

## Recommended Fixes (Priority Order)

### 🔴 HIGH (impacts results)
1. **Validate prices > 0** in momentum calculation
2. **Handle all-NaN momentum** gracefully
3. **Add minimum equity threshold** to prevent divide-by-zero
4. **Validate config parameters** (no negative costs, min universe size)
5. **Skip first rebalance if insufficient history**

### 🟡 MEDIUM (edge cases)
6. Cap extreme returns (e.g., max ±100% per day)
7. Better handling of delisted stocks
8. Detect and warn about NaN in daily returns
9. Add outlier detection for gap days

### 🟢 LOW (nice-to-have)
10. Minimum equity threshold with graceful degradation
11. Config validation with Pydantic validators
12. Logging for quarter-end discovery

---

## How to Apply Fixes

Add to `Config` class in `config.py`:
```python
from pydantic import field_validator

@field_validator('transaction_cost_bps', 'slippage_bps')
def validate_non_negative(cls, v):
    if v < 0:
        raise ValueError("Must be >= 0")
    return v

@field_validator('entry_count')
def validate_entry_count(cls, v, info):
    if v <= 0:
        raise ValueError("entry_count must be > 0")
    return v
```

Add to `signals.py`:
```python
def quarterly_total_return(prices, lookback_quarters=1):
    # Validate prices > 0
    if (prices <= 0).any().any():
        logger.warning("Found prices <= 0; will cause NaN returns")
    
    # Compute returns (existing logic)
    returns = (prices - shifted) / shifted
    
    # Check for all-NaN result
    if returns.isna().all().all():
        raise ValueError("All returns are NaN; check data quality")
    
    return returns
```

Add to `engine.py`:
```python
MIN_EQUITY = 1.0  # Minimum equity threshold

if portfolio_equity < MIN_EQUITY:
    logger.error(f"Equity dropped below minimum: ${portfolio_equity:.2f}")
    # Option 1: Liquidate positions and stop
    # Option 2: Continue with warning
```

---

## Test Plan

Run backtest with edge case datasets:
1. ✅ Single ticker (should work)
2. ✅ Very short history (5 days) → should detect insufficient data
3. ✅ All NaN prices in window → should handle gracefully
4. ✅ Extreme returns (±200%) → should detect
5. ✅ Stock delisted mid-period → should drop position
6. ✅ Negative config values → should reject in Config validation

