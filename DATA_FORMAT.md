# CSV Data Format Guide for Momentum Backtester

## Supported Column Names

### Date Column (Required)
Any of these names (case-insensitive):
- `date`
- `datetime`
- `timestamp`
- `time`

### Ticker Column (Required)
Any of these names (case-insensitive):
- `ticker`
- `symbol`
- `code`

### Close Price Column (Required)
Must contain `close` in the name (case-insensitive):
- `close`
- `Close`
- `CLOSE`
- `Adj Close` ❌ (NOT matched, use adj_close instead)

### Adjusted Close Price Column (Optional)
- `adj_close`
- `Adj_Close`
- `adjusted_close`
- If missing, uses `close` as fallback

### Volume Column (Optional)
- `volume`
- `Volume`
- `VOLUME`
- If missing, defaults to 0

---

## Examples

### Format 1: Standard OHLCV
```csv
date,ticker,open,high,low,close,adj_close,volume
2023-01-03,AAPL,150.93,151.94,150.71,151.94,151.94,47794600
2023-01-04,AAPL,150.76,152.17,150.25,151.97,151.97,54191700
```

### Format 2: Minimal (Date, Ticker, Close Only)
```csv
date,ticker,close
2023-01-03,AAPL,151.94
2023-01-04,AAPL,151.97
```

### Format 3: Alternative Column Names
```csv
timestamp,symbol,adj_close,vol
2023-01-03,AAPL,151.94,47794600
2023-01-04,AAPL,151.97,54191700
```

### Format 4: Long Format (From Database Export)
```csv
Date,Code,Price,AdjPrice,Shares_Traded
2023-01-03,AAPL,151.94,151.94,47794600
2023-01-04,AAPL,151.97,151.97,54191700
```

---

## Data Requirements

✅ **Minimum Requirements:**
- At least 3-4 quarters of historical data (63+ days)
- 10+ tickers for meaningful momentum analysis
- No major data gaps (sparse data causes issues)

❌ **What Will Cause Errors:**
- Missing date column
- Missing ticker/symbol column
- Missing close price
- Non-numeric price/volume data
- Duplicate date-ticker combinations (multiple rows per day for same ticker)

---

## Troubleshooting

### Error: "No date column found"
**Solution:** Ensure your date column is named `date`, `datetime`, `timestamp`, or `time`.

### Error: "No ticker/symbol column found"
**Solution:** Ensure your ticker column is named `ticker`, `symbol`, or `code`.

### Error: "No close price column found"
**Solution:** Ensure you have a column with "close" in the name (not "adj_close").

### Error: "Some date values could not be parsed"
**Solution:** Check date format. Use standard formats like:
- `2023-01-03` (YYYY-MM-DD) ✓
- `01/03/2023` (MM/DD/YYYY) ✓
- `2023-01-03 09:30:00` (with time) ✓

### Error: "No valid price data found"
**Solution:** Check that price columns contain numeric values, not text.

---

## Tips

1. **Use descriptive column names** - the loader will find the right columns
2. **Sort by ticker and date** before uploading (recommended but not required)
3. **Remove duplicates** - one row per ticker per day only
4. **Check date range** - at least 250+ trading days for good results
5. **Use adjusted close** if available (handles splits/dividends)

