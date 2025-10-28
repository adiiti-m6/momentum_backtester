import pandas as pd
from pathlib import Path
from typing import Tuple
from pandas import DataFrame, Series
import logging

logger = logging.getLogger(__name__)


def load_price_panel(path: str) -> DataFrame:
    """
    Load price data from CSV file or directory of per-ticker CSVs.
    
    Accepts:
    - Single CSV with columns [date, ticker, open, high, low, close, adj_close, volume]
    - Directory of per-ticker CSV files with the same schema
    
    Args:
        path: Path to CSV file or directory of CSVs.
    
    Returns:
        DataFrame in long format with columns [date, ticker, adj_close, close, volume].
        - date: DatetimeIndex or column (sorted)
        - ticker: str
        - adj_close: float (uses close if adj_close missing)
        - close: float
        - volume: int
    
    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns missing or data validation fails.
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if path_obj.is_file():
        dfs = [pd.read_csv(path_obj)]
    else:
        csv_files = list(path_obj.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {path}")
        dfs = [pd.read_csv(f) for f in sorted(csv_files)]
        logger.info(f"Loaded {len(dfs)} CSV files from {path}")
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Validate required columns
    required_cols = {"date", "ticker", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Missing required columns. Required: {required_cols}, "
            f"Got: {set(df.columns)}"
        )
    
    # Handle adj_close: use it if present, otherwise use close
    if "adj_close" not in df.columns:
        logger.warning("adj_close column not found; using close as adj_close")
        df["adj_close"] = df["close"]
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some date values could not be parsed")
    
    # Drop duplicates
    df_clean = df[["date", "ticker", "adj_close", "close", "volume"]].copy()
    df_clean = df_clean.drop_duplicates(subset=["date", "ticker"], keep="first")
    
    # Enforce dtypes
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    df_clean["ticker"] = df_clean["ticker"].astype(str)
    df_clean["adj_close"] = pd.to_numeric(df_clean["adj_close"], errors="coerce")
    df_clean["close"] = pd.to_numeric(df_clean["close"], errors="coerce")
    df_clean["volume"] = pd.to_numeric(df_clean["volume"], errors="coerce").astype("Int64")
    
    # Check for NaNs in key columns
    if df_clean[["adj_close", "close", "volume"]].isna().any().any():
        raise ValueError("NaN values found in adj_close, close, or volume after parsing")
    
    # Sort by ticker and date
    df_clean = df_clean.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    logger.info(f"Loaded price panel: {len(df_clean)} rows, {df_clean['ticker'].nunique()} tickers")
    return df_clean


def to_price_matrix(df_long: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Pivot long-format price data to wide matrices.
    
    Args:
        df_long: DataFrame with columns [date, ticker, price, volume].
                 (price can be adj_close, close, or any selected price column)
    
    Returns:
        Tuple of (prices_wide, volumes_wide):
        - prices_wide: DataFrame indexed by date, columns=tickers, values=price
        - volumes_wide: DataFrame indexed by date, columns=tickers, values=volume
    
    Raises:
        ValueError: If required columns missing or data issues detected.
    """
    # Check for new format (with 'price' column) or old format (with 'adj_close')
    if 'price' in df_long.columns:
        price_col = 'price'
        required_cols = {"date", "ticker", "price", "volume"}
    elif 'adj_close' in df_long.columns:
        price_col = 'adj_close'
        required_cols = {"date", "ticker", "adj_close", "volume"}
    else:
        raise ValueError(
            f"No price column found. Expected 'price' or 'adj_close'. "
            f"Got: {set(df_long.columns)}"
        )
    
    if not required_cols.issubset(df_long.columns):
        raise ValueError(
            f"Missing required columns. Required: {required_cols}, "
            f"Got: {set(df_long.columns)}"
        )
    
    # Pivot prices
    prices_wide = df_long.pivot_table(
        index="date",
        columns="ticker",
        values=price_col,
        aggfunc="first"
    )
    
    # Pivot volumes
    volumes_wide = df_long.pivot_table(
        index="date",
        columns="ticker",
        values="volume",
        aggfunc="first"
    )
    
    logger.info(
        f"Pivoted to matrices: {len(prices_wide)} dates × {len(prices_wide.columns)} tickers"
    )
    return prices_wide, volumes_wide


def validate_panel(prices: DataFrame, min_history_days: int = 63) -> None:
    """
    Validate price matrix for data quality and completeness.
    
    Checks:
    - Index is a DatetimeIndex
    - Dates are monotonic increasing
    - Sufficient history per ticker (at least min_history_days non-NaN values)
    - No NaN values in key rows (optional strict mode)
    
    Args:
        prices: DataFrame indexed by date, columns=tickers, values=adj_close.
        min_history_days: Minimum number of non-NaN observations required per ticker.
    
    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price index must be a DatetimeIndex")
    
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be monotonic increasing")
    
    # Check history per ticker
    for ticker in prices.columns:
        non_nan_count = prices[ticker].notna().sum()
        if non_nan_count < min_history_days:
            raise ValueError(
                f"Ticker {ticker} has only {non_nan_count} non-NaN values, "
                f"requires at least {min_history_days}"
            )
    
    logger.info(
        f"Validation passed: {len(prices)} dates, {len(prices.columns)} tickers, "
        f"min history {prices.notna().sum(axis=0).min()} days"
    )
