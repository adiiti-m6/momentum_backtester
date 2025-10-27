import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import logging

logger = logging.getLogger(__name__)


def quarterly_total_return(prices: DataFrame, lookback_quarters: int = 1) -> DataFrame:
    """
    Compute total return from t-lookback_quarters to t on quarter-end rows.
    
    Does not forward fill across quarter ends; each return is computed from the
    lookback period's starting price to current price without intermediate data.
    
    Args:
        prices: DataFrame indexed by date, columns=tickers, values=adj_close.
        lookback_quarters: Number of quarters to look back (default 1).
    
    Returns:
        DataFrame with same index and columns as prices, containing quarterly returns.
        NaN where insufficient history or at quarter ends with no lookback data.
    
    Raises:
        ValueError: If prices is not a DataFrame or lookback_quarters <= 0.
    """
    if not isinstance(prices, DataFrame):
        raise ValueError("prices must be a DataFrame")
    if lookback_quarters <= 0:
        raise ValueError("lookback_quarters must be > 0")
    
    # For each row, get the price from lookback_quarters ago
    # Assuming quarterly frequency, lookback_quarters = 1 means shift by 1
    shifted_prices = prices.shift(lookback_quarters)
    
    # Compute total return: (price_t - price_t_lag) / price_t_lag
    returns = (prices - shifted_prices) / shifted_prices
    
    logger.info(f"Computed quarterly returns with lookback={lookback_quarters}")
    return returns


def rank_by_momentum(momentum_df: DataFrame) -> DataFrame:
    """
    Rank securities by momentum within each date (cross-section).
    
    Lower rank is better (rank 1 = best momentum).
    Ties broken by ticker symbol (alphabetically earlier = better).
    
    Args:
        momentum_df: DataFrame indexed by date, columns=tickers, values=momentum.
    
    Returns:
        DataFrame with same structure, values=ranks (1=best, ascending).
    
    Raises:
        ValueError: If momentum_df is not a DataFrame.
    """
    if not isinstance(momentum_df, DataFrame):
        raise ValueError("momentum_df must be a DataFrame")
    
    # Rank within each row (across tickers), descending by momentum value
    # method='first' for tiebreaker, then we'll adjust alphabetically
    def rank_row_with_tiebreak(row):
        """Rank a row, breaking ties by ticker name."""
        # Get indices sorted by momentum (descending), then by ticker (ascending)
        valid_idx = row.notna()
        valid_data = row[valid_idx]
        
        # Sort by value descending, then by index (ticker) ascending
        sorted_tickers = valid_data.sort_values(ascending=False).index.tolist()
        sorted_tickers_by_name = sorted(sorted_tickers, key=lambda x: valid_data[x], reverse=True)
        sorted_tickers_by_name = sorted(
            sorted_tickers_by_name,
            key=lambda x: (valid_data[x], x),
            reverse=True
        )
        
        # Assign ranks
        ranks = pd.Series(dtype='float64', index=row.index)
        for rank, ticker in enumerate(sorted_tickers_by_name, start=1):
            ranks[ticker] = rank
        return ranks
    
    ranks = momentum_df.apply(rank_row_with_tiebreak, axis=1)
    
    logger.info("Ranked momentum across tickers, ties broken by symbol")
    return ranks


def top_n_and_threshold(ranks: DataFrame, n: int, threshold: int) -> tuple:
    """
    Create boolean DataFrames for entry set and within-threshold set.
    
    Args:
        ranks: DataFrame indexed by date, columns=tickers, values=ranks (1=best).
        n: Number of top securities for entry set.
        threshold: Rank threshold (inclusive); all ranks <= threshold marked True.
    
    Returns:
        Tuple of (entry_bool, threshold_bool):
        - entry_bool: Boolean DataFrame, True for top n within each date
        - threshold_bool: Boolean DataFrame, True for all ranks <= threshold
    
    Raises:
        ValueError: If n <= 0 or threshold <= 0 or threshold < n.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    if threshold < n:
        raise ValueError("threshold must be >= n")
    
    # Entry: top n (rank <= n)
    entry_bool = ranks <= n
    
    # Threshold: within threshold (rank <= threshold)
    threshold_bool = ranks <= threshold
    
    logger.info(f"Created entry set (top {n}) and threshold set (rank <= {threshold})")
    return entry_bool, threshold_bool
