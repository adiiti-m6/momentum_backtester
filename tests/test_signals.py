import pytest
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from src.core.signals import (
    quarterly_total_return,
    rank_by_momentum,
    top_n_and_threshold
)


class TestQuarterlyTotalReturn:
    """Test quarterly_total_return function."""
    
    def test_basic_return_calculation(self):
        """Test basic return calculation."""
        dates = pd.date_range("2023-01-01", periods=4, freq="QE")
        prices = DataFrame({
            "AAPL": [100.0, 110.0, 121.0, 133.1],
            "MSFT": [200.0, 210.0, 220.5, 231.525]
        }, index=dates)
        
        returns = quarterly_total_return(prices, lookback_quarters=1)
        
        # Second row should be (110-100)/100 = 0.1
        assert np.isclose(returns.loc[dates[1], "AAPL"], 0.1)
        # First row should be NaN (no lookback)
        assert pd.isna(returns.loc[dates[0], "AAPL"])
    
    def test_multi_quarter_lookback(self):
        """Test with lookback > 1."""
        dates = pd.date_range("2023-01-01", periods=5, freq="QE")
        prices = DataFrame({
            "TICK": [100.0, 105.0, 110.25, 115.76, 121.54]
        }, index=dates)
        
        returns = quarterly_total_return(prices, lookback_quarters=2)
        
        # Row 2: (110.25 - 100) / 100 = 0.1025
        assert np.isclose(returns.loc[dates[2], "TICK"], 0.1025)
        # First two rows should be NaN
        assert pd.isna(returns.loc[dates[0], "TICK"])
        assert pd.isna(returns.loc[dates[1], "TICK"])
    
    def test_nan_handling(self):
        """Test handling of NaN values in prices."""
        dates = pd.date_range("2023-01-01", periods=3, freq="QE")
        prices = DataFrame({
            "A": [100.0, np.nan, 110.0],
            "B": [200.0, 210.0, 220.0]
        }, index=dates)
        
        returns = quarterly_total_return(prices, lookback_quarters=1)
        
        # A[1] is NaN, so return should be NaN
        assert pd.isna(returns.loc[dates[1], "A"])
        # B[1] should be (210-200)/200 = 0.05
        assert np.isclose(returns.loc[dates[1], "B"], 0.05)
    
    def test_invalid_lookback(self):
        """Test error handling for invalid lookback."""
        prices = DataFrame({"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3))
        
        with pytest.raises(ValueError):
            quarterly_total_return(prices, lookback_quarters=0)
        
        with pytest.raises(ValueError):
            quarterly_total_return(prices, lookback_quarters=-1)


class TestRankByMomentum:
    """Test rank_by_momentum function."""
    
    def test_simple_ranking(self):
        """Test basic ranking without ties."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        momentum = DataFrame({
            "A": [0.05],
            "B": [0.10],
            "C": [0.03]
        }, index=dates)
        
        ranks = rank_by_momentum(momentum)
        
        # B (0.10) should be rank 1, A (0.05) rank 2, C (0.03) rank 3
        assert ranks.loc[dates[0], "B"] == 1
        assert ranks.loc[dates[0], "A"] == 2
        assert ranks.loc[dates[0], "C"] == 3
    
    def test_nan_handling_in_ranking(self):
        """Test that NaN values are handled correctly."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        momentum = DataFrame({
            "A": [0.05],
            "B": [np.nan],
            "C": [0.03]
        }, index=dates)
        
        ranks = rank_by_momentum(momentum)
        
        # A should be rank 1, C rank 2, B should be NaN
        assert ranks.loc[dates[0], "A"] == 1
        assert ranks.loc[dates[0], "C"] == 2
        assert pd.isna(ranks.loc[dates[0], "B"])
    
    def test_tiebreaker_by_ticker(self):
        """Test that ties are broken by ticker name (alphabetically)."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        momentum = DataFrame({
            "B": [0.05],
            "A": [0.05],  # Same momentum as B
            "C": [0.03]
        }, index=dates)
        
        ranks = rank_by_momentum(momentum)
        
        # With equal momentum, A should rank before B (alphabetically)
        assert ranks.loc[dates[0], "A"] < ranks.loc[dates[0], "B"]
        assert ranks.loc[dates[0], "C"] == 3


class TestTopNAndThreshold:
    """Test top_n_and_threshold function."""
    
    def test_basic_top_n(self):
        """Test basic top N selection."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        ranks = DataFrame({
            "A": [1],
            "B": [2],
            "C": [3],
            "D": [4]
        }, index=dates)
        
        entry_bool, threshold_bool = top_n_and_threshold(ranks, n=2, threshold=4)
        
        # Entry: top 2
        assert entry_bool.loc[dates[0], "A"] == True
        assert entry_bool.loc[dates[0], "B"] == True
        assert entry_bool.loc[dates[0], "C"] == False
        
        # Threshold: rank <= 4 (all)
        assert threshold_bool.loc[dates[0], :].all()
    
    def test_threshold_exclusive(self):
        """Test threshold set excluding some ranks."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        ranks = DataFrame({
            "A": [1],
            "B": [2],
            "C": [3],
            "D": [4],
            "E": [5]
        }, index=dates)
        
        entry_bool, threshold_bool = top_n_and_threshold(ranks, n=2, threshold=3)
        
        # Threshold: rank <= 3
        assert threshold_bool.loc[dates[0], "A"] == True
        assert threshold_bool.loc[dates[0], "B"] == True
        assert threshold_bool.loc[dates[0], "C"] == True
        assert threshold_bool.loc[dates[0], "D"] == False
        assert threshold_bool.loc[dates[0], "E"] == False
    
    def test_nan_in_ranks(self):
        """Test handling of NaN ranks."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        ranks = DataFrame({
            "A": [1],
            "B": [np.nan],
            "C": [2]
        }, index=dates)
        
        entry_bool, threshold_bool = top_n_and_threshold(ranks, n=2, threshold=3)
        
        # NaN comparisons should result in False
        assert pd.isna(threshold_bool.loc[dates[0], "B"])
    
    def test_error_invalid_params(self):
        """Test error handling for invalid parameters."""
        ranks = DataFrame({"A": [1], "B": [2]}, index=pd.date_range("2023-01-01", periods=1))
        
        with pytest.raises(ValueError):
            top_n_and_threshold(ranks, n=0, threshold=2)
        
        with pytest.raises(ValueError):
            top_n_and_threshold(ranks, n=2, threshold=0)
        
        with pytest.raises(ValueError):
            top_n_and_threshold(ranks, n=3, threshold=2)  # threshold < n
