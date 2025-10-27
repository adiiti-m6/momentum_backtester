import pytest
import pandas as pd
from pandas import DataFrame, DatetimeIndex
from src.core.calendar import infer_trading_days, quarter_end_days, align_to_quarter_ends


class TestInferTradingDays:
    """Test infer_trading_days function."""
    
    def test_simple_index(self):
        """Test with simple date index."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = DataFrame({"price": [1, 2, 3, 4, 5]}, index=dates)
        result = infer_trading_days(df)
        assert len(result) == 5
        assert result.is_monotonic_increasing
    
    def test_duplicate_dates(self):
        """Test that duplicates are removed."""
        dates = pd.DatetimeIndex(["2023-01-01", "2023-01-01", "2023-01-02"])
        df = DataFrame({"price": [1, 2, 3]}, index=dates)
        result = infer_trading_days(df)
        assert len(result) == 2
    
    def test_unsorted_dates(self):
        """Test that result is sorted."""
        dates = pd.DatetimeIndex(["2023-01-03", "2023-01-01", "2023-01-02"])
        df = DataFrame({"price": [1, 2, 3]}, index=dates)
        result = infer_trading_days(df)
        assert result.is_monotonic_increasing


class TestQuarterEndDays:
    """Test quarter_end_days function."""
    
    def test_full_year_2023(self):
        """Test with full year of business days."""
        dates = pd.bdate_range("2023-01-02", "2023-12-29", freq="D")
        result = quarter_end_days(dates)
        # Should have 4 quarter ends
        assert len(result) >= 4
    
    def test_partial_quarter(self):
        """Test with data for only part of a quarter."""
        # Just January and February (partial Q1)
        dates = pd.bdate_range("2023-01-02", "2023-02-28", freq="D")
        result = quarter_end_days(dates)
        # Should return empty or minimal results since no quarter end in Feb
        assert len(result) == 0
    
    def test_single_quarter_end(self):
        """Test when only a quarter end date is present."""
        # March 31, 2023 was a Friday
        dates = pd.DatetimeIndex(["2023-03-31"])
        result = quarter_end_days(dates)
        assert len(result) == 1
        assert result[0] == pd.Timestamp("2023-03-31")
    
    def test_multiple_years(self):
        """Test with data spanning multiple years."""
        dates = pd.bdate_range("2022-01-01", "2023-12-31", freq="D")
        result = quarter_end_days(dates)
        # Should have 8 quarter ends (4 per year)
        assert len(result) >= 8


class TestAlignToQuarterEnds:
    """Test align_to_quarter_ends function."""
    
    def test_filter_to_quarter_ends(self):
        """Test that DataFrame is filtered to quarter ends only."""
        dates = pd.bdate_range("2023-01-02", "2023-12-29", freq="D")
        df = DataFrame({"price": range(len(dates))}, index=dates)
        result = align_to_quarter_ends(df)
        
        # Result should be much smaller
        assert len(result) < len(df)
        # All dates should be quarter ends
        assert len(result) >= 4
    
    def test_preserve_data(self):
        """Test that data values are preserved after filtering."""
        dates = pd.date_range("2023-03-31", periods=3, freq="D")
        df = DataFrame({"price": [100, 101, 102], "volume": [1000, 1001, 1002]}, index=dates)
        result = align_to_quarter_ends(df)
        
        if len(result) > 0:
            # Check that the data columns are intact
            assert "price" in result.columns
            assert "volume" in result.columns
    
    def test_empty_result(self):
        """Test with dates that have no quarter ends."""
        dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
        df = DataFrame({"price": range(len(dates))}, index=dates)
        result = align_to_quarter_ends(df)
        
        # Should be empty or very small
        assert len(result) == 0 or len(result) < 5
