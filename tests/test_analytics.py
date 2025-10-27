import pytest
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from src.core.analytics import (
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_quarterly_hit_ratio,
    calculate_factor_exposures,
    compute_performance_metrics,
    create_quarterly_performance_table,
    create_annual_performance_table,
    create_summary_report
)


@pytest.fixture
def simple_returns():
    """Create simple, deterministic returns."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    # 10% annual return = ~0.038% daily
    daily_return = 0.10 / 252
    returns = Series([daily_return] * 252, index=dates)
    return returns


@pytest.fixture
def volatile_returns():
    """Create volatile returns with positive mean."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)
    returns = Series(
        np.random.normal(0.0005, 0.02, 252),
        index=dates
    )
    return returns


@pytest.fixture
def drawdown_returns():
    """Create returns with significant drawdown."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    returns = Series([0.01] * 50 + [-0.02] * 50, index=dates)  # Up 50%, down ~63%
    return returns


class TestCalculateCAGR:
    """Test CAGR calculation."""
    
    def test_flat_returns(self):
        """Test CAGR with zero returns."""
        returns = Series([0.0] * 252, index=pd.date_range("2023-01-01", periods=252))
        cagr = calculate_cagr(returns)
        assert cagr == 0.0
    
    def test_positive_returns(self, simple_returns):
        """Test CAGR with consistent positive returns."""
        cagr = calculate_cagr(simple_returns)
        # 10% annual should give ~10% CAGR
        assert 0.09 < cagr < 0.11
    
    def test_negative_returns(self):
        """Test CAGR with negative returns."""
        returns = Series([-0.001] * 252, index=pd.date_range("2023-01-01", periods=252))
        cagr = calculate_cagr(returns)
        assert cagr < 0
    
    def test_empty_series(self):
        """Test error handling for empty series."""
        with pytest.raises(ValueError):
            calculate_cagr(Series(dtype=float))
    
    def test_all_nan_series(self):
        """Test error handling for all NaN series."""
        returns = Series([np.nan] * 10)
        with pytest.raises(ValueError):
            calculate_cagr(returns)


class TestCalculateSharpeRatio:
    """Test Sharpe Ratio calculation."""
    
    def test_zero_volatility(self):
        """Test Sharpe with zero volatility."""
        returns = Series([0.001] * 252, index=pd.date_range("2023-01-01", periods=252))
        # Edge case: constant returns have zero std dev
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0
    
    def test_positive_excess_return(self, simple_returns):
        """Test Sharpe with positive returns."""
        sharpe = calculate_sharpe_ratio(simple_returns, risk_free_rate=0.0)
        assert sharpe > 0
    
    def test_with_risk_free_rate(self):
        """Test Sharpe with non-zero risk-free rate."""
        returns = Series([0.0005] * 252, index=pd.date_range("2023-01-01", periods=252))
        sharpe_low_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_high_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.05)
        # Higher risk-free rate -> lower Sharpe
        assert sharpe_high_rf < sharpe_low_rf
    
    def test_negative_returns(self):
        """Test Sharpe with negative returns."""
        returns = Series([-0.001] * 252, index=pd.date_range("2023-01-01", periods=252))
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe < 0


class TestCalculateMaxDrawdown:
    """Test Maximum Drawdown calculation."""
    
    def test_no_drawdown(self, simple_returns):
        """Test with only positive returns."""
        max_dd = calculate_max_drawdown(simple_returns)
        assert max_dd == 0.0
    
    def test_significant_drawdown(self, drawdown_returns):
        """Test with significant drawdown."""
        max_dd = calculate_max_drawdown(drawdown_returns)
        assert max_dd < -0.3  # Should be ~-60% after rise and fall
    
    def test_all_negative(self):
        """Test with all negative returns."""
        returns = Series([-0.01] * 100, index=pd.date_range("2023-01-01", periods=100))
        max_dd = calculate_max_drawdown(returns)
        assert -0.64 < max_dd < -0.62  # ~63% decline
    
    def test_empty_series(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            calculate_max_drawdown(Series(dtype=float))


class TestCalculateQuarterlyHitRatio:
    """Test Quarterly Hit Ratio calculation."""
    
    def test_all_positive_quarters(self):
        """Test with all positive quarterly returns."""
        dates = pd.date_range("2023-Q1", periods=4, freq="QE")
        returns = Series([0.01] * len(dates), index=dates)
        hit_ratio = calculate_quarterly_hit_ratio(returns)
        assert hit_ratio == 1.0
    
    def test_mixed_quarters(self):
        """Test with mix of positive and negative quarters."""
        dates = pd.date_range("2023-Q1", periods=8, freq="QE")
        returns = Series([0.05, -0.02, 0.03, 0.01, -0.01, 0.04, -0.03, 0.02], index=dates)
        hit_ratio = calculate_quarterly_hit_ratio(returns)
        # 6 positive out of 8 = 75%
        assert 0.74 < hit_ratio < 0.76
    
    def test_all_negative_quarters(self):
        """Test with all negative quarterly returns."""
        dates = pd.date_range("2023-Q1", periods=4, freq="QE")
        returns = Series([-0.01] * 4, index=dates)
        hit_ratio = calculate_quarterly_hit_ratio(returns)
        assert hit_ratio == 0.0
    
    def test_non_datetime_index(self):
        """Test error handling for non-DatetimeIndex."""
        returns = Series([0.01, -0.02, 0.03], index=[0, 1, 2])
        with pytest.raises(ValueError):
            calculate_quarterly_hit_ratio(returns)


class TestCalculateFactorExposures:
    """Test factor exposure calculation."""
    
    def test_no_factors(self):
        """Test with no factor data."""
        returns = Series([0.01] * 100, index=pd.date_range("2023-01-01", periods=100))
        exposures = calculate_factor_exposures(returns, factor_returns=None)
        assert exposures == {}
    
    def test_single_factor(self):
        """Test with a single factor."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = Series([0.01] * 100, index=dates)
        factor_ret = Series([0.01] * 100, index=dates)
        
        exposures = calculate_factor_exposures(returns, {"factor1": factor_ret})
        
        # Should have positive exposure to correlated factor
        assert "factor1" in exposures
        assert exposures["factor1"] > 0
    
    def test_multiple_factors(self):
        """Test with multiple factors."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = Series(np.random.normal(0, 0.01, 100), index=dates)
        factors = {
            "market": Series(np.random.normal(0, 0.01, 100), index=dates),
            "momentum": Series(np.random.normal(0, 0.01, 100), index=dates)
        }
        
        exposures = calculate_factor_exposures(returns, factors)
        
        assert len(exposures) == 2
        assert "market" in exposures
        assert "momentum" in exposures


class TestComputePerformanceMetrics:
    """Test comprehensive metrics computation."""
    
    def test_all_metrics(self, simple_returns):
        """Test that all metrics are computed."""
        metrics = compute_performance_metrics(simple_returns)
        
        assert "cagr" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "hit_ratio" in metrics
    
    def test_with_factors(self, simple_returns):
        """Test metrics with factor data."""
        dates = simple_returns.index
        factor_ret = Series([0.0004] * len(dates), index=dates)
        
        metrics = compute_performance_metrics(
            simple_returns,
            factor_returns={"factor1": factor_ret}
        )
        
        assert "factor_beta_factor1" in metrics


class TestCreateQuarterlyPerformanceTable:
    """Test quarterly performance table creation."""
    
    def test_table_creation(self):
        """Test creating quarterly table."""
        dates = pd.date_range("2023-Q1", periods=8, freq="QE")
        returns = Series([0.05, 0.03, -0.02, 0.04, 0.01, -0.01, 0.06, 0.02], index=dates)
        
        table = create_quarterly_performance_table(returns)
        
        assert len(table) == 8
        assert "date" in table.columns
        assert "return" in table.columns
        assert "cum_return" in table.columns
        assert all(table["date"] == dates)
    
    def test_return_percentages(self):
        """Test that returns are in percentage format."""
        dates = pd.date_range("2023-Q1", periods=4, freq="QE")
        returns = Series([0.05] * 4, index=dates)
        
        table = create_quarterly_performance_table(returns)
        
        # Should be 5.0 (%), not 0.05
        assert 4.99 < table["return"].iloc[0] < 5.01


class TestCreateAnnualPerformanceTable:
    """Test annual performance table creation."""
    
    def test_table_creation(self):
        """Test creating annual table."""
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        returns = Series([0.0001] * 365, index=dates)
        
        table = create_annual_performance_table(returns)
        
        assert len(table) >= 1
        assert "year" in table.columns
        assert "return" in table.columns
        assert "cum_return" in table.columns


class TestCreateSummaryReport:
    """Test comprehensive summary report."""
    
    def test_report_structure(self, simple_returns):
        """Test that report has all expected components."""
        report = create_summary_report(simple_returns)
        
        assert "metrics" in report
        assert "quarterly_performance" in report
        assert "annual_performance" in report
        
        assert isinstance(report["metrics"], dict)
        assert isinstance(report["quarterly_performance"], DataFrame)
        assert isinstance(report["annual_performance"], DataFrame)
    
    def test_report_with_factors(self, simple_returns):
        """Test report with factor data."""
        dates = simple_returns.index
        factor_ret = Series([0.0004] * len(dates), index=dates)
        
        report = create_summary_report(
            simple_returns,
            factor_returns={"test_factor": factor_ret}
        )
        
        assert "factor_beta_test_factor" in report["metrics"]
