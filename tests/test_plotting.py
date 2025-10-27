import pytest
import pandas as pd
from pandas import Series
import numpy as np
import plotly.graph_objects as go
from src.core.plotting import (
    plot_equity_curve,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_quarterly_hit_ratio,
    plot_metrics_dashboard
)


@pytest.fixture
def sample_equity():
    """Create sample equity series."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    values = 1e6 * (1.1 ** (np.arange(252) / 252))
    return Series(values, index=dates)


@pytest.fixture
def sample_returns():
    """Create sample returns series."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    returns = np.random.normal(0.0005, 0.01, 252)
    return Series(returns, index=dates)


@pytest.fixture
def sample_quarterly_returns():
    """Create sample quarterly returns."""
    dates = pd.date_range("2023-Q1", periods=8, freq="QE")
    returns = Series([0.05, -0.02, 0.03, 0.01, -0.01, 0.04, 0.02, -0.01], index=dates)
    return returns


class TestPlotEquityCurve:
    """Test equity curve plotting."""
    
    def test_basic_plot(self, sample_equity):
        """Test basic equity curve plot creation."""
        fig = plot_equity_curve(sample_equity)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Portfolio Equity Curve"
    
    def test_custom_title(self, sample_equity):
        """Test custom title."""
        custom_title = "My Strategy Returns"
        fig = plot_equity_curve(sample_equity, title=custom_title)
        
        assert fig.layout.title.text == custom_title
    
    def test_axes_labels(self, sample_equity):
        """Test that axes are labeled correctly."""
        fig = plot_equity_curve(sample_equity)
        
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Cumulative Return (%)"
    
    def test_empty_series(self):
        """Test error handling for empty series."""
        empty = Series(dtype=float)
        
        with pytest.raises(ValueError):
            plot_equity_curve(empty)
    
    def test_non_datetime_index(self):
        """Test error handling for non-DatetimeIndex."""
        bad_series = Series([1, 2, 3], index=[0, 1, 2])
        
        with pytest.raises(ValueError):
            plot_equity_curve(bad_series)
    
    def test_template_is_white(self, sample_equity):
        """Test that plotly_white template is used."""
        fig = plot_equity_curve(sample_equity)
        
        assert fig.layout.template.name == "plotly_white"


class TestPlotDrawdown:
    """Test drawdown plotting."""
    
    def test_basic_plot(self, sample_equity):
        """Test basic drawdown plot creation."""
        fig = plot_drawdown(sample_equity)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Drawdown Over Time"
    
    def test_custom_title(self, sample_equity):
        """Test custom title."""
        custom_title = "Portfolio Drawdown"
        fig = plot_drawdown(sample_equity, title=custom_title)
        
        assert fig.layout.title.text == custom_title
    
    def test_axes_labels(self, sample_equity):
        """Test axes labels."""
        fig = plot_drawdown(sample_equity)
        
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Drawdown (%)"
    
    def test_max_drawdown_annotation(self, sample_equity):
        """Test that max drawdown is annotated."""
        fig = plot_drawdown(sample_equity)
        
        # Check for max DD annotation
        shapes = fig.layout.shapes
        assert len(shapes) > 0
    
    def test_empty_series(self):
        """Test error handling."""
        empty = Series(dtype=float)
        
        with pytest.raises(ValueError):
            plot_drawdown(empty)


class TestPlotRollingSharp:
    """Test rolling Sharpe ratio plotting."""
    
    def test_basic_plot(self, sample_returns):
        """Test basic rolling Sharpe plot."""
        fig = plot_rolling_sharpe(sample_returns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Rolling Sharpe Ratio"
    
    def test_custom_window(self, sample_returns):
        """Test with custom rolling window."""
        fig = plot_rolling_sharpe(sample_returns, window=30)
        
        assert isinstance(fig, go.Figure)
    
    def test_custom_title(self, sample_returns):
        """Test custom title."""
        custom_title = "Risk-Adjusted Returns"
        fig = plot_rolling_sharpe(sample_returns, title=custom_title)
        
        assert fig.layout.title.text == custom_title
    
    def test_axes_labels(self, sample_returns):
        """Test axes labels."""
        fig = plot_rolling_sharpe(sample_returns)
        
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Sharpe Ratio"
    
    def test_zero_line_annotation(self, sample_returns):
        """Test that zero line is present."""
        fig = plot_rolling_sharpe(sample_returns)
        
        # Should have horizontal line at y=0
        shapes = fig.layout.shapes
        assert len(shapes) > 0
    
    def test_invalid_window(self, sample_returns):
        """Test error handling for invalid window."""
        with pytest.raises(ValueError):
            plot_rolling_sharpe(sample_returns, window=0)
        
        with pytest.raises(ValueError):
            plot_rolling_sharpe(sample_returns, window=len(sample_returns) + 1)
    
    def test_empty_series(self):
        """Test error handling for empty series."""
        empty = Series(dtype=float)
        
        with pytest.raises(ValueError):
            plot_rolling_sharpe(empty)


class TestPlotQuarterlyHitRatio:
    """Test quarterly hit ratio plotting."""
    
    def test_basic_plot(self, sample_quarterly_returns):
        """Test basic hit ratio plot."""
        fig = plot_quarterly_hit_ratio(sample_quarterly_returns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Quarterly Hit Ratio"
    
    def test_custom_title(self, sample_quarterly_returns):
        """Test custom title."""
        custom_title = "Win Rate"
        fig = plot_quarterly_hit_ratio(sample_quarterly_returns, title=custom_title)
        
        assert fig.layout.title.text == custom_title
    
    def test_axes_labels(self, sample_quarterly_returns):
        """Test axes labels."""
        fig = plot_quarterly_hit_ratio(sample_quarterly_returns)
        
        assert fig.layout.xaxis.title.text == "Quarter"
        assert fig.layout.yaxis.title.text == "Return (%)"
    
    def test_hit_ratio_annotation(self, sample_quarterly_returns):
        """Test that hit ratio is displayed."""
        fig = plot_quarterly_hit_ratio(sample_quarterly_returns)
        
        # Should have annotation with hit ratio
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert "Hit Ratio" in annotations[0].text
    
    def test_bar_colors(self, sample_quarterly_returns):
        """Test that bars are colored green/red."""
        fig = plot_quarterly_hit_ratio(sample_quarterly_returns)
        
        # Check bar colors
        bar_data = fig.data[0]
        assert hasattr(bar_data, 'marker')
        assert bar_data.marker.color is not None
    
    def test_empty_series(self):
        """Test error handling."""
        empty = Series(dtype=float)
        
        with pytest.raises(ValueError):
            plot_quarterly_hit_ratio(empty)
    
    def test_non_datetime_index(self):
        """Test error handling."""
        bad_series = Series([0.01, -0.02], index=[0, 1])
        
        with pytest.raises(ValueError):
            plot_quarterly_hit_ratio(bad_series)


class TestPlotMetricsDashboard:
    """Test the 2x2 metrics dashboard."""
    
    def test_dashboard_creation(self, sample_equity, sample_returns, sample_quarterly_returns):
        """Test that dashboard is created correctly."""
        fig = plot_metrics_dashboard(sample_equity, sample_returns, sample_quarterly_returns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4  # At least 4 traces (one per subplot)
    
    def test_dashboard_title(self, sample_equity, sample_returns, sample_quarterly_returns):
        """Test dashboard title."""
        fig = plot_metrics_dashboard(sample_equity, sample_returns, sample_quarterly_returns)
        
        assert fig.layout.title.text == "Backtest Performance Dashboard"
    
    def test_custom_dashboard_title(self, sample_equity, sample_returns, sample_quarterly_returns):
        """Test custom dashboard title."""
        custom_title = "My Strategy Analysis"
        fig = plot_metrics_dashboard(
            sample_equity,
            sample_returns,
            sample_quarterly_returns,
            title=custom_title
        )
        
        assert fig.layout.title.text == custom_title
    
    def test_subplot_titles(self, sample_equity, sample_returns, sample_quarterly_returns):
        """Test that subplot titles are present."""
        fig = plot_metrics_dashboard(sample_equity, sample_returns, sample_quarterly_returns)
        
        # Should have 4 subplots
        assert len(fig._grid_ref) == 2
    
    def test_empty_equity_series(self, sample_returns, sample_quarterly_returns):
        """Test error handling for empty equity series."""
        empty = Series(dtype=float)
        
        with pytest.raises(ValueError):
            plot_metrics_dashboard(empty, sample_returns, sample_quarterly_returns)
    
    def test_empty_returns_series(self, sample_equity, sample_quarterly_returns):
        """Test error handling for empty returns series."""
        empty = Series(dtype=float)
        
        with pytest.raises(ValueError):
            plot_metrics_dashboard(sample_equity, empty, sample_quarterly_returns)
    
    def test_empty_quarterly_series(self, sample_equity, sample_returns):
        """Test error handling for empty quarterly series."""
        empty = Series(dtype=float)
        
        with pytest.raises(ValueError):
            plot_metrics_dashboard(sample_equity, sample_returns, empty)
    
    def test_template_is_white(self, sample_equity, sample_returns, sample_quarterly_returns):
        """Test that plotly_white template is used."""
        fig = plot_metrics_dashboard(sample_equity, sample_returns, sample_quarterly_returns)
        
        assert fig.layout.template.name == "plotly_white"
    
    def test_dashboard_height(self, sample_equity, sample_returns, sample_quarterly_returns):
        """Test that dashboard has appropriate height."""
        fig = plot_metrics_dashboard(sample_equity, sample_returns, sample_quarterly_returns)
        
        assert fig.layout.height == 900
