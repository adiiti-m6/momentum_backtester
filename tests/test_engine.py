import pytest
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from src.core.engine import Trade, BacktestResult, BacktestEngine
from src.core.config import Config
from src.core.strategy import Decisions


@pytest.fixture
def sample_config():
    """Create a sample config for testing."""
    return Config(
        universe_size=10,
        entry_count=5,
        exit_rank_threshold=8,
        lookback_months=3,
        holding_quarters=2,
        transaction_cost_bps=10,
        slippage_bps=5
    )


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    tickers = [f"T{i}" for i in range(10)]
    
    prices = DataFrame(
        np.random.uniform(90, 110, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )
    
    # Make prices monotonic to reduce randomness
    prices = prices.cumsum() / 100 + 100
    
    return prices


@pytest.fixture
def simple_prices():
    """Create simple deterministic prices."""
    dates = pd.date_range("2023-Q1", periods=4, freq="QE")
    prices = DataFrame({
        "A": [100.0, 110.0, 121.0, 133.1],
        "B": [200.0, 210.0, 220.5, 231.5]
    }, index=dates)
    return prices


class TestTrade:
    """Test Trade dataclass."""
    
    def test_trade_creation(self):
        """Test creating a trade record."""
        trade = Trade(
            date=pd.Timestamp("2023-03-31"),
            ticker="AAPL",
            side="BUY",
            shares=100,
            price=150.0,
            gross_cost=15000,
            transaction_cost=150
        )
        
        assert trade.ticker == "AAPL"
        assert trade.side == "BUY"
        assert trade.transaction_cost == 150


class TestBacktestResult:
    """Test BacktestResult dataclass."""
    
    def test_compute_summary_stats(self, sample_config, simple_prices):
        """Test summary statistics computation."""
        dates = simple_prices.index
        equity = Series([1e6, 1.05e6, 1.1e6, 1.15e6], index=dates)
        returns = equity.pct_change().fillna(0.0)
        
        result = BacktestResult(
            config=sample_config,
            start_date=dates[0],
            end_date=dates[-1],
            equity_curve=equity,
            daily_returns=returns,
            daily_exposure=Series([1.0] * len(dates), index=dates),
            daily_turnover=Series([0.2] * len(dates), index=dates),
            positions=DataFrame(index=dates),
            weights=DataFrame(index=dates)
        )
        
        result.compute_summary_stats()
        
        assert result.total_return > 0
        assert result.annualized_return > 0
        assert result.avg_turnover == 0.2
    
    def test_empty_result_stats(self, sample_config):
        """Test stats with empty data."""
        result = BacktestResult(
            config=sample_config,
            start_date=pd.Timestamp("2023-01-01"),
            end_date=pd.Timestamp("2023-12-31"),
            equity_curve=Series(dtype=float),
            daily_returns=Series(dtype=float),
            daily_exposure=Series(dtype=float),
            daily_turnover=Series(dtype=float),
            positions=DataFrame(),
            weights=DataFrame()
        )
        
        result.compute_summary_stats()  # Should not raise


class TestBacktestEngine:
    """Test BacktestEngine class."""
    
    def test_initialization(self, sample_config, simple_prices):
        """Test engine initialization."""
        def dummy_strategy(date, pos, equity):
            return Decisions(
                date=date,
                universe={"A", "B"},
                buys=set(),
                sells=set(),
                target_weights={},
                cohort_ages={}
            )
        
        engine = BacktestEngine(
            prices=simple_prices,
            config=sample_config,
            strategy_callable=dummy_strategy,
            initial_equity=1e6
        )
        
        assert len(engine.prices) == len(simple_prices)
        assert engine.initial_equity == 1e6
    
    def test_invalid_prices_index(self, sample_config):
        """Test validation of price index."""
        # Non-DatetimeIndex
        prices = DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
        
        def dummy_strategy(date, pos, equity):
            return Decisions(date=date, universe=set(), buys=set(), sells=set(),
                           target_weights={}, cohort_ages={})
        
        with pytest.raises(ValueError, match="DatetimeIndex"):
            BacktestEngine(prices, sample_config, dummy_strategy)
    
    def test_invalid_rebalance_days(self, sample_config, simple_prices):
        """Test validation of rebalance days."""
        def dummy_strategy(date, pos, equity):
            return Decisions(date=date, universe=set(), buys=set(), sells=set(),
                           target_weights={}, cohort_ages={})
        
        engine = BacktestEngine(simple_prices, sample_config, dummy_strategy)
        
        # Rebalance day not in prices
        invalid_dates = pd.DatetimeIndex(["2023-01-01"])
        
        with pytest.raises(ValueError, match="not in prices"):
            engine.run(invalid_dates)
    
    def test_simple_buy_and_hold(self, sample_config, simple_prices):
        """Test simple buy-and-hold strategy."""
        def buy_and_hold_strategy(date, pos, equity):
            if not pos:  # First rebalance
                return Decisions(
                    date=date,
                    universe={"A", "B"},
                    buys={"A"},
                    sells=set(),
                    target_weights={"A": 1.0},
                    cohort_ages={"A": 0}
                )
            else:
                # Hold
                return Decisions(
                    date=date,
                    universe={"A", "B"},
                    buys=set(),
                    sells=set(),
                    target_weights={"A": 1.0},
                    cohort_ages={"A": 1}
                )
        
        engine = BacktestEngine(
            prices=simple_prices,
            config=sample_config,
            strategy_callable=buy_and_hold_strategy,
            initial_equity=1e6
        )
        
        rebalance_days = pd.DatetimeIndex([simple_prices.index[0]])
        result = engine.run(rebalance_days)
        
        assert result.total_return > 0
        assert len(result.trades) > 0
    
    def test_compute_trades_with_costs(self, sample_config):
        """Test trade computation with costs and slippage."""
        engine = BacktestEngine(
            prices=DataFrame({"A": [100]}, index=[pd.Timestamp("2023-01-01")]),
            config=sample_config,
            strategy_callable=lambda d, p, e: None,
            initial_equity=1e6
        )
        
        current_positions = {}
        target_shares = {"A": 100.0}
        date = pd.Timestamp("2023-01-01")
        available_prices = Series({"A": 100.0})
        
        trades = engine._compute_trades(current_positions, target_shares, date, available_prices)
        
        assert len(trades) == 1
        trade = trades[0]
        assert trade.ticker == "A"
        assert trade.side == "BUY"
        assert trade.shares == 100.0
        assert trade.transaction_cost > 0  # Should have transaction cost
    
    def test_dict_to_dataframe(self):
        """Test helper conversion function."""
        data = {
            pd.Timestamp("2023-01-01"): {"A": 0.5, "B": 0.5},
            pd.Timestamp("2023-01-02"): {"A": 0.6, "B": 0.4}
        }
        
        df = BacktestEngine._dict_to_dataframe(data)
        
        assert isinstance(df, DataFrame)
        assert len(df) == 2
        assert "A" in df.columns
        assert "B" in df.columns
    
    def test_empty_portfolio_handling(self, sample_config):
        """Test handling of empty portfolio (all cash)."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        prices = DataFrame({"A": [100, 101], "B": [200, 202]}, index=dates)
        
        def no_positions_strategy(date, pos, equity):
            return Decisions(
                date=date,
                universe={"A", "B"},
                buys=set(),
                sells=set(),
                target_weights={},  # Empty = all cash
                cohort_ages={}
            )
        
        engine = BacktestEngine(prices, sample_config, no_positions_strategy)
        result = engine.run(pd.DatetimeIndex([dates[0]]))
        
        # Should return ~0 return with empty portfolio
        assert result.total_return >= -0.01  # Allow small numerical error
    
    def test_missing_price_handling(self, sample_config):
        """Test handling of missing prices."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        prices = DataFrame({
            "A": [100, np.nan, 102],
            "B": [200, 202, 204]
        }, index=dates)
        
        def hold_both_strategy(date, pos, equity):
            return Decisions(
                date=date,
                universe={"A", "B"},
                buys=set(),
                sells=set(),
                target_weights={"A": 0.5, "B": 0.5},
                cohort_ages={"A": 0, "B": 0}
            )
        
        engine = BacktestEngine(prices, sample_config, hold_both_strategy)
        result = engine.run(pd.DatetimeIndex([dates[0]]))
        
        # Should handle NaN prices gracefully
        assert result is not None
        assert len(result.trades) > 0
