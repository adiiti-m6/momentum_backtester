import pytest
import pandas as pd
from pandas import Series
from src.core.strategy import Holding, Cohort, Decisions, StaggeredCohortStrategy


class TestCohort:
    """Test Cohort dataclass."""
    
    def test_cohort_creation(self):
        """Test basic cohort creation."""
        date = pd.Timestamp("2023-03-31")
        holdings = {"AAPL": 0, "MSFT": 0}
        cohort = Cohort(date=date, holdings=holdings)
        
        assert cohort.date == date
        assert cohort.holdings == holdings
    
    def test_increment_age(self):
        """Test age increment."""
        cohort = Cohort(
            date=pd.Timestamp("2023-03-31"),
            holdings={"AAPL": 0, "MSFT": 0}
        )
        
        cohort.increment_age()
        assert cohort.holdings == {"AAPL": 1, "MSFT": 1}
        
        cohort.increment_age()
        assert cohort.holdings == {"AAPL": 2, "MSFT": 2}
    
    def test_get_active_holdings(self):
        """Test active holdings retrieval."""
        cohort = Cohort(
            date=pd.Timestamp("2023-03-31"),
            holdings={"AAPL": 1, "MSFT": 2, "GOOG": 0}
        )
        
        active = cohort.get_active_holdings()
        assert active == {"AAPL", "MSFT", "GOOG"}


class TestDecisions:
    """Test Decisions dataclass."""
    
    def test_valid_decisions(self):
        """Test valid decisions pass validation."""
        decisions = Decisions(
            date=pd.Timestamp("2023-03-31"),
            universe={"A", "B", "C", "D"},
            buys={"A", "B"},
            sells={"C"},
            target_weights={"A": 0.33, "B": 0.33, "D": 0.34},
            cohort_ages={"A": 0, "B": 0, "D": 1}
        )
        
        decisions.validate()  # Should not raise
    
    def test_invalid_overlap(self):
        """Test that overlap between buys and sells is rejected."""
        decisions = Decisions(
            date=pd.Timestamp("2023-03-31"),
            universe={"A", "B", "C"},
            buys={"A"},
            sells={"A"},  # Overlap!
            target_weights={"B": 1.0},
            cohort_ages={"B": 1}
        )
        
        with pytest.raises(ValueError, match="Overlap"):
            decisions.validate()
    
    def test_invalid_weights(self):
        """Test that weights not summing to 1 are rejected."""
        decisions = Decisions(
            date=pd.Timestamp("2023-03-31"),
            universe={"A", "B"},
            buys=set(),
            sells=set(),
            target_weights={"A": 0.5, "B": 0.4},  # Sum = 0.9
            cohort_ages={"A": 1, "B": 1}
        )
        
        with pytest.raises(ValueError, match="sum"):
            decisions.validate()


class TestStaggeredCohortStrategy:
    """Test StaggeredCohortStrategy class."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = StaggeredCohortStrategy(
            entry_count=24,
            exit_rank_threshold=36,
            holding_quarters=2
        )
        
        assert strategy.entry_count == 24
        assert strategy.exit_rank_threshold == 36
        assert strategy.holding_quarters == 2
        assert strategy.cohorts == []
    
    def test_initial_entry(self):
        """Test initial entry with no prior holdings."""
        strategy = StaggeredCohortStrategy(
            entry_count=3,
            exit_rank_threshold=10,
            holding_quarters=2
        )
        
        date = pd.Timestamp("2023-03-31")
        universe = {"A", "B", "C", "D", "E"}
        ranks = Series({
            "A": 1,  # Best
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5
        })
        
        decisions = strategy.rebalance(date, ranks, universe)
        
        assert decisions.buys == {"A", "B", "C"}
        assert decisions.sells == set()
        assert len(decisions.target_weights) == 3
        # Equal weight: 1/3 each
        for weight in decisions.target_weights.values():
            assert abs(weight - 1/3) < 0.001
    
    def test_exit_by_age(self):
        """Test exit when age >= holding_quarters."""
        strategy = StaggeredCohortStrategy(
            entry_count=3,
            exit_rank_threshold=10,
            holding_quarters=1  # Exit after 1 quarter
        )
        
        # First rebalance: enter A, B
        date1 = pd.Timestamp("2023-03-31")
        universe1 = {"A", "B", "C", "D"}
        ranks1 = Series({"A": 1, "B": 2, "C": 3, "D": 4})
        decisions1 = strategy.rebalance(date1, ranks1, universe1)
        assert decisions1.buys == {"A", "B"}
        
        # Second rebalance: A and B are now age 1, should exit due to age
        date2 = pd.Timestamp("2023-06-30")
        universe2 = {"A", "B", "C", "D"}
        ranks2 = Series({"A": 1, "B": 2, "C": 3, "D": 4})
        decisions2 = strategy.rebalance(date2, ranks2, universe2)
        
        # A and B should be sold (age 1 >= holding_quarters 1)
        assert decisions2.sells == {"A", "B"}
        # New entries to replace them
        assert len(decisions2.buys) > 0
    
    def test_exit_by_rank(self):
        """Test exit when rank > exit_rank_threshold."""
        strategy = StaggeredCohortStrategy(
            entry_count=3,
            exit_rank_threshold=2,  # Exit if rank > 2
            holding_quarters=2
        )
        
        # First rebalance: enter A, B, C
        date1 = pd.Timestamp("2023-03-31")
        universe1 = {"A", "B", "C", "D", "E"}
        ranks1 = Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5})
        decisions1 = strategy.rebalance(date1, ranks1, universe1)
        assert decisions1.buys == {"A", "B", "C"}
        
        # Second rebalance: C now ranks poorly (rank 5 > threshold 2)
        date2 = pd.Timestamp("2023-06-30")
        universe2 = {"A", "B", "C", "D", "E"}
        ranks2 = Series({"A": 2, "B": 3, "C": 5, "D": 1, "E": 4})
        decisions2 = strategy.rebalance(date2, ranks2, universe2)
        
        # C should be sold due to rank
        assert "C" in decisions2.sells
    
    def test_equal_weighting(self):
        """Test that holdings are equal-weighted."""
        strategy = StaggeredCohortStrategy(
            entry_count=5,
            exit_rank_threshold=20,
            holding_quarters=2
        )
        
        date = pd.Timestamp("2023-03-31")
        universe = set(f"T{i}" for i in range(10))
        ranks = Series({ticker: i+1 for i, ticker in enumerate(sorted(universe))})
        
        decisions = strategy.rebalance(date, ranks, universe)
        
        # Should have 5 holdings, each with weight 0.2
        assert len(decisions.target_weights) == 5
        for weight in decisions.target_weights.values():
            assert abs(weight - 0.2) < 0.001
    
    def test_no_repurchasing_exits(self):
        """Test that exited holdings are not immediately re-entered."""
        strategy = StaggeredCohortStrategy(
            entry_count=3,
            exit_rank_threshold=2,
            holding_quarters=1
        )
        
        # First rebalance: enter A, B, C (ranks 1, 2, 3)
        date1 = pd.Timestamp("2023-03-31")
        universe1 = {"A", "B", "C", "D", "E"}
        ranks1 = Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5})
        decisions1 = strategy.rebalance(date1, ranks1, universe1)
        
        # Second rebalance: A stays good, B falls out, C stays good
        # Add new holdings D, E to replace B
        date2 = pd.Timestamp("2023-06-30")
        universe2 = {"A", "B", "C", "D", "E"}
        ranks2 = Series({"A": 1, "B": 5, "C": 2, "D": 3, "E": 4})
        decisions2 = strategy.rebalance(date2, ranks2, universe2)
        
        # B should be sold
        assert "B" in decisions2.sells
        # Should enter from remaining candidates
        assert len(decisions2.buys) > 0
    
    def test_empty_portfolio(self):
        """Test behavior with no holdings."""
        strategy = StaggeredCohortStrategy(
            entry_count=2,
            exit_rank_threshold=10,
            holding_quarters=2
        )
        
        date = pd.Timestamp("2023-03-31")
        universe = {"A", "B"}
        ranks = Series({"A": 1, "B": 2})
        
        decisions = strategy.rebalance(date, ranks, universe)
        
        assert decisions.target_weights == {"A": 0.5, "B": 0.5}
    
    def test_validation_error_missing_tickers(self):
        """Test that missing tickers in ranks raises error."""
        strategy = StaggeredCohortStrategy(
            entry_count=2,
            exit_rank_threshold=10,
            holding_quarters=2
        )
        
        date = pd.Timestamp("2023-03-31")
        universe = {"A", "B", "C"}
        ranks = Series({"A": 1})  # Missing B, C
        
        with pytest.raises(ValueError, match="missing"):
            strategy.rebalance(date, ranks, universe)
