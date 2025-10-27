from dataclasses import dataclass, field
from typing import Dict, Set, List
from pandas import DataFrame, Series
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class Holding:
    """Represents a single held security."""
    ticker: str
    age: int  # Quarters held


@dataclass
class Cohort:
    """Represents a cohort of holdings from a single rebalance date."""
    date: pd.Timestamp
    holdings: Dict[str, int]  # ticker -> age
    
    def increment_age(self) -> None:
        """Increment age of all holdings in cohort."""
        for ticker in self.holdings:
            self.holdings[ticker] += 1
    
    def get_active_holdings(self) -> Set[str]:
        """Return set of active tickers in this cohort."""
        return set(self.holdings.keys())


@dataclass
class Decisions:
    """Encapsulates rebalance decisions for a single date."""
    date: pd.Timestamp
    universe: Set[str]  # Available tickers on this date
    buys: Set[str]
    sells: Set[str]
    target_weights: Dict[str, float]  # ticker -> weight
    cohort_ages: Dict[str, int]  # ticker -> age (quarters held)
    
    def validate(self) -> None:
        """Validate decisions for consistency."""
        # No overlap between buys and sells
        if self.buys & self.sells:
            raise ValueError(f"Overlap between buys {self.buys} and sells {self.sells}")
        
        # All buys and sells in universe
        if not self.buys.issubset(self.universe):
            raise ValueError(f"Buys {self.buys} not subset of universe {self.universe}")
        if not self.sells.issubset(self.universe):
            raise ValueError(f"Sells {self.sells} not subset of universe {self.universe}")
        
        # Weights sum to 1
        weight_sum = sum(self.target_weights.values())
        if not (0.99 < weight_sum < 1.01):  # Allow small floating point error
            raise ValueError(f"Weights sum to {weight_sum}, not 1.0")
        
        # All holdings have ages
        holdings = set(self.target_weights.keys())
        if holdings != set(self.cohort_ages.keys()):
            raise ValueError(
                f"Holdings {holdings} not equal to cohort ages keys {set(self.cohort_ages.keys())}"
            )


class StaggeredCohortStrategy:
    """
    Staggered cohort momentum strategy with aging.
    
    On each quarter-end:
    1. Compute momentum rank (1-quarter lookback)
    2. Select top entry_count tickers not currently held -> create new cohort
    3. Exit positions if: rank > exit_rank_threshold OR age >= holding_quarters
    4. Age all existing cohorts
    5. Equal-weight all active holdings
    """
    
    def __init__(
        self,
        entry_count: int,
        exit_rank_threshold: int,
        holding_quarters: int
    ):
        """
        Initialize strategy parameters.
        
        Args:
            entry_count: Number of top tickers to enter each quarter.
            exit_rank_threshold: Exit if rank > this threshold.
            holding_quarters: Maximum quarters to hold a position.
        """
        self.entry_count = entry_count
        self.exit_rank_threshold = exit_rank_threshold
        self.holding_quarters = holding_quarters
        self.cohorts: List[Cohort] = []
    
    def get_current_holdings(self) -> Dict[str, int]:
        """Return dict of all current holdings and their ages."""
        holdings = {}
        for cohort in self.cohorts:
            holdings.update(cohort.holdings)
        return holdings
    
    def rebalance(
        self,
        date: pd.Timestamp,
        ranks: Series,
        universe: Set[str]
    ) -> Decisions:
        """
        Execute rebalance on a quarter-end date.
        
        Args:
            date: Rebalance date.
            ranks: Series indexed by ticker, values=rank (1=best). Must contain all universe tickers.
            universe: Set of tickers available on this date with valid lookback.
        
        Returns:
            Decisions object with buys, sells, target weights, and ages.
        
        Raises:
            ValueError: If universe not covered by ranks or other validation issues.
        """
        # Validate inputs
        if not universe.issubset(set(ranks.index)):
            missing = universe - set(ranks.index)
            raise ValueError(f"Ranks missing tickers: {missing}")
        
        # Step 1: Age existing cohorts
        for cohort in self.cohorts:
            cohort.increment_age()
        
        # Step 2: Determine holdings to exit
        current_holdings = self.get_current_holdings()
        sells = set()
        
        for ticker, age in current_holdings.items():
            # Exit if age >= holding_quarters
            if age >= self.holding_quarters:
                sells.add(ticker)
                continue
            
            # Exit if rank > exit_rank_threshold (or rank is NaN)
            if pd.isna(ranks.get(ticker)) or ranks[ticker] > self.exit_rank_threshold:
                sells.add(ticker)
        
        # Step 3: Remove sold positions from cohorts
        for cohort in self.cohorts:
            for ticker in sells:
                cohort.holdings.pop(ticker, None)
        
        # Remove empty cohorts
        self.cohorts = [c for c in self.cohorts if c.holdings]
        
        # Step 4: Determine holdings to enter
        current_holdings = self.get_current_holdings()
        candidates = universe - set(current_holdings.keys())  # Not currently held
        
        # Rank candidates by momentum (only consider those in candidates set)
        candidate_ranks = ranks[ranks.index.isin(candidates)].dropna()
        candidate_ranks = candidate_ranks.sort_values()  # Ascending (1=best)
        
        # Select top entry_count candidates
        buys = set(candidate_ranks.head(self.entry_count).index)
        
        # Step 5: Create new cohort for entries
        if buys:
            new_cohort = Cohort(
                date=date,
                holdings={ticker: 0 for ticker in buys}
            )
            self.cohorts.append(new_cohort)
        
        # Step 6: Compute target weights (equal-weight all active holdings)
        all_holdings = self.get_current_holdings()
        n_holdings = len(all_holdings)
        
        if n_holdings == 0:
            target_weights = {}
        else:
            weight = 1.0 / n_holdings
            target_weights = {ticker: weight for ticker in all_holdings}
        
        # Build decisions object
        decisions = Decisions(
            date=date,
            universe=universe,
            buys=buys,
            sells=sells,
            target_weights=target_weights,
            cohort_ages=self.get_current_holdings()
        )
        
        decisions.validate()
        logger.info(
            f"Rebalance {date}: {len(buys)} buys, {len(sells)} sells, "
            f"{len(target_weights)} holdings"
        )
        
        return decisions
