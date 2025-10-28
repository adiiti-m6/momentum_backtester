from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pandas import DataFrame, Series, DatetimeIndex
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade execution."""
    date: pd.Timestamp
    ticker: str
    side: str  # 'BUY' or 'SELL'
    shares: float
    price: float  # Execution price after slippage
    gross_cost: float  # Shares * price
    transaction_cost: float  # In dollars


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: 'Config'  # type hint
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    
    # Time series
    equity_curve: Series  # date -> equity value
    daily_returns: Series  # date -> daily return %
    daily_exposure: Series  # date -> gross exposure (sum of abs weights)
    daily_turnover: Series  # date -> turnover %
    
    # Position tracking
    positions: DataFrame  # date x ticker -> shares
    weights: DataFrame  # date x ticker -> weight (value / equity)
    
    # Trade ledger
    trades: List[Trade] = field(default_factory=list)
    
    # Ticker performance tracking
    ticker_quarterly_returns: Optional[DataFrame] = None  # quarter x ticker -> return %
    
    # Summary stats
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_turnover: float = 0.0
    
    def compute_summary_stats(self, risk_free_rate: float = 0.0) -> None:
        """Compute summary statistics."""
        if len(self.daily_returns) == 0:
            return
        
        # Total return
        self.total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1.0
        
        # Annualized return (assuming 252 trading days per year)
        days = (self.end_date - self.start_date).days
        years = days / 365.25
        self.annualized_return = (1 + self.total_return) ** (1 / years) - 1.0 if years > 0 else 0.0
        
        # Sharpe ratio
        daily_ret = self.daily_returns
        excess_ret = daily_ret - risk_free_rate / 252
        if excess_ret.std() > 0:
            self.sharpe_ratio = np.sqrt(252) * excess_ret.mean() / excess_ret.std()
        
        # Max drawdown
        cumulative = (1 + daily_ret).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        self.max_drawdown = drawdown.min()
        
        # Average turnover
        self.avg_turnover = self.daily_turnover.mean() if len(self.daily_turnover) > 0 else 0.0


class BacktestEngine:
    """
    Vectorized backtest engine for momentum strategies.
    
    Handles:
    - Daily mark-to-market P&L
    - Transaction costs and slippage
    - Weight rebalancing with trade execution
    - Position tracking and ledger
    - Exposure and turnover computation
    """
    
    def __init__(
        self,
        prices: DataFrame,
        config: 'Config',
        strategy_callable,
        volumes: Optional[DataFrame] = None,
        initial_equity: float = 1e6
    ):
        """
        Initialize backtest engine.
        
        Args:
            prices: DataFrame indexed by date, columns=tickers, values=adj_close.
            config: Config object with backtest parameters.
            strategy_callable: Callable(date, prices_window, volumes_window) -> Decisions.
            volumes: Optional DataFrame indexed by date, columns=tickers, values=volume.
            initial_equity: Starting equity in dollars.
        """
        self.prices = prices.copy()
        self.config = config
        self.strategy = strategy_callable
        self.volumes = volumes.copy() if volumes is not None else None
        self.initial_equity = initial_equity
        
        # Validation
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise ValueError("Prices index must be DatetimeIndex")
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("Prices must be sorted by date")
        
        logger.info(
            f"Initialized engine: {len(prices)} dates, {len(prices.columns)} tickers, "
            f"equity={initial_equity}"
        )
    
    def run(self, rebalance_days: DatetimeIndex) -> BacktestResult:
        """
        Run backtest over specified rebalance days.
        
        Args:
            rebalance_days: DatetimeIndex of quarter-end rebalance dates.
                           Must be subset of prices.index.
        
        Returns:
            BacktestResult with full backtest history.
        
        Raises:
            ValueError: If rebalance_days not in prices.index or other validation issues.
        """
        # Validate rebalance days
        if not rebalance_days.isin(self.prices.index).all():
            invalid = rebalance_days[~rebalance_days.isin(self.prices.index)]
            raise ValueError(f"Rebalance days not in prices: {invalid}")
        
        rebalance_days = rebalance_days.sort_values()
        
        # Minimum equity threshold to prevent division errors
        MIN_EQUITY = 1.0
        
        # Initialize tracking structures
        equity_curve = []
        dates_list = []
        daily_returns_list = []
        daily_pnl_list = []
        positions_dict = {}  # date -> {ticker -> shares}
        weights_dict = {}  # date -> {ticker -> weight}
        trades_list = []
        daily_exposure_list = []
        daily_turnover_list = []
        
        # Current state
        current_positions = {}  # ticker -> shares
        current_equity = self.initial_equity
        last_rebalance_idx = 0
        
        # Process each date
        for date_idx, date in enumerate(self.prices.index):
            # Check if this is a rebalance date
            is_rebalance = date in rebalance_days
            
            if is_rebalance:
                # Call strategy to get target weights
                try:
                    decisions = self.strategy(date, current_positions, current_equity)
                    target_weights = decisions.target_weights
                except Exception as e:
                    logger.error(f"Strategy failed on {date}: {e}")
                    raise
                
                # Get available prices on rebalance day
                available_prices = self.prices.loc[date]
                available_tickers = set(available_prices.index[available_prices.notna()])
                
                # Filter target weights to available tickers
                filtered_weights = {
                    t: w for t, w in target_weights.items()
                    if t in available_tickers
                }
                
                # Re-normalize weights
                weight_sum = sum(filtered_weights.values())
                if weight_sum > 1e-6:
                    filtered_weights = {t: w / weight_sum for t, w in filtered_weights.items()}
                else:
                    filtered_weights = {}
                
                # Compute target shares
                target_shares = {}
                for ticker, weight in filtered_weights.items():
                    price = available_prices[ticker]
                    if price > 0:
                        target_shares[ticker] = (current_equity * weight) / price
                
                # Compute trades (current -> target)
                trades = self._compute_trades(
                    current_positions,
                    target_shares,
                    date,
                    available_prices
                )
                
                # Execute trades and update equity
                for trade in trades:
                    current_equity -= trade.transaction_cost
                    trades_list.append(trade)
                
                # Update positions to target (after all trades executed)
                current_positions = target_shares.copy()
                
                # Compute turnover
                gross_value_traded = sum(t.gross_cost for t in trades)
                turnover = gross_value_traded / current_equity if current_equity > 0 else 0.0
                
                last_rebalance_idx = date_idx
            
            # Mark to market for all dates
            available_prices = self.prices.loc[date]
            
            # Compute portfolio value at market prices
            current_value = 0.0
            
            for ticker, shares in list(current_positions.items()):
                if pd.notna(available_prices.get(ticker)):
                    price = available_prices[ticker]
                    current_value += shares * price
                else:
                    # Price unavailable; drop position (mark to zero)
                    logger.warning(f"Price missing for {ticker} on {date}; dropping position")
                    current_positions.pop(ticker, None)
            
            # Portfolio equity = cash + position values
            # Start with whatever cash we have left after trades
            portfolio_equity = current_equity + current_value
            
            # Check for critically low equity
            if portfolio_equity < MIN_EQUITY and date_idx > 0:
                logger.error(f"Portfolio equity dropped to ${portfolio_equity:.2f} on {date}; stopping")
                portfolio_equity = 0.0  # Mark as failed
            
            # Compute daily return (only if we have positions or previous equity)
            if date_idx > 0:
                prev_equity = equity_curve[-1]
                daily_return = (portfolio_equity - prev_equity) / prev_equity if prev_equity > MIN_EQUITY else 0.0
            else:
                daily_return = 0.0
            
            # Record portfolio equity for this date
            equity_curve.append(portfolio_equity)
            dates_list.append(date)
            daily_returns_list.append(daily_return)
            daily_pnl_list.append(current_value - (equity_curve[-2] if len(equity_curve) > 1 else portfolio_equity))
            
            # Compute exposure (gross notional)
            gross_exposure = sum(abs(shares * available_prices.get(ticker, 0.0)) for ticker, shares in current_positions.items())
            daily_exposure_list.append(gross_exposure)
            
            # Store positions and weights
            positions_dict[date] = current_positions.copy()
            weights_dict[date] = {
                t: (s * available_prices.get(t, 0.0)) / portfolio_equity if portfolio_equity > 0 else 0.0
                for t, s in current_positions.items()
            }
            
            # Store turnover (only on rebalance dates)
            if is_rebalance:
                daily_turnover_list.append(turnover)
            else:
                daily_turnover_list.append(0.0)
        
        # Convert to DataFrames
        equity_series = Series(equity_curve, index=pd.DatetimeIndex(dates_list))
        returns_series = Series(daily_returns_list, index=pd.DatetimeIndex(dates_list))
        exposure_series = Series(daily_exposure_list, index=pd.DatetimeIndex(dates_list))
        turnover_series = Series(daily_turnover_list, index=pd.DatetimeIndex(dates_list))
        
        # Positions and weights DataFrames
        positions_df = self._dict_to_dataframe(positions_dict)
        weights_df = self._dict_to_dataframe(weights_dict)
        
        # Calculate ticker-level quarterly returns
        ticker_quarterly_returns = self._compute_ticker_quarterly_returns(positions_df, rebalance_days)
        
        # Build result
        result = BacktestResult(
            config=self.config,
            start_date=self.prices.index[0],
            end_date=self.prices.index[-1],
            equity_curve=equity_series,
            daily_returns=returns_series,
            daily_exposure=exposure_series,
            daily_turnover=turnover_series,
            positions=positions_df,
            weights=weights_df,
            trades=trades_list,
            ticker_quarterly_returns=ticker_quarterly_returns
        )
        
        result.compute_summary_stats()
        logger.info(f"Backtest complete: total return {result.total_return:.2%}")
        
        return result
    
    def _compute_trades(
        self,
        current_positions: Dict[str, float],
        target_shares: Dict[str, float],
        date: pd.Timestamp,
        available_prices: Series
    ) -> List[Trade]:
        """
        Compute trades from current to target positions with costs and slippage.
        
        Args:
            current_positions: Current holdings {ticker -> shares}.
            target_shares: Target holdings {ticker -> shares}.
            date: Rebalance date.
            available_prices: Prices available on this date.
        
        Returns:
            List of Trade objects executed.
        """
        trades = []
        all_tickers = set(current_positions.keys()) | set(target_shares.keys())
        
        for ticker in all_tickers:
            current_shares = current_positions.get(ticker, 0.0)
            target = target_shares.get(ticker, 0.0)
            delta = target - current_shares
            
            if abs(delta) < 1e-6:
                continue  # No trade
            
            # Get execution price with slippage
            base_price = available_prices.get(ticker, np.nan)
            if pd.isna(base_price) or base_price <= 0:
                continue  # Skip if price unavailable
            
            # Slippage direction
            if delta > 0:
                # BUY: slippage increases price
                exec_price = base_price * (1 + self.config.slippage_bps / 10000)
                side = 'BUY'
            else:
                # SELL: slippage decreases price
                exec_price = base_price * (1 - self.config.slippage_bps / 10000)
                side = 'SELL'
            
            # Transaction cost
            gross_cost = abs(delta) * exec_price
            transaction_cost = gross_cost * (self.config.transaction_cost_bps / 10000)
            
            trade = Trade(
                date=date,
                ticker=ticker,
                side=side,
                shares=abs(delta),
                price=exec_price,
                gross_cost=gross_cost,
                transaction_cost=transaction_cost
            )
            trades.append(trade)
        
        return trades
    
    @staticmethod
    def _dict_to_dataframe(data_dict: Dict[pd.Timestamp, Dict[str, float]]) -> DataFrame:
        """Convert {date -> {ticker -> value}} to DataFrame."""
        if not data_dict:
            return DataFrame()
        
        df = DataFrame(data_dict).T
        df.index = pd.to_datetime(df.index)
        return df.fillna(0.0)
    
    def _compute_ticker_quarterly_returns(
        self,
        positions_df: DataFrame,
        rebalance_days: DatetimeIndex
    ) -> DataFrame:
        """
        Compute quarterly returns for each ticker that was held.
        
        For each ticker in each quarter, calculate:
        Return = (Price at Quarter End - Price at Quarter Start) / Price at Quarter Start
        
        Only includes tickers that were actually held during the quarter.
        
        Args:
            positions_df: DataFrame of positions (date x ticker)
            rebalance_days: Quarter-end rebalance dates
            
        Returns:
            DataFrame with quarters as rows, tickers as columns, values = quarterly returns
        """
        if len(rebalance_days) == 0:
            return DataFrame()
        
        quarterly_returns = {}
        
        # Process each quarter (from one rebalance to the next)
        for i in range(len(rebalance_days)):
            quarter_end = rebalance_days[i]
            
            # Get quarter start (previous rebalance or beginning of data)
            if i == 0:
                quarter_start = self.prices.index[0]
            else:
                quarter_start = rebalance_days[i - 1]
            
            # Get tickers held during this quarter
            # Look at positions on the quarter_end date
            if quarter_end in positions_df.index:
                held_tickers = positions_df.loc[quarter_end]
                held_tickers = held_tickers[held_tickers > 0].index.tolist()
            else:
                held_tickers = []
            
            # Calculate returns for each held ticker
            ticker_returns = {}
            for ticker in held_tickers:
                if ticker not in self.prices.columns:
                    continue
                
                # Get prices at quarter start and end
                ticker_prices = self.prices[ticker]
                
                # Find actual available prices closest to quarter boundaries
                start_price = None
                end_price = None
                
                # Get start price (first available on or after quarter_start)
                start_mask = ticker_prices.index >= quarter_start
                if start_mask.any():
                    start_price = ticker_prices[start_mask].dropna().iloc[0] if len(ticker_prices[start_mask].dropna()) > 0 else None
                
                # Get end price (last available on or before quarter_end)
                end_mask = ticker_prices.index <= quarter_end
                if end_mask.any():
                    end_price = ticker_prices[end_mask].dropna().iloc[-1] if len(ticker_prices[end_mask].dropna()) > 0 else None
                
                # Calculate return
                if start_price is not None and end_price is not None and start_price > 0:
                    ticker_return = (end_price - start_price) / start_price
                    ticker_returns[ticker] = ticker_return
            
            # Store this quarter's returns
            quarterly_returns[quarter_end] = ticker_returns
        
        # Convert to DataFrame
        result_df = DataFrame(quarterly_returns).T
        result_df.index.name = 'Quarter End'
        
        return result_df
