import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import logging

from src.core.config import Config
from src.core.data_loader import to_price_matrix, validate_panel
from src.core.calendar import quarter_end_days
from src.core.signals import rank_by_momentum, top_n_and_threshold
from src.core.strategy import Decisions
from src.core.engine import BacktestEngine
from src.core.analytics import compute_performance_metrics
from src.core.plotting import (
    plot_equity_curve,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_quarterly_hit_ratio,
    plot_metrics_dashboard
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("test")
# Page config
st.set_page_config(
    page_title="Momentum Backtester",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quantitative Momentum Backtester")
st.markdown("Test your momentum strategy with real price data.")

# ============================================================================
# SIDEBAR: Input Controls
# ============================================================================

with st.sidebar:
    st.header("Configuration")
    
    # Data input
    st.subheader("1. Data Input")
    uploaded_file = st.file_uploader(
        "Upload CSV (date, ticker, open, high, low, close, adj_close, volume)",
        type=["csv"]
    )
    
    # Price column selector
    price_column_choice = st.selectbox(
        "Price Column to Use",
        options=["adj_close", "close"],
        index=0,
        help="Select which price column to use for backtest calculations",
        key="price_column_selector"
    )
    
    # Clear results if price column changed
    if "last_price_column" not in st.session_state:
        st.session_state.last_price_column = price_column_choice
    elif st.session_state.last_price_column != price_column_choice:
        # Price column changed - clear cached results
        if "result" in st.session_state:
            del st.session_state.result
        if "config" in st.session_state:
            del st.session_state.config
        st.session_state.last_price_column = price_column_choice
        st.info(f"🔄 Price column changed to **{price_column_choice}** - please re-run backtest")
    
    # Strategy parameters
    st.subheader("2. Strategy Parameters")
    
    universe_size = st.number_input(
        "Universe Size",
        value=120,
        min_value=1,
        help="Total number of tickers to consider"
    )
    
    entry_count = st.number_input(
        "Entry Count",
        value=24,
        min_value=1,
        max_value=universe_size,
        help="Number of top tickers to buy each quarter"
    )
    
    exit_rank_threshold = st.number_input(
        "Exit Rank Threshold",
        value=36,
        min_value=entry_count,
        help="Exit if rank falls below this"
    )
    
    lookback_months = st.number_input(
        "Lookback Period (months)",
        value=3,
        min_value=1,
        help="Quarters to look back for momentum"
    )
    
    holding_quarters = st.number_input(
        "Holding Period (quarters)",
        value=2,
        min_value=1,
        help="Maximum quarters to hold a position"
    )
    
    # Cost parameters
    st.subheader("3. Cost Parameters")
    
    transaction_cost_bps = st.number_input(
        "Transaction Cost (bps)",
        value=10,
        min_value=0,
        help="Transaction cost in basis points"
    )
    
    slippage_bps = st.number_input(
        "Slippage (bps)",
        value=5,
        min_value=0,
        help="Market impact/slippage in basis points"
    )
    
    # Initial capital
    st.subheader("4. Initial Capital")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        value=1_000_000,
        min_value=10_000,
        step=100_000,
        format="%d"
    )


# ============================================================================
# MAIN APP: Data Loading and Caching
# ============================================================================

@st.cache_data
def load_and_process_data(file_content, filename, price_column='adj_close'):
    """Load and process price data with flexible column handling.
    
    Args:
        file_content: CSV file content
        filename: Name of the uploaded file
        price_column: Which price column to use ('adj_close' or 'close')
    
    Returns:
        DataFrame with columns [date, ticker, price, volume]
        where 'price' is the selected price column
        
    Note: The cache key includes price_column, so changing it will reload the data.
    """
    # Add debug to confirm cache miss
    st.write(f"🔄 Loading data with price_column=**{price_column}**")
    
    try:
        # Read CSV - keep original types first to preserve numeric dates
        df = pd.read_csv(StringIO(file_content), dtype=str)  # Read everything as string first
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Rename columns to lowercase for easier matching
        df.columns = df.columns.str.lower().str.strip()
        
        # Show available columns for debugging
        with st.expander("📋 Column Detection", expanded=False):
            st.write(f"**Available columns:** {list(df.columns)}")
            st.write(f"**Column data types:** {dict(df.dtypes)}")
            
            # Show available price columns
            price_cols = [col for col in df.columns if 'close' in col.lower()]
            if price_cols:
                st.write(f"**Available price columns:** {price_cols}")
                st.info(f"Selected for backtest: **{price_column}**")
        
        # Try to find date column (case-insensitive, common variations)
        date_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ['date', 'datetime', 'timestamp', 'time']):
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"No date column found. Available: {list(df.columns)}")
        
        # Standardize column names
        df = df.rename(columns={
            date_col: 'date',
            'datetime': 'date',
            'timestamp': 'date', 
            'time': 'date'
        })
        
        # Parse dates - handle multiple formats
        # Show raw value first
        with st.expander("🔧 Date Parsing Debug", expanded=False):
            st.write(f"Raw date column (first 5): {df['date'].head().tolist()}")
        
        # Try multiple parsing strategies
        # Strategy 1: Direct datetime parsing (for ISO format like 2024-01-02)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
        
        # Strategy 2: If all NaN, try numeric conversion (Excel serial dates)
        if df['date'].isna().all():
            try:
                # Convert strings to float (numeric dates)
                numeric_dates = pd.to_numeric(df['date'], errors='coerce')
                
                with st.expander("🔧 Date Parsing Debug", expanded=False):
                    st.write(f"Numeric conversion: {numeric_dates.head().tolist()}")
                
                # Excel serial dates are between 1 and 60000
                if numeric_dates.notna().any():
                    sample = numeric_dates.dropna().iloc[0]
                    
                    if 1 < sample < 100000:
                        # Treat as Excel serial date
                        df['date'] = pd.to_datetime(numeric_dates, unit='D', origin='1899-12-30', errors='coerce')
                    elif sample > 1000000000:
                        # Treat as Unix timestamp
                        df['date'] = pd.to_datetime(numeric_dates, unit='s', errors='coerce')
            except:
                pass
        
        # Strategy 3: If still NaN, try other common formats
        if df['date'].isna().all():
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%m-%d-%Y', '%Y-%d-%m']:
                df['date'] = pd.to_datetime(df['date'].astype(str), format=fmt, errors='coerce')
                if df['date'].notna().any():
                    break
        
        if df['date'].isna().all():
            raise ValueError(f"Could not parse dates. Sample values: {df['date'].head().tolist()}")
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Find ticker column
        ticker_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ['ticker', 'symbol', 'code', 'name']):
                ticker_col = col
                break
        
        if ticker_col is None:
            raise ValueError(f"No ticker column found. Available: {list(df.columns)}")
        
        df = df.rename(columns={ticker_col: 'ticker'})
        df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
        
        # Find close price column (prefer 'close' not 'adj close')
        close_col = None
        for col in df.columns:
            col_lower = col.lower()
            # Match 'close' but not part of 'adj close'
            if col_lower == 'close' or (col_lower.endswith('close') and 'adj' not in col_lower):
                close_col = col
                break
        
        if close_col is None:
            raise ValueError(f"No close price column found. Available: {list(df.columns)}")
        
        df = df.rename(columns={close_col: 'close'})
        
        # Find adj_close if it exists
        adj_close_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'adj' in col_lower and 'close' in col_lower:
                adj_close_col = col
                break
        
        if adj_close_col:
            df = df.rename(columns={adj_close_col: 'adj_close'})
        else:
            df['adj_close'] = df['close']
        
        # Find volume if it exists
        volume_col = None
        for col in df.columns:
            if 'volume' in col.lower():
                volume_col = col
                break
        
        if volume_col:
            df = df.rename(columns={volume_col: 'volume'})
        else:
            df['volume'] = 0  # Default volume if not provided
        
        # Convert to numeric types
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        # Handle missing close prices
        df['close'] = df['close'].fillna(df['adj_close'])
        df['adj_close'] = df['adj_close'].fillna(df['close'])
        
        # Remove rows with NaN prices
        df = df.dropna(subset=['close', 'adj_close'])
        
        # Create a 'price' column based on user selection
        if price_column == 'adj_close':
            df['price'] = df['adj_close']
            st.info(f"✓ Using **Adjusted Close** prices for backtest")
        elif price_column == 'close':
            df['price'] = df['close']
            st.info(f"✓ Using **Close** prices for backtest")
        else:
            # Default to adj_close
            df['price'] = df['adj_close']
            st.warning(f"Unknown price column '{price_column}', defaulting to Adjusted Close")
        
        # Select and reorder required columns
        df = df[['date', 'ticker', 'price', 'volume']].copy()
        
        # Sort and validate
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No valid price data found after processing")
        
        return df
    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}")


def run_backtest(df_long, config, initial_capital):
    """Run the backtest."""
    try:
        st.write(f"**Engine initialization:** initial_capital = ${initial_capital:,.2f}")
        
        # Debug: Show what price column we're using
        st.write(f"**Price data info:**")
        st.write(f"- First ticker: {df_long['ticker'].iloc[0]}")
        st.write(f"- First date: {df_long['date'].iloc[0]}")
        st.write(f"- First price value: ${df_long['price'].iloc[0]:.2f}")
        st.write(f"- Mean price across all data: ${df_long['price'].mean():.2f}")
        
        # Convert to price matrix
        prices, volumes = to_price_matrix(df_long)
        
        # Debug: Show price matrix info
        st.write(f"**Price matrix info:**")
        st.write(f"- Shape: {prices.shape}")
        st.write(f"- First ticker mean price: ${prices.iloc[:, 0].mean():.2f}")
        
        # Validate
        validate_panel(prices, min_history_days=config.min_history_days)
        
        # Get quarter-end dates
        qe_days = quarter_end_days(prices.index)
        
        if len(qe_days) < 1:
            raise ValueError("Insufficient data for at least 1 quarter-end rebalance")
        
        # Momentum strategy with ranking
        def strategy_callable(date, current_positions, current_equity):
            """Momentum-based strategy: rank by returns and select top performers."""
            from src.core.strategy import Decisions
            
            try:
                # Get available prices on this date
                if date not in prices.index:
                    return Decisions(
                        date=date,
                        universe=set(),
                        buys=set(),
                        sells=set(),
                        target_weights={},
                        cohort_ages={}
                    )
                
                available_prices = prices.loc[date]
                universe = set(available_prices.index[available_prices.notna()])
                
                if len(universe) < config.entry_count:
                    st.warning(f"⚠️ Not enough tickers ({len(universe)}) to meet entry_count ({config.entry_count})")
                    return Decisions(
                        date=date,
                        universe=universe,
                        buys=set(),
                        sells=set(),
                        target_weights={},
                        cohort_ages={}
                    )
                
                # Get lookback period start date (in months)
                lookback_start = date - pd.DateOffset(months=config.lookback_months)
                
                # Get historical prices for momentum calculation
                mask = (prices.index >= lookback_start) & (prices.index <= date)
                lookback_prices = prices.loc[mask]
                
                if len(lookback_prices) < 2:
                    st.warning(f"⚠️ Insufficient history at {date} (only {len(lookback_prices)} days available)")
                    return Decisions(
                        date=date,
                        universe=universe,
                        buys=set(),
                        sells=set(),
                        target_weights={},
                        cohort_ages={}
                    )
                
                # Calculate returns from lookback start to current date
                # Get first and last prices over the lookback window
                returns_dict = {}
                for ticker in universe:
                    ticker_data = lookback_prices[ticker].dropna()
                    if len(ticker_data) >= 2:
                        start_price = ticker_data.iloc[0]
                        end_price = ticker_data.iloc[-1]
                        if start_price > 0:
                            ret = (end_price - start_price) / start_price
                            returns_dict[ticker] = ret
                
                if len(returns_dict) == 0:
                    st.warning(f"⚠️ No valid returns calculated at {date}")
                    return Decisions(
                        date=date,
                        universe=universe,
                        buys=set(),
                        sells=set(),
                        target_weights={},
                        cohort_ages={}
                    )
                
                # Rank by momentum (higher return = better = lower rank number)
                sorted_by_return = sorted(returns_dict.items(), key=lambda x: x[1], reverse=True)
                entry_tickers = set([ticker for ticker, ret in sorted_by_return[:config.entry_count]])
                entry_tickers = entry_tickers.intersection(universe)  # Ensure available
                
                # Equal weight on selected tickers
                if len(entry_tickers) > 0:
                    weight = 1.0 / len(entry_tickers)
                    target_weights = {ticker: weight for ticker in entry_tickers}
                else:
                    target_weights = {}
                
                return Decisions(
                    date=date,
                    universe=universe,
                    buys=entry_tickers,
                    sells=set(),
                    target_weights=target_weights,
                    cohort_ages={ticker: 0 for ticker in entry_tickers}
                )
            except Exception as e:
                st.error(f"Strategy error on {date}: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return Decisions(
                    date=date,
                    universe=set(),
                    buys=set(),
                    sells=set(),
                    target_weights={},
                    cohort_ages={}
                )
        
        # Run engine
        engine = BacktestEngine(
            prices=prices,
            config=config,
            strategy_callable=strategy_callable,
            volumes=volumes,
            initial_equity=initial_capital
        )
        
        result = engine.run(qe_days)
        
        # Debug: Check if we got any trades
        st.write(f"**Debug Info:**")
        st.write(f"- Quarter-end dates found: {len(qe_days)}")
        st.write(f"- Total trades executed: {len(result.trades)}")
        st.write(f"- Final equity: ${result.equity_curve.iloc[-1]:,.2f}")
        st.write(f"- Starting equity: ${result.equity_curve.iloc[0]:,.2f}")
        
        if len(result.trades) == 0:
            st.warning("⚠️ No trades executed! Check your data or strategy parameters.")
        
        return result
    
    except Exception as e:
        raise RuntimeError(f"Backtest failed: {str(e)}")


# ============================================================================
# MAIN UI
# ============================================================================

if uploaded_file is not None:
    # Load data
    with st.spinner("Loading and processing data..."):
        try:
            file_content = uploaded_file.read().decode("utf-8")
            df_long = load_and_process_data(file_content, uploaded_file.name, price_column=price_column_choice)
            
            st.success(f"✓ Loaded {len(df_long)} rows, {df_long['ticker'].nunique()} tickers")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df_long.head(20), use_container_width=True)
                st.write(f"Date range: {df_long['date'].min().date()} to {df_long['date'].max().date()}")
                st.write(f"Tickers: {', '.join(sorted(df_long['ticker'].unique()))}")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("📋 Expected CSV columns (case-insensitive):")
            st.code("date, ticker, close, [adj_close], [volume]")
            st.info("Or use alternative names:")
            st.code("datetime/timestamp (for date), symbol/code (for ticker), close price column")
            st.stop()
    
    # Run backtest button
    if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
        
        # Create config
        try:
            config = Config(
                universe_size=int(universe_size),
                entry_count=int(entry_count),
                exit_rank_threshold=int(exit_rank_threshold),
                lookback_months=int(lookback_months),
                holding_quarters=int(holding_quarters),
                transaction_cost_bps=float(transaction_cost_bps),
                slippage_bps=float(slippage_bps),
                min_history_days=63
            )
        except Exception as e:
            st.error(f"❌ Invalid configuration: {str(e)}")
            st.stop()
        
        # Run backtest
        with st.spinner("Running backtest..."):
            try:
                st.write(f"**Backtest Parameters:**")
                st.write(f"- Initial Capital: ${initial_capital:,.2f}")
                st.write(f"- Universe Size: {universe_size}")
                st.write(f"- Entry Count: {entry_count}")
                st.write(f"- Data points: {len(df_long)}")
                
                result = run_backtest(df_long, config, initial_capital)
                st.session_state.result = result
                st.session_state.config = config
                st.success("✓ Backtest completed successfully!")
            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                st.stop()
    
    # Display results if available
    if "result" in st.session_state:
        result = st.session_state.result
        config = st.session_state.config
        
        # ====================================================================
        # RESULTS SECTION
        # ====================================================================
        
        st.divider()
        st.header("📊 Results")
        
        # Configuration summary
        with st.expander("⚙️ Configuration Summary", expanded=False):
            config_dict = config.summary()
            cols = st.columns(2)
            for i, (key, value) in enumerate(config_dict.items()):
                with cols[i % 2]:
                    st.metric(label=key.replace("_", " ").title(), value=value)
        
        # Key metrics
        st.subheader("📈 Key Performance Metrics")
        
        # Debug: Daily returns analysis
        st.write("**Daily Returns Debug:**")
        st.write(f"- Total trading days: {len(result.daily_returns)}")
        st.write(f"- Date range: {result.daily_returns.index.min()} to {result.daily_returns.index.max()}")
        st.write(f"- Days with non-zero returns: {(result.daily_returns.abs() > 1e-10).sum()}")
        st.write(f"- Days with NaN returns: {result.daily_returns.isna().sum()}")
        st.write(f"- Days with zero returns: {(result.daily_returns.abs() <= 1e-10).sum()}")
        st.write(f"- Mean daily return: {result.daily_returns.mean():.6f}")
        st.write(f"- Std daily return: {result.daily_returns.std():.6f}")
        
        with st.expander("📋 First 20 Daily Returns", expanded=False):
            first_returns = pd.DataFrame({
                'Date': result.daily_returns.index[:20].strftime('%Y-%m-%d'),
                'Return': (result.daily_returns.iloc[:20] * 100).round(4)
            })
            st.dataframe(first_returns, use_container_width=True)
        
        with st.expander("📋 Last 20 Daily Returns", expanded=False):
            last_returns = pd.DataFrame({
                'Date': result.daily_returns.index[-20:].strftime('%Y-%m-%d'),
                'Return': (result.daily_returns.iloc[-20:] * 100).round(4)
            })
            st.dataframe(last_returns, use_container_width=True)
        
        metrics = compute_performance_metrics(result.daily_returns)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CAGR",
                f"{metrics['cagr']:.2%}",
                delta=None,
                help="Compound Annual Growth Rate"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                delta=None,
                help="Risk-adjusted return"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.2%}",
                delta=None,
                help="Maximum peak-to-trough decline"
            )
        
        with col4:
            st.metric(
                "Hit Ratio",
                f"{metrics['hit_ratio']:.1%}",
                delta=None,
                help="% of quarters with positive returns"
            )
        
        # Equity curve and other plots
        st.subheader("📉 Performance Charts")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Equity Curve",
            "Drawdown",
            "Rolling Sharpe",
            "Quarterly Returns",
            "Dashboard"
        ])
        
        with tab1:
            try:
                fig = plot_equity_curve(result.equity_curve)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting equity curve: {str(e)}")
        
        with tab2:
            try:
                fig = plot_drawdown(result.equity_curve)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting drawdown: {str(e)}")
        
        with tab3:
            try:
                fig = plot_rolling_sharpe(result.daily_returns)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting rolling Sharpe: {str(e)}")
        
        with tab4:
            try:
                # Calculate quarterly returns with NaN handling
                returns_filled = result.daily_returns.fillna(0.0)
                quarterly_returns = (1 + returns_filled).resample('QE').prod() - 1
                
                # Debug: Show quarterly stats
                st.write("**Quarterly Returns Breakdown:**")
                st.write(f"- Total quarters: {len(quarterly_returns)}")
                st.write(f"- Positive quarters: {(quarterly_returns > 0).sum()}")
                st.write(f"- Negative quarters: {(quarterly_returns < 0).sum()}")
                st.write(f"- Best quarter: {quarterly_returns.max():.2%}")
                st.write(f"- Worst quarter: {quarterly_returns.min():.2%}")
                
                fig = plot_quarterly_hit_ratio(quarterly_returns)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show quarterly returns table
                with st.expander("📊 View All Quarterly Returns", expanded=False):
                    quarterly_df = pd.DataFrame({
                        'Quarter End': quarterly_returns.index.strftime('%Y-%m-%d'),
                        'Return (%)': (quarterly_returns * 100).round(2)
                    })
                    st.dataframe(quarterly_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting quarterly returns: {str(e)}")
        
        with tab5:
            try:
                quarterly_returns = result.daily_returns.resample('QE').apply(
                    lambda x: (1 + x).prod() - 1
                )
                fig = plot_metrics_dashboard(
                    result.equity_curve,
                    result.daily_returns,
                    quarterly_returns
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting dashboard: {str(e)}")
        
        # Holdings and trades tables
        st.subheader("📋 Holdings & Trades")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Holdings (Final)**")
            if len(result.positions) > 0:
                final_positions = result.positions.iloc[-1].sort_values(ascending=False)
                holdings_df = pd.DataFrame({
                    "Ticker": final_positions.index,
                    "Shares": final_positions.values
                }).head(10)
                st.dataframe(holdings_df, use_container_width=True)
            else:
                st.info("No holdings in final period")
        
        with col2:
            st.write("**Recent Trades (Last 10)**")
            if len(result.trades) > 0:
                trades_df = pd.DataFrame([
                    {
                        "Date": t.date.date(),
                        "Ticker": t.ticker,
                        "Side": t.side,
                        "Shares": f"{t.shares:.0f}",
                        "Price": f"${t.price:.2f}",
                        "Cost": f"${t.transaction_cost:.2f}"
                    }
                    for t in result.trades[-10:]
                ])
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trades executed")
        
        # Download section
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Trades CSV
        if len(result.trades) > 0:
            trades_export = pd.DataFrame([
                {
                    "Date": t.date.isoformat(),
                    "Ticker": t.ticker,
                    "Side": t.side,
                    "Shares": t.shares,
                    "Price": t.price,
                    "Gross Cost": t.gross_cost,
                    "Transaction Cost": t.transaction_cost
                }
                for t in result.trades
            ])
            
            csv_trades = trades_export.to_csv(index=False)
            with col1:
                st.download_button(
                    label="📊 Download Trades",
                    data=csv_trades,
                    file_name="trades.csv",
                    mime="text/csv"
                )
        
        # Holdings CSV
        if len(result.positions) > 0:
            holdings_export = result.positions.copy()
            holdings_export.index = holdings_export.index.strftime('%Y-%m-%d')
            csv_holdings = holdings_export.to_csv()
            with col2:
                st.download_button(
                    label="📊 Download Holdings",
                    data=csv_holdings,
                    file_name="holdings.csv",
                    mime="text/csv"
                )
        
        # Daily returns CSV
        returns_export = result.daily_returns.copy()
        returns_export.index = returns_export.index.strftime('%Y-%m-%d')
        csv_returns = returns_export.to_csv()
        with col3:
            st.download_button(
                label="📊 Download Returns",
                data=csv_returns,
                file_name="daily_returns.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Upload a CSV file to get started")
    
    st.markdown("""
    ## Expected CSV Format
    
    ### Required Columns (case-insensitive, any order):
    - **Date**: `Date`, `datetime`, `timestamp`, or `time`
    - **Ticker**: `Ticker`, `symbol`, or `code`
    - **Close**: `Close` (closing price)
    
    ### Optional Columns (any order):
    - **High, Low, Open**: OHLC prices
    - **Volume**: Trading volume
    
    ### Column Order Doesn't Matter!
    Your CSV can have columns in **any order**. The app will find them automatically.
    
    ### Example Format (What We're Looking For):
    """)
    
    example_df = pd.DataFrame({
        'Date': ['2024-01-02', '2024-01-03', '2024-01-04'],
        'Close': [25.56478, 25.9558, 26.76032],
        'High': [25.97378, 25.9558, 26.79403],
        'Low': [25.5513, 25.64119, 26.14233],
        'Open': [25.61871, 25.69287, 26.22548],
        'Volume': ['1.67E+08', '1.67E+08', '2.37E+08'],
        'Ticker': ['AAPL', 'AAPL', 'AAPL']
    })
    st.dataframe(example_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Sample CSV (Your Format)",
            data=open("sample_data_formatted.csv").read(),
            file_name="sample_data.csv",
            mime="text/csv"
        )
    with col2:
        st.info("✅ This sample matches your data format exactly!")
