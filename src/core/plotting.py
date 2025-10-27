import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


def plot_equity_curve(equity_series: Series, title: str = "Portfolio Equity Curve") -> go.Figure:
    """
    Plot portfolio equity curve over time (visualizes CAGR).
    
    Args:
        equity_series: Series indexed by date, values=portfolio equity.
        title: Plot title (default "Portfolio Equity Curve").
    
    Returns:
        Plotly Figure.
    
    Raises:
        ValueError: If equity_series is empty or not a Series.
    """
    if not isinstance(equity_series, Series) or len(equity_series) == 0:
        raise ValueError("equity_series must be a non-empty Series")
    
    if not isinstance(equity_series.index, pd.DatetimeIndex):
        raise ValueError("equity_series index must be DatetimeIndex")
    
    # Compute cumulative return
    initial_value = equity_series.iloc[0]
    cumulative_return = (equity_series / initial_value - 1) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_series.index,
        y=cumulative_return.values,
        mode='lines',
        name='Equity Curve',
        line=dict(width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True
    )
    
    logger.info(f"Created equity curve plot: {len(equity_series)} data points")
    return fig


def plot_drawdown(equity_series: Series, title: str = "Drawdown Over Time") -> go.Figure:
    """
    Plot portfolio drawdown (%) over time (visualizes Maximum Drawdown).
    
    Drawdown is computed as: (current value - peak) / peak * 100
    
    Args:
        equity_series: Series indexed by date, values=portfolio equity.
        title: Plot title (default "Drawdown Over Time").
    
    Returns:
        Plotly Figure with drawdown as line chart.
    
    Raises:
        ValueError: If equity_series is empty or not a Series.
    """
    if not isinstance(equity_series, Series) or len(equity_series) == 0:
        raise ValueError("equity_series must be a non-empty Series")
    
    if not isinstance(equity_series.index, pd.DatetimeIndex):
        raise ValueError("equity_series index must be DatetimeIndex")
    
    # Compute running maximum
    running_max = equity_series.expanding().max()
    
    # Compute drawdown %
    drawdown = (equity_series - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_series.index,
        y=drawdown.values,
        fill='tozeroy',
        name='Drawdown',
        line=dict(width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Compute and display max drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    fig.add_hline(
        y=max_dd,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max DD: {max_dd:.2f}%",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True
    )
    
    logger.info(f"Created drawdown plot: max DD {max_dd:.2f}% on {max_dd_date.date()}")
    return fig


def plot_rolling_sharpe(
    returns_series: Series,
    window: int = 63,
    title: str = "Rolling Sharpe Ratio"
) -> go.Figure:
    """
    Plot rolling Sharpe ratio over time (visualizes risk-adjusted return).
    
    Rolling Sharpe is computed as: annualized excess return / annualized volatility
    over a rolling window.
    
    Args:
        returns_series: Series indexed by date, values=daily returns (as decimals).
        window: Rolling window in trading days (default 63 = ~3 months).
        title: Plot title.
    
    Returns:
        Plotly Figure.
    
    Raises:
        ValueError: If returns_series is empty or window is invalid.
    """
    if not isinstance(returns_series, Series) or len(returns_series) == 0:
        raise ValueError("returns_series must be a non-empty Series")
    
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        raise ValueError("returns_series index must be DatetimeIndex")
    
    if window <= 0 or window > len(returns_series):
        raise ValueError(f"window must be > 0 and <= {len(returns_series)}")
    
    # Compute rolling Sharpe ratio
    rolling_mean = returns_series.rolling(window).mean()
    rolling_std = returns_series.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)  # Annualize
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=returns_series.index,
        y=rolling_sharpe.values,
        mode='lines',
        name=f'Sharpe (rolling {window}d)',
        line=dict(width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Sharpe: %{y:.2f}<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Sharpe = 0",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True
    )
    
    logger.info(f"Created rolling Sharpe plot: window={window} days")
    return fig


def plot_quarterly_hit_ratio(
    quarterly_returns: Series,
    title: str = "Quarterly Hit Ratio"
) -> go.Figure:
    """
    Plot quarterly hit ratio as a bar chart.
    
    Hit ratio = % of quarters with positive returns.
    
    Args:
        quarterly_returns: Series indexed by quarter-end dates, values=quarterly returns.
        title: Plot title.
    
    Returns:
        Plotly Figure with bar chart.
    
    Raises:
        ValueError: If quarterly_returns is empty.
    """
    if not isinstance(quarterly_returns, Series) or len(quarterly_returns) == 0:
        raise ValueError("quarterly_returns must be a non-empty Series")
    
    if not isinstance(quarterly_returns.index, pd.DatetimeIndex):
        raise ValueError("quarterly_returns index must be DatetimeIndex")
    
    # Compute hit ratio
    n_positive = (quarterly_returns > 0).sum()
    n_total = len(quarterly_returns)
    hit_ratio = (n_positive / n_total) * 100
    
    # Create bar chart with colors
    colors = ['green' if ret > 0 else 'red' for ret in quarterly_returns.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=quarterly_returns.index,
        y=quarterly_returns.values * 100,  # Convert to %
        name='Quarterly Return',
        marker_color=colors,
        hovertemplate='<b>%{x|%Y-Q%q}</b><br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Add hit ratio annotation
    fig.add_annotation(
        text=f"Hit Ratio: {hit_ratio:.1f}% ({n_positive}/{n_total} quarters positive)",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.08,
        showarrow=False,
        font=dict(size=12),
        xanchor="center"
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Quarter",
        yaxis_title="Return (%)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True,
        xaxis_tickformat="%Y-Q%q"
    )
    
    logger.info(f"Created hit ratio plot: {hit_ratio:.1f}% ({n_positive}/{n_total})")
    return fig


def plot_metrics_dashboard(
    equity_series: Series,
    returns_series: Series,
    quarterly_returns: Series,
    title: str = "Backtest Performance Dashboard"
) -> go.Figure:
    """
    Create a 2x2 dashboard with all four key plots.
    
    Args:
        equity_series: Series of portfolio equity over time.
        returns_series: Series of daily returns.
        quarterly_returns: Series of quarterly returns.
        title: Dashboard title.
    
    Returns:
        Plotly Figure with 2x2 subplots.
    
    Raises:
        ValueError: If any input series is invalid.
    """
    # Validate inputs
    if not isinstance(equity_series, Series) or len(equity_series) == 0:
        raise ValueError("equity_series must be a non-empty Series")
    if not isinstance(returns_series, Series) or len(returns_series) == 0:
        raise ValueError("returns_series must be a non-empty Series")
    if not isinstance(quarterly_returns, Series) or len(quarterly_returns) == 0:
        raise ValueError("quarterly_returns must be a non-empty Series")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Equity Curve (CAGR)",
            "Rolling Sharpe Ratio",
            "Drawdown",
            "Quarterly Returns"
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )
    
    # 1. Equity curve
    initial_value = equity_series.iloc[0]
    cumulative_return = (equity_series / initial_value - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=equity_series.index,
            y=cumulative_return.values,
            mode='lines',
            name='Equity Curve',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Rolling Sharpe
    rolling_mean = returns_series.rolling(63).mean()
    rolling_std = returns_series.rolling(63).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    fig.add_trace(
        go.Scatter(
            x=returns_series.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Sharpe (63d)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Drawdown
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    fig.add_trace(
        go.Scatter(
            x=equity_series.index,
            y=drawdown.values,
            fill='tozeroy',
            name='Drawdown',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Quarterly hit ratio bars
    colors = ['green' if ret > 0 else 'red' for ret in quarterly_returns.values]
    fig.add_trace(
        go.Bar(
            x=quarterly_returns.index,
            y=quarterly_returns.values * 100,
            name='Q Return',
            marker_color=colors,
            hovertemplate='<b>%{x|%Y-Q%q}</b><br>%{y:.2f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    fig.update_xaxes(title_text="Quarter", row=2, col=2)
    fig.update_yaxes(title_text="Return (%)", row=2, col=2)
    
    fig.update_layout(
        title_text=title,
        template="plotly_white",
        height=900,
        showlegend=True,
        hovermode="x unified"
    )
    
    logger.info("Created metrics dashboard with 2x2 subplots")
    return fig
