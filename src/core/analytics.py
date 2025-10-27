import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_cagr(returns: Series, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        returns: Series of daily returns (as decimals, e.g., 0.01 = 1%).
        periods_per_year: Trading periods per year (default 252 for daily data).
    
    Returns:
        CAGR as decimal (e.g., 0.10 = 10%).
    
    Raises:
        ValueError: If returns is empty or all returns are NaN.
    """
    if len(returns) == 0 or returns.isna().all():
        raise ValueError("Returns series is empty or all NaN")
    
    # Compute cumulative return
    cumulative_return = (1 + returns).prod() - 1
    
    # Compute number of years
    n_periods = len(returns)
    n_years = n_periods / periods_per_year
    
    if n_years <= 0:
        return 0.0
    
    # CAGR = (end value / start value) ^ (1 / years) - 1
    cagr = (1 + cumulative_return) ** (1 / n_years) - 1
    
    return cagr


def calculate_sharpe_ratio(
    returns: Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio (excess return per unit of volatility).
    
    Args:
        returns: Series of daily returns (as decimals).
        risk_free_rate: Annual risk-free rate (default 0.0).
        periods_per_year: Trading periods per year (default 252).
    
    Returns:
        Sharpe ratio (annualized).
    
    Raises:
        ValueError: If returns is empty or all NaN.
    """
    if len(returns) == 0 or returns.isna().all():
        raise ValueError("Returns series is empty or all NaN")
    
    # Daily risk-free rate
    daily_rf_rate = risk_free_rate / periods_per_year
    
    # Excess returns
    excess_returns = returns - daily_rf_rate
    
    # Annualized Sharpe
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    if std_excess <= 0:
        return 0.0
    
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_max_drawdown(returns: Series) -> float:
    """
    Calculate Maximum Drawdown.
    
    Maximum drawdown is the largest peak-to-trough decline during the period.
    
    Args:
        returns: Series of daily returns (as decimals).
    
    Returns:
        Maximum drawdown as decimal (e.g., -0.30 = 30% drawdown).
    
    Raises:
        ValueError: If returns is empty or all NaN.
    """
    if len(returns) == 0 or returns.isna().all():
        raise ValueError("Returns series is empty or all NaN")
    
    # Compute cumulative returns (wealth index)
    cumulative_returns = (1 + returns).cumprod()
    
    # Compute running maximum
    running_max = cumulative_returns.expanding().max()
    
    # Compute drawdown at each point
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Maximum drawdown is the most negative
    max_dd = drawdown.min()
    
    return max_dd


def calculate_quarterly_hit_ratio(returns: Series) -> float:
    """
    Calculate Quarterly Hit Ratio: % of quarters with positive returns.
    
    Args:
        returns: Series of daily returns (as decimals), indexed by date.
    
    Returns:
        Hit ratio as decimal (e.g., 0.75 = 75% of quarters positive).
    
    Raises:
        ValueError: If returns is empty or all NaN.
    """
    if len(returns) == 0 or returns.isna().all():
        raise ValueError("Returns series is empty or all NaN")
    
    # Resample returns to quarterly (end of quarter)
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns index must be DatetimeIndex for quarterly resampling")
    
    quarterly_returns = (1 + returns).resample('QE').prod() - 1
    
    if len(quarterly_returns) == 0:
        return 0.0
    
    # Count positive quarters
    n_positive = (quarterly_returns > 0).sum()
    n_total = len(quarterly_returns)
    
    hit_ratio = n_positive / n_total if n_total > 0 else 0.0
    
    return hit_ratio


def calculate_factor_exposures(
    returns: Series,
    factor_returns: Optional[Dict[str, Series]] = None
) -> Dict[str, float]:
    """
    Calculate factor exposures (beta, factor loading).
    
    Placeholder for factor analysis. If factor data not provided, returns empty dict.
    
    Args:
        returns: Series of strategy returns (daily, as decimals).
        factor_returns: Optional dict of factor names -> factor return series.
                       Must have same index as returns.
    
    Returns:
        Dict of factor name -> exposure (loading/beta).
        Empty dict if no factor data provided.
    """
    if factor_returns is None or len(factor_returns) == 0:
        logger.info("No factor data provided; returning empty factor exposures")
        return {}
    
    exposures = {}
    
    for factor_name, factor_ret in factor_returns.items():
        # Align indices
        aligned_returns, aligned_factors = returns.align(factor_ret, join='inner')
        
        if len(aligned_returns) < 2:
            continue
        
        # Simple beta calculation: cov(strategy, factor) / var(factor)
        covariance = aligned_returns.cov(aligned_factors)
        factor_variance = aligned_factors.var()
        
        if factor_variance > 0:
            beta = covariance / factor_variance
            exposures[factor_name] = beta
        else:
            exposures[factor_name] = 0.0
    
    return exposures


def compute_performance_metrics(
    returns: Series,
    risk_free_rate: float = 0.0,
    factor_returns: Optional[Dict[str, Series]] = None,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Compute all performance metrics at once.
    
    Args:
        returns: Series of daily returns (as decimals), indexed by date.
        risk_free_rate: Annual risk-free rate (default 0.0).
        factor_returns: Optional dict of factor return series.
        periods_per_year: Trading periods per year.
    
    Returns:
        Dict with keys: cagr, sharpe_ratio, max_drawdown, hit_ratio, plus factor exposures.
    
    Raises:
        ValueError: If returns is empty or all NaN.
    """
    if len(returns) == 0 or returns.isna().all():
        raise ValueError("Returns series is empty or all NaN")
    
    metrics = {
        'cagr': calculate_cagr(returns, periods_per_year),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),
        'hit_ratio': calculate_quarterly_hit_ratio(returns)
    }
    
    # Add factor exposures
    factor_exposures = calculate_factor_exposures(returns, factor_returns)
    for factor_name, exposure in factor_exposures.items():
        metrics[f'factor_beta_{factor_name}'] = exposure
    
    return metrics


def create_quarterly_performance_table(returns: Series) -> DataFrame:
    """
    Create a quarterly performance summary table.
    
    Args:
        returns: Series of daily returns (as decimals), indexed by date.
    
    Returns:
        DataFrame with columns:
        - date: Quarter end date
        - return: Quarterly return %
        - cum_return: Cumulative return from start %
    
    Raises:
        ValueError: If returns is empty or indexed incorrectly.
    """
    if len(returns) == 0:
        raise ValueError("Returns series is empty")
    
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns index must be DatetimeIndex")
    
    # Resample to quarterly
    quarterly_returns = (1 + returns).resample('QE').prod() - 1
    
    # Compute cumulative return from start
    cumulative = (1 + quarterly_returns).cumprod() - 1
    
    # Build table
    table = DataFrame({
        'date': quarterly_returns.index,
        'return': quarterly_returns.values * 100,  # Convert to %
        'cum_return': cumulative.values * 100
    })
    
    table['return'] = table['return'].round(2)
    table['cum_return'] = table['cum_return'].round(2)
    
    return table


def create_annual_performance_table(returns: Series) -> DataFrame:
    """
    Create an annual performance summary table.
    
    Args:
        returns: Series of daily returns (as decimals), indexed by date.
    
    Returns:
        DataFrame with columns:
        - year: Calendar year
        - return: Annual return %
        - cum_return: Cumulative return from start %
    
    Raises:
        ValueError: If returns is empty or indexed incorrectly.
    """
    if len(returns) == 0:
        raise ValueError("Returns series is empty")
    
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns index must be DatetimeIndex")
    
    # Resample to annual (year end)
    annual_returns = (1 + returns).resample('YE').prod() - 1
    
    # Compute cumulative
    cumulative = (1 + annual_returns).cumprod() - 1
    
    # Build table
    table = DataFrame({
        'year': annual_returns.index.year,
        'return': annual_returns.values * 100,
        'cum_return': cumulative.values * 100
    })
    
    table['return'] = table['return'].round(2)
    table['cum_return'] = table['cum_return'].round(2)
    
    return table


def create_summary_report(
    returns: Series,
    risk_free_rate: float = 0.0,
    factor_returns: Optional[Dict[str, Series]] = None
) -> Dict:
    """
    Create a comprehensive summary report.
    
    Args:
        returns: Series of daily returns.
        risk_free_rate: Annual risk-free rate.
        factor_returns: Optional factor return series.
    
    Returns:
        Dict with:
        - metrics: Performance metrics
        - quarterly_table: DataFrame of quarterly returns
        - annual_table: DataFrame of annual returns
    """
    metrics = compute_performance_metrics(returns, risk_free_rate, factor_returns)
    quarterly_table = create_quarterly_performance_table(returns)
    annual_table = create_annual_performance_table(returns)
    
    report = {
        'metrics': metrics,
        'quarterly_performance': quarterly_table,
        'annual_performance': annual_table
    }
    
    logger.info(
        f"Summary report: CAGR={metrics['cagr']:.2%}, "
        f"Sharpe={metrics['sharpe_ratio']:.2f}, "
        f"MaxDD={metrics['max_drawdown']:.2%}"
    )
    
    return report
