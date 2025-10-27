import pandas as pd
from pandas import DataFrame, DatetimeIndex


def infer_trading_days(df_prices: DataFrame) -> DatetimeIndex:
    """
    Infer trading days from a multi-ticker price table.
    
    Args:
        df_prices: DataFrame with DatetimeIndex and multi-level columns (ticker, price field)
                   or simple columns with dates as index.
    
    Returns:
        Sorted unique DatetimeIndex of trading days.
    """
    return pd.DatetimeIndex(sorted(df_prices.index.unique()))


def quarter_end_days(index: DatetimeIndex) -> DatetimeIndex:
    """
    Return business days that are the last business day of Q1/Q2/Q3/Q4.
    
    Args:
        index: DatetimeIndex to filter.
    
    Returns:
        DatetimeIndex of quarter-end business days present in the input index.
    """
    # Get the last business day of each quarter for each year in the index
    quarter_end_dates = pd.DatetimeIndex([])
    
    for year in index.year.unique():
        for quarter in range(1, 5):
            # Determine the last month of the quarter
            last_month = quarter * 3
            # Create a date for the last day of that month
            last_day_of_quarter = pd.Timestamp(year=year, month=last_month, day=1)
            # Move to the last day of the month
            last_day_of_quarter = last_day_of_quarter + pd.offsets.MonthEnd(0)
            # Move to the last business day of that month
            last_business_day = last_day_of_quarter
            while last_business_day not in index and last_business_day >= pd.Timestamp(year=year, month=last_month - 2, day=1):
                last_business_day = last_business_day - pd.DateOffset(days=1)
            
            if last_business_day in index:
                quarter_end_dates = quarter_end_dates.append(pd.DatetimeIndex([last_business_day]))
    
    return pd.DatetimeIndex(sorted(quarter_end_dates.unique()))


def align_to_quarter_ends(df: DataFrame) -> DataFrame:
    """
    Subset a DataFrame to rows on quarter-end days.
    
    Args:
        df: DataFrame with DatetimeIndex.
    
    Returns:
        DataFrame subset to quarter-end business days.
    """
    qe_days = quarter_end_days(df.index)
    return df.loc[df.index.intersection(qe_days)]
