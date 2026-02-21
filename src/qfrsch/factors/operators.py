"""
Operators Library for Factor Computation
Provides primitives for time-series and cross-sectional operations.
All operators are vectorized using pandas groupby.transform().
"""

from __future__ import annotations

from typing import Union, Callable
import pandas as pd
import numpy as np


# ==================== Time-series Operators ====================
# These operators work within each ticker across time


def ts_mean(series: pd.Series, window: int, groupby_key: str = 'ticker') -> pd.Series:
    """
    Compute rolling mean within each ticker group.
    
    Parameters
    ----------
    series : pd.Series
        Input series with MultiIndex (date, ticker) or regular index.
        Should have 'ticker' information accessible via groupby_key.
    window : int
        Rolling window size (in periods/days)
    groupby_key : str, optional
        Key for grouping (default: 'ticker')
        
    Returns
    -------
    pd.Series
        Rolling mean values, same shape as input
        
    Notes
    -----
    Uses groupby().transform() for vectorized operation.
    Handles alignment automatically within each ticker group.
    
    Examples
    --------
    >>> ts_mean(df['close'], window=10)
    """
    if groupby_key not in series.index.names and groupby_key != 'ticker':
        raise ValueError(f"'{groupby_key}' not found in series index or context")
    
    # Handle both MultiIndex and single-level index
    if isinstance(series.index, pd.MultiIndex):
        ticker_idx = series.index.get_level_values('ticker')
        return series.groupby(ticker_idx).transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    else:
        # If no MultiIndex, try to infer from context (called within a function)
        # For simplicity, just apply rolling without grouping
        return series.rolling(window=window, min_periods=1).mean()


def ts_std(series: pd.Series, window: int, groupby_key: str = 'ticker') -> pd.Series:
    """
    Compute rolling standard deviation within each ticker group.
    
    Parameters
    ----------
    series : pd.Series
        Input series with MultiIndex (date, ticker) or regular index
    window : int
        Rolling window size
    groupby_key : str, optional
        Key for grouping (default: 'ticker')
        
    Returns
    -------
    pd.Series
        Rolling standard deviation values
        
    Notes
    -----
    Uses Bessel's correction (ddof=1) for unbiased estimator.
    """
    if isinstance(series.index, pd.MultiIndex):
        ticker_idx = series.index.get_level_values('ticker')
        return series.groupby(ticker_idx).transform(
            lambda x: x.rolling(window=window, min_periods=1).std(ddof=1)
        )
    else:
        return series.rolling(window=window, min_periods=1).std(ddof=1)


def ts_rank(series: pd.Series, window: int, groupby_key: str = 'ticker') -> pd.Series:
    """
    Compute rolling percentile rank (0-1) within each ticker group.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    window : int
        Rolling window size
    groupby_key : str, optional
        Key for grouping (default: 'ticker')
        
    Returns
    -------
    pd.Series
        Percentile rank within rolling window (0-1), where 1 is highest
        
    Notes
    -----
    Rank is normalized to [0, 1] range using: rank / (window + 1)
    Handles ties using 'average' method.
    """
    def _rolling_rank(x):
        return pd.Series(
            x.rolling(window=window, min_periods=1)
              .apply(lambda w: pd.Series(w).rank(method='average').iloc[-1] / (len(w) + 1))
              .values,
            index=x.index
        )
    
    if isinstance(series.index, pd.MultiIndex):
        ticker_idx = series.index.get_level_values('ticker')
        return series.groupby(ticker_idx).transform(_rolling_rank)
    else:
        return _rolling_rank(series)


def ts_delta(series: pd.Series, window: int, groupby_key: str = 'ticker') -> pd.Series:
    """
    Compute difference: x_t - x_{t-window}
    
    Parameters
    ----------
    series : pd.Series
        Input series
    window : int
        Number of periods back to compute difference
    groupby_key : str, optional
        Key for grouping (default: 'ticker')
        
    Returns
    -------
    pd.Series
        Difference series
        
    Examples
    --------
    >>> ts_delta(df['close'], window=1)  # Daily returns
    """
    if isinstance(series.index, pd.MultiIndex):
        ticker_idx = series.index.get_level_values('ticker')
        return series.groupby(ticker_idx).transform(
            lambda x: x - x.shift(window)
        )
    else:
        return series - series.shift(window)


def ts_delay(series: pd.Series, window: int, groupby_key: str = 'ticker') -> pd.Series:
    """
    Compute lagged values: x_{t-window}
    
    Parameters
    ----------
    series : pd.Series
        Input series
    window : int
        Number of periods to lag
    groupby_key : str, optional
        Key for grouping (default: 'ticker')
        
    Returns
    -------
    pd.Series
        Lagged series
        
    Examples
    --------
    >>> ts_delay(df['close'], window=1)  # Previous day's close
    """
    if isinstance(series.index, pd.MultiIndex):
        ticker_idx = series.index.get_level_values('ticker')
        return series.groupby(ticker_idx).transform(
            lambda x: x.shift(window)
        )
    else:
        return series.shift(window)


def ts_corr(s1: pd.Series, s2: pd.Series, window: int, 
            groupby_key: str = 'ticker') -> pd.Series:
    """
    Compute rolling correlation between two series within each ticker group.
    
    Parameters
    ----------
    s1 : pd.Series
        First series
    s2 : pd.Series
        Second series (must have same index as s1)
    window : int
        Rolling window size
    groupby_key : str, optional
        Key for grouping (default: 'ticker')
        
    Returns
    -------
    pd.Series
        Rolling correlation values (-1 to 1)
        
    Raises
    ------
    ValueError
        If s1 and s2 have different lengths/indices
        
    Notes
    -----
    Pearson correlation coefficient is used.
    """
    if len(s1) != len(s2):
        raise ValueError("s1 and s2 must have the same length")
    
    if isinstance(s1.index, pd.MultiIndex):
        ticker_idx = s1.index.get_level_values('ticker')
        
        def _compute_corr(group_indices):
            indices = group_indices
            corr_vals = []
            
            for i in indices:
                start_idx = max(0, i - window + 1)
                s1_window = s1.iloc[start_idx:i+1].values
                s2_window = s2.iloc[start_idx:i+1].values
                
                if len(s1_window) > 1 and not np.isnan(s1_window).all() and not np.isnan(s2_window).all():
                    try:
                        corr = np.corrcoef(s1_window, s2_window)[0, 1]
                        corr_vals.append(corr)
                    except:
                        corr_vals.append(np.nan)
                else:
                    corr_vals.append(np.nan)
            
            return corr_vals
        
        # Get unique tickers and compute correlation for each
        result_dict = {}
        for ticker in ticker_idx.unique():
            mask = ticker_idx == ticker
            indices = np.where(mask)[0]
            corr_vals = _compute_corr(indices)
            for idx, val in zip(indices, corr_vals):
                result_dict[idx] = val
        
        # Reorder by original index
        result = pd.Series([result_dict.get(i, np.nan) for i in range(len(s1))], 
                          index=s1.index)
        return result
    else:
        return s1.rolling(window=window, min_periods=1).corr(s2)


# ==================== Cross-sectional Operators ====================
# These operators work across all tickers on a given date


def cs_rank(series: pd.Series) -> pd.Series:
    """
    Compute cross-sectional percentile rank (0-1) on each date.
    
    Parameters
    ----------
    series : pd.Series
        Input series with MultiIndex (date, ticker) or with 'date' context
        
    Returns
    -------
    pd.Series
        Percentile rank across all tickers on each date (0-1),
        where 1 is the highest value
        
    Notes
    -----
    Uses 'average' method for handling ties.
    Each date gets independent ranking.
    
    Examples
    --------
    >>> cs_rank(result_series)  # Rank across all stocks each day
    """
    if isinstance(series.index, pd.MultiIndex):
        date_idx = series.index.get_level_values('date')
        # Rank within each date group, normalize to [0,1]
        rank_result = series.groupby(date_idx).rank(method='average')
        count_result = series.groupby(date_idx).transform('count')
        return (rank_result - 1) / count_result.replace(0, 1)
    else:
        # No date grouping context, return simple percentile rank
        n = len(series)
        ranks = series.rank(method='average')
        return (ranks - 1) / max(n - 1, 1)


# ==================== Mathematical Operators ====================
# Simple wrappers around numpy functions


def op_log(series: pd.Series, base: float = np.e) -> pd.Series:
    """
    Element-wise natural logarithm (or log to specified base).
    
    Parameters
    ----------
    series : pd.Series
        Input series
    base : float, optional
        Logarithm base (default: e for natural log)
        
    Returns
    -------
    pd.Series
        Logarithm values
        
    Examples
    --------
    >>> op_log(df['market_cap'])  # Log market cap
    >>> op_log(df['price'], base=10)  # Log base 10
    """
    if base == np.e:
        return np.log(series)
    else:
        return np.log(series) / np.log(base)


def op_abs(series: pd.Series) -> pd.Series:
    """
    Element-wise absolute value.
    
    Parameters
    ----------
    series : pd.Series
        Input series
        
    Returns
    -------
    pd.Series
        Absolute values
    """
    return np.abs(series)


def op_sign(series: pd.Series) -> pd.Series:
    """
    Element-wise sign function: returns -1, 0, or 1.
    
    Parameters
    ----------
    series : pd.Series
        Input series
        
    Returns
    -------
    pd.Series
        Sign values (-1, 0, 1)
    """
    return np.sign(series)


def op_sqrt(series: pd.Series) -> pd.Series:
    """
    Element-wise square root (non-negative values only).
    
    Parameters
    ----------
    series : pd.Series
        Input series (should be non-negative)
        
    Returns
    -------
    pd.Series
        Square root values
        
    Warnings
    --------
    Invalid values (negative numbers) will produce NaN.
    """
    return np.sqrt(series)


def op_exp(series: pd.Series) -> pd.Series:
    """
    Element-wise exponential function e^x.
    
    Parameters
    ----------
    series : pd.Series
        Input series
        
    Returns
    -------
    pd.Series
        Exponential values
    """
    return np.exp(series)


# ==================== Utility Functions ====================


def create_multiindex_series(df: pd.DataFrame, col: str, 
                             date_col: str = 'date', 
                             ticker_col: str = 'ticker') -> pd.Series:
    """
    Helper to create MultiIndex Series from DataFrame column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with date and ticker columns
    col : str
        Column name to extract as Series
    date_col : str, optional
        Name of date column (default: 'date')
    ticker_col : str, optional
        Name of ticker column (default: 'ticker')
        
    Returns
    -------
    pd.Series
        Series with MultiIndex (date, ticker)
        
    Examples
    --------
    >>> s = create_multiindex_series(df, 'close')
    >>> result = ts_mean(s, window=10)
    """
    return df.set_index([date_col, ticker_col])[col]


# ==================== Operator Composition Helper ====================


def nest_operators(base_series: pd.Series, 
                   operators: list[Callable]) -> pd.Series:
    """
    Apply a sequence of operators in composition (from left to right).
    
    Parameters
    ----------
    base_series : pd.Series
        Input series
    operators : list of callable
        List of operator functions to apply sequentially
        
    Returns
    -------
    pd.Series
        Result after applying all operators
        
    Examples
    --------
    >>> result = nest_operators(
    ...     df['close'],
    ...     [
    ...         lambda x: ts_mean(x, window=10),
    ...         cs_rank,
    ...         op_log
    ...     ]
    ... )
    """
    result = base_series
    for operator in operators:
        result = operator(result)
    return result
