"""
Utility Helper Functions
Supports data resampling and frequency conversion without index drift.
"""

from __future__ import annotations

from typing import Literal
import pandas as pd
import numpy as np


def resample_to_freq(
    df: pd.DataFrame,
    freq: Literal['D', 'W', 'M', 'Q', 'Y'],
    agg: Literal['last', 'mean', 'first'] = 'last'
) -> pd.DataFrame:
    """
    Resample MultiIndex DataFrame to specified frequency without drift.
    
    Handles MultiIndex (date, ticker) data by resampling dates while
    preserving ticker grouping. This ensures date indices remain on
    actual trading days (for 'last' aggregation).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with MultiIndex (date, ticker).
        Date level must be DatetimeIndex, ticker can be any hashable type.
    freq : {'D', 'W', 'M', 'Q', 'Y'}
        Target frequency.
        - 'D': Daily (no resampling)
        - 'W': Weekly (last trading day of week)
        - 'M': Monthly (last trading day of month)
        - 'Q': Quarterly (last trading day of quarter)
        - 'Y': Yearly (last trading day of year)
    agg : {'last', 'mean', 'first'}, default='last'
        Aggregation function.
        - 'last': Take last value in period
        - 'mean': Take mean value in period
        - 'first': Take first value in period
        
    Returns
    -------
    resampled : pd.DataFrame
        Resampled DataFrame with MultiIndex (date, ticker).
        Date index contains only actual dates from original data.
        Shape: (n_new_dates * n_tickers, n_features)
        
    Raises
    ------
    ValueError
        If df doesn't have MultiIndex (date, ticker) or freq is invalid.
        
    Examples
    --------
    >>> # Daily data → Weekly (last trading day per week)
    >>> daily_df = pd.DataFrame(
    ...     {'factor': [0.1, 0.2, 0.15]},
    ...     index=pd.MultiIndex.from_product(
    ...         [pd.date_range('2020-01-01', periods=3), ['A', 'B']],
    ...         names=['date', 'ticker']
    ...     )
    ... )
    >>> weekly = resample_to_freq(daily_df, freq='W')
    
    Notes
    -----
    This function prevents index drift by:
    1. Grouping by ticker
    2. Resampling each ticker's time series separately
    3. Taking the specified aggregation (default: last actual value in period)
    4. Reconstructing MultiIndex from results
    
    For 'last' aggregation, dates are guaranteed to be actual trading days
    from the original index (no artificial dates created).
    """
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['date', 'ticker']:
        raise ValueError("df must have MultiIndex with names ['date', 'ticker']")
    
    if freq == 'D':
        # No resampling needed
        return df.copy()
    
    # Map frequency to pandas rule (pandas 3.0+ uses new offset aliases)
    freq_map = {'W': 'W', 'M': 'ME', 'Q': 'QE', 'Y': 'YE'}
    if freq not in freq_map:
        raise ValueError(f"freq must be 'D', 'W', 'M', 'Q', or 'Y', got {freq}")
    
    pandas_freq = freq_map[freq]
    
    # Resample each ticker separately
    def resample_ticker(ticker_df):
        """Resample single ticker's time series."""
        if agg == 'last':
            # For 'last', resample to get last actual date in each period
            return ticker_df.resample(pandas_freq).last()
        elif agg == 'mean':
            return ticker_df.resample(pandas_freq).mean()
        elif agg == 'first':
            return ticker_df.resample(pandas_freq).first()
        else:
            raise ValueError(f"agg must be 'last', 'mean', or 'first', got {agg}")
    
    # Group by ticker and resample
    resampled_list = []
    for ticker, ticker_df in df.groupby(level='ticker'):
        # Remove ticker level for resampling
        ticker_df_ts = ticker_df.droplevel('ticker')
        ticker_df_ts.index.name = 'date'
        
        # Resample
        resampled_ts = resample_ticker(ticker_df_ts)
        
        if agg == 'last':
            # For 'last' aggregation with non-daily frequencies, map resampled dates
            # back to the actual last trading dates in each period
            # Group original data by period to find actual last dates
            period_groups = ticker_df_ts.groupby(pd.Grouper(freq=pandas_freq))
            actual_last_dates = []
            actual_last_rows = []
            for period_end, group in period_groups:
                if len(group) > 0:
                    # Get the last date and row in this period
                    last_idx = group.index[-1]
                    actual_last_dates.append(last_idx)
                    actual_last_rows.append(group.iloc[-1])
            
            if actual_last_dates:
                # Use actual dates from original data instead of resampled dates
                resampled_ts = pd.DataFrame(actual_last_rows, index=actual_last_dates)
        
        # Create MultiIndex properly: for each date, add ticker level
        multi_idx = pd.MultiIndex.from_arrays(
            [resampled_ts.index, [ticker] * len(resampled_ts)],
            names=['date', 'ticker']
        )
        resampled_ts.index = multi_idx
        
        resampled_list.append(resampled_ts)
    
    # Combine all tickers
    result = pd.concat(resampled_list, axis=0).sort_index()
    return result


def shift_labels_by_freq(
    labels: pd.Series | pd.DataFrame,
    freq: Literal['D', 'W', 'M', 'Q', 'Y'],
    shift_periods: int = 1
) -> pd.Series:
    """
    Shift labels backward in time to align with features.
    
    When using forward-looking labels (e.g., T+1 returns), this function
    shifts them back by shift_periods to align with factors at time T.
    This prevents look-ahead bias.
    
    Parameters
    ----------
    labels : pd.Series or pd.DataFrame
        Labels indexed by date. Can be MultiIndex (date, ticker) or
        single level.
    freq : {'D', 'W', 'M', 'Q', 'Y'}
        Frequency of shift.
    shift_periods : int, default=1
        Number of periods to shift backward.
        Positive value = shift backward (move to earlier dates).
        
    Returns
    -------
    shifted : pd.Series
        Shifted labels with same index as input.
        First shift_periods rows will be NaN.
        
    Examples
    --------
    >>> # T+1 returns → shift to T (align with T's factors)
    >>> labels = pd.Series([0.01, 0.02, -0.01], 
    ...                    index=pd.date_range('2020-01-02', periods=3))
    >>> shifted = shift_labels_by_freq(labels, freq='D', shift_periods=1)
    >>> # shifted index: 2020-01-01, 2020-01-02, 2020-01-03
    >>> # shifted values: NaN, 0.01, 0.02
    
    Notes
    -----
    The shift is performed per ticker if MultiIndex is present,
    ensuring alignment within each ticker.
    """
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    
    # Handle MultiIndex case
    if isinstance(labels.index, pd.MultiIndex):
        shifted_list = []
        for ticker, ticker_series in labels.groupby(level='ticker'):
            # Shift within ticker
            ticker_series_shifted = ticker_series.droplevel('ticker').shift(shift_periods)
            ticker_series_shifted = pd.Series(
                ticker_series_shifted.values,
                index=pd.MultiIndex.from_product(
                    [ticker_series_shifted.index, [ticker]],
                    names=['date', 'ticker']
                )
            )
            shifted_list.append(ticker_series_shifted)
        
        return pd.concat(shifted_list, axis=0).sort_index()
    else:
        # Single index case
        return labels.shift(shift_periods)


def align_datasets(
    X: pd.DataFrame,
    y: pd.Series,
    dropna: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align feature matrix and labels by index.
    
    Ensures X and y have matching index and removes misaligned rows.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels.
    dropna : bool, default=True
        If True, remove rows with any NaN values.
        
    Returns
    -------
    X_aligned : pd.DataFrame
        Aligned features.
    y_aligned : pd.Series
        Aligned labels.
        
    Examples
    --------
    >>> X_aligned, y_aligned = align_datasets(X, y)
    """
    # Find common index
    common_idx = X.index.intersection(y.index)
    
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    if dropna:
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_aligned = X_aligned[mask]
        y_aligned = y_aligned[mask]
    
    return X_aligned, y_aligned


def compute_ic(
    predictions: pd.Series,
    actuals: pd.Series,
    method: Literal['spearman', 'pearson'] = 'spearman'
) -> float:
    """
    Compute Information Coefficient (IC).
    
    IC measures rank correlation between predictions and realized returns.
    Commonly used to evaluate factor quality.
    
    Parameters
    ----------
    predictions : pd.Series
        Model predictions or factor scores.
    actuals : pd.Series
        Realized returns or actual values.
    method : {'spearman', 'pearson'}, default='spearman'
        Correlation method. Spearman (rank) is more robust to outliers.
        
    Returns
    -------
    ic : float
        Information Coefficient in range [-1, 1].
        Positive = predictive signal, Negative = reverse signal.
        Close to 0 = no predictive power.
        
    Examples
    --------
    >>> ic = compute_ic(model_scores, realized_returns)
    >>> print(f"IC: {ic:.4f}")
    """
    # Remove NaN and align
    valid_mask = ~(predictions.isna() | actuals.isna())
    pred_clean = predictions[valid_mask]
    actual_clean = actuals[valid_mask]
    
    if len(pred_clean) < 2:
        return np.nan
    
    if method == 'spearman':
        return pred_clean.rank().corr(actual_clean.rank())
    elif method == 'pearson':
        return pred_clean.corr(actual_clean)
    else:
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method}")


def compute_turnover(
    weights_current: pd.Series,
    weights_previous: pd.Series
) -> float:
    """
    Compute portfolio turnover.
    
    Turnover = sum(|w_t - w_{t-1}|) / 2
    Ranges from 0 (no change) to 1 (complete portfolio replacement).
    
    Parameters
    ----------
    weights_current : pd.Series
        Current period weights indexed by ticker.
    weights_previous : pd.Series
        Previous period weights indexed by ticker.
        
    Returns
    -------
    turnover : float
        Portfolio turnover in range [0, 1].
        
    Examples
    --------
    >>> turnover = compute_turnover(weights_today, weights_yesterday)
    >>> print(f"Turnover: {turnover:.2%}")
    """
    # Align weights
    common_tickers = weights_current.index.intersection(weights_previous.index)
    
    if len(common_tickers) == 0:
        return 1.0  # Complete turnover if no overlap
    
    w_curr = weights_current.loc[common_tickers]
    w_prev = weights_previous.loc[common_tickers]
    
    turnover = np.abs(w_curr - w_prev).sum() / 2
    return float(turnover)
