"""
Post-Processing Module for Factors
Provides utilities for data cleaning, standardization, and risk neutralization.
"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple
import pandas as pd
import numpy as np


# ==================== Winsorization ====================


def winsorize(series: pd.Series, 
              method: str = 'quantile',
              limits: Tuple[float, float] = (0.01, 0.99),
              group_key: str = 'date') -> pd.Series:
    """
    Remove extreme values (outliers) from a series.
    
    Parameters
    ----------
    series : pd.Series
        Input series with possible MultiIndex (date, ticker)
    method : str, optional
        Winsorization method:
        - 'quantile': Clip at quantile limits (default)
        - 'std': Clip at k standard deviations from mean
    limits : tuple of float, optional
        For 'quantile': (lower_q, upper_q) in range [0,1]
        For 'std': (multi_lower, multi_upper) standard deviations
        Default: (0.01, 0.99) keeps 1st and 99th percentile
    group_key : str, optional
        Grouping key for independent winsorization
        - 'date': Apply per-date winsorization (cross-sectional)
        - 'ticker': Apply per-ticker winsorization (time-series)
        - None: Apply global winsorization
        
    Returns
    -------
    pd.Series
        Winsorized series
        
    Notes
    -----
    Typical approach in quantitative finance is cross-sectional
    winsorization (group_key='date'), treating extreme values
    relative to market conditions on each date.
    
    Examples
    --------
    >>> # Standard 1st/99th percentile cross-sectional winsorize
    >>> factor_clean = winsorize(factor_raw, method='quantile', 
    ...                           limits=(0.01, 0.99), group_key='date')
    
    >>> # Time-series winsorize: 3-sigma bounds per stock
    >>> factor_clean = winsorize(factor_raw, method='std', 
    ...                           limits=(3.0, 3.0), group_key='ticker')
    """
    
    if method == 'quantile':
        lower_q, upper_q = limits
        
        if group_key is None:
            # Global quantile-based clipping
            lower = series.quantile(lower_q)
            upper = series.quantile(upper_q)
            return series.clip(lower=lower, upper=upper)
        
        elif isinstance(series.index, pd.MultiIndex):
            # Group-based quantile clipping
            group_idx = series.index.get_level_values(group_key)
            return series.groupby(group_idx).transform(
                lambda x: x.clip(lower=x.quantile(lower_q), 
                                upper=x.quantile(upper_q))
            )
        else:
            # Fallback: global
            lower = series.quantile(lower_q)
            upper = series.quantile(upper_q)
            return series.clip(lower=lower, upper=upper)
    
    elif method == 'std':
        lower_std, upper_std = limits
        
        if group_key is None:
            # Global std-based clipping
            mean = series.mean()
            std = series.std()
            lower = mean - lower_std * std
            upper = mean + upper_std * std
            return series.clip(lower=lower, upper=upper)
        
        elif isinstance(series.index, pd.MultiIndex):
            # Group-based std clipping
            group_idx = series.index.get_level_values(group_key)
            return series.groupby(group_idx).transform(
                lambda x: x.clip(lower=x.mean() - lower_std * x.std(),
                                upper=x.mean() + upper_std * x.std())
            )
        else:
            # Fallback: global
            mean = series.mean()
            std = series.std()
            lower = mean - lower_std * std
            upper = mean + upper_std * std
            return series.clip(lower=lower, upper=upper)
    
    else:
        raise ValueError(f"Unknown winsorization method: {method}")


# ==================== Standardization ====================


def standardize(series: pd.Series,
                method: str = 'zscore',
                group_key: str = 'date') -> pd.Series:
    """
    Standardize a series to have zero mean and unit variance.
    
    Parameters
    ----------
    series : pd.Series
        Input series with possible MultiIndex (date, ticker)
    method : str, optional
        Standardization method:
        - 'zscore': (x - mean) / std (default)
        - 'minmax': (x - min) / (max - min), scaled to [0, 1]
        - 'robust': (x - median) / IQR (robust to outliers)
    group_key : str, optional
        Grouping key for independent standardization:
        - 'date': Per-date (cross-sectional) standardization
        - 'ticker': Per-ticker (time-series) standardization
        - None: Global standardization
        
    Returns
    -------
    pd.Series
        Standardized series
        
    Notes
    -----
    In factor research, cross-sectional standardization (group_key='date')
    is standard practice: each factor is Z-scored on each date so that
    the factor has zero mean and unit variance across the market.
    
    Examples
    --------
    >>> # Standard cross-sectional Z-score each date
    >>> factor_std = standardize(factor, method='zscore', group_key='date')
    
    >>> # Robust standardization using median/IQR
    >>> factor_std = standardize(factor, method='robust', group_key='date')
    """
    
    if method == 'zscore':
        if group_key is None:
            # Global z-score
            mean = series.mean()
            std = series.std()
            std = std if std != 0 else 1.0  # Avoid division by zero
            return (series - mean) / std
        
        elif isinstance(series.index, pd.MultiIndex):
            # Group z-score
            group_idx = series.index.get_level_values(group_key)
            return series.groupby(group_idx).transform(
                lambda x: (x - x.mean()) / (x.std() or 1.0)
            )
        else:
            # Fallback: global
            mean = series.mean()
            std = series.std()
            std = std if std != 0 else 1.0
            return (series - mean) / std
    
    elif method == 'minmax':
        if group_key is None:
            # Global min-max scaling
            min_val = series.min()
            max_val = series.max()
            range_val = max_val - min_val
            range_val = range_val if range_val != 0 else 1.0
            return (series - min_val) / range_val
        
        elif isinstance(series.index, pd.MultiIndex):
            # Group min-max scaling
            group_idx = series.index.get_level_values(group_key)
            return series.groupby(group_idx).transform(
                lambda x: (x - x.min()) / (x.max() - x.min() or 1.0)
            )
        else:
            # Fallback: global
            min_val = series.min()
            max_val = series.max()
            range_val = max_val - min_val
            range_val = range_val if range_val != 0 else 1.0
            return (series - min_val) / range_val
    
    elif method == 'robust':
        if group_key is None:
            # Global robust scaling
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            iqr = iqr if iqr != 0 else 1.0
            return (series - median) / iqr
        
        elif isinstance(series.index, pd.MultiIndex):
            # Group robust scaling
            group_idx = series.index.get_level_values(group_key)
            return series.groupby(group_idx).transform(
                lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25) or 1.0)
            )
        else:
            # Fallback: global
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            iqr = iqr if iqr != 0 else 1.0
            return (series - median) / iqr
    
    else:
        raise ValueError(f"Unknown standardization method: {method}")


# ==================== Risk Neutralization ====================


def neutralize(target_series: pd.Series,
               risk_df: pd.DataFrame,
               method: str = 'ols',
               group_key: str = 'date') -> pd.Series:
    """
    Neutralize target factor against risk factors via linear regression.
    
    Performs cross-sectional linear regression on each date:
        target = intercept + beta1*risk_factor1 + beta2*risk_factor2 + ... + residuals
    Returns the residuals, which are orthogonal to risk factors.
    
    Parameters
    ----------
    target_series : pd.Series
        Target factor to be neutralized, with MultiIndex (date, ticker)
    risk_df : pd.DataFrame
        Risk factors with MultiIndex (date, ticker) or regular columns.
        Each column is a risk factor.
        Can include:
        - Continuous factors (e.g., size, beta, liquidity)
        - Categorical/dummy factors (e.g., industry dummies from pd.get_dummies)
    method : str, optional
        Regression method:
        - 'ols': Ordinary Least Squares (default)
        - 'lstsq': NumPy least squares (for singular/near-singular matrices)
    group_key : str, optional
        Grouping key for independent regressions (default: 'date')
        
    Returns
    -------
    pd.Series
        Residuals (neutralized factor), same shape and index as target_series
        
    Raises
    ------
    ValueError
        If target_series and risk_df have incompatible indices
    
    Notes
    -----
    1. This implements "style neutralization" in quantitative finance:
       Remove the component explained by known risk factors.
    
    2. Typical risk factors for equity markets:
       - Industry: One-hot encoded dummies
       - Market cap (size)
       - Volatility
       - Momentum
       - Value (book-to-market)
       - Liquidity
    
    3. The resulting residuals have zero correlation with risk factors
       (within each cross-section).
    
    4. For computational efficiency, uses groupby().apply() for
       per-date regression.
    
    Examples
    --------
    >>> # Neutralize alpha against industry and style factors
    >>> industry_dummies = pd.get_dummies(df['industry'], prefix='ind')
    >>> risk_factors = pd.concat([
    ...     industry_dummies,
    ...     df[['size', 'volatility', 'momentum']]
    ... ], axis=1)
    >>> alpha_neutral = neutralize(alpha, risk_factors, group_key='date')
    
    See Also
    --------
    winsorize : Remove outliers before/after neutralization
    standardize : Standardize result after neutralization
    """
    
    # Ensure indices are aligned
    if len(target_series) != len(risk_df):
        raise ValueError("target_series and risk_df must have same length")
    
    if isinstance(target_series.index, pd.MultiIndex):
        date_idx = target_series.index.get_level_values(group_key)
    else:
        raise ValueError("target_series must have MultiIndex with date information")
    
    # Align risk_df with target_series if needed
    if isinstance(risk_df.index, pd.MultiIndex):
        risk_df_aligned = risk_df
    else:
        # Assume same order as target_series
        risk_df_aligned = risk_df.copy()
        risk_df_aligned.index = target_series.index
    
    def _regress_group(group_idx):
        """Perform cross-sectional regression for each date"""
        mask = (date_idx == group_idx)
        y = target_series.loc[mask].values
        X = risk_df_aligned.loc[mask].values
        
        # Skip if too few observations
        if len(y) < 2:
            return pd.Series(np.nan, index=target_series.loc[mask].index)
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        y_valid = y[valid_mask]
        X_valid = X[valid_mask]
        
        if len(y_valid) < 2 or X_valid.shape[0] < X_valid.shape[1]:
            # Insufficient observations or rank deficiency
            return pd.Series(np.nan, index=target_series.loc[mask].index)
        
        try:
            if method == 'ols':
                # Add intercept column
                X_with_const = np.column_stack([np.ones(X_valid.shape[0]), X_valid])
                coeffs = np.linalg.lstsq(X_with_const, y_valid, rcond=None)[0]
                # Prediction without intercept 
                fitted = X_with_const @ coeffs
            elif method == 'lstsq':
                # Use scipy's lstsq
                X_with_const = np.column_stack([np.ones(X_valid.shape[0]), X_valid])
                coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y_valid, rcond=None)
                fitted = X_with_const @ coeffs
            else:
                raise ValueError(f"Unknown method: {method}")
            
            residuals = y_valid - fitted
            
            # Map back to original indices
            result_full = np.full(len(y), np.nan)
            result_full[valid_mask] = residuals
            
            return pd.Series(result_full, index=target_series.loc[mask].index)
        
        except np.linalg.LinAlgError:
            # Singular matrix
            return pd.Series(np.nan, index=target_series.loc[mask].index)
    
    # Apply regression to each date group
    unique_dates = date_idx.unique()
    results = []
    for date_val in unique_dates:
        results.append(_regress_group(date_val))
    
    # Concatenate and sort back to original order
    neutralized = pd.concat(results).sort_index()
    
    return neutralized[target_series.index]


# ==================== Utility: Check for Multi-Index ====================


def ensure_multiindex(df: pd.DataFrame, 
                      date_col: str = 'date',
                      ticker_col: str = 'ticker') -> pd.DataFrame:
    """
    Ensure DataFrame has MultiIndex (date, ticker).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str, optional
        Date column name
    ticker_col : str, optional
        Ticker column name
        
    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex (date, ticker)
        
    Examples
    --------
    >>> df_indexed = ensure_multiindex(df)
    """
    if isinstance(df.index, pd.MultiIndex):
        return df
    else:
        return df.set_index([date_col, ticker_col])


# ==================== Combined Pipeline ====================


def process_factor(raw_factor: pd.Series,
                   winsorize_method: str = 'quantile',
                   winsorize_limits: Tuple[float, float] = (0.01, 0.99),
                   standardize_method: str = 'zscore',
                   risk_df: Optional[pd.DataFrame] = None,
                   neutralize_method: str = 'ols') -> pd.Series:
    """
    Apply standard factor post-processing pipeline:
    1. Winsorization (remove outliers)
    2. Standardization (optional)
    3. Risk neutralization (optional)
    
    Parameters
    ----------
    raw_factor : pd.Series
        Raw factor values (MultiIndex: date, ticker)
    winsorize_method : str, optional
        Winsorization method (see winsorize)
    winsorize_limits : tuple, optional
        Winsorization limits
    standardize_method : str, optional
        Standardization method (see standardize)
    risk_df : pd.DataFrame, optional
        Risk factor DataFrame for neutralization. If None, skip neutralization.
    neutralize_method : str, optional
        Neutralization method (see neutralize)
        
    Returns
    -------
    pd.Series
        Post-processed factor
        
    Examples
    --------
    >>> processed = process_factor(
    ...     raw_alpha,
    ...     risk_df=pd.concat([industry_dummies, style_factors], axis=1)
    ... )
    """
    result = raw_factor.copy()
    
    # Step 1: Winsorize
    result = winsorize(result, method=winsorize_method, 
                      limits=winsorize_limits, group_key='date')
    
    # Step 2: Standardize
    result = standardize(result, method=standardize_method, group_key='date')
    
    # Step 3: Neutralize against risk factors
    if risk_df is not None:
        result = neutralize(result, risk_df, method=neutralize_method, group_key='date')
        result = standardize(result, method=standardize_method, group_key='date')
    
    return result
