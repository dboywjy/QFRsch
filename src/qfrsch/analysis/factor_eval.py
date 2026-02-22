"""
Factor Evaluation Module
Implements IC analysis, quantile-based backtest, Fama-MacBeth regression,
and factor stability metrics.
"""

from __future__ import annotations

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from qfrsch.analysis.metrics import calculate_annual_return, calculate_annual_volatility


def calculate_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'pearson'
) -> pd.Series:
    """
    Calculate daily Information Coefficient (IC).
    
    IC measures the correlation between factor values and forward returns on each date.
    
    Parameters
    ----------
    factor_values : pd.DataFrame
        Factor values. Index: date, columns: tickers, values: factor values.
    forward_returns : pd.DataFrame
        Forward returns. Index: date, columns: tickers, values: returns.
    method : str, default='pearson'
        Correlation method: 'pearson' or 'spearman'.
        
    Returns
    -------
    pd.Series
        Daily IC values. Index: date.
        
    Examples
    --------
    >>> ic_series = calculate_ic(factor_df, returns_df)
    >>> print(f"Mean IC: {ic_series.mean():.4f}")
    """
    
    # Align indices
    common_dates = factor_values.index.intersection(forward_returns.index)
    factor_aligned = factor_values.loc[common_dates]
    returns_aligned = forward_returns.loc[common_dates]
    
    ic_list = []
    
    for date in common_dates:
        factor_row = factor_aligned.loc[date]
        returns_row = returns_aligned.loc[date]
        
        # Remove NaN values
        valid_mask = ~(factor_row.isna() | returns_row.isna())
        if valid_mask.sum() < 2:
            continue
        
        factor_clean = factor_row[valid_mask]
        returns_clean = returns_row[valid_mask]
        
        if method == 'spearman':
            ic_value = stats.spearmanr(factor_clean, returns_clean)[0]
        else:  # pearson
            ic_value = factor_clean.corr(returns_clean)
        
        ic_list.append(ic_value)
    
    if not ic_list:
        return pd.Series(dtype=float)
    
    ic_series = pd.Series(ic_list, index=common_dates[:len(ic_list)])
    return ic_series


def calculate_rank_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame
) -> pd.Series:
    """
    Calculate daily Rank Information Coefficient (Rank IC).
    
    Rank IC uses Spearman correlation (rank-based), which is more robust to outliers.
    
    Parameters
    ----------
    factor_values : pd.DataFrame
        Factor values.
    forward_returns : pd.DataFrame
        Forward returns.
        
    Returns
    -------
    pd.Series
        Daily Rank IC values.
    """
    
    return calculate_ic(factor_values, forward_returns, method='spearman')


def calculate_ic_statistics(ic_series: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive IC statistics.
    
    Parameters
    ----------
    ic_series : pd.Series
        Daily IC values from calculate_ic().
        
    Returns
    -------
    dict
        Statistics including:
        - ic_mean: Average IC
        - ic_std: IC standard deviation
        - ic_positive_pct: Percentage of days with IC > 0
        - ic_strong_positive_pct: Percentage of days with IC > 0.03
        - ic_ir: IC Information Ratio (mean IC / std IC)
    """
    
    if len(ic_series) == 0:
        return {
            'ic_mean': 0.0,
            'ic_std': 0.0,
            'ic_positive_pct': 0.0,
            'ic_strong_positive_pct': 0.0,
            'ic_ir': 0.0,
        }
    
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_positive_pct = (ic_series > 0).sum() / len(ic_series)
    ic_strong_positive_pct = (ic_series > 0.03).sum() / len(ic_series)
    
    if ic_std == 0:
        ic_ir = 0.0
    else:
        ic_ir = ic_mean / ic_std
    
    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_positive_pct': ic_positive_pct,
        'ic_strong_positive_pct': ic_strong_positive_pct,
        'ic_ir': ic_ir,
    }


def quantile_backtest(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    num_quantiles: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Perform quantile-based factor backtest.
    
    Divides the universe into quantiles based on factor values and analyzes returns
    for each quantile group.
    
    Parameters
    ----------
    factor_values : pd.DataFrame
        Factor values. Index: date, columns: tickers.
    forward_returns : pd.DataFrame
        Forward returns. Index: date, columns: tickers.
    num_quantiles : int, default=5
        Number of quantile groups (e.g., 5 for quintiles).
        
    Returns
    -------
    dict
        Contains:
        - quantile_returns: DataFrame of daily returns per quantile [date x quantiles]
        - quantile_cumret: DataFrame of cumulative returns per quantile
        - quantile_annual_ret: Series of annualized returns per quantile
        - quantile_annual_vol: Series of annualized volatility per quantile
        - high_minus_low: Series of (highest - lowest) quantile daily returns
        
    Examples
    --------
    >>> result = quantile_backtest(factor_df, returns_df, num_quantiles=5)
    >>> print(result['quantile_annual_ret'])
    """
    
    common_dates = factor_values.index.intersection(forward_returns.index)
    factor_aligned = factor_values.loc[common_dates]
    returns_aligned = forward_returns.loc[common_dates]
    
    quantile_returns_dict = {q: [] for q in range(1, num_quantiles + 1)}
    quantile_dates = []
    
    for date in common_dates:
        factor_row = factor_aligned.loc[date]
        returns_row = returns_aligned.loc[date]
        
        # Remove NaN values
        valid_mask = ~(factor_row.isna() | returns_row.isna())
        if valid_mask.sum() < num_quantiles:
            continue
        
        factor_clean = factor_row[valid_mask]
        returns_clean = returns_row[valid_mask]
        
        # Create quantile labels
        quantile_labels = pd.qcut(factor_clean, q=num_quantiles, labels=False, duplicates='drop')
        
        # Calculate average return per quantile
        for q in range(num_quantiles):
            quantile_mask = quantile_labels == q
            if quantile_mask.sum() > 0:
                avg_ret = returns_clean[quantile_mask].mean()
            else:
                avg_ret = np.nan
            
            quantile_returns_dict[q + 1].append(avg_ret)
        
        quantile_dates.append(date)
    
    # Convert to DataFrames
    quantile_returns_df = pd.DataFrame(quantile_returns_dict, index=quantile_dates)
    quantile_returns_df = quantile_returns_df.astype(float)
    
    # Calculate cumulative returns
    quantile_cumret_df = (1 + quantile_returns_df).cumprod()
    
    # Calculate annualized metrics
    quantile_annual_ret = quantile_returns_df.apply(
        lambda x: calculate_annual_return(x.dropna(), periods_per_year=252)
    )
    quantile_annual_vol = quantile_returns_df.apply(
        lambda x: calculate_annual_volatility(x.dropna(), periods_per_year=252)
    )
    
    # Calculate high minus low returns
    high_col = num_quantiles
    low_col = 1
    high_minus_low = quantile_returns_df[high_col] - quantile_returns_df[low_col]
    
    return {
        'quantile_returns': quantile_returns_df,
        'quantile_cumret': quantile_cumret_df,
        'quantile_annual_ret': quantile_annual_ret,
        'quantile_annual_vol': quantile_annual_vol,
        'high_minus_low': high_minus_low,
    }


def fama_macbeth_regression(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    control_factors: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Implement two-step Fama-MacBeth regression.
    
    Step 1: For each date, regress returns on factor values.
    Step 2: Calculate time-series average of factor loadings and their t-statistics.
    
    This extracts the factor risk premium and tests its significance.
    
    Parameters
    ----------
    factor_values : pd.DataFrame
        Main factor to test. Index: date, columns: tickers.
    forward_returns : pd.DataFrame
        Forward returns. Index: date, columns: tickers.
    control_factors : pd.DataFrame, optional
        Control factors for multi-factor regression. Index: date, columns: factor names.
        
    Returns
    -------
    dict
        Factor risk premium, t-statistic, p-value, etc.
        
    Examples
    --------
    >>> fm_result = fama_macbeth_regression(factor_df, returns_df)
    >>> print(f"Factor Risk Premium: {fm_result['factor_premium']:.4f}")
    >>> print(f"T-statistic: {fm_result['t_stat']:.4f}")
    """
    
    common_dates = factor_values.index.intersection(forward_returns.index)
    factor_aligned = factor_values.loc[common_dates]
    returns_aligned = forward_returns.loc[common_dates]
    
    factor_loadings = []
    
    for date in common_dates:
        factor_row = factor_aligned.loc[date]
        returns_row = returns_aligned.loc[date]
        
        # Remove NaN values
        valid_mask = ~(factor_row.isna() | returns_row.isna())
        if valid_mask.sum() < 2:
            continue
        
        factor_clean = factor_row[valid_mask].values
        returns_clean = returns_row[valid_mask].values
        
        # Add constant for intercept
        X = add_constant(factor_clean)
        
        # Add control factors if provided
        if control_factors is not None:
            control_row = control_factors.loc[date].values
            X = np.column_stack([X, control_row])
        
        try:
            # Cross-sectional regression
            model = OLS(returns_clean, X).fit()
            factor_loadings.append(model.params[1])  # Loading on main factor
        except Exception:
            continue
    
    if not factor_loadings:
        return {
            'factor_premium': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'num_obs': 0,
        }
    
    factor_loadings = np.array(factor_loadings)
    
    # Time-series statistics
    mean_loading = factor_loadings.mean()
    std_loading = factor_loadings.std()
    t_stat = mean_loading / (std_loading / np.sqrt(len(factor_loadings)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(factor_loadings) - 1))
    
    return {
        'factor_premium': mean_loading,
        'loading_std': std_loading,
        't_stat': t_stat,
        'p_value': p_value,
        'num_obs': len(factor_loadings),
        'loadings': factor_loadings,
    }


def calculate_factor_stability_coefficient(
    factor_values: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    Calculate Factor Stability Coefficient (FSC).
    
    Measures how stable a factor is over time by comparing current factor exposure
    to historical average.
    
    FSC = 1 - Std(Recent Factor) / Std(Historical Factor)
    
    Parameters
    ----------
    factor_values : pd.DataFrame
        Factor values. Index: date, columns: tickers.
    window : int, default=60
        Rolling window size (days) for historical period.
        
    Returns
    -------
    pd.Series
        Daily FSC values. Index: date.
    """
    
    fsc_list = []
    dates = []
    
    for i in range(window, len(factor_values)):
        current_window = factor_values.iloc[i - window:i]
        recent_values = factor_values.iloc[i]
        
        # Calculate volatility of factor across assets
        historical_vol = current_window.std().mean()
        
        if historical_vol == 0:
            fsc = 0.0
        else:
            recent_vol = recent_values.std()
            fsc = 1 - (recent_vol / historical_vol)
        
        fsc_list.append(fsc)
        dates.append(factor_values.index[i])
    
    fsc_series = pd.Series(fsc_list, index=dates)
    return fsc_series


def calculate_factor_autocorrelation(
    factor_values: pd.DataFrame,
    lag: int = 1
) -> float:
    """
    Calculate factor autocorrelation.
    
    Higher autocorrelation indicates the factor is more stable (less mean-reversion).
    
    Parameters
    ----------
    factor_values : pd.DataFrame
        Factor values. Index: date, columns: tickers.
    lag : int, default=1
        Number of days to lag.
        
    Returns
    -------
    float
        Average autocorrelation across all tickers.
    """
    
    autocorrs = []
    
    for ticker in factor_values.columns:
        ticker_values = factor_values[ticker]
        if len(ticker_values) > lag:
            autocorr = ticker_values.autocorr(lag=lag)
            if not np.isnan(autocorr):
                autocorrs.append(autocorr)
    
    if not autocorrs:
        return 0.0
    
    return np.mean(autocorrs)


def calculate_factor_turnover(
    factor_values: pd.DataFrame,
    num_positions: int,
    lag: int = 1
) -> pd.Series:
    """
    Calculate portfolio turnover if we built equal-weight portfolio based on top/bottom assets.
    
    Parameters
    ----------
    factor_values : pd.DataFrame
        Factor values. Index: date, columns: tickers.
    num_positions : int
        Number of top/bottom positions to hold.
    lag : int, default=1
        Number of days to lag for turnover calculation.
        
    Returns
    -------
    pd.Series
        Daily turnover ratios.
    """
    
    turnover_list = []
    dates = []
    
    for i in range(lag, len(factor_values)):
        current_row = factor_values.iloc[i]
        previous_row = factor_values.iloc[i - lag]
        
        # Get top and bottom positions
        current_top = set(current_row.nlargest(num_positions).index)
        current_bottom = set(current_row.nsmallest(num_positions).index)
        current_positions = current_top | current_bottom
        
        previous_top = set(previous_row.nlargest(num_positions).index)
        previous_bottom = set(previous_row.nsmallest(num_positions).index)
        previous_positions = previous_top | previous_bottom
        
        # Calculate turnover (positions that changed)
        if len(current_positions) > 0:
            changed = len(current_positions - previous_positions) + len(previous_positions - current_positions)
            turnover = changed / (2 * num_positions) if num_positions > 0 else 0.0
        else:
            turnover = 0.0
        
        turnover_list.append(turnover)
        dates.append(factor_values.index[i])
    
    turnover_series = pd.Series(turnover_list, index=dates)
    return turnover_series


__all__ = [
    'calculate_ic',
    'calculate_rank_ic',
    'calculate_ic_statistics',
    'quantile_backtest',
    'fama_macbeth_regression',
    'calculate_factor_stability_coefficient',
    'calculate_factor_autocorrelation',
    'calculate_factor_turnover',
]
