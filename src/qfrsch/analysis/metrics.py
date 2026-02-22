"""
Analysis Metrics Module
Core indicator calculation library for performance analysis.
Provides comprehensive metrics for returns, risk, and statistical testing.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def calculate_annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns. Index: date, values: return rate (e.g., 0.01 for 1%).
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Annualized return as a ratio.
        
    Examples
    --------
    >>> returns = pd.Series([0.01, -0.005, 0.02, 0.01, -0.005])
    >>> annual_ret = calculate_annual_return(returns)
    """
    
    if len(returns) == 0:
        return 0.0
    
    total_return = (1 + returns).prod() - 1
    num_days = len(returns)
    annual_return = (1 + total_return) ** (periods_per_year / num_days) - 1
    return annual_return


def calculate_annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility (standard deviation).
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Annualized volatility as a ratio.
    """
    
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Sharpe Ratio = (Annual Return - Risk Free Rate) / Annual Volatility
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    risk_free_rate : float, default=0.02
        Annual risk-free rate.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Sharpe ratio.
    """
    
    if len(returns) == 0:
        return 0.0
    
    annual_return = calculate_annual_return(returns, periods_per_year)
    annual_vol = calculate_annual_volatility(returns, periods_per_year)
    
    if annual_vol == 0:
        return 0.0
    
    return (annual_return - risk_free_rate) / annual_vol


def calculate_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio.
    
    Similar to Sharpe, but uses downside volatility (only negative returns) instead of total volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    target_return : float, default=0.0
        Target return threshold.
    risk_free_rate : float, default=0.02
        Annual risk-free rate.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Sortino ratio.
    """
    
    if len(returns) < 2:
        return 0.0
    
    annual_return = calculate_annual_return(returns, periods_per_year)
    
    # Calculate downside deviation
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        downside_vol = 0.0
    else:
        downside_vol = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year)
    
    if downside_vol == 0:
        return 0.0
    
    return (annual_return - risk_free_rate) / downside_vol


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar Ratio = Annual Return / Absolute Value of Max Drawdown
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Calmar ratio.
    """
    
    if len(returns) == 0:
        return 0.0
    
    annual_return = calculate_annual_return(returns, periods_per_year)
    equity_curve = (1 + returns).cumprod()
    
    # Calculate max drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    mdd = drawdown.min()
    
    if mdd == 0:
        return 0.0
    
    return annual_return / abs(mdd)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
        
    Returns
    -------
    float
        Maximum drawdown as a negative ratio (e.g., -0.15 for -15%).
    """
    
    if len(returns) == 0:
        return 0.0
    
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    
    return drawdown.min()


def calculate_excess_return(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculate excess returns (Active Return).
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy daily returns. Index: date.
    benchmark_returns : pd.Series
        Benchmark daily returns. Index: date.
        
    Returns
    -------
    pd.Series
        Daily excess returns.
    """
    
    # Align indices
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_ret_aligned = strategy_returns.loc[common_index]
    benchmark_ret_aligned = benchmark_returns.loc[common_index]
    
    excess = strategy_ret_aligned - benchmark_ret_aligned
    return excess


def calculate_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio (IR).
    
    IR = (Annual Excess Return) / (Tracking Error)
    where Tracking Error = Std Dev of Excess Returns
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy daily returns.
    benchmark_returns : pd.Series
        Benchmark daily returns.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Information ratio.
    """
    
    excess = calculate_excess_return(strategy_returns, benchmark_returns)
    
    if len(excess) == 0:
        return 0.0
    
    annual_excess_return = (1 + excess).prod() ** (periods_per_year / len(excess)) - 1
    tracking_error = excess.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    return annual_excess_return / tracking_error


def calculate_excess_volatility(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate excess volatility.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy daily returns.
    benchmark_returns : pd.Series
        Benchmark daily returns.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Annualized excess volatility.
    """
    
    excess = calculate_excess_return(strategy_returns, benchmark_returns)
    
    if len(excess) < 2:
        return 0.0
    
    return excess.std() * np.sqrt(periods_per_year)


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate percentage of positive days.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
        
    Returns
    -------
    float
        Win rate as ratio (e.g., 0.55 for 55%).
    """
    
    if len(returns) == 0:
        return 0.0
    
    positive_days = (returns > 0).sum()
    return positive_days / len(returns)


def calculate_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate portfolio beta relative to benchmark.
    
    Beta = Covariance(Strategy, Benchmark) / Variance(Benchmark)
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy daily returns.
    benchmark_returns : pd.Series
        Benchmark daily returns.
        
    Returns
    -------
    float
        Beta coefficient.
    """
    
    # Align indices
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_ret = strategy_returns.loc[common_index]
    benchmark_ret = benchmark_returns.loc[common_index]
    
    if len(strategy_ret) < 2:
        return 0.0
    
    covariance = pd.Series(strategy_ret.values, index=strategy_ret.index).cov(
        pd.Series(benchmark_ret.values, index=strategy_ret.index)
    )
    variance = benchmark_ret.var()
    
    if variance == 0:
        return 0.0
    
    return covariance / variance


def calculate_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Jensen's alpha.
    
    Alpha = Annual Strategy Return - (Risk Free Rate + Beta * (Annual Benchmark Return - Risk Free Rate))
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy daily returns.
    benchmark_returns : pd.Series
        Benchmark daily returns.
    risk_free_rate : float, default=0.02
        Annual risk-free rate.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Jensen's alpha.
    """
    
    strategy_annual_ret = calculate_annual_return(strategy_returns, periods_per_year)
    benchmark_annual_ret = calculate_annual_return(benchmark_returns, periods_per_year)
    beta = calculate_beta(strategy_returns, benchmark_returns)
    
    alpha = strategy_annual_ret - (risk_free_rate + beta * (benchmark_annual_ret - risk_free_rate))
    return alpha


def newey_west_ttest(
    returns: pd.Series,
    test_value: float = 0.0,
    lags: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Newey-West adjusted T-test.
    
    Corrects for heteroskedasticity and autocorrelation in the time series.
    Particularly useful for detecting significant alpha in returns.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns (e.g., excess returns or alpha).
    test_value : float, default=0.0
        Under null hypothesis, returns have this value. Usually 0.
    lags : int, optional
        Number of lags for Newey-West correction. If None, defaults to int(np.sqrt(len(returns))).
        
    Returns
    -------
    tuple[float, float, float]
        (t_statistic, p_value, annual_return)
        Returns are annualized.
    """
    
    if len(returns) < 3:
        return 0.0, 1.0, 0.0
    
    if lags is None:
        lags = max(1, int(np.sqrt(len(returns))))
    
    # Annualize returns
    annual_ret = calculate_annual_return(returns, periods_per_year=252)
    
    # Calculate standard error with Newey-West adjustment
    mean_ret = returns.mean()
    n = len(returns)
    
    # Variance calculation with autocorrelation adjustment
    var_sum = returns.var()
    
    for lag in range(1, lags + 1):
        acf_lag = returns.autocorr(lag=lag)
        weight = 1 - (lag / (lags + 1))
        var_sum += 2 * weight * (returns.cov(returns.shift(lag)))
    
    se = np.sqrt(var_sum / n)
    
    if se == 0:
        return 0.0, 1.0, annual_ret
    
    # T-statistic
    t_stat = (mean_ret - test_value) / se
    
    # P-value (two-tailed)
    dof = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
    
    return t_stat, p_value, annual_ret


def calculate_correlation_with_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate correlation between strategy and benchmark returns.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy daily returns.
    benchmark_returns : pd.Series
        Benchmark daily returns.
        
    Returns
    -------
    float
        Correlation coefficient.
    """
    
    # Align indices
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_ret = strategy_returns.loc[common_index]
    benchmark_ret = benchmark_returns.loc[common_index]
    
    if len(strategy_ret) < 2:
        return 0.0
    
    return strategy_ret.corr(benchmark_ret)


def calculate_recovery_factor(returns: pd.Series) -> float:
    """
    Calculate Recovery Factor.
    
    Recovery Factor = Total Return / Absolute Maximum Drawdown
    Higher is better. Indicates how much total return recovered from worst loss.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
        
    Returns
    -------
    float
        Recovery factor.
    """
    
    if len(returns) == 0:
        return 0.0
    
    total_return = (1 + returns).prod() - 1
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    mdd = abs(drawdown.min())
    
    if mdd == 0:
        return 0.0
    
    return total_return / mdd


def calculate_monthly_returns(returns: pd.Series) -> pd.Series:
    """
    Aggregate daily returns to monthly returns.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns. Index must be datetime.
        
    Returns
    -------
    pd.Series
        Monthly returns. Index: month-end date.
    """
    
    if len(returns) == 0:
        return pd.Series()
    
    # Ensure daily returns are indexed by date
    returns_with_date = returns.copy()
    returns_with_date.index = pd.to_datetime(returns_with_date.index)
    
    # Calculate cumulative return per month
    monthly_ret = returns_with_date.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    return monthly_ret


def calculate_drawdown_duration(returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate average and maximum drawdown duration (in days).
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
        
    Returns
    -------
    tuple[float, float]
        (average_drawdown_duration, max_drawdown_duration)
    """
    
    if len(returns) == 0:
        return 0.0, 0.0
    
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.expanding().max()
    
    # Identify drawdown periods (when equity < running max)
    in_drawdown = equity_curve < running_max
    
    # Calculate duration of each drawdown period
    duration_groups = (in_drawdown != in_drawdown.shift()).cumsum()
    durations = in_drawdown.groupby(duration_groups).sum()
    
    if len(durations) == 0:
        return 0.0, 0.0
    
    avg_duration = durations[durations > 0].mean() if any(durations > 0) else 0.0
    max_duration = durations.max()
    
    return float(avg_duration), float(max_duration)


__all__ = [
    'calculate_annual_return',
    'calculate_annual_volatility',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_max_drawdown',
    'calculate_excess_return',
    'calculate_information_ratio',
    'calculate_excess_volatility',
    'calculate_win_rate',
    'calculate_beta',
    'calculate_alpha',
    'newey_west_ttest',
    'calculate_correlation_with_benchmark',
    'calculate_recovery_factor',
    'calculate_monthly_returns',
    'calculate_drawdown_duration',
]
