"""
Backtest Utility Functions Module
Provides helper functions for calculating performance metrics and statistics.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown of equity curve.
    
    Maximum Drawdown (MDD) is the largest peak-to-trough decline in portfolio value,
    typically expressed as a negative percentage.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio total value. Index: date, values: equity.
        
    Returns
    -------
    float
        Maximum drawdown as a ratio. Example: -0.15 means -15%.
        Returns 0.0 if equity never declined.
        
    Examples
    --------
    >>> dates = pd.date_range('2020-01-01', periods=5)
    >>> equity = pd.Series([100, 120, 100, 110, 130], index=dates)
    >>> mdd = calculate_max_drawdown(equity)
    >>> print(f"MDD: {mdd:.4f}")  # Approximately -0.1667 (from 120 to 100)
    MDD: -0.1667
    """
    
    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown from peak
    drawdown = (equity_curve - running_max) / running_max
    
    # Return minimum (most negative) drawdown
    return drawdown.min()


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio of return series.
    
    Sharpe Ratio measures risk-adjusted return, defined as:
    (avg_return - risk_free_rate) / std_return
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns. Index: date, values: return rate (e.g., 0.01 for 1%).
    risk_free_rate : float, default=0.02
        Annual risk-free rate. Example: 0.02 = 2% per year.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Sharpe ratio. Higher is better.
        Returns 0.0 if standard deviation is zero.
        
    Examples
    --------
    >>> dates = pd.date_range('2020-01-01', periods=252)
    >>> returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    >>> sr = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    >>> print(f"Sharpe Ratio: {sr:.4f}")
    """
    
    if len(returns) == 0:
        return 0.0
    
    # Annualize daily statistics
    annual_return = returns.mean() * periods_per_year
    annual_std = returns.std() * np.sqrt(periods_per_year)
    
    # Avoid division by zero
    if annual_std == 0:
        return 0.0
    
    # Calculate Sharpe ratio
    return (annual_return - risk_free_rate) / annual_std


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (return over max drawdown).
    
    Calmar Ratio = Annual Return / Absolute Value of Max Drawdown
    Measures return relative to downside risk.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns. Index: date, values: return rate.
    periods_per_year : int, default=252
        Number of trading days per year.
        
    Returns
    -------
    float
        Calmar ratio. Returns 0.0 if max drawdown is zero or returns are zero.
        
    Examples
    --------
    >>> cr = calculate_calmar_ratio(returns)
    >>> print(f"Calmar Ratio: {cr:.4f}")
    """
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate equity curve from returns
    equity_curve = (1 + returns).cumprod()
    mdd = calculate_max_drawdown(equity_curve)
    annual_return = returns.mean() * periods_per_year
    
    # Avoid division by zero
    if mdd == 0:
        return 0.0
    
    return annual_return / abs(mdd)


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate percentage of profitable days.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns. Index: date, values: return rate.
        
    Returns
    -------
    float
        Win rate as a ratio. Example: 0.55 means 55% of days were profitable.
        
    Examples
    --------
    >>> wr = calculate_win_rate(returns)
    >>> print(f"Win Rate: {wr:.2%}")
    Win Rate: 52.50%
    """
    
    if len(returns) == 0:
        return 0.0
    
    positive_days = (returns > 0).sum()
    return positive_days / len(returns)


def calculate_growth_metrics(
    equity_curve: pd.Series,
    initial_capital: float
) -> Tuple[float, float]:
    """
    Calculate total return and annualized return.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio total value.
    initial_capital : float
        Starting capital.
        
    Returns
    -------
    tuple[float, float]
        (total_return, annualized_return)
        Both expressed as ratios. Example: 0.15 means 15%.
        
    Examples
    --------
    >>> total_ret, ann_ret = calculate_growth_metrics(equity_curve, 1_000_000)
    >>> print(f"Total: {total_ret:.2%}, Annualized: {ann_ret:.2%}")
    """
    
    if len(equity_curve) < 1 or initial_capital <= 0:
        return 0.0, 0.0
    
    final_value = equity_curve.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # Annualize return: (final/initial) ^ (252/days) - 1
    num_days = len(equity_curve) - 1
    if num_days == 0:
        annualized_return = 0.0
    else:
        annualized_return = ((final_value / initial_capital) ** (252 / num_days)) - 1
    
    return total_return, annualized_return


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
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
        
    Examples
    --------
    >>> vol = calculate_volatility(returns)
    >>> print(f"Volatility: {vol:.2%}")
    Volatility: 18.50%
    """
    
    if len(returns) == 0:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


__all__ = [
    'calculate_max_drawdown',
    'calculate_sharpe_ratio',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_growth_metrics',
    'calculate_volatility',
]
