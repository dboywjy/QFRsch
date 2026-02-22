"""
Backtest Data Structures Module
Defines configuration and result models for backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import date

import pandas as pd


@dataclass
class BacktestConfig:
    """
    Configuration parameters for backtest engine.
    
    This dataclass encapsulates all hyper-parameters needed to run a backtest,
    including initial capital, transaction costs, and risk-free rate.
    
    Attributes
    ----------
    initial_capital : float, default=1_000_000
        Starting capital in currency units.
    commission_rate : float, default=0.001
        Commission rate as a percentage of trade volume.
        Example: 0.001 means 0.1% commission.
    slippage_rate : float, default=0.0005
        Slippage rate for price impact during trade execution.
        Example: 0.0005 means 0.05% slippage.
    rebalance_frequency : str, default='D'
        Frequency of portfolio rebalancing.
        Options: 'D' (daily), 'W' (weekly), 'M' (monthly).
    risk_free_rate : float, default=0.02
        Annual risk-free rate for Sharpe ratio calculation.
        Example: 0.02 means 2% annualized.
    benchmark_ticker : str, optional
        Ticker symbol of benchmark asset for comparison.
    execution_price : str, default='close'
        Which price to use for order execution: 'open' or 'close'.
    kwargs : dict
        Additional configuration parameters for extensibility.
        
    Examples
    --------
    >>> config = BacktestConfig(
    ...     initial_capital=500_000,
    ...     commission_rate=0.002,
    ...     slippage_rate=0.001
    ... )
    """
    
    initial_capital: float = 1_000_000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    rebalance_frequency: str = 'D'
    risk_free_rate: float = 0.02
    benchmark_ticker: Optional[str] = None
    execution_price: str = 'close'
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """
    Container for backtest results and analysis data.
    
    Stores comprehensive backtest output including daily equity curve,
    transaction history, holdings snapshots, and performance metrics.
    
    Attributes
    ----------
    equity_curve : pd.DataFrame
        Daily equity evolution. Index: date, columns: ['total_value', 'cash', 'positions_value'].
    trades : pd.DataFrame
        Transaction log. Columns: ['date', 'ticker', 'action', 'quantity', 'price', 'commission', 'cash_impact'].
    holdings : pd.DataFrame
        Daily holdings snapshot. Index: date, columns: tickers with quantities.
    daily_returns : pd.Series
        Daily portfolio return series. Index: date, values: return rate.
    daily_weights : pd.DataFrame
        Daily portfolio weights. Index: date, columns: tickers with weight ratios.
    metrics : Dict[str, Any]
        Summary performance metrics: total_return, sharpe_ratio, max_drawdown, etc.
    config : BacktestConfig
        Original backtest configuration used for this run.
        
    Examples
    --------
    >>> result.equity_curve.head()
             total_value       cash  positions_value
    date
    2020-01-... 1000000.0 999000.0        1000.0
    
    >>> print(f"Total Return: {result.metrics['total_return']:.2%}")
    Total Return: 5.23%
    """
    
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    daily_returns: pd.Series
    daily_weights: pd.DataFrame
    metrics: Dict[str, Any]
    config: BacktestConfig


__all__ = ['BacktestConfig', 'BacktestResult']
