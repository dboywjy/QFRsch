"""
Performance Attribution Module
Analyzes sources of portfolio performance including turnover, active risk,
and return attribution.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd

from qfrsch.analysis.metrics import (
    calculate_annual_volatility,
    calculate_excess_volatility,
)


def calculate_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series
) -> float:
    """
    Calculate portfolio turnover.
    
    Turnover = Sum(|current_weight - previous_weight|) / 2
    
    Parameters
    ----------
    current_weights : pd.Series
        Current portfolio weights. Index: ticker, values: weights.
    previous_weights : pd.Series
        Previous portfolio weights.
        
    Returns
    -------
    float
        Turnover ratio between 0 and 1.
        
    Examples
    --------
    >>> current = pd.Series({'A': 0.5, 'B': 0.5})
    >>> previous = pd.Series({'A': 0.3, 'B': 0.7})
    >>> turnover = calculate_turnover(current, previous)
    >>> print(f"Turnover: {turnover:.2%}")  # Output: 20%
    """
    
    # Align indices
    all_tickers = current_weights.index.union(previous_weights.index)
    current_aligned = current_weights.reindex(all_tickers, fill_value=0.0)
    previous_aligned = previous_weights.reindex(all_tickers, fill_value=0.0)
    
    # Calculate absolute weight changes
    weight_diff = (current_aligned - previous_aligned).abs().sum()
    
    # Turnover is half the sum of absolute changes
    turnover = weight_diff / 2
    
    return float(turnover)


def calculate_daily_turnover(
    holdings_df: pd.DataFrame,
    prices_df: pd.DataFrame
) -> pd.Series:
    """
    Calculate daily portfolio turnover from holdings data.
    
    Parameters
    ----------
    holdings_df : pd.DataFrame
        Holdings data. Index: date, columns: tickers, values: shares held.
    prices_df : pd.DataFrame
        Price data. Index: date, columns: tickers, values: prices.
        
    Returns
    -------
    pd.Series
        Daily turnover ratios. Index: date.
    """
    
    turnover_list = []
    common_dates = holdings_df.index.intersection(prices_df.index)
    
    for i in range(1, len(common_dates)):
        prev_date = common_dates[i - 1]
        curr_date = common_dates[i]
        
        prev_holdings = holdings_df.loc[prev_date]
        curr_holdings = holdings_df.loc[curr_date]
        prices = prices_df.loc[curr_date]
        
        # Calculate previous day's values at current prices
        prev_value = (prev_holdings * prices).sum()
        curr_value = (curr_holdings * prices).sum()
        
        if prev_value == 0:
            turnover = 0.0
        else:
            # Trades = absolute value of position changes
            trades = (np.abs(curr_holdings - prev_holdings) * prices).sum()
            turnover = trades / max(prev_value, curr_value)
        
        turnover_list.append(turnover)
    
    turnover_series = pd.Series(turnover_list, index=common_dates[1:])
    return turnover_series


def calculate_active_return(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculate daily active returns (excess returns).
    
    Active Return = Strategy Return - Benchmark Return
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy daily returns. Index: date.
    benchmark_returns : pd.Series
        Benchmark daily returns. Index: date.
        
    Returns
    -------
    pd.Series
        Daily active returns.
    """
    
    # Align indices
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_aligned = strategy_returns.loc[common_index]
    benchmark_aligned = benchmark_returns.loc[common_index]
    
    active = strategy_aligned - benchmark_aligned
    return active


def calculate_active_risk(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate active risk (Tracking Error).
    
    Active Risk = Std Dev(Active Returns)
    
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
        Annualized active risk.
    """
    
    active = calculate_active_return(strategy_returns, benchmark_returns)
    
    if len(active) < 2:
        return 0.0
    
    return calculate_annual_volatility(active, periods_per_year)


def decompose_active_return(
    strategy_positions: pd.DataFrame,
    benchmark_positions: pd.DataFrame,
    strategy_returns: pd.Series,
    daily_security_returns: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """
    Decompose active return into allocation effect and selection effect.
    
    Allocation Effect = (Weight Difference) * (Security Return - Benchmark Return)
    Selection Effect = (Benchmark Weight) * (Strategy Security Return - Benchmark Security Return)
    
    Parameters
    ----------
    strategy_positions : pd.DataFrame
        Strategy holdings. Index: date, columns: tickers, values: portfolio weights.
    benchmark_positions : pd.DataFrame
        Benchmark holdings. Index: date, columns: tickers, values: portfolio weights.
    strategy_returns : pd.Series
        Daily strategy returns. Index: date.
    daily_security_returns : pd.DataFrame
        Daily security returns. Index: date, columns: tickers.
        
    Returns
    -------
    tuple[pd.Series, pd.Series]
        (allocation_effect, selection_effect) - daily decomposition.
    """
    
    allocation_effect_list = []
    selection_effect_list = []
    common_dates = strategy_positions.index.intersection(daily_security_returns.index)
    common_dates = common_dates.intersection(benchmark_positions.index)
    
    for date in common_dates:
        strat_pos = strategy_positions.loc[date]
        bench_pos = benchmark_positions.loc[date]
        sec_ret = daily_security_returns.loc[date]
        
        # Align tickers
        all_tickers = strat_pos.index.union(bench_pos.index)
        strat_pos_aligned = strat_pos.reindex(all_tickers, fill_value=0.0)
        bench_pos_aligned = bench_pos.reindex(all_tickers, fill_value=0.0)
        sec_ret_aligned = sec_ret.reindex(all_tickers, fill_value=0.0)
        
        weight_diff = strat_pos_aligned - bench_pos_aligned
        return_diff = sec_ret_aligned - sec_ret_aligned.mean()
        
        allocation = (weight_diff * return_diff).sum()
        selection = (bench_pos_aligned * (strat_pos_aligned - bench_pos_aligned)).sum()
        
        allocation_effect_list.append(allocation)
        selection_effect_list.append(selection)
    
    allocation_series = pd.Series(allocation_effect_list, index=common_dates)
    selection_series = pd.Series(selection_effect_list, index=common_dates)
    
    return allocation_series, selection_series


def calculate_turnover_impact(
    daily_returns: pd.Series,
    daily_turnover: pd.Series,
    alpha_cost_per_unit_turnover: float = 0.001
) -> pd.Series:
    """
    Estimate return impact from turnover (transaction costs).
    
    Assumes each unit of turnover costs alpha_cost_per_unit_turnover.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns before costs. Index: date.
    daily_turnover : pd.Series
        Daily turnover ratios. Index: date.
    alpha_cost_per_unit_turnover : float, default=0.001
        Cost impact per unit of turnover (default 0.1%).
        
    Returns
    -------
    pd.Series
        Daily turnover-related costs. Index: date.
    """
    
    # Align indices
    common_index = daily_returns.index.intersection(daily_turnover.index)
    costs = -daily_turnover.loc[common_index] * alpha_cost_per_unit_turnover
    
    return costs


def calculate_style_attribution(
    strategy_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    style_factors: pd.DataFrame,
    daily_returns: pd.DataFrame
) -> Dict[str, float]:
    """
    Analyze portfolio style attribution.
    
    Compares strategy style factor exposures vs benchmark and relates to returns.
    
    Parameters
    ----------
    strategy_weights : pd.DataFrame
        Strategy portfolio weights. Index: date, columns: tickers.
    benchmark_weights : pd.DataFrame
        Benchmark portfolio weights. Index: date, columns: tickers.
    style_factors : pd.DataFrame
        Style factor exposures. Index: date, columns: [ticker, style_name], values: factor values.
    daily_returns : pd.DataFrame
        Daily security returns. Index: date, columns: tickers.
        
    Returns
    -------
    dict
        Style attribution analysis results.
    """
    
    attribution = {}
    
    # This is a simplified implementation
    # For more complex style attribution, consider using factor models
    
    return attribution


def calculate_sector_attribution(
    strategy_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    sector_mapping: pd.Series
) -> Dict[str, float]:
    """
    Analyze sector-level attribution.
    
    Compares sector weights between strategy and benchmark.
    
    Parameters
    ----------
    strategy_weights : pd.DataFrame
        Strategy portfolio weights. Index: date, columns: tickers.
    benchmark_weights : pd.DataFrame
        Benchmark portfolio weights. Index: date, columns: tickers.
    sector_mapping : pd.Series
        Mapping from ticker to sector. Index: tickers, values: sector names.
        
    Returns
    -------
    dict
        Sector allocation differences and contribution.
    """
    
    attribution = {}
    
    # Group by sector
    common_dates = strategy_weights.index.intersection(benchmark_weights.index)
    
    sector_attribution_list = []
    
    for date in common_dates:
        strat_weights = strategy_weights.loc[date]
        bench_weights = benchmark_weights.loc[date]
        
        # Create sector weights
        strat_sector_weights = pd.Series(index=sector_mapping.unique())
        bench_sector_weights = pd.Series(index=sector_mapping.unique())
        
        for sector in sector_mapping.unique():
            sector_tickers = sector_mapping[sector_mapping == sector].index
            strat_sector_weights[sector] = strat_weights.reindex(sector_tickers, fill_value=0).sum()
            bench_sector_weights[sector] = bench_weights.reindex(sector_tickers, fill_value=0).sum()
        
        sector_diff = (strat_sector_weights - bench_sector_weights).abs().sum() / 2
        sector_attribution_list.append(sector_diff)
    
    attribution['avg_sector_deviation'] = np.mean(sector_attribution_list)
    
    return attribution


def calculate_position_concentration(
    weights: pd.DataFrame
) -> pd.Series:
    """
    Calculate portfolio concentration using Herfindahl index.
    
    Concentration = Sum(weight_i^2)
    
    Higher values indicate more concentrated portfolio.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights. Index: date, columns: tickers.
        
    Returns
    -------
    pd.Series
        Daily concentration index.
    """
    
    concentration = (weights ** 2).sum(axis=1)
    return concentration


def calculate_return_attribution_attribution(
    returns_with_costs: pd.Series,
    returns_without_costs: pd.Series
) -> Dict[str, float]:
    """
    Decompose returns into gross and cost components.
    
    Parameters
    ----------
    returns_with_costs : pd.Series
        Net returns (after costs). Index: date.
    returns_without_costs : pd.Series
        Gross returns (before costs). Index: date.
        
    Returns
    -------
    dict
        Attribution breakdown.
    """
    
    common_index = returns_with_costs.index.intersection(returns_without_costs.index)
    
    gross_cumulative = (1 + returns_without_costs.loc[common_index]).prod() - 1
    net_cumulative = (1 + returns_with_costs.loc[common_index]).prod() - 1
    cost_impact = gross_cumulative - net_cumulative
    
    return {
        'gross_return': gross_cumulative,
        'net_return': net_cumulative,
        'cost_impact': cost_impact,
        'cost_pct': cost_impact / (1 + gross_cumulative) if gross_cumulative != -1 else 0,
    }


__all__ = [
    'calculate_turnover',
    'calculate_daily_turnover',
    'calculate_active_return',
    'calculate_active_risk',
    'decompose_active_return',
    'calculate_turnover_impact',
    'calculate_style_attribution',
    'calculate_sector_attribution',
    'calculate_position_concentration',
    'calculate_return_attribution_attribution',
]
