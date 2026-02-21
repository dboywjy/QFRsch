"""
Alpha Zoo Module
Collection of example alpha factors.
Demonstrates how to compose operators to build research-grade factors.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import FactorBase
from .operators import (
    ts_rank, ts_mean, ts_std, cs_rank, op_log, op_abs, op_sign,
    create_multiindex_series, ts_delay, ts_delta
)
from .processor import winsorize, standardize


# ==================== Simple Examples ====================


class Alpha001(FactorBase):
    """
    Simple composed alpha: cs_rank(ts_rank(log(close), 10))
    
    Represents ranking of recent price momentum in cross-section.
    
    Logic:
    1. Take log of close prices
    2. Compute 10-day rolling rank within each ticker
    3. Rank across all tickers each day
    
    Intuition: Momentum signal that's robust to scale
    """
    
    def __init__(self, name: str = "alpha_001", 
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {'window': 10}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Alpha001.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'close' column
        """
        self._validate_data(df)
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        window = self.params.get('window', 10)
        
        # Create MultiIndex series
        close_series = create_multiindex_series(df, 'close')
        
        # Step 1: Log transform
        log_close = op_log(close_series)
        
        # Step 2: Time-series rank within each ticker
        ts_ranked = ts_rank(log_close, window=window)
        
        # Step 3: Cross-sectional rank each day
        cs_ranked = cs_rank(ts_ranked)
        
        # Post-process
        return self._post_process(cs_ranked, df)
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Standard post-processing"""
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        return standardize(series, method='zscore', group_key='date')


class Alpha002(FactorBase):
    """
    Mean reversion: -ts_rank(ts_std(returns, 20), 5)
    
    Logic:
    1. Compute daily returns
    2. Calculate 20-day rolling volatility
    3. Rank volatility over 5 days
    4. Negate (lower volatility = higher alpha)
    
    Intuition: High volatility likely to revert downward
    (Low volatility regime likely to persist, stocks perform better)
    """
    
    def __init__(self, name: str = "alpha_002",
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {'vol_window': 20, 'rank_window': 5}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Alpha002.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'close' column
        """
        self._validate_data(df)
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        vol_window = self.params.get('vol_window', 20)
        rank_window = self.params.get('rank_window', 5)
        
        # Create indexed series
        indexed = df.set_index(['date', 'ticker'])
        
        # Step 1: Compute returns
        returns = indexed['close'].groupby('ticker').pct_change()
        
        # Step 2: Rolling volatility
        volatility = ts_std(returns, window=vol_window)
        
        # Step 3: Rank volatility (lower = better)
        vol_ranked = ts_rank(volatility, window=rank_window)
        
        # Step 4: Negate
        alpha = -vol_ranked
        
        # Post-process
        return self._post_process(alpha, df)
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        return standardize(series, method='zscore', group_key='date')


class Alpha003(FactorBase):
    """
    Mean reversion by size: -sign(delta) * rank(delta)
    
    Logic:
    1. Compute daily returns (delta)
    2. Cross-sectionally rank absolute returns
    3. Negate if returns were positive (mean reversion)
    
    Intuition: 
    - Stocks that went up tend to go down (and vice versa)
    - Effect is stronger for extreme moves
    - Size-neutral within day
    """
    
    def __init__(self, name: str = "alpha_003",
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Alpha003.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'close' column
        """
        self._validate_data(df)
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        # Create indexed series
        close_series = create_multiindex_series(df, 'close')
        
        # Step 1: Daily returns
        returns = ts_delta(close_series, window=1)
        
        # Step 2: Negate sign of returns
        sign = op_sign(returns)
        
        # Step 3: Cross-sectional rank
        cs_ranked = cs_rank(op_abs(returns))
        
        # Step 4: Combine: negate winners
        alpha = -sign * cs_ranked
        
        # Post-process
        return self._post_process(alpha, df)
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        return standardize(series, method='zscore', group_key='date')


# ==================== Intermediate Examples ====================


class Alpha004(FactorBase):
    """
    Earnings surprise proxy using volatility change.
    
    -ts_corr(rank(open), rank(volume), 10)
    
    Logic:
    1. Rank opening prices within each stock (20-day window)
    2. Rank volumes
    3. Compute rolling correlation
    4. Negate (low correlation between price and volume = surprise?)
    
    Note: In practice, would use actual earnings data.
    This is a proxy using price/volume dynamics.
    """
    
    def __init__(self, name: str = "alpha_004",
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {'corr_window': 10, 'rank_window': 20}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Alpha004.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'open' and 'volume' columns
        """
        self._validate_data(df)
        
        required = {'open', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame missing: {required - set(df.columns)}")
        
        corr_window = self.params.get('corr_window', 10)
        rank_window = self.params.get('rank_window', 20)
        
        # Create indexed series
        indexed = df.set_index(['date', 'ticker'])
        
        # Rank open and volume within each ticker
        open_ranked = ts_rank(indexed['open'], window=rank_window)
        vol_ranked = ts_rank(indexed['volume'], window=rank_window)
        
        # Compute rolling correlation
        from .operators import ts_corr
        alpha = -ts_corr(open_ranked, vol_ranked, window=corr_window)
        
        # Post-process
        return self._post_process(alpha, df)
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        return standardize(series, method='zscore', group_key='date')


# ==================== Advanced Examples ====================


class AlphaCombo(FactorBase):
    """
    Combined multi-factor alpha using weighted average.
    
    Demonstrates how to combine multiple alpha signals.
    
    Formula: 
    Alpha = w1*Alpha001 + w2*Alpha002 + w3*Alpha003
    
    Parameters
    ----------
    weights : dict, optional
        Dict mapping factor_name -> weight
        Default: equal weights
    """
    
    def __init__(self, name: str = "alpha_combo",
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {
                'weights': {
                    'alpha_001': 1.0,
                    'alpha_002': 1.0,
                    'alpha_003': 1.0
                }
            }
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute combined alpha factor.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must have all required columns for component alphas
        """
        self._validate_data(df)
        
        # Compute component alphas
        a1 = Alpha001(params={'window': 10}).compute(df)
        a2 = Alpha002(params={'vol_window': 20, 'rank_window': 5}).compute(df)
        a3 = Alpha003().compute(df)
        
        # Get weights
        weights = self.params.get('weights', {
            'alpha_001': 1.0,
            'alpha_002': 1.0,
            'alpha_003': 1.0
        })
        
        w1 = weights.get('alpha_001', 1.0)
        w2 = weights.get('alpha_002', 1.0)
        w3 = weights.get('alpha_003', 1.0)
        
        # Average and normalize
        total_weight = w1 + w2 + w3
        alpha = (w1 * a1 + w2 * a2 + w3 * a3) / total_weight
        
        # Final standardization
        return standardize(alpha, method='zscore', group_key='date')


# ==================== Factor Registry ====================


ALPHA_REGISTRY = {
    'alpha_001': Alpha001,
    'alpha_002': Alpha002,
    'alpha_003': Alpha003,
    'alpha_004': Alpha004,
    'alpha_combo': AlphaCombo,
}


def get_alpha(name: str, **kwargs) -> FactorBase:
    """
    Factory function to instantiate alpha factors by name.
    
    Parameters
    ----------
    name : str
        Name of alpha factor (must be in ALPHA_REGISTRY)
    **kwargs
        Arguments passed to factor constructor
        
    Returns
    -------
    FactorBase
        Instantiated factor
        
    Raises
    ------
    ValueError
        If factor name not found in registry
        
    Examples
    --------
    >>> alpha = get_alpha('alpha_001', params={'window': 15})
    >>> result = alpha.compute(df)
    """
    if name not in ALPHA_REGISTRY:
        available = ', '.join(ALPHA_REGISTRY.keys())
        raise ValueError(
            f"Unknown alpha: {name}\nAvailable: {available}"
        )
    
    return ALPHA_REGISTRY[name](**kwargs)


def list_alphas() -> list:
    """
    List all available alpha factors.
    
    Returns
    -------
    list
        Sorted list of factor names
    """
    return sorted(ALPHA_REGISTRY.keys())
