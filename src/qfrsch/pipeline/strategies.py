"""
Weight Generation Strategies
Converts model predictions (scores) into target portfolio weights.
"""

from __future__ import annotations

from typing import Optional, Literal
import warnings

import numpy as np
import pandas as pd


class TopNStrategy:
    """
    Top-N selection strategy for portfolio construction.
    
    Selects top N stocks by score for long positions and optionally
    bottom N for short positions. Allocates equal weight to each position.
    
    Parameters
    ----------
    n_stocks : int
        Number of top/bottom stocks to select. Must be positive.
    long_only : bool, default=True
        If True, only long positions (no shorts).
        If False, long top N and short bottom N.
    equal_weight : bool, default=True
        If True, equal-weight portfolio within selected stocks.
        If False, weights proportional to scores.
        
    Attributes
    ----------
    n_stocks : int
        Number of stocks selected.
    long_only : bool
        Whether strategy is long-only.
    equal_weight : bool
        Whether to use equal weighting.
        
    Examples
    --------
    >>> strategy = TopNStrategy(n_stocks=50, long_only=False)
    >>> scores = pd.Series([0.5, -0.3, 0.8, ...], index=['AAPL', 'MSFT', ...])
    >>> weights = strategy.generate_weights(scores)
    """
    
    def __init__(
        self,
        n_stocks: int,
        long_only: bool = True,
        equal_weight: bool = True
    ):
        if n_stocks <= 0:
            raise ValueError(f"n_stocks must be positive, got {n_stocks}")
        
        self.n_stocks = n_stocks
        self.long_only = long_only
        self.equal_weight = equal_weight
    
    def generate_weights(self, scores: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """
        Generate portfolio weights from scores.
        
        Parameters
        ----------
        scores : pd.Series or pd.DataFrame
            Prediction scores. If MultiIndex (date, ticker), processes each
            date separately. If single-level index (ticker), processes as
            single cross-section.
            Shape: (n_stocks,) or (n_stocks, n_dates).
            
        Returns
        -------
        weights : pd.DataFrame
            Target portfolio weights with same index structure.
            Index: MultiIndex (date, ticker) if input has dates.
            Columns: ['target_weight']
            Sum of weights per date: 1.0 (long_only) or 0.0 (long_short).
            
        Notes
        -----
        Missing scores are treated as NaN and excluded from weight calculation.
        
        Examples
        --------
        >>> # Portfolio scores for single date
        >>> scores = pd.Series([0.8, 0.5, -0.3, -0.6], 
        ...                    index=['AAPL', 'MSFT', 'IBM', 'XOM'])
        >>> weights = TopNStrategy(n_stocks=1).generate_weights(scores)
        >>> print(weights)
                  target_weight
        AAPL            0.5
        MSFT            0.5
        IBM            -0.5
        XOM            -0.5
        """
        # Handle both single date and multi-date formats
        if isinstance(scores, pd.DataFrame):
            # Each column is a date
            return pd.concat(
                [self._generate_weights_single(scores[col]) for col in scores.columns],
                keys=scores.columns,
                names=['date', 'ticker']
            )
        elif isinstance(scores.index, pd.MultiIndex):
            # MultiIndex (date, ticker) format
            weights_list = []
            for date, group in scores.groupby(level='date'):
                w = self._generate_weights_single(group.droplevel(0))
                weights_list.append(w)
            
            if weights_list:
                return pd.concat(weights_list, keys=scores.groupby(level='date').groups.keys(),
                               names=['date', 'ticker'])
            else:
                return pd.DataFrame()
        else:
            # Single cross-section
            return self._generate_weights_single(scores)
    
    def _generate_weights_single(self, scores: pd.Series) -> pd.DataFrame:
        """
        Generate weights for single date.
        
        Parameters
        ----------
        scores : pd.Series
            Scores indexed by ticker.
            
        Returns
        -------
        weights : pd.DataFrame
            Weights with columns ['target_weight'].
        """
        # Remove NaN scores
        valid_scores = scores.dropna()
        
        if len(valid_scores) == 0:
            return pd.DataFrame({'target_weight': scores * np.nan})
        
        # Sort by score (descending)
        sorted_scores = valid_scores.sort_values(ascending=False)
        
        # Select top and bottom N
        n_select = min(self.n_stocks, len(sorted_scores))
        top_n = sorted_scores.head(n_select).index
        
        # Initialize weights to zero
        weights = pd.Series(0.0, index=valid_scores.index)
        
        # Assign weights to top N (long)
        if self.equal_weight:
            weights[top_n] = 1.0 / n_select
        else:
            # Proportional to scores
            top_scores = sorted_scores.head(n_select)
            weights[top_n] = top_scores / top_scores.abs().sum()
        
        # Assign weights to bottom N (short) if not long_only
        if not self.long_only:
            bottom_n = sorted_scores.tail(n_select).index
            if self.equal_weight:
                weights[bottom_n] = -1.0 / n_select
            else:
                bottom_scores = sorted_scores.tail(n_select)
                weights[bottom_n] = -bottom_scores / bottom_scores.abs().sum()
        else:
            # Long-only: normalize to sum to 1
            weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        return pd.DataFrame({'target_weight': weights})
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TopNStrategy(n_stocks={self.n_stocks}, "
            f"long_only={self.long_only}, equal_weight={self.equal_weight})"
        )


class OptimizedStrategy:
    """
    Mean-Variance Optimization (MVO) and Risk Parity strategies.
    
    Converts scores to portfolio weights using optimization algorithms.
    Requires covariance matrix as input.
    
    Parameters
    ----------
    method : {'mvo', 'riskparity'}, default='mvo'
        Optimization method.
        - 'mvo': Mean-Variance Optimization (Markowitz)
        - 'riskparity': Equal Risk Contribution
    target_return : float, optional
        Target return constraint for MVO. If None, uses score mean.
    min_weight : float, default=-0.1
        Minimum weight allowed per asset (negative for short).
    max_weight : float, default=0.1
        Maximum weight allowed per asset.
    leverage : float, default=1.0
        Portfolio leverage (sum of abs weights).
        
    Examples
    --------
    >>> strategy = OptimizedStrategy(method='mvo', leverage=1.0)
    >>> weights = strategy.generate_weights(scores, cov_matrix)
    """
    
    def __init__(
        self,
        method: Literal['mvo', 'riskparity'] = 'mvo',
        target_return: Optional[float] = None,
        min_weight: float = -0.1,
        max_weight: float = 0.1,
        leverage: float = 1.0
    ):
        if method not in ['mvo', 'riskparity']:
            raise ValueError(f"method must be 'mvo' or 'riskparity', got {method}")
        
        self.method = method
        self.target_return = target_return
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.leverage = leverage
    
    def generate_weights(
        self,
        scores: pd.Series | pd.DataFrame,
        cov_matrix: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame:
        """
        Generate optimized portfolio weights.
        
        Parameters
        ----------
        scores : pd.Series or pd.DataFrame
            Prediction scores used as expected returns.
        cov_matrix : pd.DataFrame or np.ndarray
            Covariance matrix of asset returns.
            
        Returns
        -------
        weights : pd.DataFrame
            Optimized weights with columns ['target_weight'].
            
        Notes
        -----
        This is a simple MVO/RP implementation. For production use,
        consider using specialized packages like cvxpy or skfolio.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("scipy required for OptimizedStrategy")
        
        scores = scores.values if isinstance(scores, pd.Series) else scores
        cov_array = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
        
        n_assets = len(scores)
        
        if self.method == 'mvo':
            return self._mvo(scores, cov_array)
        else:  # riskparity
            return self._riskparity(scores, cov_array)
    
    def _mvo(self, returns: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
        """Mean-Variance Optimization."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("scipy required for optimization")
        
        n = len(returns)
        
        def portfolio_variance(w):
            return w @ cov @ w
        
        def portfolio_return(w):
            return -w @ returns  # Minimize negative return
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - self.leverage}]
        
        # Bounds
        bounds = tuple([(self.min_weight, self.max_weight)] * n)
        
        # Initial guess (equal weight)
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            lambda w: portfolio_variance(w) + 0.01 * portfolio_return(w),
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x if result.success else w0
        return pd.DataFrame({'target_weight': weights})
    
    def _riskparity(self, returns: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
        """Risk Parity (Equal Risk Contribution)."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("scipy required for optimization")
        
        n = len(returns)
        
        def risk_parity_objective(w):
            portfolio_vol = np.sqrt(w @ cov @ w)
            if portfolio_vol < 1e-6:
                return 1e6
            marginal_rc = (cov @ w) / portfolio_vol
            return np.sum((marginal_rc - (self.leverage / n)) ** 2)
        
        # Bounds
        bounds = tuple([(self.min_weight, self.max_weight)] * n)
        w0 = np.ones(n) / n
        
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds
        )
        
        weights = result.x if result.success else w0
        return pd.DataFrame({'target_weight': weights})
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptimizedStrategy(method='{self.method}', "
            f"leverage={self.leverage})"
        )
