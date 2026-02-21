"""
Pipeline Manager - Core Data Processing and Orchestration
Handles dataset preparation, resampling, and full workflow execution.
"""

from __future__ import annotations

from typing import Optional, Literal, Dict, Any
import warnings

import numpy as np
import pandas as pd

from qfrsch.utils.helpers import resample_to_freq, shift_labels_by_freq


class PipelineManager:
    """
    Central pipeline orchestrator for factor-to-portfolio transformation.
    
    Manages:
    - Multi-frequency rebalancing (D, W, M, Q, Y)
    - Label generation with proper time alignment
    - Feature-label pairing for model training
    - Full inference pipeline
    
    Parameters
    ----------
    rebalance_freq : {'D', 'W', 'M', 'Q', 'Y'}, default='D'
        Rebalancing frequency.
        - 'D': Daily
        - 'W': Weekly (last trading day)
        - 'M': Monthly (last trading day)
        - 'Q': Quarterly (last trading day)
        - 'Y': Yearly (last trading day)
    prediction_period : int, default=1
        Number of periods ahead for prediction (in rebalance_freq units).
        Example: For daily rebalancing, prediction_period=1 predicts 1-day return.
    
    Attributes
    ----------
    rebalance_freq : str
        Rebalancing frequency.
    prediction_period : int
        Prediction horizon.
    rebalance_dates_ : pd.DatetimeIndex or None
        Computed rebalancing dates (computed after make_dataset).
        
    Examples
    --------
    >>> pm = PipelineManager(rebalance_freq='W', prediction_period=1)
    >>> X, y = pm.make_dataset(factors_df, returns_df)
    >>> weights = pm.run_pipeline(X, y, model, strategy)
    
    Notes
    -----
    The key design principle: factors from date T predict returns from T+1 to T+k.
    This prevents look-ahead bias and ensures proper label alignment.
    """
    
    def __init__(
        self,
        rebalance_freq: Literal['D', 'W', 'M', 'Q', 'Y'] = 'D',
        prediction_period: int = 1
    ):
        if rebalance_freq not in ['D', 'W', 'M', 'Q', 'Y']:
            raise ValueError(
                f"rebalance_freq must be 'D', 'W', 'M', 'Q', or 'Y', "
                f"got {rebalance_freq}"
            )
        if prediction_period < 1:
            raise ValueError(f"prediction_period must be >= 1, got {prediction_period}")
        
        self.rebalance_freq = rebalance_freq
        self.prediction_period = prediction_period
        self.rebalance_dates_ = None
    
    def make_dataset(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        dropna: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix X and label vector y with proper alignment.
        
        Converts factors and returns to rebalancing frequency, then computes
        forward-looking returns as labels. Handles NaN and ensures no future
        information leakage.
        
        Parameters
        ----------
        factors : pd.DataFrame
            Factor data with MultiIndex (date, ticker) and feature columns.
            Example columns: ['factor1', 'factor2', ...]
        returns : pd.DataFrame
            Daily returns with MultiIndex (date, ticker).
            Example columns: ['returns'] or daily returns name.
        dropna : bool, default=True
            If True, remove samples with any missing values.
            
        Returns
        -------
        X : pd.DataFrame
            Feature matrix, indexed by rebalance dates.
            MultiIndex: (rebalance_date, ticker)
            Columns: factor names from factors dataframe
        y : pd.Series
            Target labels (forward returns), same index as X.
            Values: cumulative returns from T+1 to T+prediction_period.
            
        Raises
        ------
        ValueError
            If factors or returns have incompatible structure.
            
        Notes
        -----
        Critical time alignment:
        1. Resample factors to rebalance_freq (last date per period)
        2. Compute cumulative forward returns
        3. Shift labels by prediction_period to align with factors
        4. Remove NaN values
        
        Examples
        --------
        >>> # Daily data, weekly rebalancing, predict 1-week-ahead returns
        >>> X, y = pm.make_dataset(factors, returns)
        >>> print(X.shape)
        (252, 50)  # 252 rebalance dates, 50 factors per stock
        """
        # Validate inputs
        if not isinstance(factors.index, pd.MultiIndex) or factors.index.names != ['date', 'ticker']:
            raise ValueError("factors must have MultiIndex (date, ticker)")
        if not isinstance(returns.index, pd.MultiIndex) or returns.index.names != ['date', 'ticker']:
            raise ValueError("returns must have MultiIndex (date, ticker)")
        
        # Ensure datetime index by extracting unique dates per level
        # and setting levels with unique values
        factors = factors.copy()
        unique_dates_factors = pd.to_datetime(
            factors.index.get_level_values('date').unique()
        ).sort_values()
        factors.index = factors.index.set_levels(unique_dates_factors, level='date')
        
        returns = returns.copy()
        unique_dates_returns = pd.to_datetime(
            returns.index.get_level_values('date').unique()
        ).sort_values()
        returns.index = returns.index.set_levels(unique_dates_returns, level='date')
        
        # Resample factors to rebalance frequency
        X = resample_to_freq(factors, self.rebalance_freq, agg='last')
        
        # Compute cumulative forward returns
        returns_col = returns.columns[0] if isinstance(returns, pd.DataFrame) else 'returns'
        
        # Compute forward returns for each ticker separately, maintaining MultiIndex
        forward_returns_list = []
        for ticker in returns.index.get_level_values('ticker').unique():
            ticker_data = returns.loc[(slice(None), ticker), :]
            ticker_data_ts = ticker_data.droplevel('ticker')
            ticker_fwd = self._compute_forward_returns(ticker_data_ts[returns_col])
            
            # Restore MultiIndex
            ticker_fwd.index = pd.MultiIndex.from_arrays(
                [ticker_fwd.index, [ticker] * len(ticker_fwd)],
                names=['date', 'ticker']
            )
            forward_returns_list.append(ticker_fwd)
        
        forward_returns = pd.concat(forward_returns_list, axis=0).sort_index()
        forward_returns.name = returns_col
        
        # Shift returns by prediction_period (move label from T+period to T)
        y = shift_labels_by_freq(
            forward_returns,
            self.rebalance_freq,
            shift_periods=self.prediction_period
        )
        
        # Align X and y indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Store rebalance dates for reference
        self.rebalance_dates_ = X.index.get_level_values('date').unique()
        
        # Drop NaN if requested
        if dropna:
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
        
        return X, y
    
    def _compute_forward_returns(
        self,
        returns_series: pd.Series,
        n_periods: Optional[int] = None
    ) -> pd.Series:
        """
        Compute cumulative forward returns for prediction_period.
        
        Parameters
        ----------
        returns_series : pd.Series
            Daily returns indexed by date.
        n_periods : int, optional
            Number of periods to accumulate. If None, uses self.prediction_period.
            
        Returns
        -------
        forward_returns : pd.Series
            Cumulative returns, same index as input.
            First n_periods-1 values are NaN.
        """
        n_periods = n_periods or self.prediction_period
        
        # Convert daily returns to cumulative
        # (1 + r_t) * (1 + r_{t+1}) * ... - 1
        forward_cum = (1 + returns_series).rolling(
            window=n_periods, min_periods=n_periods
        ).apply(lambda x: np.prod(x) - 1, raw=False)
        
        return forward_cum
    
    def run_pipeline(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        model: Any,
        strategy: Any,
        train_start: Optional[pd.Timestamp] = None,
        train_end: Optional[pd.Timestamp] = None,
        test_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Execute full prediction pipeline: prepare → train → predict → weight.
        
        Parameters
        ----------
        factors : pd.DataFrame
            Factor data with MultiIndex (date, ticker).
        returns : pd.DataFrame
            Returns data with MultiIndex (date, ticker).
        model : ModelWrapper
            Fitted or unfitted model for prediction.
        strategy : TopNStrategy or OptimizedStrategy
            Strategy for weight generation.
        train_start : pd.Timestamp, optional
            Training data start date. If None, uses all available data.
        train_end : pd.Timestamp, optional
            Training data end date (inclusive). If None, uses all available data.
        test_date : pd.Timestamp, optional
            Inference date. If None, uses most recent date.
            
        Returns
        -------
        weights : pd.DataFrame
            Target portfolio weights with MultiIndex (date, ticker).
            Columns: ['target_weight']
            
        Examples
        --------
        >>> from qfrsch.models.base import ModelWrapper
        >>> from qfrsch.pipeline.strategies import TopNStrategy
        >>> model = ModelWrapper(model_type='ridge', alpha=0.1)
        >>> strategy = TopNStrategy(n_stocks=50, long_only=False)
        >>> weights = pm.run_pipeline(factors, returns, model, strategy)
        """
        # Prepare dataset
        X, y = self.make_dataset(factors, returns, dropna=True)
        
        # Determine date ranges
        test_date = test_date or X.index.get_level_values('date').max()
        
        # Filter training data
        train_mask = X.index.get_level_values('date') <= test_date
        if train_start is not None:
            train_mask &= X.index.get_level_values('date') >= train_start
        if train_end is not None:
            train_mask &= X.index.get_level_values('date') <= train_end
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        # Get test data (most recent rebalance date)
        test_mask = X.index.get_level_values('date') == test_date
        X_test = X[test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
            warnings.warn(f"No data available for training or test at {test_date}")
            return pd.DataFrame()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Generate predictions
        scores = model.predict(X_test)
        scores_series = pd.Series(
            scores,
            index=X_test.index.get_level_values('ticker')
        )
        
        # Convert scores to weights
        weights = strategy.generate_weights(scores_series)
        
        # Add date to index if missing
        if not isinstance(weights.index, pd.MultiIndex):
            weights.index = pd.MultiIndex.from_product(
                [[test_date], weights.index],
                names=['date', 'ticker']
            )
        
        return weights
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PipelineManager(rebalance_freq='{self.rebalance_freq}', "
            f"prediction_period={self.prediction_period})"
        )
