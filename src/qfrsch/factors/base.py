"""
Factor Base Class Module
Provides abstract base class for factor development.
All factors must inherit from FactorBase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class FactorBase(ABC):
    """
    Abstract base class for all factors in QFRsch framework.
    
    This class defines the interface that all factors must implement.
    Factors are the fundamental building blocks for quantitative strategies,
    computing derived values from market data.
    
    Attributes
    ----------
    name : str
        Unique identifier for this factor instance
    params : dict
        Configuration parameters for the factor computation
        
    Notes
    -----
    Subclasses must implement the compute() method.
    All computation must work with DataFrames containing 'date' and 'ticker' columns.
    
    Examples
    --------
    >>> class MomentumFactor(FactorBase):
    ...     def compute(self, df: pd.DataFrame) -> pd.Series:
    ...         return df.groupby('ticker')['close'].pct_change(self.params['period'])
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a factor instance.
        
        Parameters
        ----------
        name : str
            Unique identifier for this factor
        params : dict, optional
            Configuration parameters for factor computation.
            If None, an empty dict is used.
            
        Raises
        ------
        TypeError
            If name is not a string
        TypeError
            If params is not a dict or None
            
        Examples
        --------
        >>> factor = MomentumFactor(name="momentum_5d", params={"period": 5})
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        
        if params is not None and not isinstance(params, dict):
            raise TypeError(f"params must be a dict or None, got {type(params).__name__}")
        
        self.name = name
        self.params = params if params is not None else {}
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute factor values from market data.
        
        This is the core method that subclasses must implement.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data with columns ['date', 'ticker', ...].
            Must contain relevant OHLC or other market data columns.
            The 'date' column must be of datetime type.
            The 'ticker' column must be of string type.
            
        Returns
        -------
        pd.Series
            Computed factor values aligned with input DataFrame.
            Index should match the input DataFrame index.
            Values can be numeric or bool (e.g., for signals).
            
        Raises
        ------
        ValueError
            If required columns are missing or data validation fails
            
        Notes
        -----
        - Output Series should have the same length as input DataFrame
        - Missing values (NaN) are acceptable and will be handled by downstream processing
        - All nested operator calls must be supported (e.g., cs_rank(ts_mean(...)))
        """
        pass
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that input DataFrame has required structure and data types.
        
        This method checks:
        - Presence of 'date' and 'ticker' columns
        - 'date' column is of datetime type
        - 'ticker' column is of string/object type
        - DataFrame is not empty
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
            
        Raises
        ------
        ValueError
            If validation fails with descriptive error message
            
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=10),
        ...     'ticker': ['AAPL'] * 10,
        ...     'close': [150.0] * 10
        ... })
        >>> factor._validate_data(df)  # No error raised
        
        >>> df_bad = pd.DataFrame({'close': [150.0] * 10})
        >>> factor._validate_data(df_bad)  # Raises ValueError
        """
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")
        
        # Check required columns
        required_columns = {'date', 'ticker'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"DataFrame missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Validate 'date' column
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError(
                f"'date' column must be datetime type, got {df['date'].dtype}. "
                f"Try: df['date'] = pd.to_datetime(df['date'])"
            )
        
        # Validate 'ticker' column
        if not pd.api.types.is_object_dtype(df['ticker']) and not pd.api.types.is_string_dtype(df['ticker']):
            raise ValueError(
                f"'ticker' column must be string/object type, got {df['ticker'].dtype}"
            )
        
        # Check for NaN values in critical columns
        if df['date'].isna().any():
            raise ValueError("'date' column contains NaN values")
        if df['ticker'].isna().any():
            raise ValueError("'ticker' column contains NaN values")
    
    def __repr__(self) -> str:
        """
        Return string representation of factor.
        
        Returns
        -------
        str
            Formatted representation showing factor name and parameters
        """
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        if params_str:
            return f"{self.__class__.__name__}(name='{self.name}', params={{{params_str}}})"
        return f"{self.__class__.__name__}(name='{self.name}')"
