"""
Style Factors Module
Implements common risk factors for equity markets.
All style factors inherit from FactorBase and are auto-processed.
"""

from __future__ import annotations

from typing import Optional, Union, Dict, Any
import pandas as pd
import numpy as np

from .base import FactorBase
from .operators import ts_std, ts_delay, ts_mean
from .processor import winsorize, standardize


# ==================== Industry Factor ====================


class IndustryFactor(FactorBase):
    """
    Industry classification factor.
    
    Maps tickers to industry groups and generates dummy variables.
    Can be used as risk factors in neutral ization.
    
    Attributes
    ----------
    industry_map : dict
        Dictionary mapping ticker -> industry code
    industry_names : dict, optional
        Dictionary mapping industry code -> industry name (for readability)
        
    Notes
    -----
    Typical workflow:
    1. Create IndustryFactor and populate industry_map
    2. Call compute() to get main industry column
    3. Call get_dummies() to get one-hot encoded factors
    
    Examples
    --------
    >>> industry = IndustryFactor(name="industry")
    >>> industry.industry_map = {
    ...     'AAPL': 0,  # Tech
    ...     'JPM': 1,   # Finance
    ...     'XOM': 2    # Energy
    ... }
    >>> # For use in neutralization:
    >>> industry_dummies = industry.get_dummies(df)
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, params)
        self.industry_map: Dict[str, int] = {}
        self.industry_names: Dict[int, str] = {}
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Map tickers to industry codes.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'date', 'ticker' columns
            
        Returns
        -------
        pd.Series
            Industry code for each row
        """
        self._validate_data(df)
        return df['ticker'].map(self.industry_map)
    
    def get_dummies(self, df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
        """
        Generate one-hot encoded industry dummies.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with 'ticker' column
        drop_first : bool, optional
            Drop first category to avoid multicollinearity (default: True)
            
        Returns
        -------
        pd.DataFrame
            One-hot encoded industry dummies with MultiIndex (date, ticker)
            Column names are 'ind_<industry_code>'
            
        Examples
        --------
        >>> industry_dummies = industry.get_dummies(df)
        >>> # Use in neutralization
        >>> alpha_neutral = neutralize(alpha, industry_dummies)
        """
        industry_series = self.compute(df)
        dummies = pd.get_dummies(
            industry_series,
            prefix='ind',
            drop_first=drop_first,
            dtype=float
        )
        
        # Ensure MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            dummies.index = df.set_index(['date', 'ticker']).index
        else:
            dummies.index = df.index
        
        return dummies


# ==================== Continuous Style Factors ====================


class SizeFactor(FactorBase):
    """
    Market capitalization (Size) factor.
    
    Definition: log(market_cap)
    
    Characteristics:
    - Captures company size effects
    - Typically negatively correlates with returns (small-cap premium)
    - Requires 'market_cap' column in input DataFrame
    
    Notes
    -----
    Input DataFrame must contain 'market_cap' column in currency units.
    """
    
    def __init__(self, name: str = "size", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute log market cap.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'market_cap' column
            
        Returns
        -------
        pd.Series
            Log market cap values
        """
        self._validate_data(df)
        
        if 'market_cap' not in df.columns:
            raise ValueError("DataFrame must contain 'market_cap' column")
        
        # Compute log market cap
        result = np.log(df['market_cap'])
        
        # Post-process: winsorize and standardize
        result = self._post_process(result, df)
        return result
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Standard post-processing for style factors"""
        # Create MultiIndex if needed for processing
        if isinstance(df.index, pd.MultiIndex):
            indexed_series = series
        else:
            indexed_series = series.copy()
            indexed_series.index = df.set_index(['date', 'ticker']).index
        
        # Winsorize
        indexed_series = winsorize(indexed_series, method='quantile',
                                  limits=(0.01, 0.99), group_key='date')
        
        # Standardize
        indexed_series = standardize(indexed_series, method='zscore',
                                    group_key='date')
        
        return indexed_series


class VolatilityFactor(FactorBase):
    """
    Realized volatility (Volatility) factor.
    
    Definition: Rolling 20-day standard deviation of returns
    
    Characteristics:
    - Captures market risk perception
    - Higher volatility typically associated with lower returns (low-vol premium)
    - Requires 'close' price column
    """
    
    def __init__(self, name: str = "volatility", 
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {'window': 20}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute rolling volatility of returns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'close' column
            
        Returns
        -------
        pd.Series
            Rolling volatility values
        """
        self._validate_data(df)
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        window = self.params.get('window', 20)
        
        # Create MultiIndex for processing
        indexed = df.set_index(['date', 'ticker'])
        
        # Compute returns then rolling std
        returns = indexed['close'].groupby('ticker').pct_change()
        volatility = ts_std(returns, window=window)
        
        # Post-process
        volatility = self._post_process(volatility, df)
        return volatility
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        series = standardize(series, method='zscore', group_key='date')
        return series


class MomentumFactor(FactorBase):
    """
    Momentum factor.
    
    Definition: Cumulative return over past 12 months, excluding recent 1 month
    Standard in Fama-French research.
    
    Characteristics:
    - Captures trend effects
    - Positive: trending up (momentum)
    - Requires 'close' price column and sufficient history
    
    Parameters
    ----------
    lookback : int, optional
        Total lookback window in trading days (default: 252, ~1 year)
    skip_recent : int, optional
        Days to skip from most recent (default: 20, ~1 month)
    """
    
    def __init__(self, name: str = "momentum",
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {'lookback': 252, 'skip_recent': 20}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute momentum factor.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'close' column
            
        Returns
        -------
        pd.Series
            Momentum values (log returns)
        """
        self._validate_data(df)
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        lookback = self.params.get('lookback', 252)
        skip = self.params.get('skip_recent', 20)
        
        indexed = df.set_index(['date', 'ticker'])
        
        # Compute returns: price_t-skip / price_t-lookback
        price_past = ts_delay(indexed['close'], window=lookback)
        price_recent = ts_delay(indexed['close'], window=skip)
        
        momentum = np.log(price_recent / price_past)
        
        # Post-process
        momentum = self._post_process(momentum, df)
        return momentum
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        series = standardize(series, method='zscore', group_key='date')
        return series


class LiquidityFactor(FactorBase):
    """
    Liquidity factor.
    
    Definition: Average turnover over past 20 days
    
    Characteristics:
    - Higher turnover: more liquid
    - Illiquidity typically rewarded (illiquidity premium)
    - Requires 'volume' and 'market_cap' columns
    
    Notes
    -----
    Turnover = (volume * price) / market_cap
    """
    
    def __init__(self, name: str = "liquidity",
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {'window': 20}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute liquidity factor.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'volume', 'close', 'market_cap' columns
            
        Returns
        -------
        pd.Series
            Liquidity (average turnover) values
        """
        self._validate_data(df)
        
        required = {'volume', 'close', 'market_cap'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        
        window = self.params.get('window', 20)
        
        indexed = df.set_index(['date', 'ticker'])
        
        # Compute daily turnover
        turnover = (indexed['volume'] * indexed['close']) / indexed['market_cap']
        
        # Rolling average
        avg_turnover = ts_mean(turnover, window=window)
        
        # Post-process
        avg_turnover = self._post_process(avg_turnover, df)
        return avg_turnover
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        series = standardize(series, method='zscore', group_key='date')
        return series


class QualityFactor(FactorBase):
    """
    Quality factor.
    
    Definition: Return on Equity (ROE) or Debt-to-Equity ratio
    
    Characteristics:
    - High ROE: profitable company
    - Low D/E: low financial leverage
    - Typically positively associated with returns (quality premium)
    - Requires fundamental data
    
    Parameters
    ----------
    metric : str, optional
        'roe': Return on Equity (default)
        'debt_to_equity': Leverage ratio
    """
    
    def __init__(self, name: str = "quality",
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {'metric': 'roe'}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute quality factor.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain either 'roe' or 'debt_to_equity' column
            depending on params['metric']
            
        Returns
        -------
        pd.Series
            Quality factor values
        """
        self._validate_data(df)
        
        metric = self.params.get('metric', 'roe')
        
        if metric == 'roe':
            if 'roe' not in df.columns:
                raise ValueError("DataFrame must contain 'roe' column for ROE metric")
            quality = df['roe']
        
        elif metric == 'debt_to_equity':
            if 'debt_to_equity' not in df.columns:
                raise ValueError("DataFrame must contain 'debt_to_equity' column")
            # Invert D/E so higher quality = higher value
            quality = -df['debt_to_equity']
        
        else:
            raise ValueError(f"Unknown quality metric: {metric}")
        
        # Post-process
        quality = self._post_process(quality, df)
        return quality
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        series = standardize(series, method='zscore', group_key='date')
        return series


class ValueFactor(FactorBase):
    """
    Value factor.
    
    Definition: Book-to-Market ratio (B/M)
    Alternative: Price-to-Book ratio (inverse, so P/B low = P/E low = value stock)
    
    Characteristics:
    - High B/M: cheap relative to book value (value stock)
    - Typically positively associated with returns (value premium)
    - Core Fama-French factor
    - Requires 'book_value' and 'market_cap' columns
    
    Parameters
    ----------
    metric : str, optional
        'book_to_market': B/M ratio (default)
        'earnings_to_price': E/P ratio (alternative value metric)
    """
    
    def __init__(self, name: str = "value",
                 params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {'metric': 'book_to_market'}
        super().__init__(name, params)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute value factor.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain required fundamental columns
            
        Returns
        -------
        pd.Series
            Value factor values
        """
        self._validate_data(df)
        
        metric = self.params.get('metric', 'book_to_market')
        
        if metric == 'book_to_market':
            required = {'book_value', 'market_cap'}
            if not required.issubset(df.columns):
                raise ValueError(f"DataFrame missing columns: {required - set(df.columns)}")
            value = df['book_value'] / df['market_cap']
        
        elif metric == 'earnings_to_price':
            if 'earnings' not in df.columns or 'market_cap' not in df.columns:
                raise ValueError("DataFrame must contain 'earnings' and 'market_cap'")
            value = df['earnings'] / df['market_cap']
        
        else:
            raise ValueError(f"Unknown value metric: {metric}")
        
        # Post-process
        value = self._post_process(value, df)
        return value
    
    def _post_process(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        series = winsorize(series, method='quantile',
                          limits=(0.01, 0.99), group_key='date')
        series = standardize(series, method='zscore', group_key='date')
        return series


# ==================== Convenience Function ====================


def create_style_factors(df: pd.DataFrame, 
                        available_metrics: Optional[Dict[str, bool]] = None
                        ) -> Dict[str, FactorBase]:
    """
    Factory function to create commonly used style factors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Reference DataFrame to check available columns
    available_metrics : dict, optional
        Dict indicating which factors to create.
        If None, creates only factors with all required data.
        
        Example:
        {
            'size': True,
            'volatility': True,
            'momentum': True,
            'liquidity': False,  # Skip if data not available
            ...
        }
        
    Returns
    -------
    dict
        Dictionary of {factor_name: FactorBase instance}
        
    Examples
    --------
    >>> factors = create_style_factors(df)
    >>> for name, factor in factors.items():
    ...     print(f"Created {name}")
    """
    
    factors = {}
    
    # Size
    if 'market_cap' in df.columns:
        factors['size'] = SizeFactor()
    
    # Volatility
    if 'close' in df.columns:
        factors['volatility'] = VolatilityFactor()
    
    # Momentum
    if 'close' in df.columns:
        factors['momentum'] = MomentumFactor()
    
    # Liquidity
    if all(col in df.columns for col in ['volume', 'close', 'market_cap']):
        factors['liquidity'] = LiquidityFactor()
    
    # Quality
    if 'roe' in df.columns:
        factors['quality'] = QualityFactor(params={'metric': 'roe'})
    
    # Value
    if all(col in df.columns for col in ['book_value', 'market_cap']):
        factors['value'] = ValueFactor()
    
    return factors
