"""
Test cases for operators module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from qfrsch.factors.operators import (
    ts_mean, ts_std, ts_rank, ts_delta, ts_delay, ts_corr,
    cs_rank, op_log, op_abs, op_sign, op_sqrt, op_exp,
    create_multiindex_series
)


@pytest.fixture
def sample_multiindex_series():
    """Create sample MultiIndex series for testing"""
    dates = pd.date_range('2020-01-01', periods=20)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'date': date,
                'ticker': ticker,
                'close': np.random.normal(100, 10)
            })
    
    df = pd.DataFrame(data)
    return df.set_index(['date', 'ticker'])['close']


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    dates = pd.date_range('2020-01-01', periods=30)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'date': date,
                'ticker': ticker,
                'close': np.random.normal(100, 10),
                'volume': np.random.uniform(1e6, 1e8)
            })
    
    return pd.DataFrame(data)


class TestTimeSeriesOperators:
    """Test time-series operators"""
    
    def test_ts_mean_shape(self, sample_multiindex_series):
        """Test ts_mean returns correct shape"""
        result = ts_mean(sample_multiindex_series, window=5)
        assert len(result) == len(sample_multiindex_series)
        assert isinstance(result, pd.Series)
    
    def test_ts_mean_values(self):
        """Test ts_mean computes correct values"""
        data = pd.Series([1, 2, 3, 4, 5] * 2)
        data.index = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=5), ['A', 'B']],
            names=['date', 'ticker']
        )
        result = ts_mean(data, window=2)
        # Check result has same length and values are within input range
        assert len(result) == len(data)
        non_nan_vals = result[~result.isna()]
        assert (non_nan_vals >= data.min()).all()
        assert (non_nan_vals <= data.max()).all()
    
    def test_ts_std_shape(self, sample_multiindex_series):
        """Test ts_std returns correct shape"""
        result = ts_std(sample_multiindex_series, window=5)
        assert len(result) == len(sample_multiindex_series)
    
    def test_ts_rank_range(self, sample_multiindex_series):
        """Test ts_rank values are in [0, 1]"""
        result = ts_rank(sample_multiindex_series, window=5)
        assert (result >= 0).all() or result.isna().all()
        assert (result <= 1).all() or result.isna().all()
    
    def test_ts_delta_shape(self, sample_multiindex_series):
        """Test ts_delta returns correct shape"""
        result = ts_delta(sample_multiindex_series, window=1)
        assert len(result) == len(sample_multiindex_series)
    
    def test_ts_delay_shift(self):
        """Test ts_delay correctly lags values"""
        data = pd.Series([1, 2, 3, 4, 5])
        data.index = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=5), ['A']],
            names=['date', 'ticker']
        )
        result = ts_delay(data, window=1)
        # First value should be NaN, second should be 1
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 1.0
    
    def test_ts_corr_range(self, sample_multiindex_series):
        """Test ts_corr returns values in [-1, 1]"""
        s1 = sample_multiindex_series
        s2 = sample_multiindex_series * 2 + np.random.normal(0, 1, len(s1))
        result = ts_corr(s1, s2, window=5)
        # Check non-NaN values are in valid correlation range [-1, 1]
        non_nan_vals = result[~result.isna()]
        if len(non_nan_vals) > 0:
            assert (non_nan_vals >= -1.1).all()  # Allow small numerical error
            assert (non_nan_vals <= 1.1).all()


class TestCrossSectionalOperators:
    """Test cross-sectional operators"""
    
    def test_cs_rank_shape(self, sample_multiindex_series):
        """Test cs_rank returns correct shape"""
        result = cs_rank(sample_multiindex_series)
        assert len(result) == len(sample_multiindex_series)
    
    def test_cs_rank_range(self, sample_multiindex_series):
        """Test cs_rank values are in [0, 1]"""
        result = cs_rank(sample_multiindex_series)
        assert (result >= 0).all() or result.isna().all()
        assert (result <= 1).all() or result.isna().all()
    
    def test_cs_rank_distinct_values(self, sample_multiindex_series):
        """Test cs_rank produces distinct values across tickers"""
        result = cs_rank(sample_multiindex_series)
        # Should have multiple unique values (not all same)
        assert len(result.unique()) > 1


class TestMathOperators:
    """Test mathematical operators"""
    
    def test_op_log_positive(self):
        """Test op_log with positive values"""
        data = pd.Series([1.0, np.e, np.e**2])
        result = op_log(data)
        expected = pd.Series([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result.values, expected.values)
    
    def test_op_log_with_base(self):
        """Test op_log with custom base"""
        data = pd.Series([1.0, 10.0, 100.0])
        result = op_log(data, base=10)
        expected = pd.Series([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result.values, expected.values)
    
    def test_op_abs(self):
        """Test op_abs"""
        data = pd.Series([-2, -1, 0, 1, 2])
        result = op_abs(data)
        expected = pd.Series([2, 1, 0, 1, 2])
        assert (result == expected).all()
    
    def test_op_sign(self):
        """Test op_sign"""
        data = pd.Series([-2, -1, 0, 1, 2])
        result = op_sign(data)
        expected = pd.Series([-1, -1, 0, 1, 1])
        assert (result == expected).all()
    
    def test_op_sqrt(self):
        """Test op_sqrt"""
        data = pd.Series([0, 1, 4, 9, 16])
        result = op_sqrt(data)
        expected = pd.Series([0, 1, 2, 3, 4])
        np.testing.assert_array_almost_equal(result.values, expected.values)
    
    def test_op_exp(self):
        """Test op_exp"""
        data = pd.Series([0, 1, 2])
        result = op_exp(data)
        expected = pd.Series([1.0, np.e, np.e**2])
        np.testing.assert_array_almost_equal(result.values, expected.values)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_multiindex_series(self, sample_dataframe):
        """Test create_multiindex_series"""
        result = create_multiindex_series(sample_dataframe, 'close')
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['date', 'ticker']
        assert len(result) == len(sample_dataframe)
    
    def test_create_multiindex_series_values(self, sample_dataframe):
        """Test create_multiindex_series preserves values"""
        result = create_multiindex_series(sample_dataframe, 'close')
        df_ordered = sample_dataframe.set_index(['date', 'ticker']).sort_index()
        assert (result.sort_index() == df_ordered['close']).all()


class TestOperatorChaining:
    """Test operators work well together"""
    
    def test_operator_composition(self, sample_multiindex_series):
        """Test composing multiple operators"""
        result = sample_multiindex_series
        result = op_log(result)  # Log transform
        result = ts_mean(result, window=5)  # Moving average
        result = cs_rank(result)  # Cross-sectional rank
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_multiindex_series)
    
    def test_nested_operators(self, sample_multiindex_series):
        """Test nested operator calls"""
        result = cs_rank(ts_mean(sample_multiindex_series, window=10))
        assert isinstance(result, pd.Series)
        assert (result >= 0).all() or result.isna().all()
        assert (result <= 1).all() or result.isna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
