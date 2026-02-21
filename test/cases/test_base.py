"""
Test cases for FactorBase class
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from qfrsch.factors.base import FactorBase


class SimpleFactor(FactorBase):
    """Simple test factor implementation"""
    def compute(self, df: pd.DataFrame) -> pd.Series:
        self._validate_data(df)
        return df['close'].pct_change()


class TestFactorBaseInit:
    """Test FactorBase initialization"""
    
    def test_init_with_name_only(self):
        """Test initialization with just name"""
        factor = SimpleFactor(name="test_factor")
        assert factor.name == "test_factor"
        assert factor.params == {}
    
    def test_init_with_params(self):
        """Test initialization with name and params"""
        params = {"window": 10, "threshold": 0.05}
        factor = SimpleFactor(name="test_factor", params=params)
        assert factor.name == "test_factor"
        assert factor.params == params
    
    def test_init_invalid_name_type(self):
        """Test that non-string names are rejected"""
        with pytest.raises(TypeError):
            SimpleFactor(name=123)
    
    def test_init_invalid_params_type(self):
        """Test that non-dict params are rejected"""
        with pytest.raises(TypeError):
            SimpleFactor(name="test", params="invalid")


class TestValidateData:
    """Test _validate_data method"""
    
    def test_valid_data(self):
        """Test valid data passes validation"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'close': np.random.normal(100, 10, 10)
        })
        factor = SimpleFactor(name="test")
        factor._validate_data(df)  # Should not raise
    
    def test_empty_dataframe(self):
        """Test empty DataFrame raises error"""
        df = pd.DataFrame()
        factor = SimpleFactor(name="test")
        with pytest.raises(ValueError, match="cannot be empty"):
            factor._validate_data(df)
    
    def test_missing_date_column(self):
        """Test missing date column raises error"""
        df = pd.DataFrame({
            'ticker': ['AAPL'] * 10,
            'close': np.random.normal(100, 10, 10)
        })
        factor = SimpleFactor(name="test")
        with pytest.raises(ValueError, match="missing required columns"):
            factor._validate_data(df)
    
    def test_missing_ticker_column(self):
        """Test missing ticker column raises error"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'close': np.random.normal(100, 10, 10)
        })
        factor = SimpleFactor(name="test")
        with pytest.raises(ValueError, match="missing required columns"):
            factor._validate_data(df)
    
    def test_invalid_date_type(self):
        """Test non-datetime date column raises error"""
        df = pd.DataFrame({
            'date': ['2020-01-01'] * 10,  # String instead of datetime
            'ticker': ['AAPL'] * 10,
            'close': np.random.normal(100, 10, 10)
        })
        factor = SimpleFactor(name="test")
        with pytest.raises(ValueError, match="must be datetime type"):
            factor._validate_data(df)
    
    def test_nan_in_date_column(self):
        """Test NaN in date column raises error"""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2020-01-01')] * 5 + [pd.NaT] * 5,
            'ticker': ['AAPL'] * 10,
            'close': np.random.normal(100, 10, 10)
        })
        factor = SimpleFactor(name="test")
        with pytest.raises(ValueError, match="NaN values"):
            factor._validate_data(df)
    
    def test_nan_in_ticker_column(self):
        """Test NaN in ticker column raises error"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['AAPL'] * 5 + [None] * 5,
            'close': np.random.normal(100, 10, 10)
        })
        factor = SimpleFactor(name="test")
        with pytest.raises(ValueError, match="NaN values"):
            factor._validate_data(df)


class TestFactorRepr:
    """Test factor representation"""
    
    def test_repr_without_params(self):
        """Test __repr__ without parameters"""
        factor = SimpleFactor(name="test_factor")
        repr_str = repr(factor)
        assert "SimpleFactor" in repr_str
        assert "test_factor" in repr_str
    
    def test_repr_with_params(self):
        """Test __repr__ with parameters"""
        params = {"window": 10}
        factor = SimpleFactor(name="test_factor", params=params)
        repr_str = repr(factor)
        assert "SimpleFactor" in repr_str
        assert "test_factor" in repr_str
        assert "window=10" in repr_str


class TestComputeAbstract:
    """Test that compute is abstract"""
    
    def test_cannot_instantiate_base_class(self):
        """Test that FactorBase cannot be instantiated"""
        with pytest.raises(TypeError):
            FactorBase(name="test")


class TestSimpleFactorCompute:
    """Test compute method with SimpleFactor"""
    
    def test_compute_returns_series(self):
        """Test that compute returns a Series"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'close': np.linspace(100, 110, 10)
        })
        factor = SimpleFactor(name="test")
        result = factor.compute(df)
        assert isinstance(result, pd.Series)
    
    def test_compute_output_length(self):
        """Test that output length matches input"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=20),
            'ticker': ['AAPL'] * 20,
            'close': np.random.normal(100, 10, 20)
        })
        factor = SimpleFactor(name="test")
        result = factor.compute(df)
        assert len(result) == len(df)
    
    def test_compute_with_invalid_data(self):
        """Test compute raises error with invalid data"""
        df = pd.DataFrame({'close': [100, 101, 102]})  # Missing required columns
        factor = SimpleFactor(name="test")
        with pytest.raises(ValueError):
            factor.compute(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
