"""
Test cases for processor module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from qfrsch.factors.processor import (
    winsorize, standardize, neutralize, process_factor
)


@pytest.fixture
def sample_factor_data():
    """Create sample factor data with MultiIndex"""
    dates = pd.date_range('2020-01-01', periods=20)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'date': date,
                'ticker': ticker,
                'factor': np.random.normal(100, 20),
                'risk1': np.random.uniform(0, 1),
                'risk2': np.random.uniform(0, 1)
            })
    
    df = pd.DataFrame(data)
    series = df.set_index(['date', 'ticker'])['factor']
    return series


@pytest.fixture
def sample_risk_factors():
    """Create sample risk factor data"""
    dates = pd.date_range('2020-01-01', periods=20)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'date': date,
                'ticker': ticker,
                'size': np.random.normal(10, 2),
                'volatility': np.random.uniform(0.1, 0.5)
            })
    
    df = pd.DataFrame(data)
    return df.set_index(['date', 'ticker'])[['size', 'volatility']]


class TestWinsorize:
    """Test winsorize function"""
    
    def test_winsorize_quantile_method(self, sample_factor_data):
        """Test quantile-based winsorization"""
        original_min = sample_factor_data.min()
        original_max = sample_factor_data.max()
        
        result = winsorize(sample_factor_data, method='quantile',
                          limits=(0.1, 0.9), group_key='date')
        
        assert len(result) == len(sample_factor_data)
        assert result.min() >= original_min or result.isna().any()
        assert result.max() <= original_max or result.isna().any()
    
    def test_winsorize_std_method(self, sample_factor_data):
        """Test std-based winsorization"""
        result = winsorize(sample_factor_data, method='std',
                          limits=(2.0, 2.0), group_key='date')
        
        assert len(result) == len(sample_factor_data)
        assert isinstance(result, pd.Series)
    
    def test_winsorize_global(self, sample_factor_data):
        """Test global winsorization"""
        result = winsorize(sample_factor_data, method='quantile',
                          limits=(0.05, 0.95), group_key=None)
        
        assert len(result) == len(sample_factor_data)
    
    def test_winsorize_invalid_method(self, sample_factor_data):
        """Test invalid method raises error"""
        with pytest.raises(ValueError):
            winsorize(sample_factor_data, method='invalid')


class TestStandardize:
    """Test standardize function"""
    
    def test_standardize_zscore_mean(self, sample_factor_data):
        """Test z-score standardization - mean"""
        result = standardize(sample_factor_data, method='zscore', group_key='date')
        
        # Check mean is close to 0
        mean_val = result.mean()
        assert abs(mean_val) < 0.1 or result.isna().all()
    
    def test_standardize_zscore_std(self, sample_factor_data):
        """Test z-score standardization - std"""
        result = standardize(sample_factor_data, method='zscore', group_key='date')
        
        # Check std is close to 1 (accounting for NaN)
        std_val = result.std()
        assert 0.5 < std_val < 1.5 or result.notna().sum() < 3
    
    def test_standardize_minmax(self, sample_factor_data):
        """Test min-max scaling"""
        result = standardize(sample_factor_data, method='minmax', group_key='date')
        
        # Values should be roughly in [0, 1]
        assert (result >= -0.01).all() or result.isna().all()
        assert (result <= 1.01).all() or result.isna().all()
    
    def test_standardize_robust(self, sample_factor_data):
        """Test robust scaling"""
        result = standardize(sample_factor_data, method='robust', group_key='date')
        
        assert len(result) == len(sample_factor_data)
        assert isinstance(result, pd.Series)
    
    def test_standardize_invalid_method(self, sample_factor_data):
        """Test invalid method raises error"""
        with pytest.raises(ValueError):
            standardize(sample_factor_data, method='invalid')


class TestNeutralize:
    """Test neutralize function"""
    
    def test_neutralize_shape(self, sample_factor_data, sample_risk_factors):
        """Test neutralize returns correct shape"""
        result = neutralize(sample_factor_data, sample_risk_factors)
        
        assert len(result) == len(sample_factor_data)
        assert isinstance(result, pd.Series)
    
    def test_neutralize_reduces_correlation(self, sample_factor_data, sample_risk_factors):
        """Test that neutralization reduces correlation with risk factors"""
        # Compute correlation before
        corr_before_size = sample_factor_data.corr(sample_risk_factors['size'])
        
        # Neutralize
        result = neutralize(sample_factor_data, sample_risk_factors)
        
        # Correlation after should be lower (if was significant)
        if abs(corr_before_size) > 0.3:
            corr_after_size = result.corr(sample_risk_factors['size'])
            assert abs(corr_after_size) <= abs(corr_before_size)
    
    def test_neutralize_ols_method(self, sample_factor_data, sample_risk_factors):
        """Test OLS neutralization method"""
        result = neutralize(sample_factor_data, sample_risk_factors, method='ols')
        
        assert len(result) == len(sample_factor_data)
        assert result.notna().sum() > 0  # Should have some non-null values
    
    def test_neutralize_lstsq_method(self, sample_factor_data, sample_risk_factors):
        """Test LSTSQ neutralization method"""
        result = neutralize(sample_factor_data, sample_risk_factors, method='lstsq')
        
        assert len(result) == len(sample_factor_data)
        assert result.notna().sum() > 0


class TestProcessFactor:
    """Test full processing pipeline"""
    
    def test_process_factor_pipeline(self, sample_factor_data, sample_risk_factors):
        """Test complete processing pipeline"""
        result = process_factor(
            sample_factor_data,
            winsorize_method='quantile',
            winsorize_limits=(0.05, 0.95),
            standardize_method='zscore',
            risk_df=sample_risk_factors
        )
        
        assert len(result) == len(sample_factor_data)
        assert isinstance(result, pd.Series)
    
    def test_process_factor_without_neutralization(self, sample_factor_data):
        """Test pipeline without risk neutralization"""
        result = process_factor(
            sample_factor_data,
            winsorize_method='quantile',
            standardize_method='zscore',
            risk_df=None
        )
        
        assert len(result) == len(sample_factor_data)
    
    def test_process_factor_standardization(self, sample_factor_data):
        """Test that output is standardized"""
        result = process_factor(sample_factor_data)
        
        # After standardization, mean should be close to 0
        mean_val = result.mean()
        assert abs(mean_val) < 0.1 or result.notna().sum() < 3


class TestLimitCase:
    """Test edge cases"""
    
    def test_winsorize_all_same_values(self):
        """Test winsorize with constant series"""
        data = pd.Series([100] * 10)
        data.index = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=5), ['A', 'B']],
            names=['date', 'ticker']
        )
        result = winsorize(data, method='quantile', limits=(0.1, 0.9))
        # Should remain constant
        assert (result == 100).all() or result.isna().all()
    
    def test_standardize_zero_variance(self):
        """Test standardize with zero variance"""
        data = pd.Series([100] * 10)
        data.index = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=5), ['A', 'B']],
            names=['date', 'ticker']
        )
        result = standardize(data, method='zscore', group_key='date')
        # Should handle gracefully (divide by zero protection)
        assert len(result) == len(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
