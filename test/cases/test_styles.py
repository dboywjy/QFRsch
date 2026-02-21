"""
Test cases for style factors module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from qfrsch.factors.styles import (
    IndustryFactor, SizeFactor, VolatilityFactor, MomentumFactor,
    LiquidityFactor, QualityFactor, ValueFactor, create_style_factors
)


@pytest.fixture
def sample_market_data():
    """Create comprehensive sample market data"""
    dates = pd.date_range('2020-01-01', periods=60)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
    
    data = []
    np.random.seed(42)
    
    for ticker in tickers:
        price = 100
        for date in dates:
            price = price * (1 + np.random.normal(0.0005, 0.02))
            data.append({
                'date': date,
                'ticker': ticker,
                'close': price,
                'volume': np.random.uniform(1e6, 1e8),
                'market_cap': np.random.uniform(1e10, 1e12),
                'book_value': np.random.uniform(1e9, 1e11),
                'roe': np.random.uniform(0.05, 0.30),
                'earnings': np.random.uniform(1e9, 1e10),
            })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df


class TestIndustryFactor:
    """Test IndustryFactor"""
    
    def test_industry_factor_compute(self, sample_market_data):
        """Test industry factor compute"""
        ind_factor = IndustryFactor(name="industry")
        ind_factor.industry_map = {
            'AAPL': 0, 'MSFT': 0, 'GOOGL': 0,
            'TSLA': 1, 'JPM': 2
        }
        
        result = ind_factor.compute(sample_market_data)
        
        assert len(result) == len(sample_market_data)
        assert set(result.unique()) == {0, 1, 2}
    
    def test_industry_factor_get_dummies(self, sample_market_data):
        """Test industry factor get_dummies"""
        ind_factor = IndustryFactor(name="industry")
        ind_factor.industry_map = {
            'AAPL': 0, 'MSFT': 0, 'GOOGL': 0,
            'TSLA': 1, 'JPM': 2
        }
        
        dummies = ind_factor.get_dummies(sample_market_data, drop_first=False)
        
        assert isinstance(dummies, pd.DataFrame)
        assert isinstance(dummies.index, pd.MultiIndex)
        assert dummies.shape[0] == len(sample_market_data)


class TestSizeFactor:
    """Test SizeFactor"""
    
    def test_size_factor_requires_market_cap(self, sample_market_data):
        """Test SizeFactor requires market_cap column"""
        size_factor = SizeFactor()
        df_missing = sample_market_data.drop('market_cap', axis=1)
        
        with pytest.raises(ValueError, match="market_cap"):
            size_factor.compute(df_missing)
    
    def test_size_factor_compute(self, sample_market_data):
        """Test SizeFactor compute"""
        size_factor = SizeFactor()
        result = size_factor.compute(sample_market_data)
        
        assert len(result) == len(sample_market_data)
        assert result.notna().sum() > 0
    
    def test_size_factor_standardized(self, sample_market_data):
        """Test SizeFactor is standardized"""
        size_factor = SizeFactor()
        result = size_factor.compute(sample_market_data)
        
        # Should be roughly standardized (zero mean, unit variance)
        mean_val = result.mean()
        assert abs(mean_val) < 0.1 or result.isna().all()


class TestVolatilityFactor:
    """Test VolatilityFactor"""
    
    def test_volatility_factor_requires_close(self, sample_market_data):
        """Test VolatilityFactor requires close column"""
        vol_factor = VolatilityFactor()
        df_missing = sample_market_data.drop('close', axis=1)
        
        with pytest.raises(ValueError, match="close"):
            vol_factor.compute(df_missing)
    
    def test_volatility_factor_compute(self, sample_market_data):
        """Test VolatilityFactor compute"""
        vol_factor = VolatilityFactor(params={'window': 20})
        result = vol_factor.compute(sample_market_data)
        
        assert len(result) == len(sample_market_data)
        assert result.notna().sum() > 0
    
    def test_volatility_factor_positive_values(self, sample_market_data):
        """Test volatility values are reasonable"""
        vol_factor = VolatilityFactor()
        result = vol_factor.compute(sample_market_data)
        
        # After standardization, values should be in reasonable range
        non_nan_vals = result[~result.isna()]
        if len(non_nan_vals) > 0:
            assert (non_nan_vals > -5).all()
            assert (non_nan_vals < 5).all()


class TestMomentumFactor:
    """Test MomentumFactor"""
    
    def test_momentum_factor_requires_close(self, sample_market_data):
        """Test MomentumFactor requires close column"""
        mom_factor = MomentumFactor()
        df_missing = sample_market_data.drop('close', axis=1)
        
        with pytest.raises(ValueError, match="close"):
            mom_factor.compute(df_missing)
    
    def test_momentum_factor_compute(self, sample_market_data):
        """Test MomentumFactor compute"""
        mom_factor = MomentumFactor(
            params={'lookback': 30, 'skip_recent': 5}
        )
        result = mom_factor.compute(sample_market_data)
        
        assert len(result) == len(sample_market_data)


class TestLiquidityFactor:
    """Test LiquidityFactor"""
    
    def test_liquidity_factor_requires_columns(self, sample_market_data):
        """Test LiquidityFactor requires specific columns"""
        liq_factor = LiquidityFactor()
        df_missing = sample_market_data.drop('volume', axis=1)
        
        with pytest.raises(ValueError, match="missing columns"):
            liq_factor.compute(df_missing)
    
    def test_liquidity_factor_compute(self, sample_market_data):
        """Test LiquidityFactor compute"""
        liq_factor = LiquidityFactor(params={'window': 20})
        result = liq_factor.compute(sample_market_data)
        
        assert len(result) == len(sample_market_data)
        assert result.notna().sum() > 0


class TestQualityFactor:
    """Test QualityFactor"""
    
    def test_quality_factor_roe(self, sample_market_data):
        """Test QualityFactor with ROE metric"""
        qual_factor = QualityFactor(params={'metric': 'roe'})
        result = qual_factor.compute(sample_market_data)
        
        assert len(result) == len(sample_market_data)
        assert result.notna().sum() > 0
    
    def test_quality_factor_invalid_metric(self, sample_market_data):
        """Test QualityFactor with invalid metric"""
        qual_factor = QualityFactor(params={'metric': 'invalid'})
        
        with pytest.raises(ValueError, match="Unknown quality metric"):
            qual_factor.compute(sample_market_data)


class TestValueFactor:
    """Test ValueFactor"""
    
    def test_value_factor_book_to_market(self, sample_market_data):
        """Test ValueFactor with B/M metric"""
        val_factor = ValueFactor(params={'metric': 'book_to_market'})
        result = val_factor.compute(sample_market_data)
        
        assert len(result) == len(sample_market_data)
        assert result.notna().sum() > 0
    
    def test_value_factor_requires_columns(self, sample_market_data):
        """Test ValueFactor requires specific columns"""
        val_factor = ValueFactor()
        df_missing = sample_market_data.drop('book_value', axis=1)
        
        with pytest.raises(ValueError, match="missing columns"):
            val_factor.compute(df_missing)


class TestCreateStyleFactors:
    """Test create_style_factors factory function"""
    
    def test_create_all_factors(self, sample_market_data):
        """Test creating all available factors"""
        factors = create_style_factors(sample_market_data)
        
        assert isinstance(factors, dict)
        assert len(factors) > 0
        # Should include size, volatility, momentum at minimum
        assert 'size' in factors or 'volatility' in factors
    
    def test_create_factors_partial_data(self):
        """Test creating factors with partial data"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'market_cap': np.random.uniform(1e10, 1e12, 10)
        })
        df['date'] = pd.to_datetime(df['date'])
        
        factors = create_style_factors(df)
        
        # Should only create size factor
        assert 'size' in factors
        assert 'volatility' not in factors


class TestFactorIntegration:
    """Test factors work together"""
    
    def test_multiple_factors_on_same_data(self, sample_market_data):
        """Test computing multiple factors on same data"""
        factors_list = [
            SizeFactor(),
            VolatilityFactor(),
            LiquidityFactor(),
        ]
        
        results = []
        for factor in factors_list:
            result = factor.compute(sample_market_data)
            results.append(result)
        
        # Should be able to combine results
        combined = pd.concat(results, axis=1)
        assert combined.shape[0] == len(sample_market_data)
        assert combined.shape[1] == len(factors_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
