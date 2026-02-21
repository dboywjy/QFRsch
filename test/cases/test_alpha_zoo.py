"""
Test cases for alpha zoo module (example factors)
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from qfrsch.factors.alpha_zoo import (
    Alpha001, Alpha002, Alpha003, Alpha004, AlphaCombo,
    get_alpha, list_alphas
)


@pytest.fixture
def sample_ohlc_data():
    """Create realistic OHLC data"""
    dates = pd.date_range('2020-01-01', periods=80)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
    
    data = []
    np.random.seed(42)
    
    for ticker in tickers:
        price = 100
        for date in dates:
            daily_return = np.random.normal(0.0005, 0.02)
            price = price * (1 + daily_return)
            
            data.append({
                'date': date,
                'ticker': ticker,
                'open': price * (1 + np.random.normal(0, 0.005)),
                'close': price,
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'volume': np.random.uniform(1e6, 1e8),
            })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df


class TestAlpha001:
    """Test Alpha001 factor"""
    
    def test_alpha001_compute(self, sample_ohlc_data):
        """Test Alpha001 compute"""
        alpha = Alpha001(params={'window': 10})
        result = alpha.compute(sample_ohlc_data)
        
        assert len(result) == len(sample_ohlc_data)
        assert isinstance(result, pd.Series)
    
    def test_alpha001_output_range(self, sample_ohlc_data):
        """Test Alpha001 output is standardized"""
        alpha = Alpha001()
        result = alpha.compute(sample_ohlc_data)
        
        # Should be roughly standardized
        assert (result > -5).all() or result.isna().all()
        assert (result < 5).all() or result.isna().all()
    
    def test_alpha001_non_null(self, sample_ohlc_data):
        """Test Alpha001 produces non-null values"""
        alpha = Alpha001()
        result = alpha.compute(sample_ohlc_data)
        
        assert result.notna().sum() > 0


class TestAlpha002:
    """Test Alpha002 factor"""
    
    def test_alpha002_compute(self, sample_ohlc_data):
        """Test Alpha002 compute"""
        alpha = Alpha002(params={'vol_window': 20, 'rank_window': 5})
        result = alpha.compute(sample_ohlc_data)
        
        assert len(result) == len(sample_ohlc_data)
        assert isinstance(result, pd.Series)
    
    def test_alpha002_output_range(self, sample_ohlc_data):
        """Test Alpha002 output is in reasonable range"""
        alpha = Alpha002()
        result = alpha.compute(sample_ohlc_data)
        
        # Should have meaningful values
        assert result.notna().sum() > len(sample_ohlc_data) * 0.5


class TestAlpha003:
    """Test Alpha003 factor"""
    
    def test_alpha003_compute(self, sample_ohlc_data):
        """Test Alpha003 compute"""
        alpha = Alpha003()
        result = alpha.compute(sample_ohlc_data)
        
        assert len(result) == len(sample_ohlc_data)
        assert isinstance(result, pd.Series)
    
    def test_alpha003_mean_reversion(self, sample_ohlc_data):
        """Test Alpha003 mean reversion property"""
        alpha = Alpha003()
        result = alpha.compute(sample_ohlc_data)
        
        # Should have both positive and negative values
        positive_count = (result > 0).sum()
        negative_count = (result < 0).sum()
        
        assert positive_count > 0 and negative_count > 0


class TestAlpha004:
    """Test Alpha004 factor"""
    
    def test_alpha004_compute(self, sample_ohlc_data):
        """Test Alpha004 compute"""
        alpha = Alpha004()
        result = alpha.compute(sample_ohlc_data)
        
        assert len(result) == len(sample_ohlc_data)
        assert isinstance(result, pd.Series)
    
    def test_alpha004_requires_volume(self):
        """Test Alpha004 requires volume column"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'open': np.random.normal(100, 5, 10)
        })
        df['date'] = pd.to_datetime(df['date'])
        
        alpha = Alpha004()
        with pytest.raises(ValueError, match="missing"):
            alpha.compute(df)


class TestAlphaCombo:
    """Test AlphaCombo factor"""
    
    def test_alphacombo_compute(self, sample_ohlc_data):
        """Test AlphaCombo compute"""
        combo = AlphaCombo()
        result = combo.compute(sample_ohlc_data)
        
        assert len(result) == len(sample_ohlc_data)
        assert isinstance(result, pd.Series)
    
    def test_alphacombo_equal_weights(self, sample_ohlc_data):
        """Test AlphaCombo with equal weights"""
        weights = {
            'alpha_001': 1.0,
            'alpha_002': 1.0,
            'alpha_003': 1.0
        }
        combo = AlphaCombo(params={'weights': weights})
        result = combo.compute(sample_ohlc_data)
        
        assert result.notna().sum() > 0
    
    def test_alphacombo_custom_weights(self, sample_ohlc_data):
        """Test AlphaCombo with custom weights"""
        weights = {
            'alpha_001': 0.5,
            'alpha_002': 0.3,
            'alpha_003': 0.2
        }
        combo = AlphaCombo(params={'weights': weights})
        result = combo.compute(sample_ohlc_data)
        
        assert result.notna().sum() > 0


class TestAlphaFactory:
    """Test get_alpha factory function"""
    
    def test_get_alpha_alpha001(self, sample_ohlc_data):
        """Test get_alpha with alpha_001"""
        alpha = get_alpha('alpha_001')
        result = alpha.compute(sample_ohlc_data)
        
        assert result.notna().sum() > 0
    
    def test_get_alpha_alpha002(self, sample_ohlc_data):
        """Test get_alpha with alpha_002"""
        alpha = get_alpha('alpha_002')
        result = alpha.compute(sample_ohlc_data)
        
        assert result.notna().sum() > 0
    
    def test_get_alpha_with_params(self, sample_ohlc_data):
        """Test get_alpha with parameters"""
        alpha = get_alpha('alpha_001', params={'window': 15})
        result = alpha.compute(sample_ohlc_data)
        
        assert result.notna().sum() > 0
    
    def test_get_alpha_invalid_name(self):
        """Test get_alpha with invalid name"""
        with pytest.raises(ValueError, match="Unknown alpha"):
            get_alpha('alpha_999')


class TestListAlphas:
    """Test list_alphas function"""
    
    def test_list_alphas_returns_list(self):
        """Test list_alphas returns list"""
        alphas = list_alphas()
        
        assert isinstance(alphas, list)
        assert len(alphas) > 0
    
    def test_list_alphas_contains_known_alphas(self):
        """Test list_alphas contains expected factors"""
        alphas = list_alphas()
        
        assert 'alpha_001' in alphas
        assert 'alpha_002' in alphas
        assert 'alpha_003' in alphas


class TestAlphaIntegration:
    """Test alphas work correctly together"""
    
    def test_all_alphas_on_same_data(self, sample_ohlc_data):
        """Test computing all alphas on same data"""
        results = {}
        
        for alpha_name in list_alphas():
            try:
                alpha = get_alpha(alpha_name)
                result = alpha.compute(sample_ohlc_data)
                results[alpha_name] = result
            except Exception as e:
                pytest.fail(f"Alpha {alpha_name} failed: {str(e)}")
        
        assert len(results) > 0
    
    def test_alphas_consistency(self, sample_ohlc_data):
        """Test alphas produce consistent results"""
        alpha1_result1 = Alpha001().compute(sample_ohlc_data)
        alpha1_result2 = Alpha001().compute(sample_ohlc_data)
        
        # Results should be identical (deterministic)
        pd.testing.assert_series_equal(
            alpha1_result1, alpha1_result2,
            check_names=False
        )
    
    def test_alpha_cross_sectional_properties(self, sample_ohlc_data):
        """Test that alphas are properly cross-sectionally normalized"""
        alpha = Alpha001()
        result = alpha.compute(sample_ohlc_data)
        
        # By date, should have similar distribution properties
        by_date_stats = result.groupby(lambda x: x[0]).agg(['mean', 'std'])
        
        # Cross-sectional standardization should be applied
        if len(by_date_stats) > 1:
            # Means should be close to 0
            assert by_date_stats['mean'].abs().mean() < 0.5


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_alpha_with_missing_close(self):
        """Test alpha with missing close price"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'volume': np.random.uniform(1e6, 1e8, 10)
        })
        df['date'] = pd.to_datetime(df['date'])
        
        alpha = Alpha001()
        with pytest.raises(ValueError, match="close"):
            alpha.compute(df)
    
    def test_alpha_with_insufficient_history(self):
        """Test alpha with very short history"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=3),
            'ticker': ['AAPL'] * 3,
            'close': [100, 101, 102]
        })
        df['date'] = pd.to_datetime(df['date'])
        
        alpha = Alpha001()
        result = alpha.compute(df)
        
        # Should handle gracefully, even if output is mostly NaN
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
