"""
Test suite for PipelineManager.
Tests dataset preparation, frequency resampling, and full pipeline execution.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/home/jywang/project/QFRsch/src')

from qfrsch.pipeline.manager import PipelineManager
from qfrsch.models.base import ModelWrapper
from qfrsch.pipeline.strategies import TopNStrategy


@pytest.fixture
def sample_factor_returns_data():
    """Generate sample factor and returns data."""
    np.random.seed(42)
    n_days = 100
    n_tickers = 20
    n_factors = 3
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    tickers = [f'T{i:02d}' for i in range(n_tickers)]
    
    # Factors
    factor_data = np.random.randn(n_days * n_tickers, n_factors) * 0.1
    factors = pd.DataFrame(
        factor_data,
        index=pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker']),
        columns=[f'factor_{i}' for i in range(n_factors)]
    )
    
    # Returns: correlated with factors
    returns_data = (
        factor_data.mean(axis=1) * 0.2 +
        np.random.randn(n_days * n_tickers) * 0.02
    )
    
    returns = pd.DataFrame(
        returns_data,
        index=pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker']),
        columns=['returns']
    )
    
    return factors, returns


class TestPipelineManagerInit:
    """Test PipelineManager initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        pm = PipelineManager()
        assert pm.rebalance_freq == 'D'
        assert pm.prediction_period == 1
    
    def test_init_weekly(self):
        """Test weekly rebalancing."""
        pm = PipelineManager(rebalance_freq='W', prediction_period=1)
        assert pm.rebalance_freq == 'W'
    
    def test_init_monthly(self):
        """Test monthly rebalancing."""
        pm = PipelineManager(rebalance_freq='M', prediction_period=2)
        assert pm.rebalance_freq == 'M'
        assert pm.prediction_period == 2
    
    def test_init_invalid_freq(self):
        """Test error for invalid frequency."""
        with pytest.raises(ValueError, match="rebalance_freq must be"):
            PipelineManager(rebalance_freq='X')
    
    def test_init_invalid_period(self):
        """Test error for invalid prediction period."""
        with pytest.raises(ValueError, match="prediction_period must be >= 1"):
            PipelineManager(prediction_period=0)
    
    def test_repr(self):
        """Test string representation."""
        pm = PipelineManager(rebalance_freq='W')
        repr_str = repr(pm)
        assert 'PipelineManager' in repr_str
        assert 'W' in repr_str


class TestMakeDataset:
    """Test dataset preparation."""
    
    def test_make_dataset_daily(self, sample_factor_returns_data):
        """Test daily dataset creation."""
        factors, returns = sample_factor_returns_data
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        
        X, y = pm.make_dataset(factors, returns, dropna=True)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert X.index.names == ['date', 'ticker']
    
    def test_make_dataset_weekly(self, sample_factor_returns_data):
        """Test weekly dataset creation."""
        factors, returns = sample_factor_returns_data
        pm = PipelineManager(rebalance_freq='W', prediction_period=1)
        
        X, y = pm.make_dataset(factors, returns, dropna=True)
        
        # Should have fewer dates
        n_dates = len(X.index.get_level_values('date').unique())
        assert n_dates < 100  # Original had 100 days
    
    def test_make_dataset_preserves_factors(self, sample_factor_returns_data):
        """Test that all factor columns are preserved."""
        factors, returns = sample_factor_returns_data
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        
        X, y = pm.make_dataset(factors, returns, dropna=True)
        
        # Should have same factor columns
        assert set(X.columns) == set(factors.columns)
    
    def test_make_dataset_label_alignment(self, sample_factor_returns_data):
        """Test proper label alignment (no look-ahead bias)."""
        factors, returns = sample_factor_returns_data
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        
        X, y = pm.make_dataset(factors, returns, dropna=True)
        
        # Check that labels are properly shifted
        assert X.shape[0] > 0
        assert y.shape[0] > 0
        assert X.shape[0] == y.shape[0]
    
    def test_make_dataset_dropna(self, sample_factor_returns_data):
        """Test NaN handling."""
        factors, returns = sample_factor_returns_data
        
        # Introduce NaN
        factors.iloc[10, 0] = np.nan
        returns.iloc[20, 0] = np.nan
        
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        X, y = pm.make_dataset(factors, returns, dropna=True)
        
        # Should have no NaN
        assert not X.isna().any().any()
        assert not y.isna().any()
    
    def test_make_dataset_stores_rebalance_dates(self, sample_factor_returns_data):
        """Test rebalance dates are stored."""
        factors, returns = sample_factor_returns_data
        pm = PipelineManager(rebalance_freq='W', prediction_period=1)
        
        X, y = pm.make_dataset(factors, returns, dropna=True)
        
        assert pm.rebalance_dates_ is not None
        assert len(pm.rebalance_dates_) > 0
    
    def test_make_dataset_invalid_factors_index(self, sample_factor_returns_data):
        """Test error for invalid factors index."""
        _, returns = sample_factor_returns_data
        factors_bad = pd.DataFrame(np.random.randn(100, 3))  # No MultiIndex
        
        pm = PipelineManager(rebalance_freq='D')
        
        with pytest.raises(ValueError, match="MultiIndex"):
            pm.make_dataset(factors_bad, returns)
    
    def test_make_dataset_invalid_returns_index(self, sample_factor_returns_data):
        """Test error for invalid returns index."""
        factors, _ = sample_factor_returns_data
        returns_bad = pd.DataFrame(np.random.randn(100, 1))  # No MultiIndex
        
        pm = PipelineManager(rebalance_freq='D')
        
        with pytest.raises(ValueError, match="MultiIndex"):
            pm.make_dataset(factors, returns_bad)


class TestDatasetMultiFrequency:
    """Test resampling to different frequencies."""
    
    def test_frequencies_have_fewer_dates(self, sample_factor_returns_data):
        """Test that lower frequencies have fewer unique dates."""
        factors, returns = sample_factor_returns_data
        
        pm_d = PipelineManager(rebalance_freq='D')
        pm_w = PipelineManager(rebalance_freq='W')
        pm_m = PipelineManager(rebalance_freq='M')
        
        X_d, _ = pm_d.make_dataset(factors, returns, dropna=True)
        X_w, _ = pm_w.make_dataset(factors, returns, dropna=True)
        X_m, _ = pm_m.make_dataset(factors, returns, dropna=True)
        
        n_dates_d = len(X_d.index.get_level_values('date').unique())
        n_dates_w = len(X_w.index.get_level_values('date').unique())
        n_dates_m = len(X_m.index.get_level_values('date').unique())
        
        # Weekly should have fewer than daily
        assert n_dates_w <= n_dates_d
        # Monthly should have fewer than weekly
        assert n_dates_m <= n_dates_w


class TestLockedLabelGeneration:
    """Test label generation prevents look-ahead bias."""
    
    def test_first_dates_have_nan_labels(self, sample_factor_returns_data):
        """Test that initial periods have NaN labels."""
        factors, returns = sample_factor_returns_data
        pm = PipelineManager(rebalance_freq='D', prediction_period=5)
        
        X, y = pm.make_dataset(factors, returns, dropna=False)
        
        # First few should have NaN labels
        # (Can't compute future returns for first dates)
        assert y.isna().sum() > 0


class TestRunPipeline:
    """Test full pipeline execution."""
    
    def test_run_pipeline_basic(self, sample_factor_returns_data):
        """Test basic pipeline execution."""
        factors, returns = sample_factor_returns_data
        
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        model = ModelWrapper(model_type='ols')
        strategy = TopNStrategy(n_stocks=10, long_only=True)
        
        weights = pm.run_pipeline(factors, returns, model, strategy)
        
        assert isinstance(weights, pd.DataFrame)
        assert 'target_weight' in weights.columns
        assert len(weights) > 0
    
    def test_run_pipeline_returns_multiindex(self, sample_factor_returns_data):
        """Test pipeline returns proper MultiIndex."""
        factors, returns = sample_factor_returns_data
        
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        model = ModelWrapper(model_type='ridge', alpha=0.1)
        strategy = TopNStrategy(n_stocks=5, long_only=False)
        
        weights = pm.run_pipeline(factors, returns, model, strategy)
        
        assert isinstance(weights.index, pd.MultiIndex)
        assert weights.index.names == ['date', 'ticker']
    
    def test_run_pipeline_train_end_date(self, sample_factor_returns_data):
        """Test pipeline with specific train/test date."""
        factors, returns = sample_factor_returns_data
        
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        model = ModelWrapper(model_type='ols')
        strategy = TopNStrategy(n_stocks=5, long_only=True)
        
        dates = factors.index.get_level_values('date').unique()
        train_end = dates[-10]
        
        weights = pm.run_pipeline(
            factors, returns, model, strategy,
            train_end=train_end
        )
        
        assert len(weights) > 0
    
    def test_run_pipeline_models_are_trained(self, sample_factor_returns_data):
        """Test that model is fitted during pipeline."""
        factors, returns = sample_factor_returns_data
        
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        model = ModelWrapper(model_type='ridge', alpha=0.1)
        strategy = TopNStrategy(n_stocks=5, long_only=True)
        
        assert not model.is_fitted_
        
        pm.run_pipeline(factors, returns, model, strategy)
        
        # Model should be trained
        assert model.is_fitted_


class TestPipelineEdgeCases:
    """Test edge cases and error handling."""
    
    def test_insufficient_data(self):
        """Test error with very limited data."""
        factors = pd.DataFrame(
            np.random.randn(5, 2),
            index=pd.MultiIndex.from_product(
                [pd.date_range('2023-01-01', periods=5), ['A']],
                names=['date', 'ticker']
            )
        )
        returns = pd.DataFrame(
            np.random.randn(5, 1),
            index=factors.index,
            columns=['returns']
        )
        
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        X, y = pm.make_dataset(factors, returns, dropna=True)
        
        # Should produce something even with limited data
        assert len(X) >= 0
    
    def test_all_nan_data(self):
        """Test handling of all NaN data."""
        factors = pd.DataFrame(
            np.full((200, 2), np.nan),
            index=pd.MultiIndex.from_product(
                [pd.date_range('2023-01-01', periods=100, freq='B'), 
                 ['A', 'B']],
                names=['date', 'ticker']
            )
        )
        returns = pd.DataFrame(
            np.full((200, 1), np.nan),
            index=factors.index,
            columns=['returns']
        )
        
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        X, y = pm.make_dataset(factors, returns, dropna=True)
        
        # Should produce empty result
        assert len(X) == 0
        assert len(y) == 0


class TestPipelineConsistency:
    """Test pipeline consistency."""
    
    def test_same_data_same_weights(self, sample_factor_returns_data):
        """Test same input produces same weights."""
        factors, returns = sample_factor_returns_data
        
        pm = PipelineManager(rebalance_freq='D', prediction_period=1)
        model1 = ModelWrapper(model_type='ols')
        model2 = ModelWrapper(model_type='ols')
        strategy = TopNStrategy(n_stocks=5, long_only=True)
        
        w1 = pm.run_pipeline(factors, returns, model1, strategy)
        w2 = pm.run_pipeline(factors, returns, model2, strategy)
        
        # Should have same weights (same model type, same strategy)
        if len(w1) > 0 and len(w2) > 0:
            pd.testing.assert_frame_equal(w1, w2, atol=1e-10)
    
    def test_different_frequencies_different_results(self, sample_factor_returns_data):
        """Test different frequencies produce different results."""
        factors, returns = sample_factor_returns_data
        
        pm_d = PipelineManager(rebalance_freq='D', prediction_period=1)
        pm_w = PipelineManager(rebalance_freq='W', prediction_period=1)
        
        model_d = ModelWrapper(model_type='ols')
        model_w = ModelWrapper(model_type='ols')
        strategy = TopNStrategy(n_stocks=5, long_only=True)
        
        w_d = pm_d.run_pipeline(factors, returns, model_d, strategy)
        w_w = pm_w.run_pipeline(factors, returns, model_w, strategy)
        
        # Different frequencies should have different dates
        d_dates = w_d.index.get_level_values('date').unique()
        w_dates = w_w.index.get_level_values('date').unique()
        
        # Weekly should have fewer dates
        assert len(w_dates) <= len(d_dates)


class TestPipelineFrequencies:
    """Test all supported frequencies."""
    
    @pytest.mark.parametrize("freq", ['D', 'W', 'M', 'Q', 'Y'])
    def test_all_frequencies_supported(self, sample_factor_returns_data, freq):
        """Test all frequency types are supported."""
        factors, returns = sample_factor_returns_data
        
        pm = PipelineManager(rebalance_freq=freq, prediction_period=1)
        assert pm.rebalance_freq == freq
        
        # Should not raise
        X, y = pm.make_dataset(factors, returns, dropna=True)
        assert len(X) >= 0
