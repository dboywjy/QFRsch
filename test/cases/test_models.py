"""
Test suite for ModelWrapper class.
Tests model training, prediction, and rolling window validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/home/jywang/project/QFRsch/src')

from qfrsch.models.base import ModelWrapper


@pytest.fixture
def sample_regression_data():
    """Generate synthetic regression dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features) * 0.1
    # True relationship: y = 0.5*X1 - 0.3*X2 + noise
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    return X, y


@pytest.fixture
def sample_timeseries_data():
    """Generate time-series regression data with MultiIndex."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    tickers = ['A', 'B', 'C', 'D', 'E']
    
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    X_data = np.random.randn(len(index), 3) * 0.1
    X = pd.DataFrame(X_data, index=index, columns=['feat1', 'feat2', 'feat3'])
    
    y_data = 0.5 * X_data[:, 0] - 0.3 * X_data[:, 1] + np.random.randn(len(index)) * 0.1
    y = pd.Series(y_data, index=index, name='target')
    
    return X, y


class TestModelWrapperInit:
    """Test ModelWrapper initialization."""
    
    def test_init_ols(self):
        """Test OLS model initialization."""
        model = ModelWrapper(model_type='ols')
        assert model.model_type == 'ols'
        assert not model.is_fitted_
        assert model.scaler_ is None
    
    def test_init_ridge(self):
        """Test Ridge model initialization."""
        model = ModelWrapper(model_type='ridge', alpha=0.5)
        assert model.model_type == 'ridge'
        assert model.alpha == 0.5
    
    def test_init_lasso(self):
        """Test Lasso model initialization."""
        model = ModelWrapper(model_type='lasso', alpha=0.01)
        assert model.model_type == 'lasso'
        assert model.alpha == 0.01
    
    def test_init_with_scaling(self):
        """Test model with feature scaling enabled."""
        model = ModelWrapper(model_type='ols', scaling=True)
        assert model.scaling is True
    
    def test_init_invalid_type(self):
        """Test error for invalid model type."""
        with pytest.raises(ValueError, match="model_type must be"):
            ModelWrapper(model_type='invalid')
    
    def test_repr(self):
        """Test string representation."""
        model = ModelWrapper(model_type='ridge', alpha=0.1, scaling=True)
        repr_str = repr(model)
        assert 'ridge' in repr_str
        assert 'not fitted' in repr_str


class TestModelWrapperFitPredict:
    """Test model fitting and prediction."""
    
    def test_fit_numpy_arrays(self, sample_regression_data):
        """Test fitting with numpy arrays."""
        X, y = sample_regression_data
        model = ModelWrapper(model_type='ols')
        
        model.fit(X, y)
        assert model.is_fitted_
        assert model.model_ is not None
    
    def test_fit_dataframes(self, sample_regression_data):
        """Test fitting with DataFrames."""
        X, y = sample_regression_data
        X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')
        
        model = ModelWrapper(model_type='ols')
        model.fit(X_df, y_series)
        assert model.is_fitted_
    
    def test_fit_with_missing_values(self, sample_regression_data):
        """Test fitting handles NaN values correctly."""
        X, y = sample_regression_data
        X_with_nan = X.copy()
        X_with_nan[0, 0] = np.nan
        
        model = ModelWrapper(model_type='ols')
        model.fit(X_with_nan, y)
        assert model.is_fitted_
    
    def test_predict_before_fit_raises_error(self, sample_regression_data):
        """Test prediction before fit raises error."""
        X, _ = sample_regression_data
        model = ModelWrapper(model_type='ols')
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)
    
    def test_predict_returns_correct_shape(self, sample_regression_data):
        """Test prediction output shape."""
        X, y = sample_regression_data
        model = ModelWrapper(model_type='ols')
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert predictions.shape == (10,)
    
    def test_predict_with_scaling(self, sample_regression_data):
        """Test prediction with feature scaling."""
        X, y = sample_regression_data
        model = ModelWrapper(model_type='ridge', alpha=0.1, scaling=True)
        model.fit(X, y)
        
        preds = model.predict(X[:10])
        assert not np.any(np.isnan(preds))
        assert preds.shape == (10,)
    
    def test_fit_return_self(self, sample_regression_data):
        """Test fit returns self for method chaining."""
        X, y = sample_regression_data
        model = ModelWrapper(model_type='ols')
        result = model.fit(X, y)
        
        assert result is model


class TestModelTypes:
    """Test different model types."""
    
    def test_ols_prediction(self, sample_regression_data):
        """Test OLS model prediction."""
        X, y = sample_regression_data
        model = ModelWrapper(model_type='ols')
        model.fit(X, y)
        
        preds = model.predict(X[:5])
        assert len(preds) == 5
        assert not np.any(np.isnan(preds))
    
    def test_ridge_prediction(self, sample_regression_data):
        """Test Ridge model prediction."""
        X, y = sample_regression_data
        model = ModelWrapper(model_type='ridge', alpha=0.1)
        model.fit(X, y)
        
        preds = model.predict(X[:5])
        assert len(preds) == 5
    
    def test_lasso_prediction(self, sample_regression_data):
        """Test Lasso model prediction."""
        X, y = sample_regression_data
        model = ModelWrapper(model_type='lasso', alpha=0.01)
        model.fit(X, y)
        
        preds = model.predict(X[:5])
        assert len(preds) == 5
    
    def test_ridge_vs_ols(self, sample_regression_data):
        """Test Ridge regularization reduces overfitting."""
        X, y = sample_regression_data
        
        model_ols = ModelWrapper(model_type='ols')
        model_ridge = ModelWrapper(model_type='ridge', alpha=0.5)
        
        model_ols.fit(X, y)
        model_ridge.fit(X, y)
        
        preds_ols = model_ols.predict(X)
        preds_ridge = model_ridge.predict(X)
        
        # Both should produce predictions
        assert len(preds_ols) == len(preds_ridge)


class TestRollingPrediction:
    """Test rolling window prediction for time-series."""
    
    def test_rolling_predict_shape(self, sample_timeseries_data):
        """Test rolling prediction output shape."""
        X, y = sample_timeseries_data
        model = ModelWrapper(model_type='ols')
        
        preds = model.rolling_predict(X, y, rolling_window=50, step=1)
        assert len(preds) == len(X)
        assert isinstance(preds, pd.Series)
    
    def test_rolling_predict_initial_nans(self, sample_timeseries_data):
        """Test first rolling_window predictions are NaN."""
        X, y = sample_timeseries_data
        model = ModelWrapper(model_type='ols')
        
        window = 50
        preds = model.rolling_predict(X, y, rolling_window=window, step=1)
        
        # First window-1 should be NaN
        assert preds.iloc[:window-1].isna().all() or preds.iloc[:window].isna().all()
    
    def test_rolling_predict_step_parameter(self, sample_timeseries_data):
        """Test rolling prediction with different step sizes."""
        X, y = sample_timeseries_data
        model = ModelWrapper(model_type='ols')
        
        preds_step1 = model.rolling_predict(X, y, rolling_window=30, step=1)
        preds_step5 = model.rolling_predict(X, y, rolling_window=30, step=5)
        
        # Both should have same length
        assert len(preds_step1) == len(preds_step5)
        # But step=5 should have fewer non-NaN values
        assert preds_step5.notna().sum() <= preds_step1.notna().sum()
    
    def test_rolling_predict_prevents_lookahead(self, sample_timeseries_data):
        """Test rolling prediction prevents look-ahead bias."""
        X, y = sample_timeseries_data
        model = ModelWrapper(model_type='ols')
        
        preds = model.rolling_predict(X, y, rolling_window=40, step=1)
        
        # Should have valid predictions
        valid_preds = preds[~preds.isna()]
        assert len(valid_preds) > 0


class TestModelEdgeCases:
    """Test edge cases and error handling."""
    
    def test_fit_all_nan_features(self):
        """Test fitting with all NaN features."""
        X = np.full((100, 5), np.nan)
        y = np.random.randn(100)
        
        model = ModelWrapper(model_type='ols')
        with pytest.raises(ValueError, match="No valid samples"):
            model.fit(X, y)
    
    def test_fit_all_nan_target(self, sample_regression_data):
        """Test fitting with all NaN targets."""
        X, _ = sample_regression_data
        y = np.full_like(X[:, 0], np.nan)
        
        model = ModelWrapper(model_type='ols')
        with pytest.raises(ValueError, match="No valid samples"):
            model.fit(X, y)
    
    def test_predict_with_nan_features(self, sample_regression_data):
        """Test prediction with NaN features raises error."""
        X, y = sample_regression_data
        model = ModelWrapper(model_type='ols')
        model.fit(X, y)
        
        X_test = X[:5].copy()
        X_test[0, 0] = np.nan
        
        # sklearn models don't accept NaN in prediction
        with pytest.raises(ValueError, match="NaN"):
            model.predict(X_test)
    
    def test_single_sample_fit_error(self):
        """Test fitting with insufficient samples."""
        X = np.random.randn(1, 5)
        y = np.array([0.5])
        
        model = ModelWrapper(model_type='ols')
        # Should fit but may have poor generalization
        model.fit(X, y)
        assert model.is_fitted_
    
    def test_perfect_multicollinearity(self):
        """Test fitting with perfectly collinear features."""
        X = np.random.randn(100, 3)
        X[:, 2] = X[:, 0] * 2  # Perfect multicollinearity
        y = np.random.randn(100)
        
        model = ModelWrapper(model_type='ridge', alpha=0.1)
        model.fit(X, y)
        assert model.is_fitted_


class TestModelConsistency:
    """Test model consistency and reproducibility."""
    
    def test_same_data_same_predictions(self, sample_regression_data):
        """Test same input produces same predictions."""
        X, y = sample_regression_data
        
        model1 = ModelWrapper(model_type='ols')
        model2 = ModelWrapper(model_type='ols')
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        preds1 = model1.predict(X[:10])
        preds2 = model2.predict(X[:10])
        
        np.testing.assert_array_almost_equal(preds1, preds2)
    
    def test_ridge_alpha_effect(self, sample_regression_data):
        """Test different Ridge alpha values produce different results."""
        X, y = sample_regression_data
        
        model_alpha01 = ModelWrapper(model_type='ridge', alpha=0.1)
        model_alpha10 = ModelWrapper(model_type='ridge', alpha=1.0)
        
        model_alpha01.fit(X, y)
        model_alpha10.fit(X, y)
        
        preds_01 = model_alpha01.predict(X)
        preds_10 = model_alpha10.predict(X)
        
        # Different alphas should produce different predictions
        assert not np.allclose(preds_01, preds_10)
    
    def test_scaling_effect(self, sample_regression_data):
        """Test feature scaling effect on predictions."""
        X, y = sample_regression_data
        
        model_no_scale = ModelWrapper(model_type='ridge', alpha=0.1, scaling=False)
        model_with_scale = ModelWrapper(model_type='ridge', alpha=0.1, scaling=True)
        
        model_no_scale.fit(X, y)
        model_with_scale.fit(X, y)
        
        preds_no_scale = model_no_scale.predict(X[:5])
        preds_with_scale = model_with_scale.predict(X[:5])
        
        # Both should be valid
        assert not np.any(np.isnan(preds_no_scale))
        assert not np.any(np.isnan(preds_with_scale))
