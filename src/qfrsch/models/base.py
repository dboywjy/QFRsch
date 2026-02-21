"""
Model Wrapper for Unified ML Model Interface
Encapsulates sklearn and XGBoost models with standardized fit/predict.
Supports rolling window training for time-series prediction.
"""

from __future__ import annotations

from typing import Optional, Union, Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler


class ModelWrapper:
    """
    Unified wrapper for sklearn and XGBoost models.
    
    Provides consistent interface for training and prediction with support
    for rolling window training on time-series data.
    
    Parameters
    ----------
    model_type : {'ols', 'ridge', 'lasso', 'xgboost'}, default='ols'
        Type of model to use.
        - 'ols': Ordinary Least Squares (LinearRegression)
        - 'ridge': Ridge regression (L2 regularization)
        - 'lasso': Lasso regression (L1 regularization)
        - 'xgboost': XGBoost regressor
    alpha : float, default=1.0
        Regularization strength for ridge/lasso models.
        Ignored for ols and xgboost.
    scaling : bool, default=False
        Whether to standardize features before training.
    random_state : int, optional
        Random seed for reproducibility (xgboost only).
    **model_kwargs : dict
        Additional parameters passed to model constructor.
        
    Attributes
    ----------
    model_ : object
        Fitted model instance.
    scaler_ : StandardScaler or None
        Feature scaler if scaling=True, else None.
    is_fitted_ : bool
        Whether model has been fitted.
        
    Examples
    --------
    >>> wrapper = ModelWrapper(model_type='ridge', alpha=0.1)
    >>> wrapper.fit(X_train, y_train)
    >>> preds = wrapper.predict(X_test)
    """
    
    def __init__(
        self,
        model_type: Literal['ols', 'ridge', 'lasso', 'xgboost'] = 'ols',
        alpha: float = 1.0,
        scaling: bool = False,
        random_state: Optional[int] = None,
        **model_kwargs
    ):
        self.model_type = model_type
        self.alpha = alpha
        self.scaling = scaling
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        
        self.model_ = None
        self.scaler_ = None
        self.is_fitted_ = False
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build model instance based on model_type."""
        if self.model_type == 'ols':
            self.model_ = LinearRegression()
        elif self.model_type == 'ridge':
            self.model_ = Ridge(alpha=self.alpha)
        elif self.model_type == 'lasso':
            self.model_ = Lasso(alpha=self.alpha, max_iter=5000)
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model_ = xgb.XGBRegressor(
                    random_state=self.random_state,
                    **self.model_kwargs
                )
            except ImportError:
                raise ImportError(
                    "XGBoost not installed. Install with: pip install xgboost"
                )
        else:
            raise ValueError(
                f"model_type must be 'ols', 'ridge', 'lasso', or 'xgboost', "
                f"got {self.model_type}"
            )
    
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> ModelWrapper:
        """
        Train the model on input data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Training features.
        y : pd.Series or np.ndarray, shape (n_samples,)
            Training target values.
            
        Returns
        -------
        self : ModelWrapper
            Returns self for method chaining.
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Handle missing values
        mask = ~(np.isnan(X_array).any(axis=1) if X_array.ndim > 1 
                else np.isnan(X_array)) & ~np.isnan(y_array)
        X_clean = X_array[mask]
        y_clean = y_array[mask]
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples after removing NaN values")
        
        # Scale features if requested
        if self.scaling:
            self.scaler_ = StandardScaler()
            X_clean = self.scaler_.fit_transform(X_clean)
        
        # Train model
        self.model_.fit(X_clean, y_clean)
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate predictions on new data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Features to predict on.
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values.
            
        Raises
        ------
        RuntimeError
            If model has not been fitted yet.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Scale features if scaler exists
        if self.scaler_ is not None:
            X_array = self.scaler_.transform(X_array)
        
        return self.model_.predict(X_array)
    
    def rolling_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        rolling_window: int,
        step: int = 1
    ) -> pd.Series:
        """
        Rolling window prediction for time-series data.
        
        Trains model on each window and predicts the next period.
        Prevents look-ahead bias by ensuring training/test split.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Features with datetime index.
        y : pd.Series, shape (n_samples,)
            Target values with datetime index.
        rolling_window : int
            Size of training window (in periods).
        step : int, default=1
            Number of periods to advance for next prediction.
            
        Returns
        -------
        predictions : pd.Series
            Predictions aligned with original index.
            First (rolling_window + step - 1) values are NaN.
            
        Notes
        -----
        This is the core time-series validation method. Each prediction
        is generated from data NOT including the target date, preventing
        future information leakage.
        
        Examples
        --------
        >>> preds = model.rolling_predict(X, y, rolling_window=252, step=1)
        >>> # Predicts each day using only past 252 days of data
        """
        # Ensure indices are aligned
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        n_samples = len(X_aligned)
        predictions = np.full(n_samples, np.nan)
        
        # Rolling window prediction
        for i in range(rolling_window, n_samples - step + 1, step):
            # Training set: [0:i], excluding test date
            X_train = X_aligned.iloc[:i]
            y_train = y_aligned.iloc[:i]
            
            # Test set: [i + step - 1:i + step]
            test_idx = min(i + step - 1, n_samples - 1)
            X_test = X_aligned.iloc[test_idx:test_idx + 1]
            
            # Train and predict
            try:
                wrapper = ModelWrapper(
                    self.model_type,
                    alpha=self.alpha,
                    scaling=self.scaling,
                    random_state=self.random_state,
                    **self.model_kwargs
                )
                wrapper.fit(X_train, y_train)
                pred = wrapper.predict(X_test)
                predictions[test_idx] = pred[0]
            except Exception as e:
                warnings.warn(f"Prediction failed at index {test_idx}: {str(e)}")
                continue
        
        return pd.Series(predictions, index=X_aligned.index)
    
    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        params_str = f"alpha={self.alpha}" if self.alpha != 1.0 else ""
        return (
            f"ModelWrapper(model_type='{self.model_type}', "
            f"scaling={self.scaling}, {fitted_str}, {params_str})"
        )
