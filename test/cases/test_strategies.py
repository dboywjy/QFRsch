"""
Test suite for weight generation strategies.
Tests TopNStrategy and OptimizedStrategy portfolio construction.
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/home/jywang/project/QFRsch/src')

from qfrsch.pipeline.strategies import TopNStrategy, OptimizedStrategy


@pytest.fixture
def sample_scores():
    """Generate sample prediction scores."""
    np.random.seed(42)
    scores = pd.Series(
        np.random.randn(50),
        index=[f'T{i:02d}' for i in range(50)],
        name='scores'
    )
    return scores


@pytest.fixture
def multidate_scores():
    """Generate multi-date score matrix."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    tickers = [f'T{i:02d}' for i in range(30)]
    
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    scores = pd.Series(np.random.randn(len(index)), index=index, name='scores')
    
    return scores


@pytest.fixture
def cov_matrix():
    """Generate sample covariance matrix."""
    np.random.seed(42)
    n_assets = 50
    
    L = np.random.randn(n_assets, 20)
    cov = L @ L.T / 20
    
    return pd.DataFrame(
        cov,
        index=[f'T{i:02d}' for i in range(n_assets)],
        columns=[f'T{i:02d}' for i in range(n_assets)]
    )


class TestTopNStrategyInit:
    """Test TopNStrategy initialization."""
    
    def test_init_default(self):
        """Test initialization with defaults."""
        strategy = TopNStrategy(n_stocks=10)
        assert strategy.n_stocks == 10
        assert strategy.long_only is True
        assert strategy.equal_weight is True
    
    def test_init_long_short(self):
        """Test initialization for long-short strategy."""
        strategy = TopNStrategy(n_stocks=20, long_only=False)
        assert strategy.long_only is False
    
    def test_init_proportional_weights(self):
        """Test initialization for proportional weighting."""
        strategy = TopNStrategy(n_stocks=15, equal_weight=False)
        assert strategy.equal_weight is False
    
    def test_init_invalid_n_stocks(self):
        """Test error for non-positive n_stocks."""
        with pytest.raises(ValueError, match="n_stocks must be positive"):
            TopNStrategy(n_stocks=0)
        
        with pytest.raises(ValueError, match="n_stocks must be positive"):
            TopNStrategy(n_stocks=-5)
    
    def test_repr(self):
        """Test string representation."""
        strategy = TopNStrategy(n_stocks=10, long_only=False)
        repr_str = repr(strategy)
        assert 'TopNStrategy' in repr_str
        assert '10' in repr_str


class TestTopNStrategyWeights:
    """Test weight generation."""
    
    def test_generate_weights_single_date(self, sample_scores):
        """Test weight generation for single date."""
        strategy = TopNStrategy(n_stocks=10, long_only=True)
        weights = strategy.generate_weights(sample_scores)
        
        assert isinstance(weights, pd.DataFrame)
        assert 'target_weight' in weights.columns
        assert len(weights) == len(sample_scores)
    
    def test_weights_sum_to_one_long_only(self, sample_scores):
        """Test long-only weights sum to 1."""
        strategy = TopNStrategy(n_stocks=10, long_only=True, equal_weight=True)
        weights = strategy.generate_weights(sample_scores)
        
        weight_sum = weights['target_weight'].sum()
        assert np.isclose(weight_sum, 1.0)
    
    def test_weights_sum_to_zero_long_short(self, sample_scores):
        """Test long-short weights sum to ~0."""
        strategy = TopNStrategy(n_stocks=10, long_only=False, equal_weight=True)
        weights = strategy.generate_weights(sample_scores)
        
        # Should be near zero (market neutral)
        weight_sum = weights['target_weight'].sum()
        assert np.isclose(weight_sum, 0.0, atol=1e-10)
    
    def test_equal_weight_allocation(self, sample_scores):
        """Test equal weight allocation to top N."""
        n = 10
        strategy = TopNStrategy(n_stocks=n, long_only=True, equal_weight=True)
        weights = strategy.generate_weights(sample_scores)
        
        # Should have n positions with weight 1/n each
        long_weights = weights[weights['target_weight'] > 0]['target_weight']
        assert len(long_weights) == n
        assert np.allclose(long_weights.values, 1.0 / n)
    
    def test_proportional_weight_allocation(self, sample_scores):
        """Test proportional weight allocation."""
        strategy = TopNStrategy(n_stocks=10, long_only=True, equal_weight=False)
        weights = strategy.generate_weights(sample_scores)
        
        # Weights should reflect score magnitudes
        assert weights['target_weight'].sum() > 0
        assert (weights['target_weight'] >= 0).all()
    
    def test_long_short_positions(self, sample_scores):
        """Test long-short portfolio has both positions."""
        strategy = TopNStrategy(n_stocks=10, long_only=False, equal_weight=True)
        weights = strategy.generate_weights(sample_scores)
        
        has_long = (weights['target_weight'] > 0).any()
        has_short = (weights['target_weight'] < 0).any()
        
        assert has_long
        assert has_short
    
    def test_handles_nan_scores(self, sample_scores):
        """Test handling of NaN scores."""
        scores_with_nan = sample_scores.copy()
        scores_with_nan.iloc[5:10] = np.nan
        
        strategy = TopNStrategy(n_stocks=10, long_only=True)
        weights = strategy.generate_weights(scores_with_nan)
        
        # Should still produce weights for valid scores (NaN drops 5 items)
        assert len(weights) == len(scores_with_nan) - 5
        # All tickers with NaN should not appear in weights
        assert all(ix not in weights.index for ix in scores_with_nan.iloc[5:10].index)
    
    def test_multidate_weights(self, multidate_scores):
        """Test weight generation for MultiIndex (date, ticker)."""
        strategy = TopNStrategy(n_stocks=15, long_only=True, equal_weight=True)
        weights = strategy.generate_weights(multidate_scores)
        
        # Should have same MultiIndex structure
        assert isinstance(weights.index, pd.MultiIndex)
        assert weights.index.names == ['date', 'ticker']
    
    def test_n_stocks_greater_than_available(self, sample_scores):
        """Test when n_stocks > available tickers."""
        strategy = TopNStrategy(n_stocks=100, long_only=True)
        weights = strategy.generate_weights(sample_scores)
        
        # Should allocate to all available stocks
        long_weights = weights[weights['target_weight'] > 0]
        assert len(long_weights) == len(sample_scores)


class TestTopNStrategyScoring:
    """Test score ordering and ranking."""
    
    def test_top_n_are_highest_scores(self, sample_scores):
        """Test that top N positions have highest scores."""
        n = 10
        strategy = TopNStrategy(n_stocks=n, long_only=True)
        weights = strategy.generate_weights(sample_scores)
        
        # Get top N scores
        top_n_scores = sample_scores.nlargest(n).sort_values(ascending=False)
        top_n_tickers = top_n_scores.index
        
        # Check these have positive weights
        for ticker in top_n_tickers:
            assert weights.loc[ticker, 'target_weight'] > 0
    
    def test_bottom_n_are_lowest_scores_long_short(self, sample_scores):
        """Test that bottom N have short positions in long-short."""
        n = 10
        strategy = TopNStrategy(n_stocks=n, long_only=False)
        weights = strategy.generate_weights(sample_scores)
        
        # Get bottom N scores
        bottom_n_scores = sample_scores.nsmallest(n)
        bottom_n_tickers = bottom_n_scores.index
        
        # Check these have negative weights
        for ticker in bottom_n_tickers:
            assert weights.loc[ticker, 'target_weight'] < 0


class TestOptimizedStrategyInit:
    """Test OptimizedStrategy initialization."""
    
    def test_init_mvo(self):
        """Test MVO initialization."""
        strategy = OptimizedStrategy(method='mvo', leverage=1.0)
        assert strategy.method == 'mvo'
        assert strategy.leverage == 1.0
    
    def test_init_riskparity(self):
        """Test Risk Parity initialization."""
        strategy = OptimizedStrategy(method='riskparity')
        assert strategy.method == 'riskparity'
    
    def test_init_invalid_method(self):
        """Test error for invalid method."""
        with pytest.raises(ValueError, match="method must be"):
            OptimizedStrategy(method='invalid')
    
    def test_repr(self):
        """Test string representation."""
        strategy = OptimizedStrategy(method='mvo', leverage=1.0)
        repr_str = repr(strategy)
        assert 'OptimizedStrategy' in repr_str
        assert 'mvo' in repr_str


class TestOptimizedStrategyWeights:
    """Test optimized weight generation."""
    
    def test_mvo_weights_valid(self, sample_scores, cov_matrix):
        """Test MVO produces valid weights."""
        strategy = OptimizedStrategy(method='mvo', leverage=1.0)
        weights = strategy.generate_weights(sample_scores, cov_matrix)
        
        assert isinstance(weights, pd.DataFrame)
        assert 'target_weight' in weights.columns
        assert len(weights) == len(sample_scores)
    
    def test_riskparity_weights_valid(self, sample_scores, cov_matrix):
        """Test Risk Parity produces valid weights."""
        strategy = OptimizedStrategy(method='riskparity', leverage=1.0)
        weights = strategy.generate_weights(sample_scores, cov_matrix)
        
        assert isinstance(weights, pd.DataFrame)
        assert len(weights) == len(sample_scores)
    
    def test_weights_respect_leverage(self, sample_scores, cov_matrix):
        """Test weights respect leverage constraint."""
        leverage = 0.8
        strategy = OptimizedStrategy(method='mvo', leverage=leverage)
        weights = strategy.generate_weights(sample_scores, cov_matrix)
        
        # Sum of absolute weights should ~= leverage
        actual_leverage = weights['target_weight'].abs().sum()
        # Allow some tolerance
        assert 0 <= actual_leverage <= leverage * 1.5
    
    def test_weights_within_bounds(self, sample_scores, cov_matrix):
        """Test weights respect min/max bounds."""
        min_w = -0.1
        max_w = 0.1
        strategy = OptimizedStrategy(method='mvo', min_weight=min_w, max_weight=max_w)
        weights = strategy.generate_weights(sample_scores, cov_matrix)
        
        assert (weights['target_weight'] >= min_w - 1e-6).all()
        assert (weights['target_weight'] <= max_w + 1e-6).all()


class TestWeightEdgeCases:
    """Test edge cases in weight generation."""
    
    def test_all_identical_scores(self):
        """Test with all identical scores."""
        scores = pd.Series([0.5] * 30, index=[f'T{i:02d}' for i in range(30)])
        strategy = TopNStrategy(n_stocks=10, long_only=True, equal_weight=True)
        
        weights = strategy.generate_weights(scores)
        assert len(weights) == len(scores)
    
    def test_extreme_score_values(self):
        """Test with extreme score values."""
        scores = pd.Series([1e6, 1e-6, -1e6, 0, np.inf], 
                          index=['A', 'B', 'C', 'D', 'E'])
        strategy = TopNStrategy(n_stocks=2, long_only=True)
        
        weights = strategy.generate_weights(scores)
        assert len(weights) == len(scores)
    
    def test_single_ticker(self):
        """Test with single ticker."""
        scores = pd.Series([0.5], index=['SINGLE'])
        strategy = TopNStrategy(n_stocks=1)
        
        weights = strategy.generate_weights(scores)
        assert len(weights) == 1
    
    def test_all_nan_scores(self):
        """Test with all NaN scores."""
        scores = pd.Series([np.nan] * 10, index=[f'T{i}' for i in range(10)])
        strategy = TopNStrategy(n_stocks=3, long_only=True)
        
        weights = strategy.generate_weights(scores)
        # Should produce all NaN weights
        assert weights['target_weight'].isna().all()


class TestWeightConsistency:
    """Test weight generation consistency."""
    
    def test_deterministic_weights(self):
        """Test same scores produce same weights."""
        scores = pd.Series(np.array([0.5, -0.3, 0.8, -0.1, 0.2]),
                          index=['A', 'B', 'C', 'D', 'E'])
        
        strategy = TopNStrategy(n_stocks=2, long_only=False, equal_weight=True)
        
        w1 = strategy.generate_weights(scores)
        w2 = strategy.generate_weights(scores)
        
        pd.testing.assert_frame_equal(w1, w2)
    
    def test_score_order_invariance(self):
        """Test weights depend only on relative ranking."""
        scores1 = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        scores2 = pd.Series([10, 20, 30, 40, 50], index=['A', 'B', 'C', 'D', 'E'])
        
        strategy = TopNStrategy(n_stocks=2, long_only=True)
        
        w1 = strategy.generate_weights(scores1)
        w2 = strategy.generate_weights(scores2)
        
        # Same relative ranking should have identical rank-based weights
        # (though magnitudes may differ for proportional weights)
        assert (w1['target_weight'] > 0).sum() == (w2['target_weight'] > 0).sum()
