"""
Test suite for helper utilities.
Tests resampling, label shifting, alignment, and metric computation.
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/home/jywang/project/QFRsch/src')

from qfrsch.utils.helpers import (
    resample_to_freq,
    shift_labels_by_freq,
    align_datasets,
    compute_ic,
    compute_turnover
)


@pytest.fixture
def sample_multiindex_data():
    """Generate sample MultiIndex DataFrame."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    tickers = ['A', 'B', 'C', 'D', 'E']
    
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    data = pd.DataFrame(
        np.random.randn(len(index), 3),
        index=index,
        columns=['col1', 'col2', 'col3']
    )
    
    return data


class TestResampleToFreq:
    """Test frequency resampling function."""
    
    def test_resample_daily_to_daily(self, sample_multiindex_data):
        """Test daily resampling (no-op)."""
        resampled = resample_to_freq(sample_multiindex_data, freq='D')
        
        pd.testing.assert_frame_equal(resampled, sample_multiindex_data)
    
    def test_resample_daily_to_weekly(self, sample_multiindex_data):
        """Test resampling to weekly frequency."""
        resampled = resample_to_freq(sample_multiindex_data, freq='W')
        
        # Should have fewer dates
        n_dates_weekly = len(resampled.index.get_level_values('date').unique())
        n_dates_daily = len(sample_multiindex_data.index.get_level_values('date').unique())
        
        assert n_dates_weekly <= n_dates_daily
        assert isinstance(resampled.index, pd.MultiIndex)
    
    def test_resample_daily_to_monthly(self, sample_multiindex_data):
        """Test resampling to monthly frequency."""
        resampled = resample_to_freq(sample_multiindex_data, freq='M')
        
        n_dates_monthly = len(resampled.index.get_level_values('date').unique())
        n_dates_daily = len(sample_multiindex_data.index.get_level_values('date').unique())
        
        assert n_dates_monthly <= n_dates_daily
    
    def test_resample_preserves_multiindex(self, sample_multiindex_data):
        """Test resampling preserves MultiIndex structure."""
        resampled = resample_to_freq(sample_multiindex_data, freq='W')
        
        assert isinstance(resampled.index, pd.MultiIndex)
        assert resampled.index.names == ['date', 'ticker']
    
    def test_resample_preserves_columns(self, sample_multiindex_data):
        """Test resampling preserves column names."""
        resampled = resample_to_freq(sample_multiindex_data, freq='W')
        
        assert list(resampled.columns) == list(sample_multiindex_data.columns)
    
    def test_resample_agg_last(self, sample_multiindex_data):
        """Test aggregation by 'last' value."""
        resampled = resample_to_freq(sample_multiindex_data, freq='W', agg='last')
        
        # Should have MultiIndex with dates on actual trading days
        dates = resampled.index.get_level_values('date')
        assert all(d in sample_multiindex_data.index.get_level_values('date') for d in dates)
    
    def test_resample_agg_mean(self, sample_multiindex_data):
        """Test aggregation by 'mean'."""
        resampled = resample_to_freq(sample_multiindex_data, freq='W', agg='mean')
        
        assert len(resampled) > 0
        assert not resampled.isna().all().all()  # Should have valid data
    
    def test_resample_agg_first(self, sample_multiindex_data):
        """Test aggregation by 'first'."""
        resampled = resample_to_freq(sample_multiindex_data, freq='W', agg='first')
        
        assert len(resampled) > 0
    
    def test_resample_invalid_multiindex_raises_error(self):
        """Test error for non-MultiIndex input."""
        data = pd.DataFrame(np.random.randn(100, 3))
        
        with pytest.raises(ValueError, match="MultiIndex"):
            resample_to_freq(data, freq='W')
    
    def test_resample_invalid_freq_raises_error(self, sample_multiindex_data):
        """Test error for invalid frequency."""
        with pytest.raises(ValueError, match="freq must be"):
            resample_to_freq(sample_multiindex_data, freq='X')


class TestShiftLabelsByFreq:
    """Test label shifting function."""
    
    def test_shift_daily_by_one(self):
        """Test shifting daily labels by 1 period."""
        dates = pd.date_range('2023-01-01', periods=10, freq='B')
        labels = pd.Series(np.arange(1, 11), index=dates)
        
        shifted = shift_labels_by_freq(labels, freq='D', shift_periods=1)
        
        # First should be NaN, others shifted
        assert pd.isna(shifted.iloc[0])
        assert shifted.iloc[1] == labels.iloc[0]
    
    def test_shift_multiindex_labels(self):
        """Test shifting MultiIndex labels."""
        dates = pd.date_range('2023-01-01', periods=20, freq='B')
        tickers = ['A', 'B']
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        labels = pd.Series(np.random.randn(len(index)), index=index)
        
        shifted = shift_labels_by_freq(labels, freq='D', shift_periods=1)
        
        # Should preserve MultiIndex
        assert isinstance(shifted.index, pd.MultiIndex)
        assert shifted.index.names == ['date', 'ticker']
    
    def test_shift_multiple_periods(self):
        """Test shifting by multiple periods."""
        dates = pd.date_range('2023-01-01', periods=10, freq='B')
        labels = pd.Series(np.arange(1, 11), index=dates)
        
        shifted = shift_labels_by_freq(labels, freq='D', shift_periods=3)
        
        # First 3 should be NaN
        assert shifted.iloc[:3].isna().all()
        assert shifted.iloc[3] == labels.iloc[0]
    
    def test_shift_per_ticker(self):
        """Test shifting maintains per-ticker alignment."""
        dates = pd.date_range('2023-01-01', periods=10, freq='B')
        tickers = ['A', 'B']
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        values = np.concatenate([np.arange(1, 11), np.arange(11, 21)])
        labels = pd.Series(values, index=index)
        
        shifted = shift_labels_by_freq(labels, freq='D', shift_periods=1)
        
        # Check alignment per ticker
        for ticker in tickers:
            ticker_data = shifted.xs(ticker, level='ticker')
            assert pd.isna(ticker_data.iloc[0])


class TestAlignDatasets:
    """Test dataset alignment."""
    
    def test_align_perfect_alignment(self):
        """Test alignment when indices match perfectly."""
        index = pd.Index(['A', 'B', 'C', 'D', 'E'])
        X = pd.DataFrame(np.random.randn(5, 3), index=index)
        y = pd.Series(np.random.randn(5), index=index)
        
        X_aligned, y_aligned = align_datasets(X, y, dropna=False)
        
        pd.testing.assert_frame_equal(X_aligned, X)
        pd.testing.assert_series_equal(y_aligned, y)
    
    def test_align_partial_overlap(self):
        """Test alignment with partial index overlap."""
        X = pd.DataFrame(np.random.randn(10, 3), 
                        index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        y = pd.Series(np.random.randn(8),
                     index=['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        
        X_aligned, y_aligned = align_datasets(X, y, dropna=False)
        
        assert len(X_aligned) == len(y_aligned)
        assert len(X_aligned) == 8  # Intersection size
    
    def test_align_no_overlap(self):
        """Test alignment with no overlap."""
        X = pd.DataFrame(np.random.randn(5, 3), index=['A', 'B', 'C', 'D', 'E'])
        y = pd.Series(np.random.randn(5), index=['F', 'G', 'H', 'I', 'J'])
        
        X_aligned, y_aligned = align_datasets(X, y, dropna=False)
        
        assert len(X_aligned) == 0
        assert len(y_aligned) == 0
    
    def test_align_dropna_removes_nan(self):
        """Test dropna removes NaN values."""
        index = pd.Index(['A', 'B', 'C', 'D', 'E'])
        X = pd.DataFrame(np.random.randn(5, 3), index=index)
        X.iloc[1, 0] = np.nan  # Add NaN to X
        
        y = pd.Series(np.random.randn(5), index=index)
        y.iloc[2] = np.nan  # Add NaN to y
        
        X_aligned, y_aligned = align_datasets(X, y, dropna=True)
        
        # Should remove rows with NaN
        assert not X_aligned.isna().any().any()
        assert not y_aligned.isna().any()


class TestComputeIC:
    """Test Information Coefficient computation."""
    
    def test_ic_perfect_correlation(self):
        """Test IC with perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        
        ic = compute_ic(pd.Series(x), pd.Series(y), method='spearman')
        
        assert np.isclose(ic, 1.0)
    
    def test_ic_perfect_negative_correlation(self):
        """Test IC with perfect negative correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        
        ic = compute_ic(pd.Series(x), pd.Series(y), method='spearman')
        
        assert np.isclose(ic, -1.0)
    
    def test_ic_no_correlation(self):
        """Test IC with no correlation."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        ic = compute_ic(pd.Series(x), pd.Series(y), method='spearman')
        
        # Should be close to 0
        assert -0.3 < ic < 0.3
    
    def test_ic_pearson_method(self):
        """Test Pearson correlation method."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x * 2 + np.random.randn(5) * 0.01
        
        ic = compute_ic(pd.Series(x), pd.Series(y), method='pearson')
        
        # Should be close to 1
        assert ic > 0.99
    
    def test_ic_handles_nan(self):
        """Test IC handles NaN values."""
        x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        y = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
        
        ic = compute_ic(x, y)
        
        # Should return valid IC (NaN removed)
        assert not np.isnan(ic)
    
    def test_ic_insufficient_data(self):
        """Test IC with insufficient non-NaN data."""
        x = pd.Series([1.0, np.nan, np.nan])
        y = pd.Series([1.0, np.nan, np.nan])
        
        ic = compute_ic(x, y)
        
        # Should return NaN
        assert np.isnan(ic)
    
    def test_ic_invalid_method(self):
        """Test error for invalid method."""
        x = pd.Series([1, 2, 3])
        y = pd.Series([1, 2, 3])
        
        with pytest.raises(ValueError, match="method must be"):
            compute_ic(x, y, method='invalid')


class TestComputeTurnover:
    """Test portfolio turnover computation."""
    
    def test_turnover_no_change(self):
        """Test turnover when weights don't change."""
        weights = pd.Series([0.2, 0.3, 0.5], index=['A', 'B', 'C'])
        
        turnover = compute_turnover(weights, weights)
        
        assert np.isclose(turnover, 0.0)
    
    def test_turnover_complete_change(self):
        """Test turnover with complete portfolio change."""
        w_prev = pd.Series([0.5, 0.5, 0.0], index=['A', 'B', 'C'])
        w_curr = pd.Series([0.0, 0.0, 1.0], index=['A', 'B', 'C'])
        
        turnover = compute_turnover(w_curr, w_prev)
        
        # Should be 1.0 (complete replacement)
        assert np.isclose(turnover, 1.0)
    
    def test_turnover_partial_change(self):
        """Test turnover with partial portfolio change."""
        w_prev = pd.Series([0.5, 0.5, 0.0], index=['A', 'B', 'C'])
        w_curr = pd.Series([0.3, 0.4, 0.3], index=['A', 'B', 'C'])
        
        turnover = compute_turnover(w_curr, w_prev)
        
        # Should be between 0 and 1
        assert 0 < turnover < 1
    
    def test_turnover_partial_overlap(self):
        """Test turnover with partial ticker overlap."""
        w_prev = pd.Series([0.5, 0.5], index=['A', 'B'])
        w_curr = pd.Series([0.0, 0.0, 1.0], index=['B', 'C', 'D'])
        
        turnover = compute_turnover(w_curr, w_prev)
        
        # Should compute for overlapping tickers
        assert 0 <= turnover <= 1
    
    def test_turnover_no_overlap(self):
        """Test turnover with no overlapping tickers."""
        w_prev = pd.Series([0.5, 0.5], index=['A', 'B'])
        w_curr = pd.Series([0.5, 0.5], index=['C', 'D'])
        
        turnover = compute_turnover(w_curr, w_prev)
        
        # No common tickers = complete replacement
        assert turnover == 1.0


class TestHelperConsistency:
    """Test helper functions consistency."""
    
    def test_resample_then_shift(self, sample_multiindex_data):
        """Test combining resample and shift operations."""
        resampled = resample_to_freq(sample_multiindex_data, freq='W')
        
        # Extract single column as time series for shift
        ts = resampled.iloc[:, 0]
        shifted = shift_labels_by_freq(ts, freq='W', shift_periods=1)
        
        assert len(shifted) == len(ts)
    
    def test_align_after_resample(self, sample_multiindex_data):
        """Test alignment after resampling."""
        resampled = resample_to_freq(sample_multiindex_data, freq='W')
        
        X = resampled.iloc[:, :2]
        y = resampled.iloc[:, 2]
        
        X_aligned, y_aligned = align_datasets(X, y, dropna=False)
        
        assert len(X_aligned) == len(y_aligned)


class TestHelperEdgeCases:
    """Test edge cases in helper functions."""
    
    def test_resample_single_day(self):
        """Test resampling with single day of data."""
        dates = pd.date_range('2023-01-01', periods=1, freq='D')
        tickers = ['A', 'B']
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        data = pd.DataFrame(np.random.randn(2, 2), index=index)
        
        resampled = resample_to_freq(data, freq='W')
        
        assert len(resampled) == 2
    
    def test_shift_empty_series(self):
        """Test shifting empty series."""
        empty = pd.Series([], dtype=float)
        
        shifted = shift_labels_by_freq(empty, freq='D', shift_periods=1)
        
        assert len(shifted) == 0
    
    def test_ic_single_value(self):
        """Test IC with single value."""
        x = pd.Series([1.0])
        y = pd.Series([1.0])
        
        ic = compute_ic(x, y)
        
        # Should return NaN (insufficient data)
        assert np.isnan(ic)
