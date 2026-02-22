"""
Tests for Analysis Module
"""

import numpy as np
import pandas as pd
import pytest

from qfrsch.analysis import metrics, factor_eval, attribution


class TestMetrics:
    """Test metrics calculation."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = pd.Series(
            np.random.normal(0.0005, 0.01, 252),
            index=dates
        )
        return returns
    
    def test_annual_return(self, sample_returns):
        """Test annual return calculation."""
        annual_ret = metrics.calculate_annual_return(sample_returns)
        assert isinstance(annual_ret, float)
        assert -1 < annual_ret < 1  # Sanity check
    
    def test_annual_volatility(self, sample_returns):
        """Test annual volatility calculation."""
        annual_vol = metrics.calculate_annual_volatility(sample_returns)
        assert isinstance(annual_vol, float)
        assert annual_vol >= 0
    
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = metrics.calculate_sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
    
    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = metrics.calculate_sortino_ratio(sample_returns)
        assert isinstance(sortino, float)
    
    def test_calmar_ratio(self, sample_returns):
        """Test Calmar ratio calculation."""
        calmar = metrics.calculate_calmar_ratio(sample_returns)
        assert isinstance(calmar, float)
    
    def test_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        mdd = metrics.calculate_max_drawdown(sample_returns)
        assert isinstance(mdd, float)
        assert mdd <= 0
    
    def test_win_rate(self, sample_returns):
        """Test win rate calculation."""
        wr = metrics.calculate_win_rate(sample_returns)
        assert isinstance(wr, float)
        assert 0 <= wr <= 1
    
    def test_excess_return(self, sample_returns):
        """Test excess return calculation."""
        benchmark_returns = pd.Series(
            np.random.normal(0.0003, 0.008, 252),
            index=sample_returns.index
        )
        excess = metrics.calculate_excess_return(sample_returns, benchmark_returns)
        assert isinstance(excess, pd.Series)
        assert len(excess) > 0
    
    def test_beta(self, sample_returns):
        """Test beta calculation."""
        benchmark_returns = pd.Series(
            np.random.normal(0.0003, 0.008, 252),
            index=sample_returns.index
        )
        beta = metrics.calculate_beta(sample_returns, benchmark_returns)
        assert isinstance(beta, float)
    
    def test_newey_west_ttest(self, sample_returns):
        """Test Newey-West adjusted t-test."""
        t_stat, p_value, annual_ret = metrics.newey_west_ttest(sample_returns)
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert isinstance(annual_ret, float)
        assert 0 <= p_value <= 1


class TestFactorEval:
    """Test factor evaluation functions."""
    
    @pytest.fixture
    def sample_factor_data(self):
        """Generate sample factor and return data."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        tickers = ['A', 'B', 'C', 'D']
        
        factor_values = pd.DataFrame(
            np.random.randn(50, 4),
            index=dates,
            columns=tickers
        )
        
        forward_returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (50, 4)),
            index=dates,
            columns=tickers
        )
        
        return factor_values, forward_returns
    
    def test_calculate_ic(self, sample_factor_data):
        """Test IC calculation."""
        factor_values, forward_returns = sample_factor_data
        ic_series = factor_eval.calculate_ic(factor_values, forward_returns)
        assert isinstance(ic_series, pd.Series)
        assert len(ic_series) > 0
        assert (-1 <= ic_series).all() and (ic_series <= 1).all()
    
    def test_calculate_rank_ic(self, sample_factor_data):
        """Test Rank IC calculation."""
        factor_values, forward_returns = sample_factor_data
        rank_ic_series = factor_eval.calculate_rank_ic(factor_values, forward_returns)
        assert isinstance(rank_ic_series, pd.Series)
        assert len(rank_ic_series) > 0
    
    def test_ic_statistics(self, sample_factor_data):
        """Test IC statistics calculation."""
        factor_values, forward_returns = sample_factor_data
        ic_series = factor_eval.calculate_ic(factor_values, forward_returns)
        stats = factor_eval.calculate_ic_statistics(ic_series)
        
        assert 'ic_mean' in stats
        assert 'ic_std' in stats
        assert 'ic_positive_pct' in stats
        assert 'ic_ir' in stats
        assert 0 <= stats['ic_positive_pct'] <= 1
    
    def test_quantile_backtest(self, sample_factor_data):
        """Test quantile-based backtest."""
        factor_values, forward_returns = sample_factor_data
        result = factor_eval.quantile_backtest(factor_values, forward_returns, num_quantiles=3)
        
        assert 'quantile_returns' in result
        assert 'quantile_cumret' in result
        assert 'quantile_annual_ret' in result
        assert 'quantile_annual_vol' in result
        assert 'high_minus_low' in result
    
    def test_fama_macbeth_regression(self, sample_factor_data):
        """Test Fama-MacBeth regression."""
        factor_values, forward_returns = sample_factor_data
        fm_result = factor_eval.fama_macbeth_regression(factor_values, forward_returns)
        
        assert 'factor_premium' in fm_result
        assert 't_stat' in fm_result
        assert 'p_value' in fm_result
        assert 0 <= fm_result['p_value'] <= 1
    
    def test_factor_stability_coefficient(self, sample_factor_data):
        """Test factor stability coefficient."""
        factor_values, _ = sample_factor_data
        fsc = factor_eval.calculate_factor_stability_coefficient(factor_values, window=10)
        assert isinstance(fsc, pd.Series)
        assert len(fsc) > 0
    
    def test_factor_autocorrelation(self, sample_factor_data):
        """Test factor autocorrelation."""
        factor_values, _ = sample_factor_data
        autocorr = factor_eval.calculate_factor_autocorrelation(factor_values, lag=1)
        assert isinstance(autocorr, float)
        assert -1 <= autocorr <= 1


class TestAttribution:
    """Test attribution analysis."""
    
    def test_calculate_turnover(self):
        """Test turnover calculation."""
        current = pd.Series({'A': 0.5, 'B': 0.5})
        previous = pd.Series({'A': 0.3, 'B': 0.7})
        turnover = attribution.calculate_turnover(current, previous)
        
        assert isinstance(turnover, float)
        assert 0 <= turnover <= 1
        assert abs(turnover - 0.2) < 1e-6  # Should be 0.2
    
    def test_calculate_active_return(self):
        """Test active return calculation."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        strategy_ret = pd.Series(np.random.normal(0.001, 0.01, 50), index=dates)
        benchmark_ret = pd.Series(np.random.normal(0.0005, 0.008, 50), index=dates)
        
        active = attribution.calculate_active_return(strategy_ret, benchmark_ret)
        assert isinstance(active, pd.Series)
        assert len(active) > 0
    
    def test_calculate_active_risk(self):
        """Test active risk calculation."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        strategy_ret = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        benchmark_ret = pd.Series(np.random.normal(0.0005, 0.008, 100), index=dates)
        
        active_risk = attribution.calculate_active_risk(strategy_ret, benchmark_ret)
        assert isinstance(active_risk, float)
        assert active_risk >= 0
    
    def test_calculate_position_concentration(self):
        """Test position concentration calculation."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        weights = pd.DataFrame(
            np.random.uniform(0, 0.3, (10, 5)),
            index=dates
        )
        weights = weights.div(weights.sum(axis=1), axis=0)  # Normalize to sum to 1
        
        concentration = attribution.calculate_position_concentration(weights)
        assert isinstance(concentration, pd.Series)
        assert (concentration > 0).all()
        assert (concentration <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
