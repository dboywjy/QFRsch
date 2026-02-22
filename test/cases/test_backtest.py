"""
Tests for Backtest Module
"""

import numpy as np
import pandas as pd
import pytest

from qfrsch.backtest.models import BacktestConfig, BacktestResult
from qfrsch.backtest.engine import SimpleEngine


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        assert config.initial_capital == 1_000_000
        assert config.commission_rate == 0.001
        assert config.slippage_rate == 0.0005
        assert config.rebalance_frequency == 'D'
        assert config.risk_free_rate == 0.02
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=500_000,
            commission_rate=0.002,
            slippage_rate=0.001,
        )
        assert config.initial_capital == 500_000
        assert config.commission_rate == 0.002
        assert config.slippage_rate == 0.001


class TestSimpleEngine:
    """Test SimpleEngine backtest functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """
        Create sample OHLCV and target weights data.
        
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (ohlcv_df, target_weights_df)
        """
        
        # Create OHLCV data for 2 tickers over 10 days
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        
        records = []
        for date in dates:
            # Ticker A: steady uptrend
            records.append({
                'date': date,
                'ticker': 'A',
                'open': 100 + (date - dates[0]).days * 0.5,
                'high': 102 + (date - dates[0]).days * 0.5,
                'low': 99 + (date - dates[0]).days * 0.5,
                'close': 101 + (date - dates[0]).days * 0.5,
                'volume': 1000000,
            })
            
            # Ticker B: steady downtrend
            records.append({
                'date': date,
                'ticker': 'B',
                'open': 100 - (date - dates[0]).days * 0.3,
                'high': 101 - (date - dates[0]).days * 0.3,
                'low': 98 - (date - dates[0]).days * 0.3,
                'close': 99.5 - (date - dates[0]).days * 0.3,
                'volume': 1000000,
            })
        
        ohlcv_df = pd.DataFrame(records)
        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])
        ohlcv_df.set_index('date', inplace=True)
        
        # Create target weights: 60% A, 40% B
        weights_data = pd.DataFrame(
            index=pd.to_datetime(dates),
            data={
                'A': [0.6] * len(dates),
                'B': [0.4] * len(dates),
            }
        )
        
        return ohlcv_df, weights_data
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = SimpleEngine(config)
        assert engine.config.initial_capital == 1_000_000
        assert engine.ohlcv_df is None
        assert engine.target_weights_df is None
    
    def test_engine_run_basic(self, sample_data):
        """Test basic engine run."""
        ohlcv_df, target_weights_df = sample_data
        
        config = BacktestConfig(
            initial_capital=1_000_000,
            commission_rate=0.001,
            slippage_rate=0.0005,
        )
        engine = SimpleEngine(config)
        result = engine.run(ohlcv_df, target_weights_df)
        
        # Check result structure
        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity_curve, pd.DataFrame)
        assert isinstance(result.trades, pd.DataFrame)
        assert isinstance(result.holdings, pd.DataFrame)
        assert isinstance(result.daily_returns, pd.Series)
        assert isinstance(result.daily_weights, pd.DataFrame)
        assert isinstance(result.metrics, dict)
    
    def test_engine_equity_curve(self, sample_data):
        """Test equity curve tracking."""
        ohlcv_df, target_weights_df = sample_data
        
        config = BacktestConfig(initial_capital=1_000_000)
        engine = SimpleEngine(config)
        result = engine.run(ohlcv_df, target_weights_df)
        
        # Check equity curve columns
        assert 'total_value' in result.equity_curve.columns
        assert 'cash' in result.equity_curve.columns
        assert 'positions_value' in result.equity_curve.columns
        
        # Check values are non-negative
        assert (result.equity_curve['total_value'] > 0).all()
    
    def test_engine_metrics(self, sample_data):
        """Test performance metrics calculation."""
        ohlcv_df, target_weights_df = sample_data
        
        config = BacktestConfig(initial_capital=1_000_000)
        engine = SimpleEngine(config)
        result = engine.run(ohlcv_df, target_weights_df)
        
        # Check key metrics exist
        assert 'total_return' in result.metrics
        assert 'sharpe_ratio' in result.metrics
        assert 'max_drawdown' in result.metrics
        assert 'volatility' in result.metrics
        assert 'win_rate' in result.metrics
        
        # Check metric types
        assert isinstance(result.metrics['total_return'], (int, float))
        assert isinstance(result.metrics['max_drawdown'], (int, float))
    
    def test_engine_with_zero_weights(self, sample_data):
        """Test engine with zero target weights (flat)."""
        ohlcv_df, target_weights_df = sample_data
        
        # All weights to zero
        target_weights_df['A'] = 0.0
        target_weights_df['B'] = 0.0
        
        config = BacktestConfig(initial_capital=1_000_000)
        engine = SimpleEngine(config)
        result = engine.run(ohlcv_df, target_weights_df)
        
        # Should end with cash approximately equal to initial capital
        # (minus some slippage from closing initial zero positions)
        assert result.equity_curve['total_value'].iloc[-1] > 0
    
    def test_engine_invalid_input(self):
        """Test engine with empty input."""
        config = BacktestConfig()
        engine = SimpleEngine(config)
        
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            engine.run(empty_df, empty_df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
