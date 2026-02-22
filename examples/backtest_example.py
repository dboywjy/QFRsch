"""
Example: Basic Backtest Engine Usage
Demonstrates how to use SimpleEngine with OHLCV data and target weights.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from qfrsch.backtest import SimpleEngine, BacktestConfig


def create_sample_data():
    """
    Create sample OHLCV and target weights data for demonstration.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (ohlcv_df, target_weights_df)
    """
    
    # Generate 252 trading days (one year)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Simulate 3 stock assets
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Create OHLCV data
    ohlcv_records = []
    np.random.seed(42)
    
    for ticker in tickers:
        # Generate random price movement
        prices = 100 * (1 + np.random.randn(252).cumsum() * 0.01)
        
        for i, date in enumerate(dates):
            ohlcv_records.append({
                'date': date,
                'ticker': ticker,
                'open': prices[i] * (1 + np.random.uniform(-0.01, 0.01)),
                'high': prices[i] * (1 + np.random.uniform(0, 0.02)),
                'low': prices[i] * (1 - np.random.uniform(0, 0.02)),
                'close': prices[i],
                'volume': np.random.randint(1_000_000, 10_000_000),
            })
    
    ohlcv_df = pd.DataFrame(ohlcv_records)
    ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])
    ohlcv_df.set_index('date', inplace=True)
    
    # Create target weights (equal weight portfolio)
    weights_data = {
        'AAPL': [1/3] * len(dates),
        'GOOGL': [1/3] * len(dates),
        'MSFT': [1/3] * len(dates),
    }
    target_weights_df = pd.DataFrame(weights_data, index=pd.to_datetime(dates))
    
    return ohlcv_df, target_weights_df


def main():
    """Main example execution."""
    
    print("=" * 70)
    print("QFRsch Backtest Engine - Basic Example")
    print("=" * 70)
    
    # Step 1: Prepare data
    print("\n[Step 1] Preparing sample data...")
    ohlcv_df, target_weights_df = create_sample_data()
    print(f"  - OHLCV data shape: {ohlcv_df.shape}")
    print(f"  - Target weights shape: {target_weights_df.shape}")
    print(f"  - Assets: {sorted(target_weights_df.columns.tolist())}")
    print(f"  - Date range: {ohlcv_df.index.min().date()} to {ohlcv_df.index.max().date()}")
    
    # Step 2: Configure backtest
    print("\n[Step 2] Configuring backtest engine...")
    config = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=0.001,  # 0.1% commission
        slippage_rate=0.0005,   # 0.05% slippage
        risk_free_rate=0.02,    # 2% annual risk-free rate
    )
    print(f"  - Initial capital: ${config.initial_capital:,.0f}")
    print(f"  - Commission rate: {config.commission_rate:.03%}")
    print(f"  - Slippage rate: {config.slippage_rate:.03%}")
    
    # Step 3: Run backtest
    print("\n[Step 3] Running backtest simulation...")
    engine = SimpleEngine(config)
    result = engine.run(ohlcv_df, target_weights_df)
    print("  ✓ Backtest completed successfully")
    
    # Step 4: Display results
    print("\n[Step 4] Backtest Results")
    print("-" * 70)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = result.metrics
    print(f"  - Total Return:       {metrics['total_return']:>10.2%}")
    print(f"  - Annualized Return:  {metrics['annualized_return']:>10.2%}")
    print(f"  - Volatility:         {metrics['volatility']:>10.2%}")
    print(f"  - Sharpe Ratio:       {metrics['sharpe_ratio']:>10.4f}")
    print(f"  - Calmar Ratio:       {metrics['calmar_ratio']:>10.4f}")
    print(f"  - Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
    print(f"  - Win Rate:           {metrics['win_rate']:>10.2%}")
    
    # Equity curve
    print("\nEquity Curve Summary:")
    equity = result.equity_curve['total_value']
    print(f"  - Starting Value:     ${config.initial_capital:>13,.0f}")
    print(f"  - Final Value:        ${equity.iloc[-1]:>13,.0f}")
    print(f"  - Max Value:          ${equity.max():>13,.0f}")
    print(f"  - Min Value:          ${equity.min():>13,.0f}")
    
    # Daily statistics
    daily_ret = result.daily_returns
    print("\nDaily Return Statistics:")
    print(f"  - Mean Daily Return:  {daily_ret.mean():>10.4%}")
    print(f"  - Std Daily Return:   {daily_ret.std():>10.4%}")
    print(f"  - Best Day:           {daily_ret.max():>10.4%}")
    print(f"  - Worst Day:          {daily_ret.min():>10.4%}")
    
    # Trade statistics
    print("\nTrade Statistics:")
    trades = result.trades
    if not trades.empty:
        print(f"  - Total Trades:       {len(trades):>10}")
        print(f"  - Total Commission:   ${trades['commission'].sum():>13,.2f}")
        print(f"  - Buy Trades:         {len(trades[trades['action'] == 'BUY']):>10}")
        print(f"  - Sell Trades:        {len(trades[trades['action'] == 'SELL']):>10}")
    else:
        print("  - No trades executed")
    
    # Holdings snapshot
    print("\nFinal Holdings (Last Date):")
    final_holdings = result.holdings.iloc[-1]
    final_weights = result.daily_weights.iloc[-1]
    for ticker in sorted(target_weights_df.columns):
        qty = final_holdings.get(ticker, 0)
        wgt = final_weights.get(ticker, 0)
        price = ohlcv_df[ohlcv_df['ticker'] == ticker].iloc[-1]['close']
        value = qty * price if not np.isnan(qty) else 0
        print(f"  - {ticker}:  {qty:>12.2f} shares @ ${price:6.2f}  ({wgt:>6.2%})")
    
    print("\n" + "=" * 70)
    print("End of Example")
    print("=" * 70)


if __name__ == '__main__':
    main()
