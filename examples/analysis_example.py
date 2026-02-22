"""
Example: Comprehensive Analysis Workflow
Demonstrates the complete analysis pipeline from backtest results to HTML report.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from qfrsch.analysis import metrics, factor_eval, attribution, reporter


def create_sample_backtest_data():
    """Create sample backtest data and factor data."""
    
    # Time period
    dates = pd.date_range('2022-01-01', periods=504, freq='D')
    
    # Portfolio returns
    np.random.seed(42)
    portfolio_returns = pd.Series(
        np.random.normal(0.0006, 0.012, 504),
        index=dates,
        name='Portfolio Returns'
    )
    
    # Benchmark returns (slightly less volatile)
    benchmark_returns = pd.Series(
        np.random.normal(0.0004, 0.01, 504),
        index=dates,
        name='Benchmark Returns'
    )
    
    # Equity curves (cumulative)
    portfolio_equity = (1 + portfolio_returns).cumprod() * 1_000_000
    benchmark_equity = (1 + benchmark_returns).cumprod() * 1_000_000
    
    # Factor data (for 20 stocks)
    tickers = [f'Stock_{i:02d}' for i in range(1, 21)]
    
    # Create factor values (momentum-like)
    factor_values = pd.DataFrame(
        np.random.randn(504, 20),
        index=dates,
        columns=tickers
    )
    
    # Create forward returns with some correlation to factor
    forward_returns = pd.DataFrame(
        index=dates,
        columns=tickers
    )
    for i, ticker in enumerate(tickers):
        forward_returns[ticker] = (
            0.0005 * factor_values[ticker] +
            np.random.normal(0.0001, 0.02, 504)
        )
    
    return {
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'portfolio_equity': portfolio_equity,
        'benchmark_equity': benchmark_equity,
        'factor_values': factor_values,
        'forward_returns': forward_returns,
    }


def analyze_performance(data):
    """Perform comprehensive performance analysis."""
    
    print("\n" + "="*70)
    print("QFRsch Analysis Example - Performance Metrics")
    print("="*70)
    
    strategy_ret = data['portfolio_returns']
    benchmark_ret = data['benchmark_returns']
    
    # Basic metrics
    print("\n[Basic Performance Metrics]")
    print(f"  Annual Return (Strategy):    {metrics.calculate_annual_return(strategy_ret):>10.2%}")
    print(f"  Annual Return (Benchmark):   {metrics.calculate_annual_return(benchmark_ret):>10.2%}")
    
    print(f"  Annual Volatility (Strat):   {metrics.calculate_annual_volatility(strategy_ret):>10.2%}")
    print(f"  Annual Volatility (Bench):   {metrics.calculate_annual_volatility(benchmark_ret):>10.2%}")
    
    print(f"  Sharpe Ratio:                {metrics.calculate_sharpe_ratio(strategy_ret):>10.4f}")
    print(f"  Sortino Ratio:               {metrics.calculate_sortino_ratio(strategy_ret):>10.4f}")
    print(f"  Calmar Ratio:                {metrics.calculate_calmar_ratio(strategy_ret):>10.4f}")
    print(f"  Max Drawdown:                {metrics.calculate_max_drawdown(strategy_ret):>10.2%}")
    print(f"  Win Rate:                    {metrics.calculate_win_rate(strategy_ret):>10.2%}")
    
    # Relative metrics
    print("\n[Relative Metrics]")
    print(f"  Excess Return (Annual):      {metrics.calculate_excess_return(strategy_ret, benchmark_ret).mean() * 252:>10.2%}")
    print(f"  Information Ratio:           {metrics.calculate_information_ratio(strategy_ret, benchmark_ret):>10.4f}")
    print(f"  Beta:                        {metrics.calculate_beta(strategy_ret, benchmark_ret):>10.4f}")
    print(f"  Alpha (Annual):              {metrics.calculate_alpha(strategy_ret, benchmark_ret):>10.2%}")
    print(f"  Correlation with Benchmark: {metrics.calculate_correlation_with_benchmark(strategy_ret, benchmark_ret):>10.4f}")
    
    # Statistical testing
    print("\n[Statistical Testing (Newey-West Adjusted)]")
    t_stat, p_value, annual_ret = metrics.newey_west_ttest(strategy_ret)
    print(f"  T-Statistic:                 {t_stat:>10.4f}")
    print(f"  P-Value:                     {p_value:>10.4f}")
    print(f"  Significant @ 5%:            {'Yes' if p_value < 0.05 else 'No':>10}")


def analyze_factors(data):
    """Perform comprehensive factor analysis."""
    
    print("\n" + "="*70)
    print("QFRsch Analysis Example - Factor Analysis")
    print("="*70)
    
    factor_vals = data['factor_values']
    returns = data['forward_returns']
    
    # IC analysis
    print("\n[Information Coefficient (IC) Analysis]")
    ic_series = factor_eval.calculate_ic(factor_vals, returns)
    ic_stats = factor_eval.calculate_ic_statistics(ic_series)
    
    print(f"  IC Mean:                     {ic_stats['ic_mean']:>10.4f}")
    print(f"  IC Std Dev:                  {ic_stats['ic_std']:>10.4f}")
    print(f"  IC > 0 (pct):                {ic_stats['ic_positive_pct']:>10.2%}")
    print(f"  IC > 0.03 (pct):             {ic_stats['ic_strong_positive_pct']:>10.2%}")
    print(f"  IC IR:                       {ic_stats['ic_ir']:>10.4f}")
    
    # Factor stability
    print("\n[Factor Stability]")
    fsc = factor_eval.calculate_factor_stability_coefficient(factor_vals, window=60)
    autocorr = factor_eval.calculate_factor_autocorrelation(factor_vals, lag=1)
    
    print(f"  Factor Stability Coeff (avg):{fsc.mean():>10.4f}")
    print(f"  Factor Autocorrelation:      {autocorr:>10.4f}")
    
    # Quantile analysis
    print("\n[Quantile Backtest (5 Quantiles)]")
    quantile_result = factor_eval.quantile_backtest(factor_vals, returns, num_quantiles=5)
    quantile_ret = quantile_result['quantile_annual_ret']
    
    for q in range(1, 6):
        print(f"  Q{q} Annual Return:           {quantile_ret[q]:>10.2%}")
    
    high_minus_low = (quantile_result['high_minus_low'].mean() * 252)
    print(f"  Q5-Q1 Excess Return (Annual):{high_minus_low:>10.2%}")
    
    # Fama-MacBeth regression
    print("\n[Fama-MacBeth Regression]")
    fm_result = factor_eval.fama_macbeth_regression(factor_vals, returns)
    
    print(f"  Factor Risk Premium:         {fm_result['factor_premium']:>10.4f}")
    print(f"  T-Statistic:                 {fm_result['t_stat']:>10.4f}")
    print(f"  P-Value:                     {fm_result['p_value']:>10.4f}")
    print(f"  Significant @ 5%:            {'Yes' if fm_result['p_value'] < 0.05 else 'No':>10}")


def analyze_attribution(data):
    """Perform attribution analysis."""
    
    print("\n" + "="*70)
    print("QFRsch Analysis Example - Attribution Analysis")
    print("="*70)
    
    strategy_ret = data['portfolio_returns']
    benchmark_ret = data['benchmark_returns']
    
    # Active return analysis
    print("\n[Active Return Analysis]")
    active_ret = attribution.calculate_active_return(strategy_ret, benchmark_ret)
    active_risk = attribution.calculate_active_risk(strategy_ret, benchmark_ret)
    
    print(f"  Average Daily Active Return: {active_ret.mean():>10.4%}")
    print(f"  Annual Active Risk:          {active_risk:>10.2%}")
    
    # Turnover analysis
    print("\n[Turnover Analysis]")
    current_weights = pd.Series({'Stock_01': 0.3, 'Stock_02': 0.3, 'Stock_03': 0.4})
    previous_weights = pd.Series({'Stock_01': 0.2, 'Stock_02': 0.4, 'Stock_03': 0.4})
    turnover = attribution.calculate_turnover(current_weights, previous_weights)
    
    print(f"  Portfolio Turnover:          {turnover:>10.2%}")
    
    # Position concentration
    print("\n[Position Concentration]")
    dates = pd.date_range('2022-01-01', periods=252, freq='D')
    weights = pd.DataFrame(
        np.random.uniform(0.01, 0.1, (252, 20)),
        index=dates
    )
    weights = weights.div(weights.sum(axis=1), axis=0)
    
    concentration = attribution.calculate_position_concentration(weights)
    print(f"  Avg Concentration (last date): {concentration.iloc[-1]:>10.4f}")
    print(f"  Mean Concentration:          {concentration.mean():>10.4f}")


def generate_report(data):
    """Generate HTML report."""
    
    print("\n" + "="*70)
    print("QFRsch Analysis Example - Generating HTML Report")
    print("="*70)
    
    try:
        report_html = reporter.create_html_report(
            strategy_returns=data['portfolio_returns'],
            equity_curve=data['portfolio_equity'],
            factor_values=data['factor_values'],
            forward_returns=data['forward_returns'],
            benchmark_returns=data['benchmark_returns'],
            benchmark_curve=data['benchmark_equity'],
            title="QFRsch Strategy Analysis - Example Report",
            output_path='/tmp/qfrsch_analysis_report.html'
        )
        
        print("\n✓ HTML report generated successfully!")
        print("  Output: /tmp/qfrsch_analysis_report.html")
        
    except ImportError as e:
        print(f"\n⚠ Cannot generate HTML report: {e}")
        print("  Install plotly via: pip install plotly")


def main():
    """Main analysis workflow."""
    
    print("="*70)
    print("QFRsch - Comprehensive Analysis Example")
    print("="*70)
    print("\nGenerating sample data...")
    
    data = create_sample_backtest_data()
    print(f"✓ Data period: {data['portfolio_equity'].index.min().date()} to {data['portfolio_equity'].index.max().date()}")
    print(f"✓ Number of stocks in factor universe: {len(data['factor_values'].columns)}")
    
    # Perform analyses
    analyze_performance(data)
    analyze_factors(data)
    analyze_attribution(data)
    
    # Generate report
    generate_report(data)
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)


if __name__ == '__main__':
    main()
