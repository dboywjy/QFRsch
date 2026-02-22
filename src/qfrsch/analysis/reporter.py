"""
Automated Tearsheet Reporter Module
Generates comprehensive visual and textual analysis reports using Plotly.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from qfrsch.analysis import metrics, factor_eval, attribution


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
    title: str = "Portfolio Equity Curve",
    log_scale: bool = False
) -> go.Figure:
    """
    Plot equity curve with optional benchmark comparison.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value. Index: date, values: equity.
    benchmark_curve : pd.Series, optional
        Benchmark daily value for comparison.
    title : str, default="Portfolio Equity Curve"
        Chart title.
    log_scale : bool, default=False
        Use logarithmic y-axis.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive chart.
    """
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization. Install via: pip install plotly")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        name='Portfolio',
        mode='lines',
        line=dict(color='#1f77b4', width=2),
    ))
    
    if benchmark_curve is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_curve.index,
            y=benchmark_curve.values,
            name='Benchmark',
            mode='lines',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Equity Value',
        hovermode='x unified',
        template='plotly_white',
        yaxis_type='log' if log_scale else 'linear',
        height=600,
    )
    
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Portfolio Drawdown"
) -> go.Figure:
    """
    Plot portfolio drawdown over time.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns. Index: date.
    title : str, default="Portfolio Drawdown"
        Chart title.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Drawdown chart.
    """
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization. Install via: pip install plotly")
    
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        name='Drawdown',
        mode='lines',
        fill='tozeroy',
        line=dict(color='#d62728', width=1),
        fillcolor='rgba(214, 39, 40, 0.3)',
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
    )
    
    return fig


def plot_quantile_returns(
    quantile_result: Dict,
    title: str = "Factor Quantile Analysis"
) -> go.Figure:
    """
    Plot cumulative returns by quantile group.
    
    Parameters
    ----------
    quantile_result : dict
        Output from factor_eval.quantile_backtest().
    title : str, default="Factor Quantile Analysis"
        Chart title.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Quantile returns chart.
    """
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization. Install via: pip install plotly")
    
    cumret_df = quantile_result['quantile_cumret']
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for col in cumret_df.columns:
        fig.add_trace(go.Scatter(
            x=cumret_df.index,
            y=cumret_df[col],
            name=f'Quantile {col}',
            mode='lines',
            line=dict(width=2),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        template='plotly_white',
        height=500,
    )
    
    return fig


def plot_ic_distribution(
    ic_series: pd.Series,
    title: str = "Information Coefficient Distribution"
) -> go.Figure:
    """
    Plot IC distribution and statistics.
    
    Parameters
    ----------
    ic_series : pd.Series
        Daily IC values from factor_eval.calculate_ic().
    title : str, default="Information Coefficient Distribution"
        Chart title.
        
    Returns
    -------
    plotly.graph_objects.Figure
        IC distribution chart with histogram and statistics.
    """
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization. Install via: pip install plotly")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("IC Distribution", "IC Time Series")
    )
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=ic_series.values,
        nbinsx=50,
        name='IC',
        showlegend=True,
    ), row=1, col=1)
    
    # Time series
    fig.add_trace(go.Scatter(
        x=ic_series.index,
        y=ic_series.values,
        name='IC',
        mode='lines',
        line=dict(width=1),
        showlegend=False,
    ), row=1, col=2)
    
    fig.update_xaxes(title_text="IC Value", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="IC", row=1, col=2)
    
    fig.update_layout(
        title=title,
        height=500,
        template='plotly_white',
        hovermode='x unified',
    )
    
    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    title: str = "Rolling Sharpe Ratio"
) -> go.Figure:
    """
    Plot rolling Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    window : int, default=60
        Rolling window size (days).
    title : str, default="Rolling Sharpe Ratio"
        Chart title.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Rolling Sharpe chart.
    """
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for visualization. Install via: pip install plotly")
    
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: metrics.calculate_sharpe_ratio(x, periods_per_year=252)
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        name='Rolling Sharpe',
        mode='lines',
        line=dict(color='#2ca02c', width=2),
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{title} ({window}-day)",
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        template='plotly_white',
        height=400,
    )
    
    return fig


def generate_metrics_table(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Generate comprehensive metrics table.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns.
    benchmark_returns : pd.Series, optional
        Daily benchmark returns.
    risk_free_rate : float, default=0.02
        Annual risk-free rate.
        
    Returns
    -------
    dict
        Dictionary of all calculated metrics.
    """
    
    results = {}
    
    # Basic metrics
    results['Annual Return'] = metrics.calculate_annual_return(strategy_returns, 252)
    results['Annual Volatility'] = metrics.calculate_annual_volatility(strategy_returns, 252)
    results['Sharpe Ratio'] = metrics.calculate_sharpe_ratio(strategy_returns, risk_free_rate, 252)
    results['Sortino Ratio'] = metrics.calculate_sortino_ratio(strategy_returns, risk_free_rate=risk_free_rate)
    results['Calmar Ratio'] = metrics.calculate_calmar_ratio(strategy_returns, 252)
    results['Max Drawdown'] = metrics.calculate_max_drawdown(strategy_returns)
    results['Win Rate'] = metrics.calculate_win_rate(strategy_returns)
    results['Recovery Factor'] = metrics.calculate_recovery_factor(strategy_returns)
    
    # Relative metrics if benchmark provided
    if benchmark_returns is not None:
        results['Excess Return'] = (
            metrics.calculate_annual_return(strategy_returns, 252) -
            metrics.calculate_annual_return(benchmark_returns, 252)
        )
        results['Information Ratio'] = metrics.calculate_information_ratio(
            strategy_returns, benchmark_returns, 252
        )
        results['Beta'] = metrics.calculate_beta(strategy_returns, benchmark_returns)
        results['Alpha'] = metrics.calculate_alpha(strategy_returns, benchmark_returns, risk_free_rate, 252)
        results['Correlation'] = metrics.calculate_correlation_with_benchmark(
            strategy_returns, benchmark_returns
        )
    
    return results


def create_html_report(
    strategy_returns: pd.Series,
    equity_curve: pd.Series,
    factor_values: Optional[pd.DataFrame] = None,
    forward_returns: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    benchmark_curve: Optional[pd.Series] = None,
    title: str = "QFRsch Strategy Analysis Report",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate comprehensive HTML tearsheet report.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns.
    equity_curve : pd.Series
        Daily portfolio value.
    factor_values : pd.DataFrame, optional
        Factor values for IC analysis.
    forward_returns : pd.DataFrame, optional
        Forward returns for factor analysis.
    benchmark_returns : pd.Series, optional
        Benchmark daily returns.
    benchmark_curve : pd.Series, optional
        Benchmark daily value.
    title : str, default="QFRsch Strategy Analysis Report"
        Report title.
    output_path : str, optional
        Path to save HTML report. If None, returns HTML string.
        
    Returns
    -------
    str
        HTML report string, or empty string if saved to file.
    """
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for reporting. Install via: pip install plotly")
    
    # Generate plots
    equity_fig = plot_equity_curve(equity_curve, benchmark_curve)
    drawdown_fig = plot_drawdown(strategy_returns)
    sharpe_fig = plot_rolling_sharpe(strategy_returns)
    
    # Generate metrics
    metrics_dict = generate_metrics_table(strategy_returns, benchmark_returns)
    
    # Build HTML
    html_parts = [
        f"<html><head><title>{title}</title>",
        "<meta charset='utf-8'>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #666; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 10px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
        "th { background-color: #f2f2f2; }",
        ".metric-value { font-weight: bold; }",
        ".positive { color: green; }",
        ".negative { color: red; }",
        ".chart-container { margin: 30px 0; }",
        ".report-footer { margin-top: 50px; border-top: 1px solid #ddd; padding-top: 20px; color: #999; font-size: 12px; }",
        "</style>",
        "</head><body>",
    ]
    
    # Title and summary
    html_parts.append(f"<h1>{title}</h1>")
    html_parts.append(f"<p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    
    # Metrics section
    html_parts.append("<h2>Performance Metrics</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
    
    for metric_name, metric_value in metrics_dict.items():
        if isinstance(metric_value, float):
            if metric_name in ['Annual Return', 'Excess Return', 'Alpha']:
                display_value = f"{metric_value:.2%}"
                value_class = "positive" if metric_value > 0 else "negative"
            elif metric_name in ['Max Drawdown', 'Win Rate']:
                display_value = f"{metric_value:.2%}"
                value_class = "positive" if metric_name == 'Win Rate' and metric_value > 0.5 else ""
            else:
                display_value = f"{metric_value:.4f}"
                value_class = ""
            
            html_parts.append(
                f"<tr><td>{metric_name}</td>"
                f"<td class='metric-value {value_class}'>{display_value}</td></tr>"
            )
    
    html_parts.append("</table>")
    
    # Charts section
    html_parts.append("<h2>Charts</h2>")
    
    html_parts.append("<h3>Equity Curve</h3>")
    html_parts.append(f"<div class='chart-container'>{equity_fig.to_html(include_plotlyjs='cdn')}</div>")
    
    html_parts.append("<h3>Drawdown</h3>")
    html_parts.append(f"<div class='chart-container'>{drawdown_fig.to_html(include_plotlyjs=False)}</div>")
    
    html_parts.append("<h3>Rolling Sharpe Ratio</h3>")
    html_parts.append(f"<div class='chart-container'>{sharpe_fig.to_html(include_plotlyjs=False)}</div>")
    
    # Factor analysis if provided
    if factor_values is not None and forward_returns is not None:
        html_parts.append("<h2>Factor Analysis</h2>")
        
        # IC analysis
        ic_series = factor_eval.calculate_ic(factor_values, forward_returns)
        ic_stats = factor_eval.calculate_ic_statistics(ic_series)
        
        html_parts.append("<h3>Information Coefficient (IC)</h3>")
        html_parts.append("<table>")
        for key, value in ic_stats.items():
            html_parts.append(
                f"<tr><td>{key}</td><td class='metric-value'>{value:.4f}</td></tr>"
            )
        html_parts.append("</table>")
        
        ic_fig = plot_ic_distribution(ic_series)
        html_parts.append(f"<div class='chart-container'>{ic_fig.to_html(include_plotlyjs=False)}</div>")
        
        # Quantile analysis
        quantile_result = factor_eval.quantile_backtest(factor_values, forward_returns, num_quantiles=5)
        quantile_fig = plot_quantile_returns(quantile_result)
        html_parts.append("<h3>Quantile Analysis</h3>")
        html_parts.append(f"<div class='chart-container'>{quantile_fig.to_html(include_plotlyjs=False)}</div>")
    
    # Footer
    html_parts.append("<div class='report-footer'>")
    html_parts.append("<p>This report was generated automatically by QFRsch Analysis Module.</p>")
    html_parts.append(f"<p>Data period: {equity_curve.index.min().date()} to {equity_curve.index.max().date()}</p>")
    html_parts.append("</div>")
    
    html_parts.append("</body></html>")
    
    html_string = "\n".join(html_parts)
    
    # Save to file if path provided
    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_string)
        return ""
    
    return html_string


__all__ = [
    'plot_equity_curve',
    'plot_drawdown',
    'plot_quantile_returns',
    'plot_ic_distribution',
    'plot_rolling_sharpe',
    'generate_metrics_table',
    'create_html_report',
]
