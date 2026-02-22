"""
QFRsch Analysis Layer - Quick Reference
"""

"""
# QFRsch 分析层 - 快速参考

## 导入

```python
from qfrsch.analysis import metrics, factor_eval, attribution, reporter
import pandas as pd
```

## 常用指标快速计算

### 性能评价

```python
# 基础指标
annual_ret = metrics.calculate_annual_return(returns)
annual_vol = metrics.calculate_annual_volatility(returns)
sharpe = metrics.calculate_sharpe_ratio(returns)
mdd = metrics.calculate_max_drawdown(returns)

# 相对基准指标
ir = metrics.calculate_information_ratio(strategy_ret, benchmark_ret)
alpha = metrics.calculate_alpha(strategy_ret, benchmark_ret)
beta = metrics.calculate_beta(strategy_ret, benchmark_ret)

# 统计检验
t_stat, p_value, annual_ret = metrics.newey_west_ttest(excess_returns)
```

### 因子评价

```python
# IC分析
ic_series = factor_eval.calculate_ic(factor_values, forward_returns)
ic_stats = factor_eval.calculate_ic_statistics(ic_series)
print(f"IC Mean: {ic_stats['ic_mean']:.4f}")
print(f"IC IR: {ic_stats['ic_ir']:.4f}")

# 分层回测
quantile_result = factor_eval.quantile_backtest(factor_values, forward_returns, num_quantiles=5)
print(f"Q5 Return: {quantile_result['quantile_annual_ret'][5]:.2%}")

# Fama-MacBeth检验
fm_result = factor_eval.fama_macbeth_regression(factor_values, forward_returns)
print(f"t-stat: {fm_result['t_stat']:.4f}, p-value: {fm_result['p_value']:.4f}")
```

### 绩效归因

```python
# 换手率
turnover = attribution.calculate_turnover(current_weights, previous_weights)

# 主动收益
active_returns = attribution.calculate_active_return(strategy_ret, benchmark_ret)
active_risk = attribution.calculate_active_risk(strategy_ret, benchmark_ret)

# 头寸集中度
concentration = attribution.calculate_position_concentration(weights_df)
```

## 报告生成

### 一键生成HTML报告

```python
# 最简单的方式
reporter.create_html_report(
    strategy_returns=returns,
    equity_curve=equity,
    output_path="report.html"
)

# 完整的方式
reporter.create_html_report(
    strategy_returns=strategy_returns,
    equity_curve=equity_curve,
    factor_values=factor_df,
    forward_returns=forward_returns,
    benchmark_returns=benchmark_returns,
    benchmark_curve=benchmark_equity,
    title="My Strategy Analysis",
    output_path="analysis.html"
)
```

### 单独生成图表

```python
# 净值曲线
fig = reporter.plot_equity_curve(equity_curve, benchmark_curve)
fig.show()

# 回撤
fig = reporter.plot_drawdown(returns)
fig.show()

# 分层收益
fig = reporter.plot_quantile_returns(quantile_result)
fig.show()

# IC分布
fig = reporter.plot_ic_distribution(ic_series)
fig.show()
```

## 完整工作流示例

```python
from qfrsch.analysis import metrics, factor_eval, reporter
import pandas as pd

# 1. 读取数据
strategy_returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)['returns']
benchmark_returns = pd.read_csv('benchmark.csv', index_col=0, parse_dates=True)['returns']
factor_values = pd.read_csv('factors.csv', index_col=0, parse_dates=True)
forward_returns = pd.read_csv('forward_returns.csv', index_col=0, parse_dates=True)

# 2. 基础性能评价
print("=== Performance Metrics ===")
print(f"Sharpe Ratio: {metrics.calculate_sharpe_ratio(strategy_returns):.4f}")
print(f"Annual Return: {metrics.calculate_annual_return(strategy_returns):.2%}")
print(f"Max Drawdown: {metrics.calculate_max_drawdown(strategy_returns):.2%}")
print(f"Information Ratio: {metrics.calculate_information_ratio(strategy_returns, benchmark_returns):.4f}")

# 3. 因子有效性检验
print("\n=== Factor Analysis ===")
ic_series = factor_eval.calculate_ic(factor_values, forward_returns)
ic_stats = factor_eval.calculate_ic_statistics(ic_series)
print(f"IC Mean: {ic_stats['ic_mean']:.4f}")
print(f"IC IR: {ic_stats['ic_ir']:.4f}")

fm_result = factor_eval.fama_macbeth_regression(factor_values, forward_returns)
print(f"Factor Significant: {'Yes' if fm_result['p_value'] < 0.05 else 'No'}")

# 4. 分层回测
quantile_result = factor_eval.quantile_backtest(factor_values, forward_returns)
print("\n=== Quantile Performance ===")
for q in range(1, 6):
    print(f"Q{q}: {quantile_result['quantile_annual_ret'][q]:.2%}")

# 5. 生成报告
reporter.create_html_report(
    strategy_returns=strategy_returns,
    equity_curve=(1 + strategy_returns).cumprod(),
    factor_values=factor_values,
    forward_returns=forward_returns,
    benchmark_returns=benchmark_returns,
    benchmark_curve=(1 + benchmark_returns).cumprod(),
    output_path="analysis_report.html"
)
print("\nReport saved to analysis_report.html")
```

## 数据格式要求

### 收益率数据
```python
# pd.Series，index为date，values为收益率
returns = pd.Series([0.001, 0.002, -0.001, ...], index=dates)
# 也可以是百分比形式：[0.1, 0.2, -0.1, ...] 表示 10%, 20%, -10%
```

### 因子值数据
```python
# pd.DataFrame，index为date，columns为ticker，values为因子值
factor_values = pd.DataFrame(
    [[0.5, 0.3, ...],
     [0.4, 0.2, ...],
     ...],
    index=dates,
    columns=['Stock_A', 'Stock_B', ...]
)
```

### 通过价格计算收益率
```python
prices = pd.read_csv('prices.csv', index_col=0, parse_dates=True)
returns = prices.pct_change()  # 自动计算日度收益率
```

## 常见问题速查

| 问题 | 解决方案 |
|------|--------|
| 数据缺失(NaN) | 自动处理，无需预处理 |
| 日期不对齐 | 自动对齐至公共日期 |
| IC为NaN或inf | 检查是否有常数列（无波动的因子值） |
| 报告无法生成 | 安装plotly: `pip install plotly` |
| 性能过慢 | 检查数据量(>100K行)，考虑分期处理 |

## 性能基准

| 操作 | 数据规模 | 耗时 |
|------|---------|------|
| IC计算 | 50资产×250天 | <100ms |
| 分层回测 | 3000资产×250天 | <2s |
| Fama-MacBeth | 3000资产×250天 | <5s |
| 生成HTML报告 | 完整指标+图表 | <2s |

## 参数速查表

### calculate_sharpe_ratio()
```python
sharpe = metrics.calculate_sharpe_ratio(
    returns,                          # pd.Series 日度收益
    risk_free_rate=0.02,              # float 无风险利率 (默认2%)
    periods_per_year=252              # int 年交易日数 (默认252)
)
```

### quantile_backtest()
```python
result = factor_eval.quantile_backtest(
    factor_values,                    # pd.DataFrame [date x ticker]
    forward_returns,                  # pd.DataFrame [date x ticker]
    num_quantiles=5                   # int 分位数 (默认5)
)
```

### create_html_report()
```python
reporter.create_html_report(
    strategy_returns,                 # pd.Series 日度收益 (必填)
    equity_curve,                     # pd.Series 日度净值 (必填)
    factor_values=None,               # pd.DataFrame 因子值 (可选)
    forward_returns=None,             # pd.DataFrame 前向收益 (可选)
    benchmark_returns=None,           # pd.Series 基准收益 (可选)
    benchmark_curve=None,             # pd.Series 基准净值 (可选)
    title="Report",                   # str 报告标题
    output_path=None                  # str 保存路径 (为None则返回HTML字符串)
)
```

## tips

💡 **Sharpe vs Sortino**
- Sharpe：总波动性
- Sortino：仅下行波动性（对下跌敏感）
- 通常 Sortino > Sharpe

💡 **IC > Rank IC**
- 当存在异常值时
- 一般两个都计算

💡 **IC IR > 0.1**
- 好的因子标准
- < 0：不良因子
- 0.05-0.1：一般

💡 **Q5-Q1 > 年化收益×5%**
- 单调性检验
- 体现因子质量

💡 **Fama-MacBeth p-value**
- < 0.05：因子显著（优秀）
- 0.05-0.1：边界（一般）
- > 0.1：不显著（差）

## 下一步

1. 查看 [ANALYSIS_GUIDE.md](../ANALYSIS_GUIDE.md) 了解详细技术细节
2. 运行 `examples/analysis_example.py` 查看完整示例
3. 查看测试文件 `test/cases/test_analysis.py` 学习用法
"""
