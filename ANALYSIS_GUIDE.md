"""
QFRsch Analysis Layer - Comprehensive Guide
"""

"""
## 分析层 (Analysis Layer) 完整指南

### 核心目标
为 QFRsch 框架提供全方位的量化分析能力，包括：
- 性能指标计算
- 因子有效性评价
- 归因分析
- 自动化报告生成


## 模块概览

### 1. src/qfrsch/analysis/metrics.py (核心指标库)

计算投资组合的全面性能指标。

#### 基础收益/风险指标

**年化收益率 (Annual Return)**
```python
from qfrsch.analysis import metrics
import pandas as pd

returns = pd.Series([0.001, 0.002, -0.001, ...])  # 日度收益率
annual_ret = metrics.calculate_annual_return(returns, periods_per_year=252)
# Output: 0.15 (表示15%)
```

**年化波动率 (Annual Volatility)**
```python
annual_vol = metrics.calculate_annual_volatility(returns, periods_per_year=252)
# Output: 0.18 (表示18%)
```

**夏普比率 (Sharpe Ratio)**
利用无风险利率调整的风险调整收益率。
```python
sharpe = metrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
# Output: 0.85
```

**索迪诺比率 (Sortino Ratio)**
仅考虑下行波动率，对下跌风险敏感。
```python
sortino = metrics.calculate_sortino_ratio(returns, target_return=0.0, risk_free_rate=0.02)
# Output: 0.92 (通常高于Sharpe Ratio)
```

**卡玛比率 (Calmar Ratio)**
年化收益率 / 最大回撤绝对值
```python
calmar = metrics.calculate_calmar_ratio(returns, periods_per_year=252)
# Output: 1.15
```

**最大回撤 (Maximum Drawdown)**
```python
mdd = metrics.calculate_max_drawdown(returns)
# Output: -0.15 (表示-15%)
```

**胜率 (Win Rate)**
正收益日占比
```python
wr = metrics.calculate_win_rate(returns)
# Output: 0.53 (53%)
```

#### 相对基准指标

**超额收益 (Excess Return)**
```python
excess = metrics.calculate_excess_return(strategy_returns, benchmark_returns)
# pd.Series: 每日超额收益
```

**信息比率 (Information Ratio)**
年化超额收益 / 跟踪误差
```python
ir = metrics.calculate_information_ratio(strategy_returns, benchmark_returns, periods_per_year=252)
# Output: 0.35
```

**超额波动率/跟踪误差 (Excess Volatility)**
```python
excess_vol = metrics.calculate_excess_volatility(strategy_returns, benchmark_returns)
# Output: 0.05 (5% 跟踪误差)
```

**贝塔系数 (Beta)**
策略相对基准的系统性风险系数
```python
beta = metrics.calculate_beta(strategy_returns, benchmark_returns)
# Output: 0.85
```

**詹森阿尔法 (Jensen's Alpha)**
```python
alpha = metrics.calculate_alpha(strategy_returns, benchmark_returns, risk_free_rate=0.02)
# Output: 0.03 (3% 年化超额收益不可解释部分)
```

#### 统计检验

**Newey-West 调整的 T 检验**
用于检验超额收益是否统计显著，纠正异方差和自相关性。
```python
t_stat, p_value, annual_ret = metrics.newey_west_ttest(excess_returns, test_value=0.0, lags=None)
# t_stat: 检验统计量
# p_value: 双尾p值。若p_value < 0.05，则在5%显著性水平下拒绝原假设
# annual_ret: 年化收益率
```


### 2. src/qfrsch/analysis/factor_eval.py (因子评价模块)

评价单个因子的有效性和稳定性。

#### IC 分析

**日度 IC 计算**
信息系数 (IC) = 因子与后续收益的相关性
```python
from qfrsch.analysis import factor_eval
import pandas as pd

factor_values = pd.DataFrame(...)  # Index: date, Columns: tickers
forward_returns = pd.DataFrame(...)  # Index: date, Columns: tickers

ic_series = factor_eval.calculate_ic(factor_values, forward_returns, method='pearson')
# Output: pd.Series，日度IC值
```

**Rank IC 计算**
基于秩的相关性，对异常值更鲁棒。
```python
rank_ic = factor_eval.calculate_rank_ic(factor_values, forward_returns)
# Output: pd.Series，日度Rank IC值
```

**IC 统计量**
```python
ic_stats = factor_eval.calculate_ic_statistics(ic_series)
# Output: {
#     'ic_mean': 0.033,        # IC平均值
#     'ic_std': 0.124,         # IC标准差
#     'ic_positive_pct': 0.55, # IC > 0 的比例
#     'ic_strong_positive_pct': 0.42,  # IC > 0.03 的比例
#     'ic_ir': 0.266,          # IC信息比 (IC均值/IC标差)
# }
```

#### 分层回测 (Quantile Analysis)

按因子值将股票分为N组，比较各组的收益。
```python
quantile_result = factor_eval.quantile_backtest(
    factor_values, 
    forward_returns, 
    num_quantiles=5  # 分成5组
)
# Output: {
#     'quantile_returns': DataFrame,          # [date x quantiles] 各分位数日度收益
#     'quantile_cumret': DataFrame,           # 累计收益
#     'quantile_annual_ret': Series,          # 各分位数年化收益
#     'quantile_annual_vol': Series,          # 各分位数年化波动率
#     'high_minus_low': Series,               # 最高分位 - 最低分位 的超额收益
# }
```

单调性分析：Q5 年化收益应该显著高于 Q1，表明因子有效。

#### Fama-MacBeth 回归

两步法回归提取因子风险溢价及显著性。

Step 1: 每天对所有股票做回归 return_i = α + β*factor_i
Step 2: 对时间序列的β进行 t 检验

```python
fm_result = factor_eval.fama_macbeth_regression(
    factor_values, 
    forward_returns,
    control_factors=None  # 可选的控制因子
)
# Output: {
#     'factor_premium': 0.0008,     # 平均因子风险溢价
#     'loading_std': 0.0012,        # β的时间序列标准差
#     't_stat': 2.15,               # t统计量
#     'p_value': 0.032,             # p值 (显著水平 < 0.05)
#     'num_obs': 252,               # 时间序列数据点数
# }
```

#### 因子稳定性

**因子稳定性系数 (FSC)**
衡量因子某时期与历史平均的相似度。
```python
fsc = factor_eval.calculate_factor_stability_coefficient(factor_values, window=60)
# Output: pd.Series，日度FSC值
# FSC > 0.8 表示因子稳定，< 0.5 表示剧烈变动
```

**因子自相关性**
衡量因子的持续性(momentum vs. mean-reversion)
```python
autocorr = factor_eval.calculate_factor_autocorrelation(factor_values, lag=1)
# Output: float，-1 到 1 之间
# > 0.2: 因子有动量效应，值往往持续上升/下降
# < -0.2: 因子有均值回归效应
```


### 3. src/qfrsch/analysis/attribution.py (归因分析模块)

分析投资组合表现的来源。

#### 换手率分析

**单次换手率计算**
```python
from qfrsch.analysis import attribution

current_weights = pd.Series({'A': 0.5, 'B': 0.5})
previous_weights = pd.Series({'A': 0.3, 'B': 0.7})

turnover = attribution.calculate_turnover(current_weights, previous_weights)
# Output: 0.20 (20% 换手)
```

**日度换手率序列**
```python
daily_turnover = attribution.calculate_daily_turnover(holdings_df, prices_df)
# Output: pd.Series，日度换手率
```

#### 主动收益分析

**主动收益（超额收益）**
```python
active_returns = attribution.calculate_active_return(strategy_returns, benchmark_returns)
# Output: pd.Series，每日超额收益
```

**主动风险（跟踪误差）**
```python
active_risk = attribution.calculate_active_risk(strategy_returns, benchmark_returns)
# Output: 0.08 (8% 跟踪误差)
```

#### 头寸集中度分析

**Herfindahl 集中度指标**
∑(weight_i²)，0 到 1 之间
```python
concentration = attribution.calculate_position_concentration(weights_df)
# Output: pd.Series，日度集中度
# 1/N = 等权重组合的集中度 (N是持仓数量)
# 高于平均值表示组合较为集中
```

#### 收益成本分解

```python
attribution_result = attribution.calculate_return_attribution_attribution(
    returns_with_costs,      # 净收益（已扣费）
    returns_without_costs    # 总收益（未扣费）
)
# Output: {
#     'gross_return': 0.15,      # 总收益率
#     'net_return': 0.14,        # 净收益率
#     'cost_impact': 0.01,       # 成本影响
#     'cost_pct': 0.0667,        # 成本占比
# }
```


### 4. src/qfrsch/analysis/reporter.py (自动化报告模块)

生成可交互式的 HTML 分析报告。

#### 可视化函数

**净值曲线**
```python
from qfrsch.analysis import reporter

fig = reporter.plot_equity_curve(
    equity_curve=portfolio_equity,
    benchmark_curve=benchmark_equity,
    log_scale=False
)
fig.show()
```

**回撤曲线**
```python
fig = reporter.plot_drawdown(returns)
fig.show()
```

**因子分层收益柱状图**
```python
quantile_result = factor_eval.quantile_backtest(...)
fig = reporter.plot_quantile_returns(quantile_result)
fig.show()
```

**IC 分布**
```python
ic_series = factor_eval.calculate_ic(...)
fig = reporter.plot_ic_distribution(ic_series)
fig.show()
```

**滚动夏普比率**
```python
fig = reporter.plot_rolling_sharpe(returns, window=60)
fig.show()
```

#### 自动化报告生成

**生成 HTML Tearsheet**
```python
reporter.create_html_report(
    strategy_returns=strategy_returns,
    equity_curve=equity_curve,
    factor_values=factor_values,
    forward_returns=forward_returns,
    benchmark_returns=benchmark_returns,
    benchmark_curve=benchmark_curve,
    title="My Strategy Analysis",
    output_path="report.html"
)
```

输出包含：
- 完整性能指标表
- 净值曲线图
- 回撤图
- 滚动夏普比率图
- IC 分析图表
- 分层回测收益图


## 完整工作流示例

```python
from qfrsch.analysis import metrics, factor_eval, attribution, reporter
import pandas as pd

# 1. 准备数据
strategy_returns = pd.Series(...)        # 日度收益率
benchmark_returns = pd.Series(...)       # 基准日度收益率
factor_values = pd.DataFrame(...)        # 因子值 [date x tickers]
forward_returns = pd.DataFrame(...)      # 后续收益 [date x tickers]

# 2. 性能评价
print(f"Sharpe Ratio: {metrics.calculate_sharpe_ratio(strategy_returns):.4f}")
print(f"Max Drawdown: {metrics.calculate_max_drawdown(strategy_returns):.2%}")
print(f"Information Ratio: {metrics.calculate_information_ratio(strategy_returns, benchmark_returns):.4f}")

# 3. 因子分析
ic_series = factor_eval.calculate_ic(factor_values, forward_returns)
ic_stats = factor_eval.calculate_ic_statistics(ic_series)
print(f"IC Mean: {ic_stats['ic_mean']:.4f}")

quantile_result = factor_eval.quantile_backtest(factor_values, forward_returns)
print(f"Q5-Q1: {quantile_result['quantile_annual_ret'][5] - quantile_result['quantile_annual_ret'][1]:.2%}")

# 4. 生成报告
reporter.create_html_report(
    strategy_returns=strategy_returns,
    equity_curve=(1 + strategy_returns).cumprod(),
    factor_values=factor_values,
    forward_returns=forward_returns,
    benchmark_returns=benchmark_returns,
    benchmark_curve=(1 + benchmark_returns).cumprod(),
    output_path="analysis_report.html"
)
```


## 性能规范

所有分析函数都针对大规模数据进行了优化：

- **时间序列长度**：可直接处理10年以上（2500+ 个交易日）数据
- **资产数量**：可处理 3000+ 个资产的因子矩阵
- **IC 计算**：对50个资产的日度数据计算 IC，< 1 秒
- **分层回测**：对 3000 资产进行分层分析，< 2 秒
- **Fama-MacBeth 回归**：对 3000 资产进行两步回归，< 5 秒

数据对齐和 NaN 处理已内置，毋需额外预处理。


## 关键技术细节

### Newey-West 调整

纠正时间序列中的异方差和自相关性，广泛用于金融数据的 t 检验。

```
SE_NW = sqrt(Var[0] + 2*∑(1 - k/(L+1)) * Cov[k])
其中 k = 1,...,L 是滞后期数，L ≈ sqrt(N)
```

### IC 信息比 (IC IR)

衡量因子每单位波动产生的信息系数增益，一般 IC IR > 0.1 表示有价值因子。

```
IC IR = E(IC) / Std(IC)
```

### 因子稳定性系数

衡量因子在市场中的稳定性，防止选中的只是市场某个阶段的特殊因子。

```
FSC = 1 - Std(Recent Factor) / Std(Historical Factor)
```

-1 到 1 之间，越接近 1 表示因子越稳定。


## 常见问题

Q: 如何处理缺失数据？
A: 所有函数使用 NaN 掩码自动跳过缺失值。无需提前清理。

Q: 能处理多因子模型吗？
A: Fama-MacBeth 回归支持传入 control_factors 参数处理多因子。

Q: 如何选择 IC 计算方法（Pearson vs Rank）？
A: Pearson IC：假设线性关系；Rank IC：非参数，对异常值鲁棒。通常两个都计算。

Q: 分层数量怎么选择？
A: 常见是 5 分位（5-quantile）或 10 分位。数量越多，每组样本越少，波动越大。

Q: 如何理解 Information Ratio？
A: IR = (超额收益) / (跟踪误差)，衡量每单位跟踪误差获得的超额收益。
   IR > 1：优秀，IR > 0.5：良好，IR < 0：不好。


## 下一步

1. **集成到 Pipeline**：
   - Pipeline 生成因子
   - Analysis 评价因子有效性

2. **集成到 Backtest**：
   - Backtest 生成收益序列
   - Analysis 评价策略表现

3. **实时监测**：
   - 定期重新计算 IC、IR
   - 监控因子稳定性变化
   - 发现因子失效信号
"""
