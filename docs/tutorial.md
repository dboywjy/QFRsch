# QFRsch 项目教程

> QFRsch (Quantitative Factor Research System) v0.1 — 量化因子研究框架
>
> Python ≥ 3.12 | 包管理：uv | 依赖：pandas, numpy, scikit-learn, xgboost, plotly, statsmodels

---

## 1 项目结构

```
QFRsch/
├── README.md                   # 项目说明与设计文档
├── pyproject.toml              # 依赖与构建配置
├── main.py                     # 入口文件（占位）
├── src/qfrsch/                 # 核心源码
│   ├── core/calendars.py       # 交易日历
│   ├── factors/                # 因子层
│   │   ├── base.py             #   因子基类（FactorBase ABC）
│   │   ├── operators.py        #   算子库（ts_/cs_/op_）
│   │   ├── processor.py        #   因子后处理（去极值/标准化/中性化）
│   │   ├── styles.py           #   风格因子（市值/波动率/行业）
│   │   └── alpha_zoo.py        #   Alpha 因子库 + 注册表
│   ├── models/base.py          # 模型层（OLS/Ridge/Lasso/XGBoost）
│   ├── pipeline/               # 策略流水线
│   │   ├── manager.py          #   PipelineManager（数据集构建 + 全流程）
│   │   └── strategies.py       #   TopNStrategy / OptimizedStrategy(MVO/RiskParity)
│   ├── backtest/               # 回测层
│   │   ├── engine.py           #   SimpleEngine（日频事件驱动引擎）
│   │   ├── models.py           #   BacktestConfig / BacktestResult
│   │   └── utils.py            #   指标计算辅助函数
│   ├── analysis/               # 分析层
│   │   ├── metrics.py          #   绩效指标（Sharpe/Sortino/Calmar/MaxDD/Alpha/Beta...）
│   │   ├── factor_eval.py      #   因子评价（IC/RankIC/分层回测/Fama-MacBeth）
│   │   ├── attribution.py      #   归因分析（换手率/主动收益/Brinson分解）
│   │   └── reporter.py         #   HTML Tearsheet 报表（Plotly）
│   └── utils/helpers.py        # 工具函数（重采样/标签对齐/数据集对齐）
├── test/                       # 测试（296 个用例）
│   ├── conftest.py
│   └── cases/                  #   按模块组织的单元测试
├── examples/                   # 示例脚本
│   ├── analysis_example.py     #   分析层完整示例
│   ├── backtest_example.py     #   回测引擎基础示例
│   ├── validate_wrds.py        #   WRDS 官方结果验证
│   ├── wrds/                   #   WRDS 市场数据（parquet）
│   └── wrds-backtests/         #   WRDS 回测基准数据（CSV）
└── docs/                       # 文档
```

---

## 2 五层架构

```
OHLCV 数据 → 因子计算 → 模型预测 → 策略权重 → 回测模拟 → 绩效分析
   数据层       因子层      模型层       策略层       回测层      分析层
```

### 2.1 数据层 `core/`

**TradingCalendar** — 基于 `pandas-market-calendars` 的交易日历管理。

```python
from qfrsch.core.calendars import TradingCalendar

cal = TradingCalendar("NYSE")
dates = cal.get_trading_dates("2024-01-01", "2024-12-31")
cal.is_trading_day("2024-07-04")     # False（独立日）
cal.get_next_trading_date("2024-07-04")  # 2024-07-05
```

辅助函数：`to_datetime()`, `to_date()`, `to_str()`, `get_month_end()`, `get_quarter_end()` 等。

### 2.2 因子层 `factors/`

#### 因子基类

所有因子继承 `FactorBase`，必须实现 `compute(df) -> pd.Series`：

```python
from qfrsch.factors.base import FactorBase

class MyFactor(FactorBase):
    def compute(self, df):
        self._validate_data(df)
        return df.groupby('ticker')['close'].pct_change(self.params['period'])

factor = MyFactor("momentum_20d", params={"period": 20})
result = factor.compute(df)
```

#### 算子库

| 类型 | 算子 | 说明 |
|------|------|------|
| 时序 | `ts_mean`, `ts_std`, `ts_rank`, `ts_delta`, `ts_delay`, `ts_corr` | 按 ticker 分组的滚动计算 |
| 截面 | `cs_rank` | 每日截面百分位排名 (0-1) |
| 数学 | `op_log`, `op_abs`, `op_sign`, `op_sqrt`, `op_exp` | 逐元素变换 |

算子支持 MultiIndex `(date, ticker)` 和普通 Index。

```python
from qfrsch.factors.operators import ts_mean, cs_rank

avg_close = ts_mean(df['close'], window=10)   # 10日均价
rank = cs_rank(df['factor'])                   # 截面排名
```

#### 因子后处理

```python
from qfrsch.factors.processor import winsorize, standardize, neutralize

# 截面去极值 (1%/99% 分位数)
clean = winsorize(factor, method='quantile', limits=(0.01, 0.99), group_key='date')

# 截面标准化 (Z-score)
normed = standardize(factor, method='zscore', group_key='date')

# 行业/市值中性化 (OLS 残差)
neutral = neutralize(factor, risk_df, method='ols', group_key='date')
```

#### 内置因子

- **Alpha Zoo**：Alpha001–004（基于 WorldQuant 公式），`AlphaCombo` 加权合成
- **风格因子**：`SizeFactor`（log市值）、`VolatilityFactor`（20日波动率）、`IndustryFactor`（行业分类 + one-hot）

### 2.3 模型层 `models/`

**ModelWrapper** — 统一接口封装 sklearn / xgboost 模型。

```python
from qfrsch.models.base import ModelWrapper

model = ModelWrapper(model_type='ridge', alpha=1.0, scaling=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 滚动预测（防止前瞻偏差）
preds = model.rolling_predict(X, y, rolling_window=252, step=21)
```

支持：`ols`, `ridge`, `lasso`, `xgboost`

### 2.4 策略层 `pipeline/`

#### PipelineManager

核心流水线管理器，处理多频率重采样和标签对齐：

```python
from qfrsch.pipeline.manager import PipelineManager

pm = PipelineManager(rebalance_freq='M', prediction_period=1)
X, y = pm.make_dataset(factors_df, returns_df)   # 自动对齐，防前瞻
weights = pm.run_pipeline(X, y, model, strategy)
```

关键设计：T 日因子预测 T+1 到 T+k 收益，`shift_labels_by_freq()` 保证无信息泄露。

#### 策略

```python
from qfrsch.pipeline.strategies import TopNStrategy, OptimizedStrategy

# Top-N 等权
strategy = TopNStrategy(n_stocks=50, long_only=True, equal_weight=True)

# 均值方差优化
strategy = OptimizedStrategy(method='mvo', max_weight=0.05, leverage=1.0)
# 风险平价
strategy = OptimizedStrategy(method='riskparity')
```

### 2.5 回测层 `backtest/`

**SimpleEngine** — 日频事件驱动回测引擎。

```python
from qfrsch.backtest import SimpleEngine, BacktestConfig

config = BacktestConfig(
    initial_capital=1_000_000,
    commission_rate=0.001,     # 0.1% 手续费
    slippage_rate=0.0005,      # 0.05% 滑点
    risk_free_rate=0.02,
    execution_price='close',
)

engine = SimpleEngine(config)
result = engine.run(ohlcv_df, target_weights_df)

print(result.metrics['sharpe_ratio'])   # Sharpe
print(result.equity_curve.tail())       # 净值曲线
```

**输入格式**：
- `ohlcv_df`：DatetimeIndex，列 `['ticker', 'open', 'high', 'low', 'close', 'volume']`
- `target_weights_df`：DatetimeIndex，列为 ticker，值为目标权重

**输出**：`BacktestResult` 含 `equity_curve`, `trades`, `holdings`, `daily_returns`, `daily_weights`, `metrics`

### 2.6 分析层 `analysis/`

#### 绩效指标

```python
from qfrsch.analysis import metrics

metrics.calculate_annual_return(returns)
metrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
metrics.calculate_sortino_ratio(returns)
metrics.calculate_calmar_ratio(returns)
metrics.calculate_max_drawdown(returns)
metrics.calculate_win_rate(returns)

# 相对指标
metrics.calculate_alpha(strategy_ret, benchmark_ret)
metrics.calculate_beta(strategy_ret, benchmark_ret)
metrics.calculate_information_ratio(strategy_ret, benchmark_ret)

# Newey-West 调整 t 检验
t_stat, p_value, annual_ret = metrics.newey_west_ttest(returns)
```

#### 因子评价

```python
from qfrsch.analysis import factor_eval

# IC / Rank IC
ic = factor_eval.calculate_ic(factor_df, forward_returns_df)
rank_ic = factor_eval.calculate_rank_ic(factor_df, forward_returns_df)
stats = factor_eval.calculate_ic_statistics(ic)  # mean, std, IR, >0比例, >0.03比例

# 分层回测（分5组）
result = factor_eval.quantile_backtest(factor_df, forward_returns_df, num_quantiles=5)
# result['quantile_returns']    每组月度收益
# result['quantile_cumret']     累计收益
# result['high_minus_low']      多空收益

# Fama-MacBeth 回归
fm = factor_eval.fama_macbeth_regression(factor_df, forward_returns_df)

# 因子稳定性
fsc = factor_eval.calculate_factor_stability_coefficient(factor_df, window=60)
auto = factor_eval.calculate_factor_autocorrelation(factor_df, lag=1)
```

#### 归因分析

```python
from qfrsch.analysis import attribution

turnover = attribution.calculate_turnover(current_weights, previous_weights)
active_ret = attribution.calculate_active_return(strategy_ret, benchmark_ret)
tracking_error = attribution.calculate_active_risk(strategy_ret, benchmark_ret)
concentration = attribution.calculate_position_concentration(weights_df)  # Herfindahl
decomp = attribution.decompose_active_return(...)  # Brinson 分解
```

#### HTML 报表

```python
from qfrsch.analysis import reporter

reporter.create_html_report(
    strategy_returns=returns,
    equity_curve=equity,
    factor_values=factor_df,
    forward_returns=fwd_ret_df,
    benchmark_returns=bench_ret,
    title='策略分析报告',
    output_path='report.html',
)
```

生成包含净值曲线、回撤图、分层收益柱状图、IC 分布图、指标摘要表的交互式 HTML 报告。

---

## 3 完整工作流示例

```python
import pandas as pd
from qfrsch.factors.alpha_zoo import Alpha001
from qfrsch.factors.processor import winsorize, standardize
from qfrsch.models.base import ModelWrapper
from qfrsch.pipeline.manager import PipelineManager
from qfrsch.pipeline.strategies import TopNStrategy
from qfrsch.backtest import SimpleEngine, BacktestConfig
from qfrsch.analysis import metrics, factor_eval, reporter

# 1. 计算因子
alpha = Alpha001("alpha001")
factor_values = alpha.compute(ohlcv_df)
factor_clean = standardize(winsorize(factor_values))

# 2. 构建数据集
pm = PipelineManager(rebalance_freq='M', prediction_period=1)
X, y = pm.make_dataset(factors_df, returns_df)

# 3. 模型训练
model = ModelWrapper('ridge', scaling=True)
predictions = model.rolling_predict(X, y, rolling_window=252)

# 4. 生成权重
strategy = TopNStrategy(n_stocks=50, long_only=True)
weights = strategy.generate_weights(predictions)

# 5. 回测
config = BacktestConfig(initial_capital=1_000_000, commission_rate=0.001)
result = SimpleEngine(config).run(ohlcv_df, weights)

# 6. 分析
print(f"Sharpe: {result.metrics['sharpe_ratio']:.3f}")
print(f"MaxDD:  {result.metrics['max_drawdown']:.2%}")

# 7. 生成报告
reporter.create_html_report(
    strategy_returns=result.daily_returns,
    equity_curve=result.equity_curve['total_value'],
    title='策略回测报告',
    output_path='tearsheet.html',
)
```

---

## 4 示例脚本

| 脚本 | 功能 |
|------|------|
| `examples/backtest_example.py` | SimpleEngine 基础用法，使用模拟数据演示回测全流程 |
| `examples/analysis_example.py` | 分析层完整演示：绩效指标、IC分析、分层回测、归因、HTML报表 |
| `examples/validate_wrds.py` | 用 WRDS 官方 momentum12 因子回测数据验证 QFRsch 的准确性 |

运行方式：

```bash
source ./qfrsch-env/bin/activate
uv run --active python examples/backtest_example.py
```

---

## 5 WRDS 验证结果

使用 WRDS Quantitative Factors Platform 的 momentum12 信号（4929 只股票，2019–2022）进行验证：

| 方法 | 年化收益 | 年化波动 | Sharpe |
|------|---------|---------|--------|
| WRDS 官方 (Q5 Long) | 19.13% | 26.19% | 0.730 |
| `quantile_backtest()` | 21.89% | 26.89% | 0.740 |
| SimpleEngine | 21.06% | 26.27% | 0.726 |

收益率 ~2.7% 的差异来源：WRDS 使用 FFXRET（Fama-French 超额收益），本项目使用 CRSP 原始收益率；WRDS 额外过滤了交易所和股票类型。Sharpe 比率高度一致。

---

## 6 测试

```bash
source ./qfrsch-env/bin/activate
uv run --active python -m pytest test/ -x -q     # 296 个用例
```

测试覆盖所有核心模块：

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_base.py` | FactorBase 初始化、参数验证、数据校验 |
| `test_operators.py` | ts_/cs_/op_ 全部算子 |
| `test_processor.py` | 去极值、标准化、中性化 |
| `test_styles.py` | 行业/市值/波动率因子 |
| `test_alpha_zoo.py` | Alpha001–004 |
| `test_models.py` | ModelWrapper fit/predict/rolling_predict |
| `test_pipeline.py` | PipelineManager 数据集构建、标签对齐 |
| `test_strategies.py` | TopN/Optimized 权重生成 |
| `test_backtest.py` | SimpleEngine 回测全流程 |
| `test_analysis.py` | 绩效指标计算 |
| `test_calendars.py` | 交易日历 |
| `test_helpers.py` | 工具函数 |

---

## 7 环境配置

```bash
# 创建虚拟环境
python3.12 -m venv qfrsch-env
source ./qfrsch-env/bin/activate

# 安装依赖
uv sync

# 安装开发依赖
uv sync --group dev
```

依赖列表：

| 包 | 用途 |
|----|------|
| `pandas` / `numpy` | 核心数据处理 |
| `scikit-learn` | 回归模型 (OLS/Ridge/Lasso) |
| `xgboost` | 树模型 |
| `statsmodels` | Fama-MacBeth / Newey-West |
| `plotly` | 交互式可视化 / HTML 报表 |
| `pandas-market-calendars` | 交易日历 |
| `pyarrow` | Parquet 读写 |
| `pytest` (dev) | 单元测试 |
| `bt` (dev) | 回测参照对比 |
