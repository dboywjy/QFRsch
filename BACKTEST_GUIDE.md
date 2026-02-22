"""
QFRsch Backtest Layer - Quick Start Guide
"""

"""
## 核心模块概览

开发了自研的、基于Pandas的轻量化回测引擎，包含以下三个核心模块：

### 1. src/qfrsch/backtest/models.py
数据结构定义模块，包含：

- **BacktestConfig**: 回测配置类 (dataclass)
  - initial_capital: 初始资金 (默认 1,000,000)
  - commission_rate: 手续费率 (默认 0.1%)
  - slippage_rate: 滑点率 (默认 0.05%)
  - rebalance_frequency: 调仓频率 (默认每日)
  - risk_free_rate: 无风险利率 (默认 2%)
  - execution_price: 执行价格 ('open' 或 'close')

- **BacktestResult**: 回测结果容器类
  - equity_curve: 日度净值曲线 DataFrame
  - trades: 交易流水 DataFrame
  - holdings: 日度持仓快照 DataFrame
  - daily_returns: 日度收益率 Series
  - daily_weights: 日度权重 DataFrame
  - metrics: 性能指标字典 (total_return, sharpe_ratio, max_drawdown 等)
  - config: 原始配置对象


### 2. src/qfrsch/backtest/utils.py
性能计算辅助函数：

- calculate_max_drawdown(equity_curve) -> float
  计算最大回撤
  
- calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year) -> float
  计算夏普比率
  
- calculate_calmar_ratio(returns, periods_per_year) -> float
  计算卡玛比率
  
- calculate_win_rate(returns) -> float
  计算胜率
  
- calculate_growth_metrics(equity_curve, initial_capital) -> (float, float)
  计算总收益率和年化收益率
  
- calculate_volatility(returns, periods_per_year) -> float
  计算年化波动率


### 3. src/qfrsch/backtest/engine.py
核心回测引擎 - SimpleEngine 类：

**初始化**:
```python
config = BacktestConfig(initial_capital=1_000_000)
engine = SimpleEngine(config)
```

**运行回测**:
```python
result = engine.run(ohlcv_df, target_weights_df)
```

**输入数据格式**:
- ohlcv_df: DataFrame
  - Index: datetime (交易日期)
  - Columns: ['ticker', 'open', 'high', 'low', 'close', 'volume']
  
- target_weights_df: DataFrame
  - Index: datetime (权重日期)
  - Columns: 各种股票代码 (例如 'A', 'B', 'AAPL')
  - Values: 目标权重 (0.0-1.0)

**核心回测逻辑** (逐日循环):
1. 初始化现金和持仓
2. 对每一个交易日：
   a. 获取当日收盘价
   b. 计算持仓市值变化
   c. 从权重矩阵查询目标权重
   d. 计算调仓数量 (target_qty - current_qty)
   e. 应用滑点和手续费
   f. 更新持仓和现金
   g. 记录交易和状态


## 使用示例

```python
import pandas as pd
from qfrsch.backtest import SimpleEngine, BacktestConfig

# 准备数据
ohlcv_df = pd.read_csv('ohlcv.csv')  # 包含 ticker, open, high, low, close, volume
target_weights_df = pd.read_csv('weights.csv')  # 包含日期和股票权重

# 创建配置
config = BacktestConfig(
    initial_capital=1_000_000,
    commission_rate=0.001,  # 0.1%
    slippage_rate=0.0005,   # 0.05%
)

# 运行回测
engine = SimpleEngine(config)
result = engine.run(ohlcv_df, target_weights_df)

# 查看结果
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")

# 访问详细数据
print(result.equity_curve)  # 日度净值
print(result.trades)        # 交易流水
print(result.holdings)      # 持仓快照
print(result.daily_returns) # 日度收益率
```


## 设计说明

### 灵活的资产支持
- 不限制股票数量
- 从权重矩阵自动识别所有资产
- 权重缺失的资产自动设为0

### 精准的成本模型
- **手续费**: 基于成交额的百分比费用
- **滑点**: 买入价格向上调整，卖出价格向下调整
- 公式:
  - 买入执行价 = 收盘价 * (1 + slippage_rate)
  - 卖出执行价 = 收盘价 * (1 - slippage_rate)
  - 手续费 = 成交额 * commission_rate

### 向量化计算
- 尽量避免嵌套循环
- 利用 Pandas 的高效操作
- 支持大规模数据



## 测试

所有单元测试在 test/cases/test_backtest.py 中：

```bash
cd /home/jywang/project/QFRsch
source ./qfrsch-env/bin/activate
uv run --active pytest test/cases/test_backtest.py -v
```

测试覆盖:
- BacktestConfig 初始化和自定义
- SimpleEngine 初始化
- 完整回测流程
- 净值曲线追踪
- 性能指标计算
- 零权重处理（平仓）
- 异常输入处理


## 运行示例

```bash
cd /home/jywang/project/QFRsch
source ./qfrsch-env/bin/activate
uv run --active python examples/backtest_example.py
```

示例输出包括:
- 数据准备统计
- 回测配置信息
- 完整性能指标
- 交易统计
- 最终持仓


## 关键特性

1. **简单直观**: 逐日循环的实现逻辑清晰，易于理解和二次开发

2. **性能友好**: 基于 Pandas 的向量化计算，即使大规模数据也快速

3. **鲁棒性强**:
   - 日期对齐处理
   - 缺失权重默认为0
   - 价格不可用时跳过该日期
   
4. **成本模型准确**:
   - 每笔交易都精确计算手续费
   - 滑点应用正确（买卖方向相反）
   - 交易流水完整记录

5. **输出完整**:
   - 净值曲线对标资金变化
   - 交易流水对标实际成交
   - 持仓快照对标策略状态


## 常见问题

Q: 如何处理权重矩阵中缺失的日期？
A: 缺失日期会在 _prepare_data 中被跳过，只处理 OHLCV 和权重都有数据的日期。

Q: 如何处理某个股票在某天没有价格数据？
A: 会跳过该日期，等待下一个有数据的日期执行调仓。

Q: 如何修改手续费或滑点？
A: 通过 BacktestConfig 对象配置：
   config = BacktestConfig(commission_rate=0.002, slippage_rate=0.001)

Q: 如何使用不同的执行价格（如开盘价）？
A: config = BacktestConfig(execution_price='open')


## 下一步建议

1. 与 Pipeline 层集成：
   - Pipeline 输出权重矩阵
   - 直接传入 SimpleEngine.run()

2. 因子分析增强：
   - 记录每日因子值
   - 分析因子对收益的贡献

3. 虎虎性能优化：
   - 对大规模数据进行并行处理
   - 增加缓存机制

4. 风险管理扩展：
   - 增加止损机制
   - 增加头寸限制
"""
