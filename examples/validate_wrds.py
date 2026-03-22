"""
WRDS Backtest Validation Script

Validates QFRsch factor_eval.quantile_backtest() and SimpleEngine
against WRDS Quantitative Factors Platform official results.

Reference: momentum12 signal, 5-quintile sort, Jan 2019 - Jan 2022
WRDS results:
  Portfolio 5 (Long): Return=19.13%, Risk=26.19%, Sharpe=0.730
  Portfolio 1 (Short): Return=23.47%, Risk=36.70%, Sharpe=0.640
  Portfolio [5-1]:      Return=-4.34%, Risk=15.25%, Sharpe=-0.285
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd

# ============================================================
# Part 1: Load Data
# ============================================================
print("=" * 70)
print("WRDS Backtest Validation for QFRsch")
print("=" * 70)

t0 = time.time()

# Load WRDS signal
sig = pd.read_csv('examples/wrds-backtests/vub0jsn2qlk3mtvt_sig.csv')
sig['fdate'] = pd.to_datetime(sig['fdate'])
sig['PERMNO'] = sig['PERMNO'].astype(int)
print(f"Signal: {len(sig):,} rows, {sig['PERMNO'].nunique()} stocks, "
      f"{sig['fdate'].min().date()} to {sig['fdate'].max().date()}")

# Load WRDS official monthly results
wrds_res = pd.read_csv('examples/wrds-backtests/vub0jsn2qlk3mtvt.csv')
wrds_res['Date'] = pd.to_datetime(wrds_res['Date'])
wrds_res['fdate'] = pd.to_datetime(wrds_res['fdate'])
# Parse percentage columns
for col in wrds_res.columns:
    if col not in ['Date', 'fdate']:
        wrds_res[col] = wrds_res[col].astype(str).str.rstrip('%').astype(float) / 100.0
print(f"WRDS results: {len(wrds_res)} months")

# Load market data — only needed columns and date range
print("Loading market prices (this may take a moment)...")
mp = pd.read_parquet(
    'examples/wrds/market_prices/market_prices.parquet',
    columns=['permno', 'date', 'ret', 'ret_adj', 'adj_close']
)
mp['date'] = pd.to_datetime(mp['date'])
mp['permno'] = mp['permno'].astype(int)

# Filter to relevant date range: 2019-01 to 2022-02 (need one extra month for forward returns)
mp = mp[(mp['date'] >= '2019-01-01') & (mp['date'] <= '2022-02-28')].copy()
print(f"Market data: {len(mp):,} rows, {mp['permno'].nunique()} stocks, "
      f"{mp['date'].min().date()} to {mp['date'].max().date()}")

# Get common stocks
common_permnos = sorted(set(sig['PERMNO']) & set(mp['permno']))
print(f"Common stocks: {len(common_permnos)}")

# Filter to common stocks
sig = sig[sig['PERMNO'].isin(common_permnos)].copy()
mp = mp[mp['permno'].isin(common_permnos)].copy()

print(f"Data loaded in {time.time()-t0:.1f}s")

# ============================================================
# Part 2: Compute Monthly Returns from Daily Data
# ============================================================
print("\n" + "=" * 70)
print("Computing Monthly Returns")
print("=" * 70)

# Compute monthly returns per stock: (1+r1)*(1+r2)*...*(1+rn) - 1
mp['ret_clean'] = mp['ret'].fillna(0.0)
mp['year_month'] = mp['date'].dt.to_period('M')

monthly_ret = mp.groupby(['permno', 'year_month'])['ret_clean'].apply(
    lambda x: (1 + x).prod() - 1
).reset_index()
monthly_ret.columns = ['permno', 'year_month', 'monthly_ret']

# Convert period back to month-end date for alignment
monthly_ret['month_end'] = monthly_ret['year_month'].dt.to_timestamp('M')
print(f"Monthly returns: {len(monthly_ret):,} rows")

# ============================================================
# Part 3: Direct Quintile Comparison (manual, matches WRDS logic)
# ============================================================
print("\n" + "=" * 70)
print("Part A: Direct Quintile Replication (Manual)")
print("=" * 70)

# WRDS logic:
# At fdate (month-end), rank stocks by momentum12 into 5 quintiles.
# Compute next-month return for each quintile (equal-weighted).

fdate_list = sorted(sig['fdate'].unique())
quintile_returns_manual = []

for fdate in fdate_list:
    # Signal at this fdate
    sig_month = sig[sig['fdate'] == fdate][['PERMNO', 'momentum12']].dropna()
    if len(sig_month) < 50:
        continue

    # Next month's return
    next_month = (fdate + pd.offsets.MonthEnd(1))
    next_ym = next_month.to_period('M')

    ret_month = monthly_ret[monthly_ret['year_month'] == next_ym][['permno', 'monthly_ret']]
    if ret_month.empty:
        continue

    # Merge signal with returns
    merged = sig_month.merge(ret_month, left_on='PERMNO', right_on='permno', how='inner')
    if len(merged) < 50:
        continue

    # Quintile sort
    merged['quintile'] = pd.qcut(merged['momentum12'], q=5, labels=False, duplicates='drop') + 1

    # Equal-weighted return per quintile
    q_ret = merged.groupby('quintile')['monthly_ret'].mean()

    row = {'fdate': fdate, 'next_month': next_month}
    for q in range(1, 6):
        row[f'Q{q}'] = q_ret.get(q, np.nan)
    row['Q5_Q1'] = row.get('Q5', 0) - row.get('Q1', 0)
    quintile_returns_manual.append(row)

qr_manual = pd.DataFrame(quintile_returns_manual)
print(f"Generated {len(qr_manual)} monthly quintile return observations")

# Compare with WRDS monthly total returns
print("\n--- Monthly Return Comparison (Manual vs WRDS) ---")
print(f"{'Month':<12} {'Q1 (Mine)':>10} {'Q1 (WRDS)':>10} {'Q5 (Mine)':>10} {'Q5 (WRDS)':>10}")
print("-" * 55)

wrds_aligned = wrds_res.set_index('fdate')
for _, row in qr_manual.iterrows():
    fdate = row['fdate']
    if fdate in wrds_aligned.index:
        wrds_row = wrds_aligned.loc[fdate]
        print(f"{str(fdate.date()):<12} "
              f"{row['Q1']:>10.2%} {wrds_row['momentum12_TR_1']:>10.2%} "
              f"{row['Q5']:>10.2%} {wrds_row['momentum12_TR_5']:>10.2%}")

# Annualized metrics
print("\n--- Annualized Summary (Manual Replication) ---")
n_months = len(qr_manual)
for q in range(1, 6):
    q_col = f'Q{q}'
    total_ret = (1 + qr_manual[q_col]).prod() - 1
    ann_ret = (1 + total_ret) ** (12 / n_months) - 1
    ann_vol = qr_manual[q_col].std() * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    label = "Short" if q == 1 else ("Long" if q == 5 else "")
    print(f"  Portfolio {q} ({label:>5}): Return={ann_ret:>7.2%}, "
          f"Risk={ann_vol:>7.2%}, Sharpe={sharpe:>6.3f}")

# Q5 - Q1
ls_rets = qr_manual['Q5_Q1']
ls_total = (1 + ls_rets).prod() - 1
ls_ann = (1 + ls_total) ** (12 / n_months) - 1
ls_vol = ls_rets.std() * np.sqrt(12)
ls_sharpe = (ls_ann - 0.02) / ls_vol if ls_vol > 0 else 0
print(f"  Portfolio [5-1]:     Return={ls_ann:>7.2%}, "
      f"Risk={ls_vol:>7.2%}, Sharpe={ls_sharpe:>6.3f}")

# WRDS reference
print("\n--- WRDS Official Reference ---")
print("  Portfolio 1 (Short): Return=23.47%, Risk=36.70%, Sharpe=0.640")
print("  Portfolio 5 (Long):  Return=19.13%, Risk=26.19%, Sharpe=0.730")
print("  Portfolio [5-1]:     Return=-4.34%, Risk=15.25%, Sharpe=-0.285")

# ============================================================
# Part 4: Using QFRsch factor_eval.quantile_backtest()
# ============================================================
print("\n" + "=" * 70)
print("Part B: QFRsch factor_eval.quantile_backtest()")
print("=" * 70)

from qfrsch.analysis.factor_eval import quantile_backtest

# Prepare factor_values: DataFrame with index=fdate, columns=permno
factor_pivot = sig.pivot_table(index='fdate', columns='PERMNO', values='momentum12')
print(f"Factor matrix: {factor_pivot.shape}")

# Prepare forward_returns: DataFrame with same structure
# For each fdate, the "forward return" is the next month's return
forward_ret_dict = {}
for fdate in sorted(factor_pivot.index):
    next_month = (fdate + pd.offsets.MonthEnd(1))
    next_ym = next_month.to_period('M')
    ret_m = monthly_ret[monthly_ret['year_month'] == next_ym].set_index('permno')['monthly_ret']
    forward_ret_dict[fdate] = ret_m

forward_returns_df = pd.DataFrame(forward_ret_dict).T
forward_returns_df.index.name = 'fdate'
forward_returns_df.columns.name = 'permno'
print(f"Forward returns matrix: {forward_returns_df.shape}")

# Run QFRsch quantile_backtest
qbt = quantile_backtest(factor_pivot, forward_returns_df, num_quantiles=5)

print("\n--- QFRsch quantile_backtest() Results ---")
print("Annualized Returns per Quintile:")
print(qbt['quantile_annual_ret'])
print("\nAnnualized Volatility per Quintile:")
print(qbt['quantile_annual_vol'])

# Note: quantile_backtest uses daily annualization (252). For monthly data,
# we need periods_per_year=12. Let's compute manually from the monthly quintile returns.
print("\n--- Corrected for Monthly Frequency ---")
qr_df = qbt['quantile_returns']
n_obs = len(qr_df)
for q in range(1, 6):
    rets = qr_df[q].dropna()
    total = (1 + rets).prod() - 1
    ann_ret = (1 + total) ** (12 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    label = "Short" if q == 1 else ("Long" if q == 5 else "")
    print(f"  Portfolio {q} ({label:>5}): Return={ann_ret:>7.2%}, "
          f"Risk={ann_vol:>7.2%}, Sharpe={sharpe:>6.3f}")

hml = qbt['high_minus_low'].dropna()
hml_total = (1 + hml).prod() - 1
hml_ann = (1 + hml_total) ** (12 / len(hml)) - 1
hml_vol = hml.std() * np.sqrt(12)
hml_sharpe = (hml_ann - 0.02) / hml_vol if hml_vol > 0 else 0
print(f"  Portfolio [5-1]:     Return={hml_ann:>7.2%}, "
      f"Risk={hml_vol:>7.2%}, Sharpe={hml_sharpe:>6.3f}")

# ============================================================
# Part 5: SimpleEngine Backtest (Top Quintile Long-Only)
# ============================================================
print("\n" + "=" * 70)
print("Part C: SimpleEngine Backtest (Q5 Long-Only)")
print("=" * 70)

from qfrsch.backtest.engine import SimpleEngine
from qfrsch.backtest.models import BacktestConfig

# Build monthly target weights for Q5 (top quintile)
# At each fdate, identify Q5 stocks, assign equal weights
# Forward-fill these weights to all trading days in the holding month

target_weights_rows = []
for fdate in sorted(factor_pivot.index):
    scores = factor_pivot.loc[fdate].dropna()
    if len(scores) < 50:
        continue
    # Q5 = top 20% by momentum12
    threshold = scores.quantile(0.8)
    q5_stocks = scores[scores >= threshold].index.tolist()
    if len(q5_stocks) == 0:
        continue
    weight = 1.0 / len(q5_stocks)
    weights = {str(s): weight for s in q5_stocks}
    target_weights_rows.append({'fdate': fdate, 'weights': weights})

# Get all unique tickers in any Q5 portfolio
all_q5_tickers = set()
for row in target_weights_rows:
    all_q5_tickers.update(row['weights'].keys())
all_q5_tickers = sorted(all_q5_tickers)
print(f"All Q5 tickers across months: {len(all_q5_tickers)}")

# Build daily target weights by forward-filling monthly signals
# For each month, weights apply from first trading day after fdate to last trading day of next month
mp_daily = mp[['permno', 'date', 'adj_close', 'ret']].copy()
mp_daily['ticker'] = mp_daily['permno'].astype(str)

trading_dates = sorted(mp_daily['date'].unique())
trading_dates = [d for d in trading_dates if d >= pd.Timestamp('2019-02-01') and d <= pd.Timestamp('2022-01-31')]
print(f"Trading dates for backtest: {len(trading_dates)}")

# Map each trading date to its effective signal (from previous month-end)
date_to_weights = {}
signal_dates = sorted([r['fdate'] for r in target_weights_rows])

for td in trading_dates:
    # Find the most recent fdate <= td
    effective_fdate = None
    for fd in reversed(signal_dates):
        if fd <= td:
            effective_fdate = fd
            break
    if effective_fdate is None:
        continue
    
    # Find matching weights
    for row in target_weights_rows:
        if row['fdate'] == effective_fdate:
            date_to_weights[td] = row['weights']
            break

# Build target_weights_df (date × tickers)
tw_records = []
for td, wts in date_to_weights.items():
    row = {t: 0.0 for t in all_q5_tickers}
    for t, w in wts.items():
        if t in row:
            row[t] = w
    tw_records.append({'date': td, **row})

target_weights_df = pd.DataFrame(tw_records).set_index('date')
target_weights_df.index = pd.to_datetime(target_weights_df.index)
print(f"Target weights: {target_weights_df.shape}")

# Build OHLCV dataframe for SimpleEngine
# SimpleEngine expects: DatetimeIndex, columns=['ticker', 'open', 'high', 'low', 'close', 'volume']
# We only need adj_close as price for the Q5 stocks
q5_permno_list = [int(t) for t in all_q5_tickers]
ohlcv_data = mp_daily[mp_daily['permno'].isin(q5_permno_list)].copy()
ohlcv_data = ohlcv_data[(ohlcv_data['date'] >= target_weights_df.index.min()) &
                         (ohlcv_data['date'] <= target_weights_df.index.max())]

# Format for SimpleEngine
ohlcv_df = pd.DataFrame({
    'ticker': ohlcv_data['ticker'].values,
    'open': ohlcv_data['adj_close'].values,
    'high': ohlcv_data['adj_close'].values,
    'low': ohlcv_data['adj_close'].values,
    'close': ohlcv_data['adj_close'].values,
    'volume': np.ones(len(ohlcv_data)),
}, index=pd.to_datetime(ohlcv_data['date'].values))
ohlcv_df.index.name = 'date'

print(f"OHLCV data: {len(ohlcv_df):,} rows, {ohlcv_df['ticker'].nunique()} tickers")

# Run SimpleEngine with low costs (WRDS uses essentially zero costs for quantile returns)
config = BacktestConfig(
    initial_capital=10_000_000,
    commission_rate=0.0,  # WRDS quantile returns don't include transaction costs
    slippage_rate=0.0,
    rebalance_frequency='D',
    risk_free_rate=0.02,
    execution_price='close',
)

print("Running SimpleEngine...")
t1 = time.time()
engine = SimpleEngine(config)
result = engine.run(ohlcv_df, target_weights_df)
t_engine = time.time() - t1
print(f"SimpleEngine completed in {t_engine:.1f}s")

# Engine metrics
print("\n--- SimpleEngine Results (Q5 Long-Only) ---")
for k, v in result.metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# Recompute with monthly returns for fair comparison 
eq = result.equity_curve['total_value']
monthly_eq = eq.resample('ME').last()
monthly_engine_ret = monthly_eq.pct_change().dropna()

if len(monthly_engine_ret) > 0:
    se_total = (1 + monthly_engine_ret).prod() - 1
    se_ann = (1 + se_total) ** (12 / len(monthly_engine_ret)) - 1
    se_vol = monthly_engine_ret.std() * np.sqrt(12)
    se_sharpe = (se_ann - 0.02) / se_vol if se_vol > 0 else 0
    print(f"\n  Monthly-based metrics:")
    print(f"    Annual Return: {se_ann:.2%}")
    print(f"    Annual Risk:   {se_vol:.2%}")
    print(f"    Sharpe Ratio:  {se_sharpe:.3f}")

# ============================================================
# Part 6: Summary Comparison
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)

print(f"\n{'Method':<30} {'Ann Return':>12} {'Ann Risk':>12} {'Sharpe':>10}")
print("-" * 65)
print(f"{'WRDS Official (Q5 Long)':<30} {'19.13%':>12} {'26.19%':>12} {'0.730':>10}")

# Manual replication Q5
q5_manual = qr_manual['Q5']
m_total = (1 + q5_manual).prod() - 1
m_ann = (1 + m_total) ** (12 / len(q5_manual)) - 1
m_vol = q5_manual.std() * np.sqrt(12)
m_sharpe = (m_ann - 0.02) / m_vol if m_vol > 0 else 0
print(f"{'Manual Quintile (Q5)':<30} {m_ann:>11.2%} {m_vol:>11.2%} {m_sharpe:>10.3f}")

# QFRsch quantile_backtest Q5
q5_qfrsch = qbt['quantile_returns'][5].dropna()
qf_total = (1 + q5_qfrsch).prod() - 1
qf_ann = (1 + qf_total) ** (12 / len(q5_qfrsch)) - 1
qf_vol = q5_qfrsch.std() * np.sqrt(12)
qf_sharpe = (qf_ann - 0.02) / qf_vol if qf_vol > 0 else 0
print(f"{'QFRsch quantile_backtest (Q5)':<30} {qf_ann:>11.2%} {qf_vol:>11.2%} {qf_sharpe:>10.3f}")

# SimpleEngine Q5
if len(monthly_engine_ret) > 0:
    print(f"{'SimpleEngine (Q5 Long-Only)':<30} {se_ann:>11.2%} {se_vol:>11.2%} {se_sharpe:>10.3f}")

print("\n" + "=" * 70)
print(f"Total elapsed: {time.time()-t0:.1f}s")
