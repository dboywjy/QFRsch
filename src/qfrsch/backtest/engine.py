"""
Backtest Engine Module
Implements SimpleEngine class for lightweight, pandas-based backtesting.
"""

from __future__ import annotations

from typing import Tuple, Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from qfrsch.backtest.models import BacktestConfig, BacktestResult
from qfrsch.backtest.utils import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_growth_metrics,
    calculate_volatility,
)


class SimpleEngine:
    """
    Lightweight pandas-based backtest engine.
    
    Implements daily-loop simulation with support for transaction costs
    (commissions and slippage) and flexible rebalancing.
    
    The engine processes OHLCV data and target weight matrix to simulate
    portfolio evolution, tracking cash, positions, and performance metrics.
    
    Parameters
    ----------
    config : BacktestConfig
        Configuration object containing initial_capital, commission_rate, etc.
        
    Attributes
    ----------
    config : BacktestConfig
        Backtest configuration.
    ohlcv_df : pd.DataFrame
        OHLCV data. Index: date, columns: ['ticker', 'open', 'high', 'low', 'close', 'volume'].
    target_weights_df : pd.DataFrame
        Target weight matrix. Index: date, columns: tickers, values: target weight (e.g., 0.5 for 50%).
    dates : pd.DatetimeIndex
        Sorted list of all trading dates.
    tickers : list[str]
        All unique tickers in universe.
        
    Examples
    --------
    >>> config = BacktestConfig(initial_capital=1_000_000)
    >>> engine = SimpleEngine(config)
    >>> result = engine.run(ohlcv_df, target_weights_df)
    >>> print(f"Total Return: {result.metrics['total_return']:.2%}")
    """
    
    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize backtest engine with configuration.
        
        Parameters
        ----------
        config : BacktestConfig
            Configuration parameters for backtest.
        """
        self.config = config
        self.ohlcv_df: Optional[pd.DataFrame] = None
        self.target_weights_df: Optional[pd.DataFrame] = None
        self.dates: Optional[pd.DatetimeIndex] = None
        self.tickers: List[str] = []
    
    def run(
        self,
        ohlcv_df: pd.DataFrame,
        target_weights_df: pd.DataFrame
    ) -> BacktestResult:
        """
        Execute backtest simulation.
        
        Main entry point that orchestrates data preparation, daily-loop simulation,
        and result aggregation.
        
        Parameters
        ----------
        ohlcv_df : pd.DataFrame
            OHLCV data. Index: date, columns: ['ticker', 'open', 'high', 'low', 'close', 'volume'].
            Must have all required columns.
        target_weights_df : pd.DataFrame
            Target weight matrix. Index: date, columns: tickers, values: target weight.
            Missing values are treated as 0 (no position).
            
        Returns
        -------
        BacktestResult
            Complete backtest results including equity curve, trades, holdings, and metrics.
            
        Raises
        ------
        ValueError
            If ohlcv_df or target_weights_df is empty or malformed.
        """
        
        # Validate inputs
        if ohlcv_df.empty or target_weights_df.empty:
            raise ValueError("OHLCV and target weights dataframes must not be empty")
        
        # Store and prepare data
        self._prepare_data(ohlcv_df, target_weights_df)
        
        # Run daily-loop simulation
        equity_curve, trades, holdings, daily_weights = self._simulate()
        
        # Calculate returns and metrics
        daily_returns = self._calculate_daily_returns(equity_curve)
        metrics = self._calculate_metrics(equity_curve, daily_returns)
        
        # Construct and return result
        result = BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            holdings=holdings,
            daily_returns=daily_returns,
            daily_weights=daily_weights,
            metrics=metrics,
            config=self.config,
        )
        
        return result
    
    def _prepare_data(
        self,
        ohlcv_df: pd.DataFrame,
        target_weights_df: pd.DataFrame
    ) -> None:
        """
        Prepare and validate OHLCV and weight data.
        
        Ensures data is properly indexed and tickers are identified.
        
        Parameters
        ----------
        ohlcv_df : pd.DataFrame
            Raw OHLCV data.
        target_weights_df : pd.DataFrame
            Raw target weights.
        """
        
        # Ensure index is datetime
        if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
            ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
        if not isinstance(target_weights_df.index, pd.DatetimeIndex):
            target_weights_df.index = pd.to_datetime(target_weights_df.index)
        
        # Store data
        self.ohlcv_df = ohlcv_df.sort_index()
        self.target_weights_df = target_weights_df.sort_index()
        
        # Extract dates (intersection of both dataframes)
        ohlcv_dates = set(self.ohlcv_df.index.date)
        weights_dates = set(self.target_weights_df.index.date)
        common_dates = sorted(ohlcv_dates & weights_dates)
        
        if not common_dates:
            raise ValueError("No overlapping dates between OHLCV and target weights")
        
        self.dates = pd.to_datetime(common_dates)
        
        # Extract all unique tickers from target_weights_df columns
        self.tickers = sorted(self.target_weights_df.columns.tolist())
    
    def _simulate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute daily-loop backtest simulation.
        
        Core simulation logic:
        1. Initialize cash and positions
        2. For each date:
           a. Calculate returns based on price changes
           b. Update portfolio value
           c. Extract target weights for rebalancing
           d. Calculate position changes (turnover)
           e. Apply transaction costs (commission + slippage)
           f. Update holdings
           
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (equity_curve, trades, holdings, daily_weights)
        """
        
        # Initialize state tracking
        equity_records = []
        trade_records = []
        holdings_records = []
        weights_records = []
        
        # State variables
        cash = self.config.initial_capital
        holdings = {ticker: 0.0 for ticker in self.tickers}  # Current shares held
        previous_prices = {ticker: 0.0 for ticker in self.tickers}  # Last execution price
        
        # Get execution price column
        price_col = self.config.execution_price.lower()
        if price_col not in self.ohlcv_df.columns:
            price_col = 'close'
        
        # Main daily loop
        for date in self.dates:
            date_prices = self._get_prices_for_date(date, price_col)
            
            # Skip if no price data
            if date_prices is None or not date_prices:
                continue
            
            # Step 1: Calculate portfolio value change from price movements
            position_value = sum(holdings.get(ticker, 0.0) * date_prices.get(ticker, 0.0) 
                                for ticker in self.tickers)
            total_value = cash + position_value
            
            # Record daily state before rebalancing
            holdings_records.append({
                'date': date,
                'holdings': holdings.copy(),
                'total_value': total_value,
                'cash': cash,
                'position_value': position_value,
            })
            
            # Record equity before rebalancing
            equity_records.append({
                'date': date,
                'total_value': total_value,
                'cash': cash,
                'positions_value': position_value,
            })
            
            # Step 2: Get target weights for this date
            target_weights = self._get_target_weights(date)
            
            # Step 3: Calculate target quantities
            target_quantities = {}
            for ticker in self.tickers:
                target_weight = target_weights.get(ticker, 0.0)
                target_quantities[ticker] = (target_weight * total_value) / max(date_prices.get(ticker, 1.0), 0.001)
            
            # Step 4: Calculate position changes and transaction costs
            turnover = 0.0
            for ticker in self.tickers:
                current_qty = holdings.get(ticker, 0.0)
                target_qty = target_quantities.get(ticker, 0.0)
                qty_change = target_qty - current_qty
                
                if abs(qty_change) > 1e-8:  # Skip negligible changes
                    exec_price = date_prices.get(ticker, 0.0)
                    
                    # Apply slippage
                    if qty_change > 0:  # Buying
                        exec_price = exec_price * (1 + self.config.slippage_rate)
                    else:  # Selling
                        exec_price = exec_price * (1 - self.config.slippage_rate)
                    
                    # Calculate transaction cost
                    trade_value = abs(qty_change * exec_price)
                    commission = trade_value * self.config.commission_rate
                    
                    # Update holdings and cash
                    holdings[ticker] = target_qty
                    cash -= qty_change * exec_price + commission
                    turnover += trade_value
                    
                    # Record trade
                    trade_records.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'BUY' if qty_change > 0 else 'SELL',
                        'quantity': abs(qty_change),
                        'price': exec_price,
                        'commission': commission,
                        'cash_impact': -(qty_change * exec_price + commission),
                    })
                    
                    previous_prices[ticker] = exec_price
            
            # Record daily weights
            weights_records.append({
                'date': date,
                'weights': target_weights.copy(),
            })
        
        # Convert records to dataframes
        equity_df = self._records_to_equity_df(equity_records)
        trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()
        holdings_df = self._records_to_holdings_df(holdings_records)
        weights_df = self._records_to_weights_df(weights_records)
        
        return equity_df, trades_df, holdings_df, weights_df
    
    def _get_prices_for_date(self, date: pd.Timestamp, price_col: str) -> Optional[Dict[str, float]]:
        """
        Extract prices for all tickers on a given date.
        
        Parameters
        ----------
        date : pd.Timestamp
            Trading date.
        price_col : str
            Which price column to use ('open', 'close', etc.).
            
        Returns
        -------
        dict[str, float] or None
            Dictionary mapping {ticker: price}. None if no data found.
        """
        
        try:
            date_data = self.ohlcv_df.loc[self.ohlcv_df.index.normalize() == date.normalize()]
        except Exception:
            return None
        
        if date_data.empty:
            return None
        
        prices = {}
        for ticker in self.tickers:
            ticker_data = date_data[date_data['ticker'] == ticker]
            if not ticker_data.empty and price_col in ticker_data.columns:
                prices[ticker] = float(ticker_data[price_col].iloc[0])
        
        return prices if prices else None
    
    def _get_target_weights(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Extract target weights for all tickers on a given date.
        
        Parameters
        ----------
        date : pd.Timestamp
            Target date.
            
        Returns
        -------
        dict[str, float]
            Dictionary mapping {ticker: target_weight}.
            Missing tickers default to 0 (no position).
        """
        
        try:
            weight_row = self.target_weights_df.loc[date.normalize()]
        except KeyError:
            return {ticker: 0.0 for ticker in self.tickers}
        
        weights = {}
        for ticker in self.tickers:
            if ticker in weight_row.index:
                weights[ticker] = float(weight_row[ticker])
            else:
                weights[ticker] = 0.0
        
        return weights
    
    def _calculate_daily_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """
        Calculate daily portfolio returns.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            Daily equity values.
            
        Returns
        -------
        pd.Series
            Daily returns. Index: date, values: return rate.
        """
        
        returns = equity_curve['total_value'].pct_change()
        returns.iloc[0] = 0.0  # First day return is 0
        return returns
    
    def _calculate_metrics(
        self,
        equity_curve: pd.DataFrame,
        daily_returns: pd.Series
    ) -> Dict[str, any]:
        """
        Calculate comprehensive performance metrics.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            Daily equity values.
        daily_returns : pd.Series
            Daily returns.
            
        Returns
        -------
        dict
            Dictionary of metrics: total_return, sharpe_ratio, max_drawdown, etc.
        """
        
        equity_series = equity_curve['total_value']
        initial_capital = self.config.initial_capital
        
        # Growth metrics
        total_return, annualized_return = calculate_growth_metrics(equity_series, initial_capital)
        
        # Risk metrics
        max_drawdown = calculate_max_drawdown(equity_series)
        volatility = calculate_volatility(daily_returns)
        sharpe_ratio = calculate_sharpe_ratio(daily_returns, self.config.risk_free_rate)
        calmar_ratio = calculate_calmar_ratio(daily_returns)
        win_rate = calculate_win_rate(daily_returns)
        
        # Additional metrics
        final_value = equity_series.iloc[-1]
        num_trades = len(self.ohlcv_df[self.ohlcv_df['ticker'].isin(self.tickers)])
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trading_days': len(equity_series),
        }
    
    @staticmethod
    def _records_to_equity_df(records: List[Dict]) -> pd.DataFrame:
        """Convert equity records to DataFrame."""
        if not records:
            return pd.DataFrame(columns=['total_value', 'cash', 'positions_value'])
        
        df = pd.DataFrame(records)
        df.set_index('date', inplace=True)
        return df[['total_value', 'cash', 'positions_value']]
    
    @staticmethod
    def _records_to_holdings_df(records: List[Dict]) -> pd.DataFrame:
        """Convert holdings records to DataFrame."""
        if not records:
            return pd.DataFrame()
        
        dates = []
        holdings_list = []
        
        for record in records:
            dates.append(record['date'])
            holdings_list.append(record['holdings'])
        
        df = pd.DataFrame(holdings_list, index=pd.to_datetime(dates))
        return df.fillna(0.0)
    
    @staticmethod
    def _records_to_weights_df(records: List[Dict]) -> pd.DataFrame:
        """Convert weights records to DataFrame."""
        if not records:
            return pd.DataFrame()
        
        dates = []
        weights_list = []
        
        for record in records:
            dates.append(record['date'])
            weights_list.append(record['weights'])
        
        df = pd.DataFrame(weights_list, index=pd.to_datetime(dates))
        return df.fillna(0.0)


__all__ = ['SimpleEngine']
