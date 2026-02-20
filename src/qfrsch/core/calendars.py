"""
Trading Calendar Management Module
Responsible for handling all date-related issues in the quantitative system
"""

from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal



class TradingCalendar:
    """
    Trading Calendar Manager for quantitative backtesting.
    Provides utilities for handling trading dates and market calendars.
    """

    def __init__(self, calendar: str = "NYSE"):
        """
        Initialize trading calendar.

        Args:
            calendar: Market calendar name (e.g., "NYSE", "SSE", "SZSE")
        """
        self.calendar_name = calendar
        try:
            self.calendar = mcal.get_calendar(calendar)
        except Exception:
            raise ValueError(f"Calendar '{calendar}' not supported")

    def get_trading_dates(
        self, start_date: Union[str, date], end_date: Union[str, date]
    ) -> pd.DatetimeIndex:
        """
        Get all trading dates in a date range.

        Args:
            start_date: Start date (str or datetime.date)
            end_date: End date (str or datetime.date)

        Returns:
            DatetimeIndex of trading dates
        """
        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
        return schedule.index

    def is_trading_day(self, date_obj: Union[str, date]) -> bool:
        """
        Check if a date is a trading day.

        Args:
            date_obj: Date to check

        Returns:
            True if trading day, False otherwise
        """
        trading_dates = self.get_trading_dates(date_obj, date_obj)
        return len(trading_dates) > 0

    def count_trading_days(
        self, start_date: Union[str, date], end_date: Union[str, date]
    ) -> int:
        """
        Count trading days between two dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Number of trading days
        """
        return len(self.get_trading_dates(start_date, end_date))

    def get_next_trading_date(
        self, date_obj: Union[str, date], n: int = 1
    ) -> Optional[pd.Timestamp]:
        """
        Get the n-th next trading date.

        Args:
            date_obj: Reference date
            n: Number of trading days forward (default: 1)

        Returns:
            Next trading date or None if not found
        """
        date_ts = pd.Timestamp(date_obj).date()
        end_date = date_ts + timedelta(days=100)
        trading_dates = self.get_trading_dates(date_ts, end_date)
        if len(trading_dates) > n:
            return trading_dates[n]
        return None

    def get_prev_trading_date(
        self, date_obj: Union[str, date], n: int = 1
    ) -> Optional[pd.Timestamp]:
        """
        Get the n-th previous trading date.

        Args:
            date_obj: Reference date
            n: Number of trading days backward (default: 1)

        Returns:
            Previous trading date or None if not found
        """
        date_ts = pd.Timestamp(date_obj).date()
        start_date = date_ts - timedelta(days=100)
        trading_dates = self.get_trading_dates(start_date, date_ts)
        if len(trading_dates) >= n:
            return trading_dates[-n]
        return None

    def get_market_open_time(
        self, date_obj: Union[str, date]
    ) -> Optional[datetime]:
        """
        Get market open time for a specific date.

        Args:
            date_obj: Date to query

        Returns:
            Market open time or None if not trading day
        """
        try:
            schedule = self.calendar.schedule(start_date=date_obj, end_date=date_obj)
            if len(schedule) > 0:
                return schedule.iloc[0]["market_open"]
            return None
        except Exception:
            return None

    def get_market_close_time(
        self, date_obj: Union[str, date]
    ) -> Optional[datetime]:
        """
        Get market close time for a specific date.

        Args:
            date_obj: Date to query

        Returns:
            Market close time or None if not trading day
        """
        try:
            schedule = self.calendar.schedule(start_date=date_obj, end_date=date_obj)
            if len(schedule) > 0:
                return schedule.iloc[0]["market_close"]
            return None
        except Exception:
            return None

    def get_trading_schedule(
        self, start_date: Union[str, date], end_date: Union[str, date]
    ) -> pd.DataFrame:
        """
        Get full trading schedule (open and close times) for date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with market_open and market_close columns
        """
        return self.calendar.schedule(start_date=start_date, end_date=end_date)

    def get_early_closes(
        self, start_date: Union[str, date], end_date: Union[str, date]
    ) -> List[date]:
        """
        Get all early close dates in a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of dates with early closing times
        """
        schedule = self.get_trading_schedule(start_date, end_date)
        # Compare with standard close time
        standard_close = schedule.iloc[0]["market_close"].time() if len(schedule) > 0 else None
        
        early_closes = []
        for idx, row in schedule.iterrows():
            if row["market_close"].time() < standard_close:
                early_closes.append(idx.date())
        return early_closes



# ==================== Date Conversion Functions ====================


def to_datetime(date_obj: Union[str, date, datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Convert various date formats to pandas Timestamp.

    Args:
        date_obj: Date in various formats (str, date, datetime, Timestamp)

    Returns:
        pandas Timestamp object
    """
    return pd.Timestamp(date_obj)


def to_date(date_obj: Union[str, date, datetime, pd.Timestamp]) -> date:
    """
    Convert various date formats to datetime.date.

    Args:
        date_obj: Date in various formats

    Returns:
        datetime.date object
    """
    ts = pd.Timestamp(date_obj)
    return ts.date()


def to_str(date_obj: Union[str, date, datetime, pd.Timestamp], fmt: str = "%Y-%m-%d") -> str:
    """
    Convert date to string with specified format.

    Args:
        date_obj: Date in various formats
        fmt: Format string (default: "%Y-%m-%d")

    Returns:
        Formatted date string
    """
    ts = pd.Timestamp(date_obj)
    return ts.strftime(fmt)


# ==================== Date Frequency Functions ====================


def get_date_week(date_obj: Union[str, date]) -> int:
    """
    Get week number (1-52/53) of the year.

    Args:
        date_obj: Date in various formats

    Returns:
        Week number (1-based)
    """
    ts = pd.Timestamp(date_obj)
    return ts.isocalendar()[1]


def get_date_month(date_obj: Union[str, date]) -> int:
    """
    Get month number (1-12) of the date.

    Args:
        date_obj: Date in various formats

    Returns:
        Month number (1-12)
    """
    ts = pd.Timestamp(date_obj)
    return ts.month


def get_date_quarter(date_obj: Union[str, date]) -> int:
    """
    Get quarter number (1-4) of the year.

    Args:
        date_obj: Date in various formats

    Returns:
        Quarter number (1-4)
    """
    ts = pd.Timestamp(date_obj)
    return ts.quarter


def get_date_year(date_obj: Union[str, date]) -> int:
    """
    Get year of the date.

    Args:
        date_obj: Date in various formats

    Returns:
        Year
    """
    ts = pd.Timestamp(date_obj)
    return ts.year


def get_week_start(date_obj: Union[str, date]) -> date:
    """
    Get the first day (Monday) of the week.

    Args:
        date_obj: Date in various formats

    Returns:
        Monday of the week
    """
    ts = pd.Timestamp(date_obj)
    return (ts - pd.Timedelta(days=ts.weekday())).date()


def get_week_end(date_obj: Union[str, date]) -> date:
    """
    Get the last day (Sunday) of the week.

    Args:
        date_obj: Date in various formats

    Returns:
        Sunday of the week
    """
    ts = pd.Timestamp(date_obj)
    return (ts + pd.Timedelta(days=6 - ts.weekday())).date()


def get_month_start(date_obj: Union[str, date]) -> date:
    """
    Get the first day of the month.

    Args:
        date_obj: Date in various formats

    Returns:
        First day of the month
    """
    ts = pd.Timestamp(date_obj)
    return ts.replace(day=1).date()


def get_month_end(date_obj: Union[str, date]) -> date:
    """
    Get the last day of the month.

    Args:
        date_obj: Date in various formats

    Returns:
        Last day of the month
    """
    ts = pd.Timestamp(date_obj)
    return (ts + pd.offsets.MonthEnd(0)).date()


def get_quarter_start(date_obj: Union[str, date]) -> date:
    """
    Get the first day of the quarter.

    Args:
        date_obj: Date in various formats

    Returns:
        First day of the quarter
    """
    ts = pd.Timestamp(date_obj)
    quarter_month = (ts.quarter - 1) * 3 + 1
    return ts.replace(month=quarter_month, day=1).date()


def get_quarter_end(date_obj: Union[str, date]) -> date:
    """
    Get the last day of the quarter.

    Args:
        date_obj: Date in various formats

    Returns:
        Last day of the quarter
    """
    ts = pd.Timestamp(date_obj)
    quarter_month = ts.quarter * 3
    return ts.replace(month=quarter_month, day=1).date() + timedelta(days=32)
    # Add 32 days then get last day to handle month-end correctly


def get_year_start(date_obj: Union[str, date]) -> date:
    """
    Get the first day of the year.

    Args:
        date_obj: Date in various formats

    Returns:
        First day of the year
    """
    ts = pd.Timestamp(date_obj)
    return ts.replace(month=1, day=1).date()


def get_year_end(date_obj: Union[str, date]) -> date:
    """
    Get the last day of the year.

    Args:
        date_obj: Date in various formats

    Returns:
        Last day of the year
    """
    ts = pd.Timestamp(date_obj)
    return ts.replace(month=12, day=31).date()


def generate_date_range(
    start_date: Union[str, date], end_date: Union[str, date], freq: str = "D"
) -> pd.DatetimeIndex:
    """
    Generate date range with specified frequency.

    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D': daily, 'W': weekly, 'M': monthly, 'Q': quarterly, 'Y': yearly)

    Returns:
        DatetimeIndex with specified frequency
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def split_by_month(
    start_date: Union[str, date], end_date: Union[str, date]
) -> List[tuple]:
    """
    Split date range into monthly periods.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of tuples (month_start, month_end)
    """
    periods = pd.period_range(start=start_date, end=end_date, freq="M")
    result = []
    for period in periods:
        result.append((period.start_time.date(), period.end_time.date()))
    return result


def split_by_quarter(
    start_date: Union[str, date], end_date: Union[str, date]
) -> List[tuple]:
    """
    Split date range into quarterly periods.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of tuples (quarter_start, quarter_end)
    """
    periods = pd.period_range(start=start_date, end=end_date, freq="Q")
    result = []
    for period in periods:
        result.append((period.start_time.date(), period.end_time.date()))
    return result


def split_by_year(
    start_date: Union[str, date], end_date: Union[str, date]
) -> List[tuple]:
    """
    Split date range into yearly periods.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of tuples (year_start, year_end)
    """
    periods = pd.period_range(start=start_date, end=end_date, freq="Y")
    result = []
    for period in periods:
        result.append((period.start_time.date(), period.end_time.date()))
    return result


# ==================== Timezone Conversion Functions ====================


def to_utc(dt: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Convert datetime to UTC timezone.

    Args:
        dt: Datetime in any format

    Returns:
        Datetime in UTC timezone
    """
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        # Assume UTC if no timezone info
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_timezone(
    dt: Union[str, datetime, pd.Timestamp], tz: str
) -> pd.Timestamp:
    """
    Convert datetime to specified timezone.

    Args:
        dt: Datetime in any format
        tz: Timezone name (e.g., "US/Eastern", "Asia/Shanghai", "Europe/London")

    Returns:
        Datetime in specified timezone
    """
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        # Assume UTC if no timezone info
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz)


def localize(
    dt: Union[str, datetime, pd.Timestamp], tz: str = "UTC"
) -> pd.Timestamp:
    """
    Add timezone information to naive datetime.

    Args:
        dt: Naive datetime
        tz: Timezone name (default: "UTC")

    Returns:
        Timezone-aware datetime
    """
    ts = pd.Timestamp(dt)
    if ts.tz is not None:
        return ts
    return ts.tz_localize(tz)


def get_timezone_aware(
    dt: Union[str, datetime, pd.Timestamp], tz: str = "UTC"
) -> pd.Timestamp:
    """
    Ensure datetime is timezone-aware.

    Args:
        dt: Datetime in any format
        tz: Timezone name to assume if naive (default: "UTC")

    Returns:
        Timezone-aware datetime
    """
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        return ts.tz_localize(tz)
    return ts


def remove_timezone(dt: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Remove timezone information from datetime.

    Args:
        dt: Datetime in any format

    Returns:
        Naive datetime (without timezone)
    """
    ts = pd.Timestamp(dt)
    if ts.tz is not None:
        return ts.tz_localize(None)
    return ts


def get_utc_offset(dt: Union[str, datetime, pd.Timestamp], tz: str) -> str:
    """
    Get UTC offset for a specific timezone at given datetime.

    Args:
        dt: Datetime in any format
        tz: Timezone name

    Returns:
        UTC offset string (e.g., "+08:00", "-05:00")
    """
    ts = pd.Timestamp(dt, tz=tz)
    offset = ts.utcoffset()
    return str(offset)


# ==================== Market Timezone Helpers ====================


def to_market_timezone(
    dt: Union[str, datetime, pd.Timestamp], market: str
) -> pd.Timestamp:
    """
    Convert datetime to specified market's timezone.

    Args:
        dt: Datetime in any format
        market: Market name ("US", "CN", "HK", "JP", "EU", "UK")

    Returns:
        Datetime in market's local timezone

    Raises:
        ValueError: If market is not supported
    """
    market_tz_map = {
        "US": "US/Eastern",
        "CN": "Asia/Shanghai",
        "HK": "Asia/Hong_Kong",
        "JP": "Asia/Tokyo",
        "EU": "Europe/Brussels",
        "UK": "Europe/London",
    }

    if market.upper() not in market_tz_map:
        raise ValueError(f"Market '{market}' not supported")

    return to_timezone(dt, market_tz_map[market.upper()])


def get_market_open_hours(market: str) -> tuple:
    """
    Get standard market open and close hours for a market.
    Returns hours in local market time.

    Args:
        market: Market name ("NYSE", "NASDAQ", "TSE", "LSE", "SSE", "SZSE", "HKEX")

    Returns:
        Tuple of (open_hour, close_hour) in 24h format

    Examples:
        ("09:30", "16:00") for US markets
        ("09:30", "15:00") for Japan
    """
    market_hours_map = {
        "NYSE": ("09:30", "16:00"),
        "NASDAQ": ("09:30", "16:00"),
        "TSE": ("09:00", "15:00"),
        "LSE": ("08:00", "16:30"),
        "SSE": ("09:30", "15:00"),
        "SZSE": ("09:30", "15:00"),
        "HKEX": ("09:30", "16:00"),
    }

    if market.upper() not in market_hours_map:
        raise ValueError(f"Market '{market}' not supported. Use TradingCalendar.get_market_open_time() for exact times.")

    return market_hours_map[market.upper()]


