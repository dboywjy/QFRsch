"""
Test cases for Trading Calendar Management Module
"""

import pytest
from datetime import datetime, date, timedelta
import pandas as pd

from qfrsch.core.calendars import (
    TradingCalendar,
    to_datetime,
    to_date,
    to_str,
    get_date_week,
    get_date_month,
    get_date_quarter,
    get_date_year,
    get_week_start,
    get_week_end,
    get_month_start,
    get_month_end,
    get_quarter_start,
    get_quarter_end,
    get_year_start,
    get_year_end,
    generate_date_range,
    split_by_month,
    split_by_quarter,
    split_by_year,
    to_utc,
    to_timezone,
    localize,
    get_timezone_aware,
    remove_timezone,
    get_utc_offset,
    to_market_timezone,
    get_market_open_hours,
)


class TestTradingCalendar:
    """Test TradingCalendar class"""

    def test_init_valid_calendar(self):
        """Test initialization with valid calendar"""
        cal = TradingCalendar("NYSE")
        assert cal.calendar_name == "NYSE"
        assert cal.calendar is not None

    def test_init_invalid_calendar(self):
        """Test initialization with invalid calendar"""
        with pytest.raises(ValueError):
            TradingCalendar("INVALID_CALENDAR")

    def test_get_trading_dates(self):
        """Test getting trading dates in a range"""
        cal = TradingCalendar("NYSE")
        trading_dates = cal.get_trading_dates("2025-01-01", "2025-01-31")
        assert len(trading_dates) > 0
        assert isinstance(trading_dates, pd.DatetimeIndex)

    def test_is_trading_day(self):
        """Test checking if a date is trading day"""
        cal = TradingCalendar("NYSE")
        # 2025-01-02 is a Thursday (trading day)
        assert cal.is_trading_day("2025-01-02") is True
        # 2025-01-04 is a Saturday (non-trading day)
        assert cal.is_trading_day("2025-01-04") is False

    def test_count_trading_days(self):
        """Test counting trading days"""
        cal = TradingCalendar("NYSE")
        count = cal.count_trading_days("2025-01-01", "2025-01-31")
        assert count > 0
        assert isinstance(count, int)

    def test_get_next_trading_date(self):
        """Test getting next trading date"""
        cal = TradingCalendar("NYSE")
        next_date = cal.get_next_trading_date("2025-01-02", n=1)
        assert next_date is not None
        assert isinstance(next_date, pd.Timestamp)

    def test_get_prev_trading_date(self):
        """Test getting previous trading date"""
        cal = TradingCalendar("NYSE")
        prev_date = cal.get_prev_trading_date("2025-01-10", n=1)
        assert prev_date is not None
        assert isinstance(prev_date, pd.Timestamp)

    def test_get_market_open_time(self):
        """Test getting market open time"""
        cal = TradingCalendar("NYSE")
        open_time = cal.get_market_open_time("2025-01-02")
        if open_time is not None:
            assert isinstance(open_time, pd.Timestamp)

    def test_get_market_close_time(self):
        """Test getting market close time"""
        cal = TradingCalendar("NYSE")
        close_time = cal.get_market_close_time("2025-01-02")
        if close_time is not None:
            assert isinstance(close_time, pd.Timestamp)

    def test_get_trading_schedule(self):
        """Test getting full trading schedule"""
        cal = TradingCalendar("NYSE")
        schedule = cal.get_trading_schedule("2025-01-01", "2025-01-31")
        assert isinstance(schedule, pd.DataFrame)
        assert len(schedule) > 0
        assert "market_open" in schedule.columns
        assert "market_close" in schedule.columns

    def test_get_early_closes(self):
        """Test getting early close dates"""
        cal = TradingCalendar("NYSE")
        early_closes = cal.get_early_closes("2025-01-01", "2025-12-31")
        assert isinstance(early_closes, list)


class TestDateConversion:
    """Test date conversion functions"""

    def test_to_datetime_from_str(self):
        """Test converting string to datetime"""
        result = to_datetime("2025-01-15")
        assert isinstance(result, pd.Timestamp)

    def test_to_datetime_from_date(self):
        """Test converting date object to datetime"""
        d = date(2025, 1, 15)
        result = to_datetime(d)
        assert isinstance(result, pd.Timestamp)

    def test_to_date(self):
        """Test converting to date object"""
        result = to_date("2025-01-15")
        assert isinstance(result, date)
        assert result == date(2025, 1, 15)

    def test_to_str_default_format(self):
        """Test converting to string with default format"""
        result = to_str("2025-01-15")
        assert isinstance(result, str)
        assert result == "2025-01-15"

    def test_to_str_custom_format(self):
        """Test converting to string with custom format"""
        result = to_str("2025-01-15", fmt="%d/%m/%Y")
        assert result == "15/01/2025"


class TestDateFrequency:
    """Test date frequency functions"""

    def test_get_date_week(self):
        """Test getting week number"""
        week = get_date_week("2025-01-15")
        assert isinstance(week, int)
        assert 1 <= week <= 53

    def test_get_date_month(self):
        """Test getting month"""
        month = get_date_month("2025-01-15")
        assert month == 1

    def test_get_date_quarter(self):
        """Test getting quarter"""
        q1 = get_date_quarter("2025-01-15")
        assert q1 == 1
        q2 = get_date_quarter("2025-04-15")
        assert q2 == 2
        q3 = get_date_quarter("2025-07-15")
        assert q3 == 3
        q4 = get_date_quarter("2025-10-15")
        assert q4 == 4

    def test_get_date_year(self):
        """Test getting year"""
        year = get_date_year("2025-01-15")
        assert year == 2025

    def test_get_week_boundaries(self):
        """Test getting week start and end"""
        # 2025-01-15 is a Wednesday
        start = get_week_start("2025-01-15")
        end = get_week_end("2025-01-15")
        assert isinstance(start, date)
        assert isinstance(end, date)
        assert (end - start).days == 6

    def test_get_month_boundaries(self):
        """Test getting month start and end"""
        start = get_month_start("2025-01-15")
        end = get_month_end("2025-01-15")
        assert start == date(2025, 1, 1)
        assert end == date(2025, 1, 31)

    def test_get_quarter_boundaries(self):
        """Test getting quarter start and end"""
        start = get_quarter_start("2025-01-15")
        end = get_quarter_end("2025-01-15")
        assert start == date(2025, 1, 1)
        assert end.month > start.month or end.year > start.year

    def test_get_year_boundaries(self):
        """Test getting year start and end"""
        start = get_year_start("2025-06-15")
        end = get_year_end("2025-06-15")
        assert start == date(2025, 1, 1)
        assert end == date(2025, 12, 31)

    def test_generate_date_range(self):
        """Test generating date range"""
        dates = generate_date_range("2025-01-01", "2025-01-10", freq="D")
        assert len(dates) == 10
        assert isinstance(dates, pd.DatetimeIndex)

    def test_generate_date_range_weekly(self):
        """Test generating weekly date range"""
        dates = generate_date_range("2025-01-01", "2025-12-31", freq="W")
        assert len(dates) > 0
        assert isinstance(dates, pd.DatetimeIndex)

    def test_split_by_month(self):
        """Test splitting date range by month"""
        periods = split_by_month("2025-01-01", "2025-03-31")
        assert len(periods) == 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in periods)

    def test_split_by_quarter(self):
        """Test splitting date range by quarter"""
        periods = split_by_quarter("2025-01-01", "2025-12-31")
        assert len(periods) >= 4
        assert all(isinstance(p, tuple) and len(p) == 2 for p in periods)

    def test_split_by_year(self):
        """Test splitting date range by year"""
        periods = split_by_year("2024-01-01", "2026-12-31")
        assert len(periods) >= 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in periods)


class TestTimezoneConversion:
    """Test timezone conversion functions"""

    def test_to_utc_from_naive(self):
        """Test converting naive datetime to UTC"""
        result = to_utc("2025-01-15 12:00:00")
        assert result.tz is not None
        assert str(result.tz) == "UTC"

    def test_to_utc_from_aware(self):
        """Test converting aware datetime to UTC"""
        dt = pd.Timestamp("2025-01-15 12:00:00", tz="US/Eastern")
        result = to_utc(dt)
        assert str(result.tz) == "UTC"

    def test_to_timezone(self):
        """Test converting to specific timezone"""
        result = to_timezone("2025-01-15 12:00:00", "Asia/Shanghai")
        assert result.tz is not None
        assert "Shanghai" in str(result.tz)

    def test_localize(self):
        """Test localizing naive datetime"""
        result = localize("2025-01-15 12:00:00", tz="UTC")
        assert result.tz is not None

    def test_localize_already_aware(self):
        """Test localizing already timezone-aware datetime"""
        dt = pd.Timestamp("2025-01-15 12:00:00", tz="UTC")
        result = localize(dt)
        assert result.tz is not None

    def test_get_timezone_aware_naive(self):
        """Test ensuring naive datetime becomes aware"""
        result = get_timezone_aware("2025-01-15 12:00:00")
        assert result.tz is not None

    def test_get_timezone_aware_already_aware(self):
        """Test ensuring already aware datetime stays aware"""
        dt = pd.Timestamp("2025-01-15 12:00:00", tz="UTC")
        result = get_timezone_aware(dt)
        assert result.tz is not None

    def test_remove_timezone(self):
        """Test removing timezone information"""
        dt = pd.Timestamp("2025-01-15 12:00:00", tz="UTC")
        result = remove_timezone(dt)
        assert result.tz is None

    def test_get_utc_offset(self):
        """Test getting UTC offset"""
        offset = get_utc_offset("2025-01-15 12:00:00", "US/Eastern")
        assert isinstance(offset, str)
        # Should be -05:00 in January (EST)
        assert "-" in offset or "+" in offset

    def test_to_market_timezone(self):
        """Test converting to market timezone"""
        result = to_market_timezone("2025-01-15 12:00:00", "US")
        assert result.tz is not None
        assert "Eastern" in str(result.tz)

    def test_to_market_timezone_china(self):
        """Test converting to China market timezone"""
        result = to_market_timezone("2025-01-15 12:00:00", "CN")
        assert result.tz is not None
        assert "Shanghai" in str(result.tz)

    def test_to_market_timezone_invalid(self):
        """Test converting to invalid market"""
        with pytest.raises(ValueError):
            to_market_timezone("2025-01-15 12:00:00", "INVALID")

    def test_get_market_open_hours_us(self):
        """Test getting US market hours"""
        hours = get_market_open_hours("NYSE")
        assert hours == ("09:30", "16:00")

    def test_get_market_open_hours_japan(self):
        """Test getting Japan market hours"""
        hours = get_market_open_hours("TSE")
        assert hours == ("09:00", "15:00")

    def test_get_market_open_hours_invalid(self):
        """Test getting hours for invalid market"""
        with pytest.raises(ValueError):
            get_market_open_hours("INVALID")


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_workflow_trading_dates_with_timezone(self):
        """Test workflow: get trading dates and convert to market timezone"""
        cal = TradingCalendar("NYSE")
        dates = cal.get_trading_dates("2025-01-01", "2025-01-31")
        
        # Convert first date to market timezone
        first_date = dates[0]
        market_time = to_market_timezone(first_date, "US")
        
        assert market_time is not None
        assert market_time.tz is not None

    def test_workflow_monthly_analysis(self):
        """Test workflow: split into months and analyze each"""
        periods = split_by_month("2025-01-01", "2025-03-31")
        
        for start, end in periods:
            month = get_date_month(start)
            assert 1 <= month <= 12
            assert start < end

    def test_workflow_trading_schedule_with_dates(self):
        """Test workflow: get schedule and extract dates"""
        cal = TradingCalendar("NYSE")
        schedule = cal.get_trading_schedule("2025-01-01", "2025-01-31")
        
        assert len(schedule) > 0
        for idx, row in schedule.iterrows():
            assert row["market_open"] < row["market_close"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
