"""Time tools tests."""

from __future__ import annotations

from tools import time_tools


def test_time_range_to_unix_utc8_matches_expected_epoch() -> None:
    start_ts, end_ts = time_tools.time_range_to_unix_utc8(
        date_value="2021-03-04",
        start_time="18:30:00",
        end_time="19:00:00",
    )
    assert start_ts == 1614853800
    assert end_ts == 1614855600


def test_normalize_unix_like_to_utc8_supports_seconds_of_day() -> None:
    assert time_tools.normalize_unix_like_to_utc8(66600, "2021-03-04") == 1614853800
