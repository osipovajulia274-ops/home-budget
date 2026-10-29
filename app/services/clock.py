"""One place that knows what "today" is (tests replace `today`)."""
from __future__ import annotations

import datetime as dt


def today(tz_name: str = "Europe/Moscow") -> dt.date:
    try:
        from zoneinfo import ZoneInfo

        return dt.datetime.now(ZoneInfo(tz_name)).date()
    except Exception:  # no tzdata on the machine
        return dt.datetime.now(dt.timezone.utc).date()
