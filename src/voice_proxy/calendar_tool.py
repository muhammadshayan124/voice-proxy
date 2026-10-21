"""Mock calendar data for Phase 1 testing.

This exists so the escalation policy's "scheduling_open_slot" rule has real
data to check against instead of the model always having to guess and
escalate as low_confidence. Swap for a real calendar integration later —
the `check(date, start_time, end_time)` contract shouldn't need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_CALENDAR_PATH = Path(__file__).resolve().parents[2] / "config" / "calendar.yaml"


def _to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    return _to_minutes(a_start) < _to_minutes(b_end) and _to_minutes(b_start) < _to_minutes(a_end)


@dataclass
class Calendar:
    timezone: str
    events: list[dict]  # each: {date: "YYYY-MM-DD", start, end, title}

    @classmethod
    def load(cls, path: Path = DEFAULT_CALENDAR_PATH, today: Optional[date] = None) -> "Calendar":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        today = today or date.today()
        events = [
            {
                "date": (today + timedelta(days=e["date_offset_days"])).isoformat(),
                "start": e["start"],
                "end": e["end"],
                "title": e["title"],
            }
            for e in raw["events"]
        ]
        return cls(timezone=raw["timezone"], events=events)

    def today_context(self) -> str:
        today = date.today()
        return f"Today's date is {today.isoformat()} ({today.strftime('%A')}), timezone {self.timezone}."

    def check(self, date: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> dict:
        day_events = [e for e in self.events if e["date"] == date]

        if start_time and end_time:
            conflicts = [e for e in day_events if _overlaps(start_time, end_time, e["start"], e["end"])]
            return {
                "free": len(conflicts) == 0,
                "conflicts": [f'{e["start"]}-{e["end"]} {e["title"]}' for e in conflicts],
            }

        return {
            "busy_blocks": [f'{e["start"]}-{e["end"]} {e["title"]}' for e in day_events]
            or ["nothing scheduled — looks open all day"]
        }
