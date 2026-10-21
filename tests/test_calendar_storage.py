from datetime import date, timedelta

from voice_proxy.calendar_tool import Calendar, _overlaps
from voice_proxy.storage import Storage


def test_overlap_is_exclusive_at_the_boundaries():
    assert _overlaps("15:00", "16:00", "15:30", "16:30")
    assert _overlaps("15:00", "16:00", "15:10", "15:20")  # containment
    assert not _overlaps("15:00", "16:00", "16:00", "17:00")  # back to back
    assert not _overlaps("15:00", "16:00", "13:00", "15:00")


def test_check_reports_conflicts_with_the_conflicting_event(calendar):
    result = calendar.check("2030-01-02", "15:30", "16:30")
    assert result == {"free": False, "conflicts": ["15:00-16:00 Team sync"]}


def test_check_reports_a_free_slot(calendar):
    assert calendar.check("2030-01-02", "17:00", "18:00") == {"free": True, "conflicts": []}


def test_check_without_times_lists_the_days_blocks(calendar):
    result = calendar.check("2030-01-02")
    assert result["busy_blocks"] == ["15:00-16:00 Team sync", "19:00-22:00 Family dinner"]


def test_an_empty_day_is_reported_as_open(calendar):
    assert "open all day" in calendar.check("2031-05-05")["busy_blocks"][0]


def test_load_resolves_offsets_relative_to_today(tmp_path):
    path = tmp_path / "cal.yaml"
    path.write_text(
        'timezone: "UTC"\nevents:\n  - {date_offset_days: 0, start: "09:00", end: "10:00", title: A}\n'
        '  - {date_offset_days: 2, start: "09:00", end: "10:00", title: B}\n',
        encoding="utf-8",
    )
    today = date(2030, 6, 1)
    cal = Calendar.load(path, today=today)
    assert [e["date"] for e in cal.events] == ["2030-06-01", (today + timedelta(days=2)).isoformat()]


def test_storage_round_trip_and_escalation_queue(tmp_path):
    store = Storage(tmp_path / "t.db")
    conv = store.create_conversation("Sam", "friend_close", "chat")
    store.add_message(conv, 1, "assistant", "second")
    store.add_message(conv, 0, "them", "first")
    msg_id = store.add_message(conv, 2, "assistant", "holding reply")
    assert [r["content"] for r in store.get_transcript(conv)] == ["first", "second", "holding reply"]

    assert store.pending_escalations() == []
    store.add_escalation(conv, msg_id, "money_or_pricing", "high", "asked price", "holding reply")
    pending = store.pending_escalations()
    assert len(pending) == 1
    assert pending[0]["status"] == "pending" and pending[0]["message_id"] == msg_id


def test_storage_is_isolated_per_database_file(tmp_path):
    a, b = Storage(tmp_path / "a.db"), Storage(tmp_path / "b.db")
    conv = a.create_conversation("Sam", "friend_close", "chat")
    a.add_message(conv, 0, "them", "hi")
    assert b.get_transcript(conv) == []
