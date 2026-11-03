"""Bookmaker prices are fetched at 8pm, 2am and 8am Melbourne time, each once."""
import datetime as dt
from zoneinfo import ZoneInfo
import app

MEL = ZoneInfo("Australia/Melbourne")
at = lambda y, m, d, h, mi=0: dt.datetime(y, m, d, h, mi, tzinfo=MEL).astimezone(dt.timezone.utc)

def test_each_fetch_time_runs_once():
    assert app.odds_slot_due(at(2026, 11, 3, 20, 5), None) == "2026-11-03 20:00 Melbourne"
    assert app.odds_slot_due(at(2026, 11, 3, 23, 0), "2026-11-03 20:00 Melbourne") is None          # already done
    assert app.odds_slot_due(at(2026, 11, 4, 2, 1), "2026-11-03 20:00 Melbourne") == "2026-11-04 02:00 Melbourne"
    assert app.odds_slot_due(at(2026, 11, 4, 8, 30), "2026-11-04 02:00 Melbourne") == "2026-11-04 08:00 Melbourne"
    assert app.odds_slot_due(at(2026, 11, 4, 12, 0), "2026-11-04 08:00 Melbourne") is None          # nothing between 8am and 8pm

def test_a_missed_time_is_caught_up_after_a_restart():
    assert app.odds_slot_due(at(2026, 11, 3, 22, 40), "2026-11-03 08:00 Melbourne") == "2026-11-03 20:00 Melbourne"
