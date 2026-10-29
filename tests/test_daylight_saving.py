"""US Eastern game days around the clock changes (production already uses America/New_York; these guard the edges)."""
import pytest
import ingest, repair, app

# (tip-off in UTC, expected Eastern date)
CASES = [
    ("2026-03-09T03:30Z", "2026-03-08"),   # still evening of Mar 8 in EDT; a fixed UTC-5 would also say Mar 8
    ("2026-03-09T04:30Z", "2026-03-09"),   # 00:30 EDT on Mar 9; a fixed UTC-5 would wrongly say Mar 8
    ("2026-03-08T06:59Z", "2026-03-08"),   # 01:59 EST just before the clocks go forward
    ("2026-03-08T07:00Z", "2026-03-08"),   # 03:00 EDT just after
    ("2026-11-01T05:30Z", "2026-11-01"),   # 01:30 EDT (first one)
    ("2026-11-02T04:30Z", "2026-11-01"),   # 23:30 EST on Nov 1; a fixed UTC-4 would wrongly say Nov 2
    ("2026-11-02T05:00Z", "2026-11-02"),   # midnight EST
    ("2026-01-15T00:30Z", "2026-01-14"),   # ordinary winter evening game
    ("2026-07-15T00:30Z", "2026-07-14"),   # ordinary summer evening game
]

@pytest.mark.parametrize("tip,day", CASES)
def test_ingest_and_repair_agree_with_eastern_time(tip, day):
    assert ingest.et_date(tip) == day and repair.et_date(tip) == day and app.et_day(tip) == day
