"""Pre-game price refreshes (about 2 hours and 30 minutes before tip-off): once each, only with a known balance above the brake,
balance checked before every paid request, attempts saved before sending, and missed windows skipped. The 8pm/2am/8am fetches are separate."""
import datetime as dt, json, sqlite3
from contextlib import closing
import app, odds

NOW = dt.datetime(2026, 11, 5, 0, 0, tzinfo=dt.timezone.utc)
at = lambda mins: (NOW + dt.timedelta(minutes=mins)).strftime("%Y-%m-%dT%H:%MZ")
GAMES = [dict(game_id="g2h", start=at(110)), dict(game_id="g30", start=at(28)), dict(game_id="later", start=at(300)), dict(game_id="soon", start=at(5))]

def test_each_game_gets_its_two_pregame_refreshes_once():
    assert sorted(app.pregame_due(GAMES, NOW, set(), "19000")) == [("2h", "g2h"), ("30m", "g30")]
    assert app.pregame_due(GAMES, NOW, {"pre 2h g2h", "pre 30m g30"}, "19000") == []

def test_missing_invalid_or_low_credits_mean_no_extra_fetches():
    for bad in (None, "", "abc", "nan", "-5", "900"):
        assert app.pregame_due(GAMES, NOW, set(), bad) == []

def test_downtime_that_misses_a_window_is_not_caught_up_late():
    g = [dict(game_id="x", start=at(60))]                 # back up 60 min before tip: the 2-hour window has passed, the 30-minute one has not started
    assert app.pregame_due(g, NOW, set(), "19000") == []
    g = [dict(game_id="x", start=at(10))]                 # back up 10 min before tip: both windows missed
    assert app.pregame_due(g, NOW, set(), "19000") == []

def _setup(monkeypatch, balance, games):
    with closing(app.con()) as c:
        odds.init(c); c.execute("CREATE TABLE IF NOT EXISTS odds_slots(k TEXT PRIMARY KEY, v TEXT)"); c.execute("DELETE FROM odds_slots WHERE k LIKE 'pre %' OR k LIKE 'log %'")
        c.execute("INSERT OR REPLACE INTO odds_status VALUES('remaining', ?)", (str(balance),)); c.commit()
    monkeypatch.setattr(app, "projections_all", lambda: [dict(game_id=g["game_id"], start=g["start"]) for g in games])
    monkeypatch.setattr(app, "_stage", lambda name, fn: fn())

def test_two_games_crossing_the_credit_limit_only_the_first_is_fetched(monkeypatch):
    games = [dict(game_id="a", start=at(110)), dict(game_id="b", start=at(115))]
    _setup(monkeypatch, 1505, games); sent = []
    def fake(**k):
        sent.append(k["only"])
        with closing(app.con()) as c: c.execute("UPDATE odds_status SET v='1496' WHERE k='remaining'"); c.commit()    # this fetch cost 9 credits
        return dict(props=dict(ok=True, fetched=1, remaining="1496"))
    monkeypatch.setattr(app, "refresh_odds", fake)
    app._pregame_refresh(NOW)
    assert len(sent) == 1                                  # the balance was checked again before the second paid request
    with closing(app.con()) as c: assert json.loads(c.execute("SELECT v FROM odds_slots WHERE k LIKE 'pre 2h %' AND v LIKE '%ok%'").fetchone()[0])["state"] == "ok"

def test_a_restart_during_a_fetch_never_sends_it_again(monkeypatch):
    games = [dict(game_id="r", start=at(110))]
    _setup(monkeypatch, 19000, games)
    def crash(**k): raise KeyboardInterrupt("server restarted mid-fetch")
    monkeypatch.setattr(app, "refresh_odds", crash)
    try: app._pregame_refresh(NOW)
    except KeyboardInterrupt: pass
    with closing(app.con()) as c: assert json.loads(c.execute("SELECT v FROM odds_slots WHERE k='pre 2h r'").fetchone()[0])["state"] == "sent"   # saved before sending
    again = []; monkeypatch.setattr(app, "refresh_odds", lambda **k: again.append(1) or {})
    app._pregame_refresh(NOW + dt.timedelta(minutes=5))
    assert again == []

def test_team_price_wait_applies_after_a_failure_too(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k"); calls = []
    with closing(sqlite3.connect(odds.DB)) as c:
        odds.init(c); c.execute("DELETE FROM odds_status WHERE k IN ('lines_at','lines_try')"); c.execute("INSERT OR REPLACE INTO odds_status VALUES('remaining','19000')"); c.commit()
    def boom(path, **kw): calls.append(path); raise RuntimeError("HTTP 500")
    monkeypatch.setattr(odds, "_get", boom)
    assert odds.refresh_lines([], refetch_h=0.4, min_credits=1500, attempt_gap_h=0.4)["ok"] is False
    assert odds.refresh_lines([], refetch_h=0.4, min_credits=1500, attempt_gap_h=0.4).get("skipped") and len(calls) == 1      # no retry within 24 minutes

# ---- the real fetch code (app.refresh_odds -> odds.refresh / odds.refresh_lines) with fake provider replies ----
import math, pytest

def _real(monkeypatch, events_balance, odds_reply=None):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    start = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=110)).strftime("%Y-%m-%dT%H:%MZ")
    with closing(app.con()) as c:
        odds.init(c); c.execute("CREATE TABLE IF NOT EXISTS odds_slots(k TEXT PRIMARY KEY, v TEXT)"); c.execute("DELETE FROM odds_slots WHERE k LIKE 'pre %' OR k LIKE 'log %'")
        c.execute("DELETE FROM odds_fetch"); c.execute("DELETE FROM odds_status WHERE k IN ('lines_at','lines_try')")
        c.execute("INSERT OR REPLACE INTO odds_status VALUES('remaining','19000')"); c.commit()
    monkeypatch.setattr(app, "projections_all", lambda: [dict(game_id="rg", start=start, team="NY", opp="BOS", home=1, player_id="p1", player="Some Player")])
    paid = []
    def fake_get(path, **kw):
        if path.endswith("/events"):
            return [dict(id="e1", home_team="New York Knicks", away_team="Boston Celtics", commence_time=start)], events_balance
        paid.append(path)
        if odds_reply == "error": raise RuntimeError("Odds API request failed (HTTP 500): HTTPError")
        return ({"bookmakers": []} if path.endswith("/odds") and "/events/" in path else []), "18990"
    monkeypatch.setattr(odds, "_get", fake_get)
    app._pregame_refresh(dt.datetime.now(dt.timezone.utc))
    with closing(app.con()) as c:
        row = c.execute("SELECT v FROM odds_slots WHERE k='pre 2h rg'").fetchone()
    return paid, (json.loads(row[0])["state"] if row else None)

def test_real_fetch_with_a_good_balance_is_sent_and_saved_ok(monkeypatch):
    paid, state = _real(monkeypatch, "19000")
    assert any("/events/e1/odds" in p for p in paid) and state == "ok"

@pytest.mark.parametrize("bal", [None, "", "abc", "inf", "-inf", "nan", "1499"])
def test_real_fetch_skips_the_paid_request_when_the_latest_balance_is_bad(monkeypatch, bal):
    paid, state = _real(monkeypatch, bal)
    assert not any("/events/e1/odds" in p for p in paid) and state == "failed"      # the reply with no (or a bad/low) balance: nothing paid, saved as failed

def test_real_fetch_that_fails_is_saved_as_failed(monkeypatch):
    paid, state = _real(monkeypatch, "19000", odds_reply="error")
    assert state == "failed"

def test_inf_is_never_a_usable_balance():
    assert app._credits("inf") is None and app._credits(float("inf")) is None and app._credits("19000") == 19000.0
    assert app.pregame_due(GAMES, NOW, set(), "inf") == []
