"""Past prices, step 1: ONE past game is fetched once, only with a known 3,000+ balance; the cost and what came back are recorded."""
import json, sqlite3
from contextlib import closing
import app, history, odds

def _db(balance):
    c = sqlite3.connect(":memory:"); odds.init(c)
    c.execute("CREATE TABLE games(game_id TEXT, date TEXT, season TEXT, home TEXT, away TEXT, home_pts INT, away_pts INT, tip_utc TEXT)")
    c.executemany("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)", [("old1", "2026-03-12", "2026", "NY", "BOS", 110, 100, "2026-03-13T00:00Z"),
                                                              ("new1", "2026-11-01", "2027", "NY", "BOS", 100, 99, "2026-11-02T00:00Z")])
    c.execute("INSERT OR REPLACE INTO odds_status VALUES('remaining', ?)", (balance,)); c.commit(); return c

def test_one_game_cost_check_runs_once_and_records_the_cost(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k"); calls = []
    def fake(path, **kw):
        calls.append(path)
        if path.endswith("/events"):
            return {"data": [dict(id="h1", home_team="New York Knicks", away_team="Boston Celtics", commence_time="2026-03-13T00:00:00Z")]}, dict(remaining="19990", last="1")
        return {"timestamp": "2026-03-12T23:00:00Z", "data": {"bookmakers": [{"key": "sportsbet", "markets": [{"key": "player_points", "outcomes": [
            {"name": "Over", "description": "Jalen Brunson", "point": 26.5, "price": 1.87}, {"name": "Under", "description": "Jalen Brunson", "point": 26.5, "price": 1.93}]}]}]}}, dict(remaining="19950", last="40")
    monkeypatch.setattr(history, "_get", fake)
    c = _db("20000"); out = history.cost_check(c)
    assert out["state"] == "done" and out["odds_cost"] == "40" and out["prices"] == 2 and out["game"] == "BOS at NY"
    assert history.cost_check(c) == out and len(calls) == 2                       # never repeated
    assert c.execute("SELECT count(*) FROM hist_odds").fetchone()[0] == 2

def test_no_check_without_a_known_balance_of_3000(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k"); monkeypatch.setattr(history, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")))
    for bal in ("2999", "abc", "inf"):
        assert history.cost_check(_db(bal))["ran"] is False

def test_download_is_off_until_approved():
    assert history.HISTORY_MONTHLY_BUDGET == 0
