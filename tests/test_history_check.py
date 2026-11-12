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

def test_download_budget_is_option_b():
    assert history.HISTORY_MONTHLY_BUDGET == 5000 and history.DOWNLOAD_MARKETS == "player_points_rebounds_assists"


# ---------- step 2: monthly download ----------
def _season_db(balance, days=3, per_day=2):
    c = _db(balance)
    teams = [("NY", "BOS", "New York Knicks", "Boston Celtics"), ("LAL", "GS", "Los Angeles Lakers", "Golden State Warriors")]
    for d in range(days):
        for i in range(per_day):
            h, a = teams[i][:2]
            c.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)", (f"g{d}{i}", f"2026-02-{10+d:02d}", "2026", h, a, 100, 90, None))
    c.commit(); return c

def _fake_api(log, bal={"v": 20000}, fail_odds=False, cost=10):
    def fake(path, **kw):
        log.append((path, kw.get("date"), kw.get("markets")))
        if path.endswith("/events"):
            day = kw["date"][:10]; bal["v"] -= 1
            return {"data": [dict(id=f"e{day}NY", home_team="New York Knicks", away_team="Boston Celtics", commence_time=f"{day}T23:30:00Z"),
                             dict(id=f"e{day}LAL", home_team="Los Angeles Lakers", away_team="Golden State Warriors", commence_time=f"{day}T23:30:00Z")]}, dict(remaining=str(bal["v"]), last="1")
        if fail_odds: raise RuntimeError("Odds API request failed (HTTP 500): HTTPError")
        bal["v"] -= cost
        return {"timestamp": kw["date"], "data": {"bookmakers": [{"key": "sportsbet", "markets": [{"key": "player_points_rebounds_assists", "outcomes": [
            {"name": "Over", "description": "Jalen Brunson", "point": 39.5, "price": 1.87}, {"name": "Under", "description": "Jalen Brunson", "point": 39.5, "price": 1.93}]}]}]}}, dict(remaining=str(bal["v"]), last=str(cost))
    return fake

def test_one_day_per_step_pra_only_an_hour_before_tip_and_never_twice(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k"); log = []; monkeypatch.setattr(history, "_get", _fake_api(log, {"v": 20000}))
    c = _season_db("20000", days=2)                                 # 3 past game days: 1 + 2 + 2 games
    out = history.download_step(c)
    assert out["ran"] and len(log) == 1 + out["games"]               # one day per step: its game list, then each game
    assert all(m == "player_points_rebounds_assists" for p, d, m in log if p.endswith("/odds")) and log[1][1].endswith("T22:30:00Z")
    for _ in range(5): history.download_step(c)
    games = [p for p, d, m in log if p.endswith("/odds")]
    assert len(games) == 5 and len(set(games)) == 5                # every game bought exactly once
    assert history.download_step(c)["reason"] == "all past game days done"
    assert c.execute("SELECT v FROM odds_status WHERE k='remaining'").fetchone()[0] == "19947"     # the app's credit count kept up to date
    s = history.summary(c); assert s["games_stored"] == 5 and s["spent_this_month"] == 53 and s["prices_stored"] == 10

def test_stops_at_the_monthly_budget(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k"); log = []; monkeypatch.setattr(history, "_get", _fake_api(log, {"v": 20000}))
    c = _season_db("20000", days=5)
    for _ in range(10): history.download_step(c, budget=30)
    assert history.spent(c) <= 30 and len([1 for p, *_ in log if p.endswith("/odds")]) == 2

def test_never_goes_below_3000_credits_or_runs_on_an_unknown_balance(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k"); monkeypatch.setattr(history, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")))
    for bal in ("3010", "abc", "inf", None):
        c = _season_db(bal); assert history.download_step(c)["ran"] is False
    log = []; monkeypatch.setattr(history, "_get", _fake_api(log, {"v": 3015}))
    c = _season_db("3015", days=1); history.download_step(c)
    assert len([1 for p, *_ in log if p.endswith("/odds")]) == 1     # 3015 -> 3014 after the list -> 3004 after one game; a second would go below 3000
    assert int(c.execute("SELECT v FROM odds_status WHERE k='remaining'").fetchone()[0]) >= 3000

def test_a_failed_request_is_recorded_and_not_repeated_in_a_loop(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k"); log = []; monkeypatch.setattr(history, "_get", _fake_api(log, {"v": 20000}, fail_odds=True))
    c = _season_db("20000", days=1)
    out = history.download_step(c)
    assert len([1 for p, *_ in log if p.endswith("/odds")]) == 1 and history.summary(c)["failed"] == 1
    for _ in range(5): history.download_step(c)
    assert len([1 for p, *_ in log if p.endswith("/events")]) <= 3     # at most 3 tries per day, then it moves on

def test_off_when_budget_is_zero(monkeypatch):
    monkeypatch.setattr(history, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")))
    assert history.download_step(_season_db("20000"), budget=0)["ran"] is False


def test_november_has_the_approved_extra_5000_and_other_months_do_not():
    assert history.month_budget("2026-11") == 10000 and history.month_budget("2026-12") == 5000
