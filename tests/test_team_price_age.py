"""Team bets (winner, handicap, total) get the same old-price protection as player bets."""
import datetime as dt, sqlite3
import pytest
import app, odds

GP = {"g1": dict(source="market", home="NY", away="BOS", p_home=0.70, p_away=0.30, start="2999-01-01T00:00Z", home_pts=112, away_pts=106,
                 margin=5.0, total=218.0, sigma_margin=13., sigma_total=17., handicaps=[], totals=[], team_totals={}, bands=[])}
LINES = {"g1": {"h2h": {"home": (2.0, "sportsbet"), "home_best": (2.0, "sportsbet")}}}

def _value(monkeypatch, age):
    monkeypatch.setattr(app, "enriched", lambda: []); monkeypatch.setattr(app, "game_preds", lambda: GP)
    monkeypatch.setattr(app, "model_status", lambda preds=None: (object(), None))
    monkeypatch.setattr(odds, "load_lines", lambda c: LINES); monkeypatch.setattr(odds, "line_ages", lambda c: {"g1": age})
    return app.compute_value()["picks"]

def test_fresh_team_price_can_be_value(monkeypatch):
    assert [p["stat"] for p in _value(monkeypatch, 1.0)] == ["ml"]

def test_48_hour_old_team_price_is_never_value(monkeypatch):
    assert _value(monkeypatch, 48.0) == []

def test_slate_marks_old_team_prices(monkeypatch):
    gp = GP["g1"]
    assert app.match_payload(gp, {}, 48.0)["stale_prices"] is True and app.match_payload(gp, {}, 48.0)["price_age_h"] == 48.0
    assert app.match_payload(gp, {}, 1.0)["stale_prices"] is False

def test_empty_lines_answer_keeps_last_good_prices_and_true_age(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "DB", str(tmp_path / "l.db")); monkeypatch.setenv("ODDS_API_KEY", "k")
    con = sqlite3.connect(odds.DB); odds.init(con)
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)).isoformat()
    con.execute("INSERT INTO book_lines VALUES('g1','h2h','home',0.0,2.0,'sportsbet',?)", (old,)); con.commit()
    monkeypatch.setattr(odds, "_get", lambda path, **kw: ([{"home_team": "New York Knicks", "away_team": "Boston Celtics", "bookmakers": []}], 100))
    odds.refresh_lines([dict(game_id="g1", start="2999-01-01T00:00Z", home="NY", away="BOS")])
    assert con.execute("SELECT count(*) FROM book_lines").fetchone()[0] == 1
    assert 47.9 < odds.line_ages(con)["g1"] < 48.1

def test_old_player_price_is_not_value():
    assert app.is_stale(9.0) and not app.is_stale(2.0) and not app.is_stale(None)
