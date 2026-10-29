"""An empty bookmaker answer must keep the last good prices AND their true age."""
import datetime as dt, sqlite3
import odds

def _db(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "DB", str(tmp_path / "o.db")); monkeypatch.setenv("ODDS_API_KEY", "k")
    con = sqlite3.connect(odds.DB); odds.init(con); return con

def test_empty_answer_keeps_prices_and_true_age(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=20)).isoformat()
    con.execute("INSERT INTO book_odds VALUES('g9','p9','pts',24.5,'over',1.9,'sportsbet',?)", (old,)); con.commit()
    start = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5)).isoformat()
    monkeypatch.setattr(odds, "_get", lambda path, **kw: ([{"id": "e1", "home_team": "New York Knicks", "away_team": "Boston Celtics"}], 100) if path.endswith("/events") else ({"bookmakers": []}, 99))
    odds.refresh([dict(game_id="g9", start=start, home="NY", away="BOS")], {"g9": [("p9", "X Player")]})
    assert con.execute("SELECT count(*) FROM book_odds WHERE game_id='g9'").fetchone()[0] == 1       # last good price kept
    assert 19.9 < odds.fetch_ages(con)["g9"] < 20.1                                                    # and still shown as 20 hours old

def test_fresh_prices_show_small_age(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    con.execute("INSERT INTO book_odds VALUES('g1','p1','pts',20.5,'over',1.8,'sportsbet',?)", (now,)); con.commit()
    assert odds.fetch_ages(con)["g1"] < 0.01
