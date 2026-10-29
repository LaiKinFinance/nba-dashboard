"""Shared results must refresh when anything they depend on changes, and prices are re-aged as time passes."""
import sqlite3, time
import pytest
from fastapi.testclient import TestClient
import app, odds

def test_key_changes_when_anything_is_written_to_the_database(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "k.db")); c = sqlite3.connect(app.DB); c.execute("CREATE TABLE t(x)"); c.commit()
    k1 = app._enriched_key(); time.sleep(0.01); c.execute("INSERT INTO t VALUES(1)"); c.commit()
    assert app._enriched_key() != k1

def test_key_changes_with_the_clock_so_price_age_is_re_checked(monkeypatch):
    base = time.time(); monkeypatch.setattr(time, "time", lambda: base); k1 = app._enriched_key()
    monkeypatch.setattr(time, "time", lambda: base + 61); assert app._enriched_key() != k1
    monkeypatch.setattr(time, "time", lambda: base + 9 * 3600); assert app._enriched_key() != k1

def test_key_changes_when_a_new_model_is_saved(tmp_path, monkeypatch):
    import predict
    monkeypatch.setattr(predict, "MODELS", str(tmp_path / "m.pkl")); (tmp_path / "m.pkl").write_bytes(b"a"); k1 = app._enriched_key()
    time.sleep(0.01); (tmp_path / "m.pkl").write_bytes(b"bb"); assert app._enriched_key() != k1

def test_price_older_than_eight_hours_is_stale_whatever_the_page_was_built_with(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "DB", str(tmp_path / "o.db")); con = sqlite3.connect(odds.DB); odds.init(con)
    import datetime as dt
    t = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=7.9)).isoformat()
    con.execute("INSERT INTO book_odds VALUES('g','p','pts',20.5,'over',1.9,'sportsbet',?)", (t,)); con.commit()
    assert not app.is_stale(odds.fetch_ages(con)["g"]) and odds.fetch_times(con)["g"] == t
    import types; real = dt.datetime
    class Later(real):
        @classmethod
        def now(cls, tz=None): return real.now(tz) + dt.timedelta(hours=1)
    monkeypatch.setattr(odds, "dt", types.SimpleNamespace(datetime=Later, timezone=dt.timezone, timedelta=dt.timedelta))
    assert app.is_stale(odds.fetch_ages(con)["g"])               # one hour later the same stored price counts as old

def test_api_answers_are_never_cached_by_the_browser(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    assert TestClient(app.app).get("/api/health").headers["cache-control"] == "no-store"

def test_page_marks_old_prices_by_their_time_stamp_not_a_saved_flag():
    html = open("static/index.html").read()
    assert "oldAt(p.odds_at)" in html and "oldAt(((gameIdx[g]||{}).match||{}).prices_at)" in html and "setInterval(" in html
