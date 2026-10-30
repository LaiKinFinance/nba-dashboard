"""A game that has just finished is stored and its picks are marked within minutes, without waiting for the 4-hour catch-up."""
import sqlite3, json
import pytest
import app, live, ingest

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "f.db")); monkeypatch.setattr(ingest, "DB", str(tmp_path / "f.db"))
    app.init(); c = sqlite3.connect(app.DB); c.executescript(ingest.SCHEMA); ingest.ensure_columns(c); c.commit(); return c

def _board(*pairs): return lambda: [dict(game_id=g, state=s) for g, s in pairs]

def test_only_final_games_that_are_not_stored_are_picked_up(db, monkeypatch):
    db.execute("INSERT INTO games(game_id,date,home,away,home_pts,away_pts) VALUES('stored','2026-10-29','A','B',100,90)")
    db.execute("INSERT INTO unavailable_games(game_id,date,home,away,home_pts,away_pts,reason,first_seen,last_checked) VALUES('nobox','2026-10-29','A','B',1,2,'r','t','t')"); db.commit()
    monkeypatch.setattr(live, "scoreboard", _board(("stored", "post"), ("nobox", "post"), ("new", "post"), ("live", "in"), ("later", "pre")))
    assert app.finished_not_stored() == ["new"]

def test_nothing_happens_when_nothing_has_finished(db, monkeypatch):
    monkeypatch.setattr(live, "scoreboard", _board(("live", "in"))); called = []
    monkeypatch.setattr(ingest, "run", lambda *a, **k: called.append(1))
    assert app.fast_results() == dict(skipped=True) and called == []

def test_a_finished_game_is_fetched_and_graded(db, monkeypatch):
    monkeypatch.setattr(live, "scoreboard", _board(("new", "post"))); calls = []
    monkeypatch.setattr(ingest, "run", lambda *a, **k: calls.append("ingest")); monkeypatch.setattr(app, "grade_picks", lambda: calls.append("picks")); monkeypatch.setattr(app, "grade_bets", lambda: calls.append("bets"))
    assert app.fast_results() == dict(games=1) and calls == ["ingest", "picks", "bets"]
    assert app.REFRESH_LOCK.acquire(blocking=False); app.REFRESH_LOCK.release()          # the lock is let go again

def test_it_steps_aside_while_the_big_catch_up_runs(db, monkeypatch):
    monkeypatch.setattr(live, "scoreboard", _board(("new", "post"))); called = []
    monkeypatch.setattr(ingest, "run", lambda *a, **k: called.append(1))
    with app.REFRESH_LOCK: assert app.fast_results() == dict(skipped=True)
    assert called == []

def test_a_failed_scoreboard_does_nothing(db, monkeypatch):
    monkeypatch.setattr(live, "scoreboard", lambda: (_ for _ in ()).throw(OSError("down")))
    assert app.finished_not_stored() == []


def test_who_started_lists_the_five_starters_of_recent_finished_games(db):
    from fastapi.testclient import TestClient
    import datetime as dt
    app.ACCESS_CODE = ""
    now = dt.datetime.now(dt.timezone.utc); recent = (now - dt.timedelta(hours=20)).strftime("%Y-%m-%dT%H:%MZ"); old = (now - dt.timedelta(hours=120)).strftime("%Y-%m-%dT%H:%MZ")
    db.execute("INSERT INTO games(game_id,date,tip_utc,home,away,home_pts,away_pts) VALUES('new','2026-10-29',?,'ATL','CLE',114,111)", (recent,))
    db.execute("INSERT INTO games(game_id,date,tip_utc,home,away,home_pts,away_pts) VALUES('old','2026-10-20',?,'ATL','CLE',100,90)", (old,))
    for i in range(7): db.execute("INSERT INTO player_games(game_id,player_id,player,team,starter,min) VALUES('new',?,?,?,?,?)", (f"a{i}", f"Atl {i}", "ATL", 1 if i < 5 else 0, 30 - i))
    for i in range(5): db.execute("INSERT INTO player_games(game_id,player_id,player,team,starter,min) VALUES('new',?,?,?,1,?)", (f"c{i}", f"Cle {i}", "CLE", 30 - i))
    db.commit()
    out = TestClient(app.app).get("/api/started").json()
    assert [g["game_id"] for g in out] == ["new"]
    g = out[0]; assert g["home"]["score"] == 114 and g["away"]["score"] == 111
    assert [p["name"] for p in g["home"]["starters"]] == [f"Atl {i}" for i in range(5)] and len(g["away"]["starters"]) == 5
