"""Price-fetch trouble must show up as trouble, and the health page must answer even when storage is unreadable."""
import datetime as dt, os, sqlite3
import pytest
from fastapi.testclient import TestClient
import app, odds

@pytest.fixture
def blank(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "x.db")); monkeypatch.setattr(odds, "DB", str(tmp_path / "x.db"))
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60)); monkeypatch.setenv("ODDS_API_KEY", "k")
    return tmp_path

def _fixture_game():
    start = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5)).isoformat()
    return [dict(game_id="g1", start=start, home="NY", away="BOS")], {"g1": [("p1", "Some Player")]}

def test_failed_player_price_call_is_partial_not_ok(blank, monkeypatch):
    monkeypatch.setattr(app, "projections_all", lambda: [])
    monkeypatch.setattr(odds, "refresh", lambda g, p: dict(ok=False, msg="events call failed"))
    monkeypatch.setattr(odds, "refresh_lines", lambda g: dict(ok=True))
    assert app._stage("odds", app.refresh_odds) == "partial"
    assert TestClient(app.app).get("/api/health").json()["stages"]["odds"]["note"].startswith("player prices")

def test_failed_team_price_call_is_partial_not_ok(blank, monkeypatch):
    monkeypatch.setattr(app, "projections_all", lambda: [])
    monkeypatch.setattr(odds, "refresh", lambda g, p: dict(ok=True)); monkeypatch.setattr(odds, "refresh_lines", lambda g: dict(ok=False, msg="HTTP 500"))
    assert app._stage("odds", app.refresh_odds) == "partial"

def test_per_game_error_is_kept_and_last_ok_does_not_advance(blank, monkeypatch):
    con = sqlite3.connect(odds.DB); odds.init(con)
    games, pbg = _fixture_game()
    def fake(path, **kw):
        if path.endswith("/events"): return [{"id": "e1", "home_team": "New York Knicks", "away_team": "Boston Celtics"}], 100
        raise RuntimeError("Odds API request failed (HTTP 500): HTTPError")
    monkeypatch.setattr(odds, "_get", fake)
    r = odds.refresh(games, pbg); st = odds.status(con)
    assert r["ok"] is False and r["errors"] == 1 and "HTTP 500" in st["error"] and "last_ok" not in st

def test_clean_run_clears_error_and_sets_last_ok(blank, monkeypatch):
    con = sqlite3.connect(odds.DB); odds.init(con); con.execute("INSERT INTO odds_status VALUES('error','old trouble')"); con.commit()
    games, pbg = _fixture_game()
    monkeypatch.setattr(odds, "_get", lambda path, **kw: ([{"id": "e1", "home_team": "New York Knicks", "away_team": "Boston Celtics"}], 100) if path.endswith("/events") else ({"bookmakers": []}, 99))
    assert odds.refresh(games, pbg)["ok"] is True
    st = odds.status(con); assert st["error"] == "" and "last_ok" in st

def test_health_answers_when_database_cannot_be_opened(blank, monkeypatch):
    monkeypatch.setattr(app, "DB", str(blank))          # a folder is not a database
    r = TestClient(app.app).get("/api/health")
    assert r.status_code == 503 and r.json()["overall"] == "problem" and "cannot open the database" in r.json()["problems"][0]

def test_health_answers_when_backup_folder_cannot_be_read(blank, monkeypatch):
    sqlite3.connect(app.DB).close(); os.makedirs(blank / "backups")
    real = os.listdir
    monkeypatch.setattr(os, "listdir", lambda p: (_ for _ in ()).throw(PermissionError("denied")) if str(p).endswith("backups") else real(p))
    r = TestClient(app.app).get("/api/health")
    assert r.status_code == 200 and any("backup folder" in p for p in r.json()["problems"])
