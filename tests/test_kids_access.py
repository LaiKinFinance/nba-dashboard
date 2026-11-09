"""The kids' code: scores, lineups, box scores and team/player stats only. Every other page is refused by the website itself."""
import pytest
from fastapi.testclient import TestClient
import app, live

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "t.db")); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "ACCESS_CODE", "FULL"); monkeypatch.setattr(app, "KIDS_CODE", "KIDS"); app.init()
    import sqlite3; c = sqlite3.connect(app.DB)
    c.executescript("CREATE TABLE IF NOT EXISTS games(game_id TEXT, date TEXT); CREATE TABLE IF NOT EXISTS projections(game_id TEXT, made_at TEXT, payload TEXT);"
                    "CREATE TABLE IF NOT EXISTS player_games(player_id TEXT, player TEXT, team TEXT, date TEXT)"); c.commit()
    monkeypatch.setattr(app, "model_status", lambda b: (None, None))
    monkeypatch.setattr(live, "scoreboard", lambda: [dict(game_id="g1", state="in")])
    monkeypatch.setattr(live, "boxscore", lambda gid: dict(game_id=gid, teams=[]))
    monkeypatch.setattr(app, "projections_all", lambda: [dict(game_id="g1", player_id="p1", proj=dict(pts=20, reb=5, ast=5))])
    return TestClient(app.app)

K = {"X-Code": "KIDS", "X-User": "kid"}; F = {"X-Code": "FULL", "X-User": "mark"}

def test_kids_see_scores_and_box_scores_without_picks(client):
    assert client.get("/api/meta", headers=K).json() == {"role": "kids"}
    assert client.get("/api/live", headers=K).json()[0]["game_id"] == "g1"
    assert client.get("/api/live/g1", headers=K).json()["tracked"] == {}                 # no pick stars or projections
    assert client.get("/api/live/g1", headers=F).json()["tracked"]["p1"]["pts"] == 20     # the full app still gets them
    assert client.get("/api/players?q=ab", headers=K).status_code == 200

@pytest.mark.parametrize("path", ["/api/best", "/api/value", "/api/calls", "/api/track", "/api/slate", "/api/picks", "/api/leaderboard", "/api/scorecard", "/api/status", "/api/backup", "/api/refresh/status"])
def test_kids_are_refused_everything_with_odds_or_picks(client, path):
    assert client.get(path, headers=K).status_code in ((404,) if path == "/api/backup" else (401,))     # the backup needs its own separate code

def test_kids_cannot_save_or_post(client):
    assert client.post("/api/parlay", headers=K, json={"legs": []}).status_code in (401, 422)
    assert client.post("/api/picks", headers=K, json={"legs": []}).status_code in (401, 422)
    assert client.post("/api/bets", headers=K, json={"legs": []}).status_code in (401, 422)

def test_wrong_code_still_refused_and_full_role_reported(client):
    assert client.get("/api/live", headers={"X-Code": "nope"}).status_code == 401
    assert client.get("/api/meta", headers=F).json()["role"] == "full"

def test_kids_code_same_as_main_code_gives_no_kids_mode(client, monkeypatch):
    monkeypatch.setattr(app, "KIDS_CODE", "FULL")
    assert client.get("/api/meta", headers=F).json()["role"] == "full"
