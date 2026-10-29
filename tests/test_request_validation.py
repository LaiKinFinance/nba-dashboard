"""Bad bet requests are refused before any calculation (they used to give wrong answers or HTTP 500)."""
import pytest
from fastapi.testclient import TestClient
import app

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    return TestClient(app.app, raise_server_exceptions=False)

BAD = {
    "double-double with side=home": dict(game_id="g", player_id="p", stat="dd", side="home", line=0),
    "player market without player": dict(game_id="g", stat="pts", line=20.5, side="over"),
    "team total without team": dict(game_id="g", stat="tt", line=110.5, side="over"),
    "winner with over/under side": dict(game_id="g", stat="ml", side="over", line=0),
    "band with upper below lower": dict(game_id="g", stat="band", team="home", lo=10, hi=5, side="home", line=0),
    "band without team": dict(game_id="g", stat="band", lo=1, side="home", line=0),
}

@pytest.mark.parametrize("name", list(BAD))
def test_bad_leg_is_refused_with_422(client, name):
    assert client.post("/api/parlay", json=dict(legs=[BAD[name]])).status_code == 422
