"""The past-prices study export: only reachable with the private key (wrong or missing key looks like a missing page)."""
import sqlite3, pytest
from fastapi.testclient import TestClient
import app

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "t.db")); monkeypatch.setattr(app, "RATE", (10**6, 60)); app.init()
    import history; c = sqlite3.connect(app.DB); history.init(c)
    c.execute("INSERT INTO hist_odds VALUES('g1','2026-03-10T22:00:00Z','Jalen Brunson','pra',39.5,'over',1.87,'sportsbet')"); c.commit()
    return TestClient(app.app)

def test_wrong_or_missing_key_is_not_found(client):
    assert client.get("/api/export/hist_odds").status_code == 404
    assert client.get("/api/export/hist_odds?key=guess").status_code == 404

def test_right_key_returns_the_prices(client, monkeypatch):
    import hashlib
    monkeypatch.setattr(app, "EXPORT_KEY_SHA256", hashlib.sha256(b"k3y").hexdigest())
    js = client.get("/api/export/hist_odds?key=k3y").json()
    assert js["cols"][:3] == ["game_id", "snapshot", "player"] and js["rows"][0][2] == "Jalen Brunson"
