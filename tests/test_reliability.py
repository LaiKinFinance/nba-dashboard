"""Health page, isolated refresh stages, backup, fail-closed access, request limits."""
import os, sqlite3
import pytest
import app
from fastapi.testclient import TestClient

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", "t"); monkeypatch.setattr(app, "RATE", (10**6, 60))
    return TestClient(app.app, raise_server_exceptions=False)

def test_server_without_access_code_on_fly_refuses(client, monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setenv("FLY_APP_NAME", "x")
    assert client.get("/api/meta").status_code == 503

def test_one_failing_stage_does_not_stop_the_next(client):
    calls = []
    app._stage("a", lambda: 1 / 0); app._stage("b", lambda: calls.append(1))
    h = client.get("/api/health").json()
    assert "ZeroDivisionError" in h["stages"]["a"]["note"] and h["stages"]["a"]["state"] == "failed" and h["stages"]["b"]["state"] == "ok" and calls == [1]

def test_key_text_never_shown(client):
    app._stage("k", lambda: (_ for _ in ()).throw(RuntimeError("https://x/y?apiKey=abcdef123456&z=1")))
    assert "abcdef123456" not in str(client.get("/api/health").json())

def test_backup_is_readable_copy_once_per_day():
    b = app.backup_db(); assert sqlite3.connect(b).execute("select count(*) from games").fetchone()[0] > 1000 and app.backup_db() == b

def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(app, "RATE", (3, 60)); app._hits.clear()
    codes = [client.get("/api/health").status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3] == 429
    app._hits.clear()
