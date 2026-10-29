"""POST /api/refresh queues one background job and answers at once, even when training is due."""
import threading, time
import pytest
from fastapi.testclient import TestClient
import app

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60)); monkeypatch.setattr(app, "_last_manual", [0.0])
    monkeypatch.setitem(app._job, "state", "idle")
    return TestClient(app.app)

def test_refresh_answers_at_once_and_does_not_duplicate(client, monkeypatch):
    gate, started, calls = threading.Event(), threading.Event(), []
    def slow(): calls.append(1); started.set(); gate.wait(10)
    monkeypatch.setattr(app, "refresh_once", slow)
    t0 = time.time(); r1 = client.post("/api/refresh"); assert time.time() - t0 < 1.0
    assert r1.status_code == 202 and r1.json()["accepted"] is True
    assert started.wait(5)
    r2 = client.post("/api/refresh"); assert r2.status_code == 202 and r2.json()["accepted"] is False and r2.json()["already_running"] is True
    st = client.get("/api/refresh/status").json(); assert st["job"]["state"] == "running" and st["job"]["started_at"]
    gate.set()
    for _ in range(50):
        if app._job["state"] == "idle": break
        time.sleep(0.1)
    assert calls == [1] and client.get("/api/refresh/status").json()["job"]["finished_at"]

def test_manual_refresh_never_trains_inside_the_request(client, monkeypatch):
    trained = []
    monkeypatch.setattr(app, "train_quietly", lambda: trained.append(threading.current_thread().name))
    gate = threading.Event(); monkeypatch.setattr(app, "refresh_once", lambda: (gate.wait(5), app.train_quietly()))
    client.post("/api/refresh"); assert trained == []          # nothing ran on the request itself
    gate.set()
    for _ in range(50):
        if trained: break
        time.sleep(0.1)
    assert trained and trained[0] != threading.main_thread().name

def test_too_soon_after_a_manual_refresh_is_refused(client, monkeypatch):
    monkeypatch.setattr(app, "refresh_once", lambda: None)
    assert client.post("/api/refresh").status_code == 202
    for _ in range(50):
        if app._job["state"] == "idle": break
        time.sleep(0.1)
    assert client.post("/api/refresh").status_code == 429
