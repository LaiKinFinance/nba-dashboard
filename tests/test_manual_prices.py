"""'Refresh prices now' fetches once in the background and can't be repeated within an hour (so taps can't burn credits)."""
import time
from fastapi.testclient import TestClient
import app

def test_manual_refresh_runs_once_then_waits(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); app._hits.clear(); monkeypatch.setattr(app, "_odds_on", lambda: True)
    calls = []
    monkeypatch.setattr(app, "refresh_odds", lambda **k: calls.append(k) or dict(props=dict(fetched=13, remaining="19900")))
    app._manual_odds.update(running=False, at=0.0, result=None)
    c = TestClient(app.app)
    assert c.post("/api/odds/refresh").json()["started"] is True
    for _ in range(50):
        if not app._manual_odds["running"]: break
        time.sleep(0.05)
    assert len(calls) == 1 and calls[0]["refetch_h"] < 1 and app._manual_odds["result"] == dict(games=13, credits_left="19900")
    r = c.post("/api/odds/refresh").json(); assert r["started"] is False and "try again" in r["msg"]
