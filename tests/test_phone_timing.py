"""Phones report how long pages took; the health page shows the summary (no names, no secrets)."""
from fastapi.testclient import TestClient
import app

def test_phone_timings_are_summarised_on_health(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); app.PHONE_TIMES.clear(); app._hits.clear()
    c = TestClient(app.app)
    assert c.post("/api/timing", json={"t": [{"p": "slate", "ms": 900}, {"p": "slate", "ms": 300}, {"p": "tab:best/today", "ms": 1200}]}).json() == {"ok": True}
    s = app.phone_speed()
    assert s["slate"] == dict(n=2, median_ms=600, worst_ms=900) and s["tab:best/today"]["worst_ms"] == 1200
