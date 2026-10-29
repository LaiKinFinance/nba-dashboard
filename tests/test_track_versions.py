"""The Track record counts every published pick, old and new, as Mark asked."""
import pytest
from fastapi.testclient import TestClient
import app

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "t.db")); monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "grade_picks", lambda: None); monkeypatch.setattr(app, "_current_model_id", lambda: "NEW")
    app.init(); return TestClient(app.app)

def _pick(c, model, result, p=0.8, kind="leg", i=[0]):
    i[0] += 1
    c.execute("INSERT INTO pick_log(day,game_id,player_id,player,team,kind,stat,line,side,model_p,created,result,model_id) VALUES('2026-10-28',?,?,?,?,?,?,?,?,?,?,?,?)",
              ("g", f"p{i[0]}", "X", "A", kind, "reb", 5.5, "over", p, "t", result, model))

def test_track_record_counts_every_published_pick(client):
    import sqlite3
    c = sqlite3.connect(app.DB)
    for r in ("won", "won", "lost"): _pick(c, "NEW", r)
    for r in ("lost",) * 10: _pick(c, "OLD", r)
    _pick(c, None, "lost")
    c.commit(); t = client.get("/api/track").json()
    assert t["legs"]["n"] == 14 and t["legs"]["won"] == 2
    assert "earlier_versions" not in t
