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


def test_each_day_carries_the_first_start_time_so_the_page_can_use_the_phone_clock(client):
    import sqlite3, json
    c = sqlite3.connect(app.DB)
    c.execute("CREATE TABLE IF NOT EXISTS projections(game_id TEXT, player_id TEXT, payload TEXT, made_at TEXT, PRIMARY KEY(game_id, player_id))")
    c.execute("INSERT INTO projections VALUES('g','x',?,?)", (json.dumps({"start": "2026-10-30T00:00Z"}), "t"))
    _pick(c, "NEW", None); c.commit()
    day = client.get("/api/track").json()["days"][0]
    assert day["day"] == "2026-10-28" and day["start"] == "2026-10-30T00:00Z"


def test_todays_picks_cover_only_the_next_game_day():
    rows = [dict(start="2026-10-30T00:00Z", id=1), dict(start="2026-10-30T02:00Z", id=2), dict(start="2026-10-30T23:30Z", id=3), dict(start="2026-10-31T02:00Z", id=4)]
    assert [r["id"] for r in app.next_game_day(rows)] == [1, 2]
    assert app.next_game_day([]) == []
