"""The refresh must stop and say so when data repair or download is incomplete; health must tell the truth and never crash."""
import hashlib, sqlite3
import pytest
import app, repair, ingest, predict
from fastapi.testclient import TestClient

@pytest.fixture
def blank(tmp_path, monkeypatch):
    """An empty database and no saved model: the worst case for the health page."""
    monkeypatch.setattr(app, "DB", str(tmp_path / "x.db")); monkeypatch.setattr(predict, "MODELS", str(tmp_path / "none.pkl"))
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    return tmp_path

def _fake_cycle(monkeypatch, repair_result, ingest_result):
    calls = []
    monkeypatch.setattr(repair, "run_all", lambda *a, **k: repair_result)
    monkeypatch.setattr(ingest, "run", lambda *a, **k: ingest_result)
    monkeypatch.setattr(app, "train_quietly", lambda: calls.append("train"))
    monkeypatch.setattr(predict, "model_stale", lambda: False); monkeypatch.setattr(app, "_models_age", lambda: 10 ** 9)
    monkeypatch.setattr(predict, "project", lambda: calls.append("project") or []); monkeypatch.setattr(predict, "save", lambda p: None)
    for name in ("refresh_odds", "grade_picks", "log_picks", "backup_db"): monkeypatch.setattr(app, name, lambda *a, **k: None)
    return calls

def test_incomplete_repair_blocks_training_and_projection(blank, monkeypatch):
    calls = _fake_cycle(monkeypatch, {"problem": "3 games have no tip-off time"}, {"problem": None})
    app.refresh_once()
    assert calls == []
    st = TestClient(app.app).get("/api/health").json()
    assert st["stages"]["repair"]["state"] == "partial" and st["stages"]["train"]["state"] == "skipped" and st["stages"]["project"]["state"] == "skipped"
    assert st["overall"] == "problem"

def test_failed_download_blocks_training_and_projection(blank, monkeypatch):
    calls = _fake_cycle(monkeypatch, {"problem": None}, {"problem": "2 game(s) could not be downloaded"})
    app.refresh_once(); assert calls == []

def test_good_cycle_trains_and_projects(blank, monkeypatch):
    calls = _fake_cycle(monkeypatch, {"problem": None}, {"problem": None})
    app.refresh_once(); assert calls == ["train", "project"]
    assert TestClient(app.app).get("/api/health").json()["stages"]["project"]["state"] == "ok"

def test_health_never_crashes_on_empty_install(blank):
    r = TestClient(app.app).get("/api/health")
    assert r.status_code == 200 and r.json()["overall"] == "problem"

def test_partial_roster_answer_changes_nothing(tmp_path, monkeypatch):
    con = sqlite3.connect(tmp_path / "r.db"); con.execute("CREATE TABLE rosters(player_id TEXT PRIMARY KEY, team TEXT, name TEXT, fetched_at TEXT)")
    con.execute("CREATE TABLE roster_log(fetched_at TEXT, player_id TEXT, name TEXT, old_team TEXT, new_team TEXT)")
    abbrs = sorted(ingest.REAL_TEAMS); teams = [{"team": {"id": str(i), "abbreviation": a}} for i, a in enumerate(abbrs, 1)]
    def fake(empty):
        def get(url, params, tries=4):
            if url.endswith("/teams"): return {"sports": [{"leagues": [{"teams": teams}]}]}
            ab = abbrs[int(url.split("/teams/")[1].split("/")[0]) - 1]
            return {"athletes": [] if ab == empty else [{"id": f"{ab}{k}", "displayName": f"{ab} P{k}"} for k in range(15)]}
        return get
    monkeypatch.setattr(ingest, "get", fake(None)); assert repair.refresh_rosters(con) == 450
    monkeypatch.setattr(ingest, "get", fake("NY"))
    assert repair.refresh_rosters(con) == 0                                           # 435 players came back, yet one team was empty
    assert con.execute("SELECT count(*) FROM rosters WHERE team IS NOT NULL").fetchone()[0] == 450

def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    p = tmp_path / "d.db"; con = sqlite3.connect(p); con.execute("CREATE TABLE games(game_id TEXT, date TEXT)"); con.execute("INSERT INTO games VALUES('1','2026-01-01')"); con.commit(); con.close()
    monkeypatch.setattr(ingest, "DB", str(p)); before = hashlib.md5(p.read_bytes()).hexdigest()
    out = repair.dry_run()
    assert out["games"] == 1 and hashlib.md5(p.read_bytes()).hexdigest() == before

def test_chunked_oversize_body_is_refused(blank):
    c = TestClient(app.app, raise_server_exceptions=False)
    def gen(): yield b'{"x":"'; yield b"A" * 60000; yield b'"}'
    assert c.post("/api/parlay", content=b'{"x":"' + b"A" * 60000 + b'"}', headers={"content-type": "application/json"}).status_code == 413
    assert c.post("/api/parlay", content=gen(), headers={"content-type": "application/json"}).status_code == 413
