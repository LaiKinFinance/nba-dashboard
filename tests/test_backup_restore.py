"""Backup, off-server download and a safe way back that keeps newer bets and records."""
import gzip, hashlib, json, os, shutil, sqlite3, sys
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import backup_restore as br
import app, ingest

@pytest.fixture
def live(tmp_path, monkeypatch):
    """A small live data folder with the real table layout (created by the app itself)."""
    d = tmp_path / "live"; d.mkdir(); monkeypatch.setattr(app, "DB", str(d / "nba.db")); monkeypatch.setattr(ingest, "DB", str(d / "nba.db"))
    app.init(); c = sqlite3.connect(d / "nba.db"); c.executescript(ingest.SCHEMA); ingest.ensure_columns(c)
    c.execute("INSERT INTO games(game_id,date,home,away,home_pts,away_pts) VALUES('g1','2026-05-01','A','B',100,90)")
    c.execute("INSERT INTO player_games(game_id,player_id,player,pts) VALUES('g1','p1','P One',20)")
    c.execute("INSERT INTO bets(user,created,legs,stake,odds,model_prob) VALUES('mark','2026-05-01T00:00Z','[]',10,2.0,.4)")
    c.execute("INSERT INTO picks(user,created,legs,note,odds,model_prob) VALUES('mate','2026-05-01T01:00Z','[]','old pick',2.0,.4)")
    c.execute("INSERT INTO comments(pick_id,user,created,body) VALUES(1,'mark','2026-05-01T02:00Z','nice')"); c.commit(); c.close()
    (d / "models.pkl").write_bytes(b"model-v1"); return d

def rows(path, sql): return sqlite3.connect(path).execute(sql).fetchall()

def test_bundle_is_checked_and_corruption_is_detected(live, tmp_path):
    b = tmp_path / "bundle"; m = br.make_bundle(str(live), str(b))
    assert br.verify_bundle(str(b)) == [] and m["files"]["models.pkl"]["bytes"] == 8 and m["tables"]["games"] == 1
    (b / "models.pkl").write_bytes(b"tampered"); assert any("models.pkl does not match" in p for p in br.verify_bundle(str(b)))

def test_restore_keeps_newer_records_and_touches_nothing_else(live, tmp_path):
    b = tmp_path / "bundle"; br.make_bundle(str(live), str(b))
    # life goes on after the backup: new bet, new pick with a comment and a vote, a published prediction, a newly graded old bet
    c = sqlite3.connect(live / "nba.db")
    c.execute("INSERT INTO bets(user,created,legs,stake,odds,model_prob) VALUES('mark','2026-05-02T00:00Z','[{\"x\":1}]',20,3.0,.3)")
    c.execute("UPDATE bets SET result='won' WHERE created='2026-05-01T00:00Z'")
    c.execute("INSERT INTO picks(user,created,legs,note,odds,model_prob) VALUES('mate','2026-05-02T01:00Z','[]','new pick',2.5,.4)")
    c.execute("INSERT INTO comments(pick_id,user,created,body) VALUES(2,'mark','2026-05-02T02:00Z','good one')")
    c.execute("INSERT INTO votes VALUES(2,'mark',1)"); c.commit()
    # ...then a bad release damages the live data
    c.execute("DROP TABLE player_games"); c.execute("UPDATE games SET home_pts=999"); c.commit(); c.close()
    before = br.sha256(str(live / "nba.db"))
    new = tmp_path / "recovered.db"; added = br.restore(str(b), str(live / "nba.db"), str(new))
    assert br.sha256(str(live / "nba.db")) == before                                                          # the damaged live file was only read
    assert rows(new, "SELECT home_pts FROM games") == [(100,)] and rows(new, "SELECT pts FROM player_games") == [(20,)]       # old good data is back
    assert sorted(r[0] for r in rows(new, "SELECT stake FROM bets")) == [10.0, 20.0]                                    # the newer bet survived
    assert rows(new, "SELECT result FROM bets WHERE stake=10") == [("won",)]                                            # and a mark made after the backup
    assert sorted(r[0] for r in rows(new, "SELECT note FROM picks")) == ["new pick", "old pick"]
    assert sorted(r[0] for r in rows(new, "SELECT body FROM comments")) == ["good one", "nice"]
    new_pick = rows(new, "SELECT id FROM picks WHERE note='new pick'")[0][0]
    assert rows(new, "SELECT pick_id FROM comments WHERE body='good one'") == [(new_pick,)] and rows(new, "SELECT pick_id FROM votes") == [(new_pick,)]   # links re-pointed
    assert added["bets"] == 1 and added["picks"] == 1 and added["comments"] == 1 and added["bets_marked_since"] == 1
    assert rows(new, "PRAGMA integrity_check") == [("ok",)]

def test_restoring_twice_adds_nothing_twice(live, tmp_path):
    b = tmp_path / "bundle"; br.make_bundle(str(live), str(b))
    c = sqlite3.connect(live / "nba.db"); c.execute("INSERT INTO bets(user,created,legs,stake,odds,model_prob) VALUES('mark','2026-05-02T00:00Z','[]',20,3.0,.3)"); c.commit(); c.close()
    br.restore(str(b), str(live / "nba.db"), str(tmp_path / "r1.db"))
    br.restore(str(b), str(tmp_path / "r1.db"), str(tmp_path / "r2.db"))                                          # restore on top of an already-restored copy
    assert rows(tmp_path / "r2.db", "SELECT count(*) FROM bets") == [(2,)]

def test_it_refuses_to_overwrite_and_refuses_a_broken_bundle(live, tmp_path):
    b = tmp_path / "bundle"; br.make_bundle(str(live), str(b)); target = tmp_path / "t.db"; target.write_bytes(b"x")
    with pytest.raises(FileExistsError): br.restore(str(b), str(live / "nba.db"), str(target))
    (b / "nba.db").write_bytes(b"not a database")
    with pytest.raises(ValueError): br.restore(str(b), str(live / "nba.db"), str(tmp_path / "new.db"))
    assert not (tmp_path / "new.db").exists()

@pytest.fixture
def client(live, monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60)); monkeypatch.setattr(app, "_last_backup_download", [0.0])
    import predict; monkeypatch.setattr(predict, "MODELS", str(live / "models.pkl"))
    return TestClient(app.app, raise_server_exceptions=False)

def test_download_is_off_until_a_backup_code_is_set(client, monkeypatch):
    monkeypatch.setattr(app, "BACKUP_CODE", ""); assert client.get("/api/backup", headers={"X-Backup-Code": "x"}).status_code == 404

def test_download_needs_the_right_code_and_gives_a_checkable_copy(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "BACKUP_CODE", "s3cret")
    assert client.get("/api/backup", headers={"X-Backup-Code": "wrong"}).status_code == 401
    monkeypatch.setattr(app, "_last_backup_download", [0.0])
    r = client.get("/api/backup", headers={"X-Backup-Code": "s3cret"}); assert r.status_code == 200
    gz = tmp_path / "dl.db.gz"; gz.write_bytes(r.content); out = tmp_path / "from_download"
    br.unpack_download(str(gz), str(out), expected_sha=r.headers["X-SHA256-Of-Database"])
    assert br.verify_bundle(str(out)) == [] and rows(out / "nba.db", "SELECT home_pts FROM games") == [(100,)]
    assert client.get("/api/backup", headers={"X-Backup-Code": "s3cret"}).status_code == 429                  # not twice within 30 seconds

def test_model_download(client, monkeypatch):
    monkeypatch.setattr(app, "BACKUP_CODE", "s3cret"); r = client.get("/api/backup/model", headers={"X-Backup-Code": "s3cret"})
    assert r.status_code == 200 and r.content == b"model-v1" and r.headers["X-SHA256"] == hashlib.sha256(b"model-v1").hexdigest()

def test_backup_code_is_not_in_the_deploy_logs_or_page():
    assert "BACKUP_CODE" in open(".github/workflows/deploy.yml").read() and "s3cret" not in open("static/index.html").read()
