"""The check of what we actually published, and the rule that results belong to one specific model."""
import json, sqlite3
import numpy as np, pytest
from fastapi.testclient import TestClient
import app, predict, features

def _db(tmp_path, monkeypatch, shift):
    """Fake history: 60 games x 10 players; chances saved before tip-off; `shift` lowers what really happens (0 = honest)."""
    p = tmp_path / "p.db"; monkeypatch.setattr(predict, "DB", str(p)); con = sqlite3.connect(p)
    con.execute("CREATE TABLE player_games(game_id TEXT, player_id TEXT, pts INT, reb INT, ast INT)"); predict.ensure_prediction_log(con)
    rng = np.random.default_rng(0)
    for g in range(60):
        for k in range(10):
            u = rng.random()                                  # one real points total per player-game, so both lines agree with it
            pts = 20 if u < 0.6 else 12 if u < 0.85 - shift else 5
            con.execute("INSERT INTO player_games VALUES(?,?,?,0,0)", (f"g{g}", f"p{k}", pts))
            for line, p_over in ((10.5, 0.85), (15.5, 0.6)):
                con.execute("INSERT OR REPLACE INTO prediction_log(game_id,player_id,stat,line,p_over,model_id,made_at,start) VALUES(?,?,?,?,?,?,?,?)", (f"g{g}", f"p{k}", "pts", line, p_over, "M1", "t", "2000-01-01T00:00Z"))
    con.commit()

def test_honest_published_chances_show_no_gap(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch, 0.0); r = predict.published_check()["models"]["M1"]
    v = r["by_stat"]["pts"]; assert r["status"] == "ok" and abs(v["gap"]) < 0.06 and v["gap_low"] < 0.03

def test_overconfident_published_chances_are_caught_with_a_range(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch, 0.15); v = predict.published_check()["models"]["M1"]["by_stat"]["pts"]
    assert v["gap"] > 0.08 and v["gap_low"] > 0.0 and v["gap_low"] < v["gap"] < v["gap_high"]

def test_too_few_games_says_waiting(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch, 0.0); assert predict.published_check(min_games=500)["models"]["M1"]["status"] == "waiting"

def test_missing_log_says_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "DB", str(tmp_path / "empty.db")); r = predict.published_check(); assert r["status"] == "waiting" and r["models"] == {}

def test_incompatible_model_is_not_scored(tmp_path, monkeypatch):
    import pickle
    monkeypatch.setattr(predict, "MODELS", str(tmp_path / "m.pkl")); cols = predict.feature_contract(); cols["reb"] = ["something_else"]
    pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="2000-01-01", report={}, schema=features.FEATURE_SCHEMA, feature_columns=cols, model_id="M1"), open(predict.MODELS, "wb"))
    r = predict.served_check(); assert r["status"] == "incompatible" and r["model_id"] == "M1"

def test_health_ignores_check_results_from_another_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "h.db")); monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(predict, "load_models", lambda strict=True: dict(model_id="NEW", schema=features.FEATURE_SCHEMA, feature_columns=predict.feature_contract()))
    con = sqlite3.connect(app.DB); con.execute("CREATE TABLE model_checks(checked_at TEXT, payload TEXT)")
    con.execute("INSERT INTO model_checks VALUES('2026-01-01', ?)", (json.dumps(dict(kind="published", status="ok", model_id="OLD", by_stat={"pts": dict(gap=.2, gap_low=.1, gap_high=.3)}, games=99)),)); con.commit()
    h = TestClient(app.app).get("/api/health").json()
    assert h["served_check"]["published"] is None and h["served_check"]["ignored_old_model_results"] == 1
    assert not any("too high" in p for p in h["problems"])


def test_each_model_version_is_checked_on_its_own(tmp_path, monkeypatch):
    """Version M1 is honest, version M2 is over-confident: M2's problems must not be hidden by M1, nor M1's good record credited to M2."""
    p = tmp_path / "v.db"; monkeypatch.setattr(predict, "DB", str(p)); con = sqlite3.connect(p)
    con.execute("CREATE TABLE player_games(game_id TEXT, player_id TEXT, pts INT, reb INT, ast INT)"); predict.ensure_prediction_log(con)
    rng = np.random.default_rng(1)
    for g in range(80):
        mid, shift = ("M1", 0.0) if g < 40 else ("M2", 0.2)
        for k in range(10):
            u = rng.random(); pts = 20 if u < 0.6 else 12 if u < 0.85 - shift else 5
            con.execute("INSERT INTO player_games VALUES(?,?,?,0,0)", (f"g{g}", f"p{k}", pts))
            for line, po in ((10.5, 0.85), (15.5, 0.6)):
                con.execute("INSERT INTO prediction_log(game_id,player_id,stat,line,p_over,model_id,made_at,start) VALUES(?,?,?,?,?,?,?,?)", (f"g{g}", f"p{k}", "pts", line, po, mid, "t", "2000-01-01T00:00Z"))
    con.commit(); r = predict.published_check()["models"]
    assert set(r) == {"M1", "M2"} and r["M1"]["games"] == 40 and r["M2"]["games"] == 40
    assert abs(r["M1"]["by_stat"]["pts"]["gap"]) < 0.08 and r["M2"]["by_stat"]["pts"]["gap"] > 0.12

def test_published_record_cannot_change_after_the_game_starts(tmp_path):
    con = sqlite3.connect(tmp_path / "f.db"); predict.ensure_prediction_log(con)
    con.execute("INSERT INTO prediction_log VALUES('past','p','pts',10.5,0.8,'M1','t','2000-01-01T00:00Z')")
    con.execute("INSERT INTO prediction_log VALUES('soon','p','pts',10.5,0.8,'M1','t','2999-01-01T00:00Z')"); con.commit()
    for sql in ("UPDATE prediction_log SET p_over=0.1 WHERE game_id='past'", "DELETE FROM prediction_log WHERE game_id='past'"):
        with pytest.raises(sqlite3.DatabaseError): con.execute(sql)
    assert con.execute("SELECT p_over FROM prediction_log WHERE game_id='past'").fetchone()[0] == 0.8
    con.execute("UPDATE prediction_log SET p_over=0.7 WHERE game_id='soon'"); assert con.execute("SELECT p_over FROM prediction_log WHERE game_id='soon'").fetchone()[0] == 0.7

def test_save_does_not_log_or_overwrite_a_game_that_has_started(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "DB", str(tmp_path / "s.db")); con = sqlite3.connect(predict.DB); predict.ensure_prediction_log(con)
    con.execute("INSERT INTO prediction_log VALUES('past','p','pts',10.5,0.8,'M1','t','2000-01-01T00:00Z')"); con.commit(); con.close()
    def pred(gid, start, p): return dict(game_id=gid, player_id="p", start=start, model_id="M2", ladders={"pts": [dict(line=10.5, p_over=p)]}, home=1)
    monkeypatch.setattr(predict.project, "game_preds", {}, raising=False)
    predict.save([pred("past", "2000-01-01T00:00Z", 0.3), pred("future", "2999-01-01T00:00Z", 0.6)])
    rows = dict(sqlite3.connect(predict.DB).execute("SELECT game_id, p_over FROM prediction_log").fetchall())
    assert rows == {"past": 0.8, "future": 0.6}
