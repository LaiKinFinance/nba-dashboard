"""The check of what we actually published, and the rule that results belong to one specific model."""
import json, sqlite3
import numpy as np, pytest
from fastapi.testclient import TestClient
import app, predict, features

def _db(tmp_path, monkeypatch, shift):
    """Fake history: 60 games x 10 players; chances saved before tip-off; `shift` lowers what really happens (0 = honest)."""
    p = tmp_path / "p.db"; monkeypatch.setattr(predict, "DB", str(p)); con = sqlite3.connect(p)
    con.execute("CREATE TABLE player_games(game_id TEXT, player_id TEXT, pts INT, reb INT, ast INT)")
    con.execute("CREATE TABLE prediction_log(game_id TEXT, player_id TEXT, stat TEXT, line REAL, p_over REAL, model_id TEXT, made_at TEXT, PRIMARY KEY(game_id, player_id, stat, line))")
    rng = np.random.default_rng(0)
    for g in range(60):
        for k in range(10):
            u = rng.random()                                  # one real points total per player-game, so both lines agree with it
            pts = 20 if u < 0.6 else 12 if u < 0.85 - shift else 5
            con.execute("INSERT INTO player_games VALUES(?,?,?,0,0)", (f"g{g}", f"p{k}", pts))
            for line, p_over in ((10.5, 0.85), (15.5, 0.6)):
                con.execute("INSERT OR REPLACE INTO prediction_log VALUES(?,?,?,?,?,?,?)", (f"g{g}", f"p{k}", "pts", line, p_over, "M1", "t"))
    con.commit()

def test_honest_published_chances_show_no_gap(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch, 0.0); r = predict.published_check()
    v = r["by_stat"]["pts"]; assert r["status"] == "ok" and abs(v["gap"]) < 0.06 and v["gap_low"] < 0.03

def test_overconfident_published_chances_are_caught_with_a_range(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch, 0.15); v = predict.published_check()["by_stat"]["pts"]
    assert v["gap"] > 0.08 and v["gap_low"] > 0.0 and v["gap_low"] < v["gap"] < v["gap_high"]

def test_too_few_games_says_waiting(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch, 0.0); assert predict.published_check(min_games=500)["status"] == "waiting"

def test_missing_log_says_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "DB", str(tmp_path / "empty.db")); assert predict.published_check()["status"] == "waiting"

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
