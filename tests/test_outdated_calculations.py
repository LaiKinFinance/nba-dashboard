"""Nothing turns an old model's output into fresh chances; started games keep their own frozen numbers."""
import pickle
import pytest
from fastapi.testclient import TestClient
import app, predict, features

FUTURE, PAST = "2999-01-01T00:00Z", "2000-01-01T00:00Z"

@pytest.fixture
def current(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODELS", str(tmp_path / "m.pkl"))
    pickle.dump(dict(models={}, alphas={"pts": .1}, corr={}, trained_through="x", report={}, schema=features.FEATURE_SCHEMA, feature_columns=predict.feature_contract(), model_id="NEW"), open(predict.MODELS, "wb"))

def _proj(**kw): return dict(game_id="g", player_id="p", **kw)

def test_upcoming_games_from_another_model_are_withheld(current, monkeypatch):
    monkeypatch.setattr(app, "projections", lambda: [_proj(start=FUTURE, model_id="OLD", alphas={"pts": .05})])
    assert app.guarded_projections() == []

def test_upcoming_games_from_current_model_pass(current, monkeypatch):
    monkeypatch.setattr(app, "projections", lambda: [_proj(start=FUTURE, model_id="NEW", alphas={"pts": .1})])
    assert len(app.guarded_projections()) == 1

def test_started_game_keeps_its_own_numbers_and_is_labelled_frozen(current, monkeypatch):
    monkeypatch.setattr(app, "projections", lambda: [_proj(start=PAST, model_id="OLD", alphas={"pts": .05})])
    out = app.guarded_projections(); assert out[0]["frozen"] is True and out[0]["alphas"] == {"pts": .05}

def test_started_game_without_stored_numbers_is_left_out_not_recalculated(current, monkeypatch):
    monkeypatch.setattr(app, "projections", lambda: [_proj(start=PAST, model_id=None)])
    assert app.guarded_projections() == []

def test_old_schema_model_withholds_everything_upcoming(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODELS", str(tmp_path / "m.pkl"))
    pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="x", report={}), open(predict.MODELS, "wb"))
    monkeypatch.setattr(app, "projections", lambda: [_proj(start=FUTURE, model_id=None, alphas={"pts": .1})])
    assert app.guarded_projections() == []

def test_no_model_file_is_refused_not_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODELS", str(tmp_path / "none.pkl"))
    monkeypatch.setattr(predict, "train", lambda *a, **k: (_ for _ in ()).throw(AssertionError("trained")))
    with pytest.raises(Exception) as e: app.guarded_projections()
    assert getattr(e.value, "status_code", None) == 503

def test_cannot_build_a_bet_on_a_game_that_has_started(current, monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "projections", lambda: [_proj(start=PAST, model_id="NEW", alphas={"pts": .1}, team="A")])
    r = TestClient(app.app, raise_server_exceptions=False).post("/api/parlay", json=dict(legs=[dict(game_id="g", player_id="p", stat="pts", line=20.5, side="over")]))
    assert r.status_code == 400 and "already started" in r.json()["detail"]
