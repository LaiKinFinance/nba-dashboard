"""Loading never trains; bet-probability routes refuse a model that does not match the projections."""
import pickle
import pytest
from fastapi import HTTPException
import app, predict, features

@pytest.fixture
def models_path(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODELS", str(tmp_path / "m.pkl")); return predict.MODELS

def _save(path, **kw):
    b = dict(models={}, alphas={}, corr={}, trained_through="x", report={}, schema=features.FEATURE_SCHEMA, feature_columns=predict.feature_contract(), model_id="abc", **kw)
    pickle.dump(b, open(path, "wb"))

def test_loading_a_missing_model_never_trains(models_path, monkeypatch):
    monkeypatch.setattr(predict, "train", lambda *a, **k: (_ for _ in ()).throw(AssertionError("trained inside a load")))
    with pytest.raises(predict.ModelMissing): predict.load_models()
    with pytest.raises(predict.ModelMissing): predict.load_models(strict=False)

FUTURE, PAST = "2999-01-01T00:00Z", "2000-01-01T00:00Z"

def test_matching_model_and_projections_are_current(models_path):
    _save(models_path); assert app.model_status([dict(model_id="abc", start=FUTURE)]) [1] is None

def test_old_schema_model_is_refused_for_bets(models_path):
    pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="x", report={}), open(models_path, "wb"))
    with pytest.raises(HTTPException) as e: app.require_current_model([])
    assert e.value.status_code == 503

def test_projection_from_another_model_is_refused(models_path):
    _save(models_path)
    with pytest.raises(HTTPException) as e: app.require_current_model([dict(model_id="old", start=FUTURE)])
    assert e.value.status_code == 503

def test_games_already_started_do_not_block(models_path):
    _save(models_path); assert app.model_status([dict(model_id="old", start=PAST)])[1] is None

def test_missing_model_is_refused_not_trained(models_path):
    with pytest.raises(HTTPException): app.require_current_model([])

def test_best_and_value_return_empty_with_a_notice_when_stale(models_path, monkeypatch):
    monkeypatch.setattr(app, "enriched", lambda: [])
    monkeypatch.setattr(app, "model_status", lambda preds=None: (None, "Predictions are being refreshed"))
    b = app.compute_best(); v = app.compute_value()
    assert b["legs"] == [] and b["combo"]["legs"] == [] and "refreshed" in b["notice"]
    assert v["picks"] == [] and "refreshed" in v["notice"]
