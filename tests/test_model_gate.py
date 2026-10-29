"""Saved-model version check and the validation gate."""
import os, pickle, pytest
import predict, features

@pytest.fixture
def models_path(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODELS", str(tmp_path / "models.pkl")); return predict.MODELS

def test_old_model_refused_for_projection_but_readable(models_path):
    pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="2026-01-01", report={}), open(models_path, "wb"))
    assert predict.model_stale()
    with pytest.raises(predict.StaleModel): predict.load_models()
    assert predict.load_models(strict=False)["trained_through"] == "2026-01-01"

def test_current_model_accepted(models_path):
    pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="x", report={}, schema=features.FEATURE_SCHEMA), open(models_path, "wb"))
    assert not predict.model_stale() and predict.load_models()["trained_through"] == "x"

def test_gate_rejects_model_worse_than_plain_average():
    good = {"stats": {t: dict(mae_model=4.0, mae_10game_avg=4.1) for t in predict.TARGETS}}
    bad = {"stats": {**good["stats"], "pts": dict(mae_model=6.0, mae_10game_avg=4.1)}}
    assert predict.validate(good) == (True, []) and predict.validate(bad) == (False, ["pts"])
