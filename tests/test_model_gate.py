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
    pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="x", report={}, schema=features.FEATURE_SCHEMA, feature_columns=predict.feature_contract()), open(models_path, "wb"))
    assert not predict.model_stale() and predict.load_models()["trained_through"] == "x"

GOOD_ALPHAS = {t: 0.1 for t in predict.ALL}

def _report(**over):
    stats = {t: dict(mae_model=4.0, mae_10game_avg=4.1) for t in predict.TARGETS}
    stats.update(dd=dict(brier=0.08), td=dict(brier=0.008))
    return {"n_test": 5000, "stats": {**stats, **over}}

def test_gate_accepts_a_good_report():
    assert predict.validate(_report(), GOOD_ALPHAS) == (True, [])

def test_gate_rejects_model_worse_than_plain_average():
    ok, why = predict.validate(_report(pts=dict(mae_model=6.0, mae_10game_avg=4.1)), GOOD_ALPHAS)
    assert not ok and any("pts" in w for w in why)

@pytest.mark.parametrize("bad", [float("nan"), 0.0, None, float("inf")])
def test_gate_rejects_missing_or_invalid_errors(bad):
    assert not predict.validate(_report(pts=dict(mae_model=bad, mae_10game_avg=4.1)), GOOD_ALPHAS)[0]

@pytest.mark.parametrize("bad", [float("nan"), None, float("inf")])
def test_gate_rejects_missing_or_invalid_scores(bad):
    assert not predict.validate(_report(dd=dict(brier=bad)), GOOD_ALPHAS)[0]

def test_gate_rejects_too_few_test_games():
    r = _report(); r["n_test"] = 0
    assert not predict.validate(r, GOOD_ALPHAS)[0]
    r.pop("n_test"); assert not predict.validate(r, GOOD_ALPHAS)[0]

@pytest.mark.parametrize("n", [float("nan"), float("inf"), 1000.5, -5, 999, None, "5000", True])
def test_gate_rejects_invalid_sample_counts(n):
    r = _report(); r["n_test"] = n
    assert not predict.validate(r, GOOD_ALPHAS)[0]

def test_gate_accepts_whole_number_float_count():
    r = _report(); r["n_test"] = 5000.0
    assert predict.validate(r, GOOD_ALPHAS)[0]

def test_gate_requires_every_expected_spread_value():
    partial = {k: v for k, v in GOOD_ALPHAS.items() if k != "blk"}
    ok, why = predict.validate(_report(), partial); assert not ok and any("blk" in w for w in why)
    assert not predict.validate(_report(), None)[0]

@pytest.mark.parametrize("bad", [float("nan"), 0.0, -0.1, float("inf"), 6.0])
def test_gate_rejects_invalid_spread_values(bad):
    assert not predict.validate(_report(), dict(GOOD_ALPHAS, pts=bad))[0]


def test_model_with_different_input_columns_counts_as_stale(models_path):
    cols = predict.feature_contract(); cols["pts"] = cols["pts"][:-1]
    pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="x", report={}, schema=features.FEATURE_SCHEMA, feature_columns=cols), open(models_path, "wb"))
    assert predict.model_stale() and "different input columns" in predict.contract_problem(predict.load_models(strict=False))
