"""Speed-ups must not change any number: shared results refresh when data changes, pages are compressed, heavy work stays out of the web process."""
import time
import pytest
from fastapi.testclient import TestClient
import app, model

def test_enriched_is_computed_once_until_data_changes(monkeypatch):
    calls = []; key = [("a",)]
    monkeypatch.setattr(app, "_enriched_build", lambda: calls.append(1) or [{"x": len(calls)}]); monkeypatch.setattr(app, "_enriched_key", lambda: key[0])
    app._enr.update(key=None, val=None)
    a = app.enriched(); b = app.enriched(); assert a is b and len(calls) == 1
    key[0] = ("b",); c = app.enriched(); assert len(calls) == 2 and c is not a          # new projections or prices: recomputed

def test_batched_chances_equal_one_at_a_time():
    import numpy as np
    alphas = {t: 0.1 for t in model.ALL}; alphas["pts_fix"] = dict(version="x", b=0.05, alpha=0.12, q=0.05, kappa=0.2)
    Ls = np.floor(14.3) + np.arange(-14, 15) + 0.5; Ls = Ls[Ls >= 0.5]
    for stat in ("pts", "reb"):
        batch = np.atleast_1d(model.over_prob(stat, 14.3, alphas, Ls))
        single = np.array([float(model.over_prob(stat, 14.3, alphas, float(L))) for L in Ls])
        assert np.allclose(batch, single, atol=1e-12)

def test_large_responses_are_compressed(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    r = TestClient(app.app).get("/", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip"

def test_projection_and_checks_run_in_a_background_process(monkeypatch):
    seen = []
    monkeypatch.setattr(app, "run_quietly", lambda code, timeout=0: seen.append(code) or "[]")
    app.project_quietly(); app.train_quietly()
    assert any("predict.save(predict.project())" in c for c in seen) and any("predict.train()" in c for c in seen)
