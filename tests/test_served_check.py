"""Do predicted chances match what happens? Checked on data the model was not trained on."""
import numpy as np
import predict, model

def _simulate(alpha_true, alpha_used, n=60000, seed=1):
    rng = np.random.default_rng(seed); mu = rng.uniform(4, 25, n)
    r = 1 / alpha_true; y = rng.negative_binomial(r, r / (r + mu))
    line = np.floor(mu) + 0.5 + rng.integers(-3, 4, n)
    p = model.nb_over(mu, alpha_used, line); return p, (y > line).astype(int)

def test_honest_spread_gives_small_gap():
    p, y = _simulate(0.1, 0.1); s = predict.calibration_summary(p, y)
    assert abs(s["gap"]) < 0.01 and abs(s["high_pred"] - s["high_actual"]) < 0.02

def test_too_small_spread_shows_overconfidence_in_high_chances():
    p, y = _simulate(0.15, 0.01); s = predict.calibration_summary(p, y)
    assert s["high_pred"] - s["high_actual"] > 0.04

def test_served_check_says_waiting_when_few_later_games(monkeypatch):
    monkeypatch.setattr(predict, "load_models", lambda strict=False: dict(trained_through="2999-01-01", models={}, alphas={}))
    out = predict.served_check(); assert out["status"] == "waiting" and out["later_rows"] == 0
