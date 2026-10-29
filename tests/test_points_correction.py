"""The corrected points distribution is used everywhere, is only kept if it earns its place, and no points VALUE claims are made."""
import numpy as np, pytest
from scipy import stats
import app, model, parlay, combos, predict

FIX = dict(version=model.PTS_FIX_VERSION, b=0.05, alpha=0.12, q=0.05, kappa=0.2)
PLAIN = {t: 0.1 for t in model.ALL}
FIXED = dict(PLAIN, pts_fix=FIX)

def test_without_a_correction_nothing_changes():
    assert model.over_prob("pts", 15.0, PLAIN, 12.5) == pytest.approx(float(model.nb_over(15.0, 0.1, 12.5)))

def test_correction_changes_points_only():
    assert model.over_prob("pts", 15.0, FIXED, 12.5) != pytest.approx(float(model.nb_over(15.0, 0.1, 12.5)), abs=1e-4)
    assert model.over_prob("reb", 8.0, FIXED, 6.5) == pytest.approx(float(model.nb_over(8.0, 0.1, 6.5)))

@pytest.mark.parametrize("line", [8.5, 12.5, 16.5, 20.5])
def test_simulated_samples_match_the_formula(line):
    u = np.random.default_rng(1).random(400000); x = model.samples("pts", u, 15.0, FIXED)
    assert (x > line).mean() == pytest.approx(float(model.over_prob("pts", 15.0, FIXED, line)), abs=0.004)

def test_single_bets_and_parlays_use_the_same_corrected_distribution():
    pred = dict(proj={**{k: 3.0 for k in parlay.ALL}, "pts": 15.0}, team="A", p_dd=.1, p_td=.01)
    corr = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}
    leg = dict(player_id="p", game_id="g", stat="pts", line=12.5, side="over")
    r = parlay.parlay([leg], {("g", "p"): pred}, FIXED, corr, n=300000)
    exact = float(model.over_prob("pts", 15.0, FIXED, 12.5))
    assert r["leg_probs"][0] == pytest.approx(exact) and r["joint"] == pytest.approx(exact, abs=0.004)

def test_combined_totals_use_the_corrected_points():
    same = np.eye(3).tolist(); mu = {"pts": 15.0, "reb": 5.0, "ast": 4.0}
    a = {l["line"]: l["p_over"] for l in combos.combo_lines(mu, PLAIN, same)["pra"]}
    b = {l["line"]: l["p_over"] for l in combos.combo_lines(mu, FIXED, same)["pra"]}
    common = [k for k in a if k in b]; assert common and any(abs(a[k] - b[k]) > 0.005 for k in common)

def _data(n, heavy_tail, seed=0):
    rng = np.random.default_rng(seed); mu = rng.uniform(6, 25, n); r = 1 / 0.12
    bad = rng.random(n) < (0.08 if heavy_tail else 0.0); m = np.where(bad, 0.2 * mu, mu)
    return mu, rng.negative_binomial(r, r / (r + m))

def test_correction_is_kept_when_it_helps_on_later_games():
    mu, y = _data(6000, True); fix = model.choose_points_fix(mu, y)
    assert fix and fix["version"] == model.PTS_FIX_VERSION and 0.02 < fix["q"] < 0.2 and fix["evidence"]["accepted"]

def test_correction_is_dropped_when_it_does_not_help():
    mu, y = _data(6000, False); assert model.choose_points_fix(mu, y) is None

def test_too_little_data_gives_no_correction():
    mu, y = _data(300, True); assert model.choose_points_fix(mu, y) is None

def test_gate_rejects_a_broken_correction():
    ok, why = predict.validate({"n_test": 5000, "stats": {**{t: dict(mae_model=4., mae_10game_avg=4.1) for t in predict.TARGETS}, "dd": dict(brier=.08), "td": dict(brier=.008)}},
                               dict(PLAIN, pts_fix=dict(FIX, q=float("nan"))))
    assert not ok and any("pts_fix" in w for w in why)

def test_no_points_value_or_safe_claims(monkeypatch):
    assert app.POINTS_CLAIMS_ENABLED is False
    p = dict(player_id="p", game_id="g", player="X", team="A", opp="B", home=1, start="2999-01-01T00:00Z", injury=None, starting=True, min=30, odds_age_h=1.0,
             strip={"pts": [dict(line=12.5, p_over=.9, price=2.0, book="sportsbet", edge=.4, hit=9, n=10)]}, book={}, proj={"pts": 15.0, "reb": 5., "ast": 4.},
             p_dd=.1, p_td=.01, show_dd=False, show_td=False, ladders={"pts": [dict(line=12.5, p_over=.9)], "reb": [], "ast": []}, sd={}, last10=[], evidence=[], absent=[], opp_factor={})
    monkeypatch.setattr(app, "enriched", lambda: [p]); monkeypatch.setattr(app, "game_preds", lambda: {})
    monkeypatch.setattr(app, "model_status", lambda preds=None: (object(), None)); monkeypatch.setattr(app, "require_current_model", lambda preds=None: dict(alphas=PLAIN, corr={}))
    assert app.compute_value()["picks"] == [] and app.compute_best()["legs"] == []

def test_split_between_fitting_and_checking_never_cuts_a_game():
    groups = np.repeat(np.arange(400), 10)                       # 400 games, 10 players each, in date order
    for frac in (0.5, 0.6, 0.63, 0.7):
        k = model.game_cut(groups, frac)
        assert k % 10 == 0 and groups[k - 1] != groups[k]        # the cut falls between games

def test_correction_choice_uses_whole_games():
    mu, y = _data(6000, True); groups = np.repeat(np.arange(600), 10)
    fix = model.choose_points_fix(mu, y, groups=groups)
    assert fix and fix["evidence"]["holdout_rows"] % 10 == 0     # the checking part holds whole games only
