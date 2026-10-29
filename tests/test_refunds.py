"""A tie on a whole-number line refunds the stake. Chances, fair price and value must treat it that way."""
import numpy as np, pytest
from fastapi.testclient import TestClient
import app, parlay, model

ALPHAS = {k: 0.1 for k in parlay.ALL}
CORR = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}
PRED = dict(proj={**{k: 3.0 for k in parlay.ALL}, "pts": 20.0}, team="A", p_dd=.1, p_td=.01)
GP = dict(source="market", home="NY", away="BOS", p_home=.6, p_away=.4, start="2999-01-01T00:00Z", home_pts=112., away_pts=106.,
          margin=5.0, total=218.0, sigma_margin=13., sigma_total=17., handicaps=[], totals=[], team_totals={}, bands=[])

def test_player_whole_line_has_a_refund_chance_that_matches_the_distribution():
    leg = dict(player_id="p", game_id="g", stat="pts", line=20.0, side="over")
    r = parlay.parlay([leg], {("g", "p"): PRED}, ALPHAS, CORR, n=200000)
    exact = float(model.nb_over(20.0, 0.1, 19.0) - model.nb_over(20.0, 0.1, 20.0))      # P(X = 20)
    assert r["push_probs"][0] == pytest.approx(exact, abs=0.01)

def test_half_point_line_never_refunds():
    leg = dict(player_id="p", game_id="g", stat="pts", line=20.5, side="over")
    assert parlay.parlay([leg], {("g", "p"): PRED}, ALPHAS, CORR, n=20000)["push_probs"] == [0.0]

@pytest.mark.parametrize("stat,side,line", [("spread", "home", -5.0), ("spread", "away", -5.0), ("total", "over", 218.0), ("total", "under", 218.0)])
def test_team_whole_line_chances_add_to_one(stat, side, line):
    w, pu = app.team_leg_outcome(dict(stat=stat, side=side, line=line), GP)
    other = app.team_leg_outcome(dict(stat=stat, side={"home": "away", "away": "home", "over": "under", "under": "over"}[side], line=line), GP)
    assert pu > 0.02 and 0 < w < 1 and w + other[0] + pu == pytest.approx(1.0, abs=1e-9)

def test_team_half_line_has_no_refund():
    assert app.team_leg_outcome(dict(stat="spread", side="home", line=-4.5), GP)[1] == 0.0

def test_refund_value_matches_hand_calculation_for_independent_legs():
    legs = [dict(player_id="p", game_id="g", stat="pts", line=20.0, side="over")]
    r = parlay.parlay(legs, {("g", "p"): PRED}, ALPHAS, CORR, n=300000, prices=[1.9], team_extra=[(0.5, 0.1, 2.0)])
    pw = r["leg_probs"][0]; pp = r["push_probs"][0]
    expected = (pw * 1.9 + pp * 1.0) * (0.5 * 2.0 + 0.1 * 1.0) - 1
    assert r["ev"] == pytest.approx(expected, abs=0.02)

def test_refund_value_is_unavailable_without_leg_prices():
    assert parlay.refund_ev([np.array([True])], [np.array([False])], [None], []) is None

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "projections", lambda: [dict(PRED, game_id="g", player_id="p")]); monkeypatch.setattr(app, "game_preds", lambda: {"g": GP})
    monkeypatch.setattr(app, "require_current_model", lambda preds=None: dict(alphas=ALPHAS, corr=CORR))
    return TestClient(app.app, raise_server_exceptions=False)

def test_api_without_leg_prices_gives_no_value_on_a_refundable_bet(client):
    r = client.post("/api/parlay", json=dict(odds=3.0, legs=[dict(game_id="g", player_id="p", stat="pts", line=20.0, side="over"), dict(game_id="g", stat="spread", side="home", line=-5.0)])).json()
    assert r["value"] is None and "price of each leg" in r["value_note"] and r["fair_odds"] is None and "refund" in r["refund_note"].lower()

def test_api_with_leg_prices_gives_refund_aware_value(client):
    r = client.post("/api/parlay", json=dict(legs=[dict(game_id="g", player_id="p", stat="pts", line=20.0, side="over", price=1.9), dict(game_id="g", stat="spread", side="home", line=-5.0, price=1.9)])).json()
    assert "assumption" in r["value"] and r["fair_odds"] is None

def test_api_half_point_bets_keep_the_normal_value(client):
    r = client.post("/api/parlay", json=dict(odds=3.0, legs=[dict(game_id="g", player_id="p", stat="pts", line=20.5, side="over")])).json()
    assert r["value"]["ev"] is not None and r["fair_odds"] is not None and "refund_note" not in r
