"""Nothing is calculated or recommended for a game that has started."""
import datetime as dt
import numpy as np, pytest
from fastapi.testclient import TestClient
import app, parlay, odds

NOW = dt.datetime.now(dt.timezone.utc)
iso = lambda d: d.strftime("%Y-%m-%dT%H:%MZ")
PAST, FUTURE = iso(NOW - dt.timedelta(minutes=20)), iso(NOW + dt.timedelta(hours=3))
ALPHAS = {k: 0.1 for k in parlay.ALL}; CORR = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}

def gp(start): return dict(source="market", home="NY", away="BOS", p_home=.75, p_away=.25, start=start, home_pts=112., away_pts=106., margin=5.0, total=218.0,
                           sigma_margin=13., sigma_total=17., handicaps=[], totals=[], team_totals={}, bands=[], game_id="x")

def player(game, start):
    ladder = [dict(line=3.5, p_over=.7, hit=9, n=10, price=2.4, book="sportsbet", edge=.3, alt=None, boost=0, fade=0)]
    return dict(player_id="p" + game, game_id=game, player="X", team="A", opp="B", home=1, start=start, injury=None, starting=True, min=30, odds_age_h=1.0, odds_at=None,
                strip={"reb": ladder, "ast": [], "pts": []}, book={}, proj={"pts": 15., "reb": 8., "ast": 4.}, p_dd=.1, p_td=.01, show_dd=False, show_td=False,
                ladders={"pts": [], "reb": ladder, "ast": []}, sd={}, last10=[], evidence=[], absent=[], opp_factor={}, lineup_status="")

@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "model_status", lambda preds=None: (object(), None)); monkeypatch.setattr(app, "require_current_model", lambda preds=None: dict(alphas=ALPHAS, corr=CORR))
    pl = [player("old", PAST), player("new", FUTURE)]
    monkeypatch.setattr(app, "enriched", lambda: pl); monkeypatch.setattr(app, "projections", lambda: pl)
    monkeypatch.setattr(app, "game_preds", lambda: {"old": gp(PAST), "new": gp(FUTURE)})
    monkeypatch.setattr(odds, "load_lines", lambda c: {g: {"h2h": {"home": (2.0, "sportsbet"), "home_best": (2.0, "sportsbet")}} for g in ("old", "new")})
    monkeypatch.setattr(odds, "line_ages", lambda c: {"old": 1.0, "new": 1.0})
    return TestClient(app.app, raise_server_exceptions=False)

def test_has_started():
    assert app.has_started(PAST) and not app.has_started(FUTURE) and app.has_started("not a time")

def test_best_picks_leave_started_games_out(setup):
    legs = app.compute_best()["legs"]
    assert legs and all(l["game_id"] == "new" for l in legs)

def test_value_picks_leave_started_games_out_player_and_team(setup):
    picks = app.compute_value()["picks"]
    assert any(p["kind"] == "team" for p in picks) and any(p["kind"] == "player" for p in picks)
    assert all(p["game_id"] == "new" for p in picks)

@pytest.mark.parametrize("leg", [
    dict(game_id="old", player_id="pold", stat="reb", line=3.5, side="over"),
    dict(game_id="old", stat="ml", side="home"),
    dict(game_id="old", stat="spread", side="home", line=-4.5),
    dict(game_id="old", stat="total", side="over", line=218.5),
])
def test_bet_calculation_refused_for_started_game(setup, leg):
    r = setup.post("/api/parlay", json=dict(legs=[leg])); assert r.status_code == 400 and "already started" in r.json()["detail"]

def test_one_started_leg_blocks_the_whole_slip(setup):
    r = setup.post("/api/parlay", json=dict(legs=[dict(game_id="new", stat="ml", side="home"), dict(game_id="old", stat="ml", side="home")]))
    assert r.status_code == 400

@pytest.mark.parametrize("leg", [dict(game_id="new", player_id="pnew", stat="reb", line=3.5, side="over"), dict(game_id="new", stat="ml", side="home")])
def test_games_not_started_still_work(setup, leg):
    assert setup.post("/api/parlay", json=dict(legs=[leg])).status_code == 200
