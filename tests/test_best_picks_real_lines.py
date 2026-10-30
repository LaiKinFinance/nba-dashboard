"""Best picks use lines a bookmaker offers, skip tiny lines, and keep model-only lines out of the combined pick."""
import datetime as dt
import numpy as np, pytest
import app, parlay

NOW = dt.datetime.now(dt.timezone.utc); FUTURE = (NOW + dt.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
ALPHAS = {k: 0.1 for k in parlay.ALL}; CORR = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}

def entry(line, p, price=None): return dict(line=line, p_over=p, hit=9, n=10, price=price, book="sportsbet" if price else None, edge=0, alt=None, boost=0, fade=0)

def player(pid, game, book_reb=(), ladder_reb=(), age=1.0):
    return dict(player_id=pid, game_id=game, player=pid, team="A", opp="B", home=1, start=FUTURE, injury=None, starting=True, min=30, odds_age_h=age, odds_at=None,
                strip={"reb": list(book_reb), "ast": [], "pts": []}, book={"reb": list(book_reb)} if book_reb else {}, proj={"pts": 15., "reb": 8., "ast": 4.}, p_dd=.1, p_td=.01,
                show_dd=False, show_td=False, ladders={"pts": [], "reb": list(ladder_reb), "ast": []}, sd={}, last10=[], evidence=[], absent=[], opp_factor={}, lineup_status="")

@pytest.fixture
def best(monkeypatch):
    pl = [player("real", "g1", book_reb=[entry(3.5, .70, 1.8), entry(4.5, .45, 2.6)], ladder_reb=[entry(3.5, .70)]),
          player("modelonly", "g2", ladder_reb=[entry(4.5, .75)]),                     # no bookmaker line at all
          player("tiny", "g3", book_reb=[entry(0.5, .83, 1.05)], ladder_reb=[entry(0.5, .83)]),   # only a 0.5 line
          player("stale", "g4", book_reb=[entry(3.5, .72, 1.7)], ladder_reb=[entry(2.5, .78)], age=30.0)]   # price is too old to trust
    monkeypatch.setattr(app, "model_status", lambda preds=None: (object(), None)); monkeypatch.setattr(app, "require_current_model", lambda preds=None: dict(alphas=ALPHAS, corr=CORR))
    monkeypatch.setattr(app, "enriched", lambda: pl); monkeypatch.setattr(app, "playing_picks", lambda: [])
    return app.compute_best()

def test_the_combined_pick_uses_only_lines_a_bookmaker_offers(best):
    assert [l["player_id"] for l in best["combo"]["legs"]] == ["real"] and best["combo"]["legs"][0]["price"] == 1.8

def test_a_leg_with_no_bookmaker_line_is_never_shown(best):
    assert {l["player_id"] for l in best["legs"]} == {"real"} and all(l["real"] and l["price"] for l in best["legs"])

def test_a_stale_price_means_no_pick(best):
    assert "stale" not in {l["player_id"] for l in best["legs"]}

def test_tiny_lines_are_never_suggested(best):
    assert all(l["line"] >= app.MIN_BEST_LINE for l in best["legs"]) and "tiny" not in {l["player_id"] for l in best["legs"]}


def _p(pid, ast=(), reb=(), start=None, **k):
    d = player(pid, "g1", book_reb=list(reb)); d["ladders"]["ast"] = list(ast); d["book"]["ast"] = list(ast); d["start"] = start or FUTURE; d.update(k); return d

def test_each_game_gets_three_picks_from_three_players_with_evidence():
    players = [_p("a", reb=[entry(3.5, .70, 1.8)], ast=[entry(2.5, .80, 1.4)]), _p("b", reb=[entry(4.5, .66, 2.0)]), _p("c", ast=[entry(3.5, .72, 1.7)]), _p("d", reb=[entry(2.5, .64, 1.6)])]
    for p in players: p["proj"] = {"pts": 15., "reb": 8., "ast": 4.}; p["last10"] = [dict(reb=5, ast=3, pts=10)] * 10
    out = app.best3_for_game(players)
    assert len(out) == 3 and len({o["player_id"] for o in out}) == 3                      # one pick per player, never the same player twice
    assert all(o["evidence"][0].startswith("Model gives a") and any("last 10 games" in e for e in o["evidence"]) for o in out)

def test_no_picks_for_a_game_that_has_started():
    past = (NOW - dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ")
    assert app.best3_for_game([_p("a", reb=[entry(3.5, .70, 1.8)], start=past)]) == []

def test_no_picks_at_all_when_no_bookmaker_line_was_found():
    p = player("x", "g1", ladder_reb=[entry(3.5, .70)])                       # only the model's own ladder, no bookmaker price
    p["book"] = {}; assert app.best3_for_game([p]) == [] and app._best_candidates([p]) == []
