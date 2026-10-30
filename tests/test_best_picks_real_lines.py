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

def test_model_only_lines_are_listed_after_real_ones_and_flagged(best):
    by = {l["player_id"]: l for l in best["legs"]}
    assert by["real"]["real"] is True and by["modelonly"]["real"] is False and by["modelonly"]["price"] is None
    assert [l["real"] for l in best["legs"]] == sorted((l["real"] for l in best["legs"]), reverse=True)

def test_a_stale_price_counts_as_no_bookmaker_line(best):
    assert {l["player_id"]: l["real"] for l in best["legs"]}["stale"] is False

def test_tiny_lines_are_never_suggested(best):
    assert all(l["line"] >= app.MIN_BEST_LINE for l in best["legs"]) and "tiny" not in {l["player_id"] for l in best["legs"]}
