"""On the player cards, whole-number bookmaker lines are not priced or offered; half-point lines still are."""
import sqlite3
import numpy as np, pytest
import app, odds

def pred():
    ladder = [dict(line=L, p_over=.5) for L in (3.5, 4.5, 5.5)]
    return dict(player_id="p", game_id="g", player="X", team="A", opp="B", home=1, start="2999-01-01T00:00Z", injury=None, starting=True, min=30, lineup_status="",
                proj={"pts": 15., "reb": 5.0, "ast": 4., "fg3m": 1.5, "stl": 1., "blk": .5}, proj_full={}, proj_base={}, p_dd=.1, p_td=.01, combo={}, alphas={k: .1 for k in ("pts", "reb", "ast", "fg3m", "stl", "blk")},
                ladders={"pts": ladder, "reb": ladder, "ast": ladder}, sd={}, evidence=[], absent=[], opp_factor={}, model_id="M")

@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "e.db")); monkeypatch.setattr(odds, "DB", str(tmp_path / "e.db")); app.init()
    c = sqlite3.connect(app.DB); c.execute("CREATE TABLE player_games(game_id TEXT, player_id TEXT, date TEXT, opp TEXT, home INT, min REAL, pts INT, reb INT, ast INT, fg3m INT, stl INT, blk INT)"); c.commit()
    monkeypatch.setattr(app, "guarded_projections", lambda: [pred()]); monkeypatch.setattr(app, "model_status", lambda preds=None: ({"alphas": {}}, None))
    book = {("g", "p"): {"reb": {4.5: {"over": (1.9, "sportsbet"), "over_best": (1.9, "sportsbet")}, 5.0: {"over": (1.8, "sportsbet"), "over_best": (1.8, "sportsbet")},
                                  6.0: {"over": (2.5, "sportsbet"), "over_best": (2.5, "sportsbet")}}}}
    monkeypatch.setattr(odds, "load", lambda con: book); monkeypatch.setattr(odds, "fetch_ages", lambda con: {"g": 1.0}); monkeypatch.setattr(odds, "fetch_times", lambda con: {"g": "t"})
    app._enr.update(key=None, val=None)

def test_whole_number_lines_are_not_priced_but_half_point_lines_are(setup):
    p = app._enriched_build()[0]
    priced = [l["line"] for l in p["strip"]["reb"] if l["price"]]
    assert priced == [4.5]                                               # 5.0 and 6.0 are whole numbers: no price, no chance, no VALUE
    assert p["level_lines"] == {"reb": [5.0, 6.0]}
    assert all(l["line"] % 1 == 0.5 for l in p["strip"]["reb"])         # the strip itself only offers half-point lines
    assert "refund" not in str(p["strip"])

def test_no_value_pick_ever_comes_from_a_whole_number_line(setup, monkeypatch):
    monkeypatch.setattr(app, "enriched", lambda: app._enriched_build()); monkeypatch.setattr(app, "game_preds", lambda: {})
    monkeypatch.setattr(odds, "load_lines", lambda c: {}); monkeypatch.setattr(odds, "line_ages", lambda c: {})
    picks = app.compute_value()["picks"]
    assert all(float(x["line"]) % 1 == 0.5 for x in picks if x["kind"] == "player")
