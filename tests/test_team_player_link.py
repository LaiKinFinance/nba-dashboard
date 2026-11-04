"""Pick list: a team pick and a player pick from the same game are worked out together (4-season link), not as if unrelated."""
import numpy as np
import parlay

ALPHAS = {k: 0.1 for k in parlay.ALL}; CORR = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}
PRED = lambda pid, home: dict(player_id=pid, game_id="g", team="PHI" if home else "BOS", home=int(home), proj={"pts": 25.0, "reb": 8.0, "ast": 5.0, "fg3m": 2.0, "stl": 1.0, "blk": 1.0}, p_dd=.3, p_td=.05)
GP = dict(p_home=0.6, p_away=0.4, margin=3.0, sigma_margin=13.0, total=225.0, sigma_total=18.0, home_pts=114.0, away_pts=111.0, bands=[])
preds = {("g", "star"): PRED("star", True), ("g", "opp"): PRED("opp", False)}
over = lambda pid: dict(game_id="g", player_id=pid, stat="pra", line=37.5, side="over")
win_home = dict(game_id="g", stat="ml", side="home", line=0)

def joint(pl_legs, tm_legs):
    return parlay.parlay(pl_legs, preds, ALPHAS, CORR, team_legs=tm_legs, gps={"g": GP}, n=200000)

def test_team_win_and_own_star_go_together_and_opposing_star_against():
    alone = joint([over("star")], [])["joint"]; team = joint([], [win_home])["joint"]
    assert abs(team - 0.6) < 0.01                                   # the team pick keeps its own chance
    own, opp = joint([over("star")], [win_home])["joint"], joint([over("opp")], [win_home])["joint"]
    assert own > alone * team * 1.05 and opp < alone * team * 0.95
