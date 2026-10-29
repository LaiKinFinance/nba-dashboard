"""Training and prediction must see the same history; unplayed games must not leak into each other's features."""
import numpy as np, pandas as pd
import features, predict

def _history():
    rows, gid = [], 0
    teams = ["AAA", "BBB", "CCC"]
    for day in range(40):
        a, b = [("AAA", "BBB"), ("BBB", "CCC"), ("CCC", "AAA")][day % 3]
        date = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d"); gid += 1
        for team, opp, home in ((a, b, 1), (b, a, 0)):
            for k in range(8):
                rows.append(dict(game_id=f"g{gid}", player_id=f"{team}{k}", player=f"{team} P{k}", team_id=team, team=team, opp=opp, home=home, date=date, season="2026",
                                 starter=int(k < 5), min=30 - k, pts=20 - k + day % 4, reb=6, ast=4, stl=1, blk=1, tov=2, fg3m=2, plusminus=0, _upcoming=0))
    return pd.DataFrame(rows)

def _placeholders(specs):
    base = _history(); rows = []
    for gid, team, opp, home, date in specs:
        for k in range(3):
            rows.append(dict(game_id=gid, player_id=f"{team}{k}", player=f"{team} P{k}", team_id=team, team=team, opp=opp, home=home, date=date, season="2026", starter=0,
                             min=0., pts=0, reb=0, ast=0, stl=0, blk=0, tov=0, fg3m=0, plusminus=0, _upcoming=1))
    return base, pd.DataFrame(rows)

def test_other_unplayed_games_do_not_change_opponent_defence():
    base, ph = _placeholders([("x1", "AAA", "BBB", 1, "2026-02-15"), ("x2", "CCC", "BBB", 1, "2026-02-16")])
    only2 = ph[ph.game_id == "x2"]
    both = features.build(pd.concat([base, ph], ignore_index=True)); clean = features.build(pd.concat([base, only2], ignore_index=True))
    a = both[(both._upcoming == 1) & (both.game_id == "x2")].sort_values("player_id")[["opp_allow_pts", "opp_allow_reb", "opp_allow_ast"]].values
    b = clean[clean._upcoming == 1].sort_values("player_id")[["opp_allow_pts", "opp_allow_reb", "opp_allow_ast"]].values
    assert np.allclose(a, b) and not np.isnan(a).any()

def test_completed_games_features_ignore_placeholders():
    base, ph = _placeholders([("x1", "AAA", "BBB", 1, "2026-02-15")])
    without = features.build(base.copy()); with_ph = features.build(pd.concat([base, ph], ignore_index=True))
    w = with_ph[with_ph._upcoming == 0].sort_values(["game_id", "player_id"]).reset_index(drop=True); o = without.sort_values(["game_id", "player_id"]).reset_index(drop=True)
    cols = ["opp_allow_pts", "opp_allow_reb", "opp_allow_ast", "pts_m10", "min_m5", "rest"]
    pd.testing.assert_frame_equal(w[cols], o[cols])

def test_players_who_left_keep_their_history():
    raw = _history(); cur = pd.DataFrame({"player_id": ["AAA0", "AAA1"], "cur_team": ["AAA", "AAA"]})
    out = predict.attach_current_team(raw, cur)
    assert len(out) == len(raw) and out.cur_team.notna().sum() < len(out)

def test_second_game_rest_counts_from_first_scheduled_game():
    f = pd.DataFrame([dict(player_id="p", date="2026-10-30", game_id="x1", rest=3.0, b2b=0), dict(player_id="p", date="2026-10-31", game_id="x2", rest=3.0, b2b=0)])
    out = predict.apply_scheduled_rest(f)
    assert list(out.rest) == [3.0, 1.0] and list(out.b2b) == [0, 1]
