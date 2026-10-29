"""Teammate-absence tracking.

A team's "regulars" are players averaging 18+ minutes recently (exponentially weighted, per team). For every historical game we record
which regulars did not play and how much scoring/rebounding/passing/minutes that removed. The same bookkeeping, using the injury and
lineup feeds, describes tonight's absences, so training and prediction use identical features."""
import numpy as np, pandas as pd

REG_MIN, WINDOW, ALPHA = 18.0, 25, 0.25   # regular threshold, how many team games back a player still counts, EWM weight
COLS = ["min", "pts", "reb", "ast"]
ABS = ["abs_min", "abs_pts", "abs_reb", "abs_ast", "abs_n"]

def track(df):
    """df: player_games. Returns (features, state): features indexed by (game_id, team) with ABS columns;
    state[team] = {player_id: {name, min, pts, reb, ast, last, n}} as of each team's latest game (for tonight's slate)."""
    df = df.sort_values(["date", "game_id"])
    feats, state, counts, seasons = [], {}, {}, {}
    # Process the league in order. A team change becomes known only when observed,
    # never from the player's next appearance or a full-season membership list.
    for g, game in df.groupby("game_id", sort=False):
        for r in game.itertuples():
            for old_team, roster in state.items():
                if old_team != r.team:
                    roster.pop(r.player_id, None)
        for team, rows in game.groupby("team", sort=False):
            season = rows.season.iloc[0]
            if seasons.get(team) != season:
                state[team] = {}; counts[team] = 0; seasons[team] = season
            track_ = state[team]; i = counts[team]
            here = set(rows.player_id)
            # who is a regular right now (state BEFORE this game), and absent from it
            a = dict(abs_min=0.0, abs_pts=0.0, abs_reb=0.0, abs_ast=0.0, abs_n=0)
            for pid, s in track_.items():
                if s["min"] >= REG_MIN and s["n"] >= 5 and i - s["last"] <= WINDOW and pid not in here:
                    for c in COLS: a["abs_" + c] += s[c]
                    a["abs_n"] += 1
            feats.append(dict(game_id=g, team=team, **a))
            for r in rows.itertuples():                 # update state AFTER the game
                s = track_.setdefault(r.player_id, dict(name=r.player, min=r.min, pts=r.pts, reb=r.reb, ast=r.ast, last=i, n=0))
                if s["n"]: 
                    for c in COLS: s[c] = (1 - ALPHA) * s[c] + ALPHA * getattr(r, c)
                s["n"] += 1; s["last"] = i; s["name"] = r.player
            counts[team] = i + 1
            for s in track_.values(): s["age"] = counts[team] - s["last"]
    return pd.DataFrame(feats, columns=["game_id", "team"] + ABS).set_index(["game_id", "team"]), state

def regulars(state_team, current_ids=None, games_back=WINDOW):
    """Tonight's regulars for a team from the tracked state (used to translate injury lists into absence features)."""
    out = {}
    for pid, s in state_team.items():
        if s["min"] >= REG_MIN and s["n"] >= 5 and s.get("age", 0) <= games_back and not s.get("moved") and (current_ids is None or pid in current_ids): out[pid] = s
    return out
