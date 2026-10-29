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
    # team changes: previous / next team seen for each player, so traded players aren't counted as absent from the old team
    df = df.assign(pteam=df.groupby("player_id").team.shift(1), nteam=df.groupby("player_id").team.shift(-1))
    played = {(g, t): set(x.player_id) for (g, t), x in df.groupby(["game_id", "team"])}
    nxt = {(r.game_id, r.player_id): r.nteam for r in df[["game_id", "player_id", "nteam"]].itertuples()}
    feats, state = [], {}
    for team, tg in df.groupby("team"):
        games = list(dict.fromkeys(tg.game_id))             # this team's games in order
        gdates = tg.drop_duplicates("game_id").set_index("game_id").date
        by_game = {g: x for g, x in tg.groupby("game_id")}
        season_of = tg.drop_duplicates("game_id").set_index("game_id").season
        season_players = {sn: set(x.player_id) for sn, x in tg.groupby("season")}   # who really played for this team that season
        track_ = {}
        for i, g in enumerate(games):
            here = played[(g, team)]
            # who is a regular right now (state BEFORE this game), and absent from it
            a = dict(abs_min=0.0, abs_pts=0.0, abs_reb=0.0, abs_ast=0.0, abs_n=0)
            for pid, s in track_.items():
                if s["min"] >= REG_MIN and s["n"] >= 5 and i - s["last"] <= WINDOW and pid not in here and pid in season_players[season_of[g]]:
                    moved = s.get("moved")                    # traded to another team after this stint
                    if moved: continue
                    for c in COLS: a["abs_" + c] += s[c]
                    a["abs_n"] += 1
            feats.append(dict(game_id=g, team=team, **a))
            for r in by_game[g].itertuples():                 # update state AFTER the game
                s = track_.setdefault(r.player_id, dict(name=r.player, min=r.min, pts=r.pts, reb=r.reb, ast=r.ast, last=i, n=0))
                if s["n"]: 
                    for c in COLS: s[c] = (1 - ALPHA) * s[c] + ALPHA * getattr(r, c)
                s["n"] += 1; s["last"] = i; s["name"] = r.player
                nt = nxt[(g, r.player_id)]; s["moved"] = bool(isinstance(nt, str) and nt != team)
        state[team] = track_
    return pd.DataFrame(feats).set_index(["game_id", "team"]), state

def regulars(state_team, current_ids=None, games_back=WINDOW):
    """Tonight's regulars for a team from the tracked state (used to translate injury lists into absence features)."""
    out = {}
    for pid, s in state_team.items():
        if s["min"] >= REG_MIN and s["n"] >= 5 and not s.get("moved") and (current_ids is None or pid in current_ids): out[pid] = s
    return out
