"""Leak-free per-player-game features: everything is computed from games strictly before the row's game."""
import numpy as np, pandas as pd
from top3 import load

STATS = ["min", "pts", "reb", "ast"]

def build(df=None):
    df = (load() if df is None else df).sort_values(["date", "game_id"]).reset_index(drop=True)
    df["pra"] = df.pts + df.reb + df.ast
    df["td"] = ((df.pts >= 10).astype(int) + (df.reb >= 10) + (df.ast >= 10) >= 3).astype(int)
    df["dd"] = ((df.pts >= 10).astype(int) + (df.reb >= 10) + (df.ast >= 10) >= 2).astype(int)
    g = df.groupby("player_id", group_keys=False)
    cols = STATS + ["pra", "td", "dd", "starter"]
    for c in cols:
        s = g[c].shift(1)  # previous games only
        for w in (5, 10, 20):
            df[f"{c}_m{w}"] = s.groupby(df.player_id).transform(lambda x: x.rolling(w, min_periods=2).mean())
        if c in ("pts", "reb", "ast", "min"):
            df[f"{c}_sd10"] = s.groupby(df.player_id).transform(lambda x: x.rolling(10, min_periods=3).std())
        df[f"{c}_exp"] = s.groupby(df.player_id).transform(lambda x: x.expanding(min_periods=3).mean())
    # td rate over a long window (rare event)
    df["td_m40"] = g["td"].shift(1).groupby(df.player_id).transform(lambda x: x.rolling(40, min_periods=5).mean())
    df["prior_games"] = g.cumcount()
    prev_date = g["date"].shift(1)
    df["rest"] = (pd.to_datetime(df.date) - pd.to_datetime(prev_date)).dt.days.clip(upper=10)
    df["b2b"] = (df.rest == 1).astype(int)

    # opponent defence: what each defending team has allowed per game (rolling 15, shifted), relative to league
    tg = df.groupby(["game_id", "team", "opp", "date"], as_index=False)[["pts", "reb", "ast"]].sum()
    dfn = tg.rename(columns={"opp": "def_team"}).sort_values(["date", "game_id"])
    lg = dfn.groupby("date")[["pts", "reb", "ast"]].mean()
    lg = lg.expanding().mean().shift(1)
    for c in ("pts", "reb", "ast"):
        dfn[f"opp_allow_{c}"] = dfn.groupby("def_team")[c].transform(lambda x: x.shift(1).rolling(30, min_periods=8).mean())
        ratio = dfn[f"opp_allow_{c}"] / dfn["date"].map(lg[c])
        # opponent defence is noisy: shrink toward league average and clip so trees never see wild values
        dfn[f"opp_allow_{c}"] = (1 + 0.5 * (ratio - 1)).clip(0.93, 1.07)
    dfn = dfn[["game_id", "def_team", "opp_allow_pts", "opp_allow_reb", "opp_allow_ast"]]
    df = df.merge(dfn, left_on=["game_id", "opp"], right_on=["game_id", "def_team"], how="left").drop(columns="def_team")
    return df

FEATURES = [f"{c}_{s}" for c in ["min", "pts", "reb", "ast", "pra", "dd", "td", "starter"] for s in ("m5", "m10", "m20", "exp")] \
    + ["pts_sd10", "reb_sd10", "ast_sd10", "min_sd10", "td_m40", "home", "rest", "b2b", "prior_games",
       "opp_allow_pts", "opp_allow_reb", "opp_allow_ast"]
