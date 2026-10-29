"""Pick each team's top-3 players by recent production (min + usage proxy + PRA)."""
import os
import sqlite3
import pandas as pd

DB = os.path.join(os.environ.get("DATA_DIR", "data"), "nba.db")

def load():
    con = sqlite3.connect(DB)
    return pd.read_sql("SELECT * FROM player_games ORDER BY date, game_id", con)

def top3(df, as_of=None, window=15, min_games=3):
    """Rank by mean of last `window` games: score = PRA + 0.5*minutes. Season blend handled by caller."""
    if as_of:
        df = df[df.date < as_of]
    df = df.assign(pra=df.pts + df.reb + df.ast)
    # team of record = most recent team
    recent = df.sort_values("date").groupby("player_id").tail(window)
    g = recent.groupby(["player_id", "player"]).agg(team=("team", "last"), games=("pra", "size"),
                                                    min=("min", "mean"), pra=("pra", "mean")).reset_index()
    g = g[g.games >= min_games]
    g["score"] = g.pra + 0.5 * g["min"]
    g["rank"] = g.groupby("team").score.rank(ascending=False, method="first")
    return g[g["rank"] <= 3].sort_values(["team", "rank"])

if __name__ == "__main__":
    t = top3(load())
    print(t.to_string(index=False))
