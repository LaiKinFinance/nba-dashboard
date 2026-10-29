"""Read-only extended audit. Run with the repository's dependencies installed."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest import REAL_TEAMS


def main():
    db = Path(os.environ.get("DATA_DIR", "data")) / "nba.db"
    with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as con:
        g = pd.read_sql("SELECT * FROM games", con)
        p = pd.read_sql("SELECT * FROM player_games", con)
        print("integrity", con.execute("PRAGMA integrity_check").fetchone()[0])
        print("indexes", con.execute("PRAGMA index_list(player_games)").fetchall())
        print("journal", con.execute("PRAGMA journal_mode").fetchone()[0])
        for table in ("projections", "game_preds", "book_odds", "book_lines"):
            try:
                print("table", table, con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            except sqlite3.OperationalError:
                print("table", table, "missing")
    print("date range", g.date.min(), g.date.max())
    print("season types", g.groupby(["season", "season_type"]).size().to_string())
    print("market coverage", g.groupby(["season", "season_type"])[["ou", "spread_home", "ml_home", "ml_away"]].agg(lambda x: x.notna().mean()).round(3).to_string())
    joined = p.merge(g, on="game_id", suffixes=("", "_g"), how="left")
    wrong = joined[(joined.date != joined.date_g) | (joined.team != joined.home_g).where(joined.home == 1, joined.team != joined.away) |
                   (joined.opp != joined.away).where(joined.home == 1, joined.opp != joined.home_g)]
    print("orphan or inconsistent player rows", len(wrong))
    print("missing boxes", len(set(g.game_id) - set(p.game_id)))
    print("same team opponents", len(g[g.home == g.away]))
    print("unexpected teams", sorted((set(g.home) | set(g.away) | set(p.team)) - REAL_TEAMS))
    print("nonpositive minutes", int((p['min'] <= 0).sum()), "over48", int((p['min'] > 48).sum()), "max", p['min'].max())
    print("subminute rows", int((p['min'] < 1).sum()))
    tm = p.groupby(["game_id", "team"]).agg(minutes=("min", "sum"), points=("pts", "sum"), players=("player_id", "size")).reset_index()
    tm = tm.merge(g[["game_id", "date", "home", "home_pts", "away_pts"]], on="game_id")
    tm["score"] = tm.home_pts.where(tm.team == tm.home, tm.away_pts)
    print("box anomalies", tm[(tm.minutes < 235) | ((tm.points-tm.score).abs() > 2)].to_json(orient="records"))
    norm = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s.lower()) if c.isascii() and c.isalnum())
    names = p.assign(normal=p.player.map(norm)).groupby("normal").player_id.nunique()
    print("normalized names with multiple ids", names[names > 1].to_dict())
    seq = p.sort_values(["date", "game_id"]).copy()
    seq["previous_team"] = seq.groupby("player_id").team.shift()
    changes = seq[seq.previous_team.notna() & (seq.team != seq.previous_team)]
    print("team changes", len(changes), changes[["date", "player", "previous_team", "team"]].tail(30).to_json(orient="records"))
    for season, x in g[g.season_type == 2].groupby("season"):
        n = pd.concat([x.home, x.away]).value_counts()
        print("schedule", season, len(x), n.to_dict())
    print("largest totals", g.assign(total=g.home_pts+g.away_pts).nlargest(8, "total")[["game_id", "date", "home", "away", "home_pts", "away_pts", "season_type"]].to_json(orient="records"))


if __name__ == "__main__":
    main()
