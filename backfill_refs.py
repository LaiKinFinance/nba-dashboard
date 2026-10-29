"""Collect referee crews and team foul / free-throw / shot-attempt totals per game from ESPN box scores (for referee analysis)."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import ingest

TEAM_COLS = {"home_fta": "REAL", "away_fta": "REAL", "home_pf": "REAL", "away_pf": "REAL", "home_fga": "REAL", "away_fga": "REAL"}

def ensure(con):
    have = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    for c, t in TEAM_COLS.items():
        if c not in have: con.execute(f"ALTER TABLE games ADD COLUMN {c} {t}")
    con.execute("CREATE TABLE IF NOT EXISTS officials(game_id TEXT, name TEXT, ord INTEGER, PRIMARY KEY(game_id, name))")

def parse(js, home, away):
    refs = [(o["displayName"], o.get("order", 0)) for o in (js.get("gameInfo", {}).get("officials") or [])]
    stat = {}
    for t in js.get("boxscore", {}).get("teams", []):
        ab = t["team"]["abbreviation"]; d = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
        ft = (d.get("freeThrowsMade-freeThrowsAttempted") or "0-0").split("-"); fg = (d.get("fieldGoalsMade-fieldGoalsAttempted") or "0-0").split("-")
        stat[ab] = dict(fta=float(ft[1]), pf=float(d.get("fouls") or 0), fga=float(fg[1]))
    return refs, stat

def run():
    con = sqlite3.connect(ingest.DB); ensure(con)
    todo = [r for r in con.execute("SELECT game_id, home, away FROM games WHERE home_fta IS NULL ORDER BY date DESC")]
    print(len(todo), "games to fill", flush=True)
    def work(r):
        try: return r, parse(ingest.get(ingest.SUM, {"event": r[0]}), r[1], r[2])
        except Exception: return r, None
    n = 0
    with ThreadPoolExecutor(6) as ex:
        for (gid, h, a), res in ex.map(work, todo):
            if not res: continue
            refs, st = res
            if h in st and a in st:
                con.execute("UPDATE games SET home_fta=?, away_fta=?, home_pf=?, away_pf=?, home_fga=?, away_fga=? WHERE game_id=?",
                            (st[h]["fta"], st[a]["fta"], st[h]["pf"], st[a]["pf"], st[h]["fga"], st[a]["fga"], gid))
            con.executemany("INSERT OR REPLACE INTO officials VALUES(?,?,?)", [(gid, nm, o) for nm, o in refs]); n += 1
            if n % 100 == 0: con.commit(); print(n, flush=True)
    con.commit(); print("filled", n)

if __name__ == "__main__":
    run()
