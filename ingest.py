"""Pull NBA box scores from ESPN's public JSON into SQLite (data/nba.db)."""
import os, sqlite3, sys, json, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests

H = {"User-Agent": "Mozilla/5.0"}
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
DB = os.path.join(os.environ.get("DATA_DIR", "data"), "nba.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS games(game_id TEXT PRIMARY KEY, date TEXT, season TEXT, season_type INT,
  home_id TEXT, home TEXT, away_id TEXT, away TEXT, home_pts INT, away_pts INT);
CREATE TABLE IF NOT EXISTS player_games(game_id TEXT, player_id TEXT, player TEXT, team_id TEXT, team TEXT,
  opp TEXT, home INT, date TEXT, season TEXT, starter INT, min REAL, pts INT, reb INT, ast INT, stl INT, blk INT,
  tov INT, fg3m INT, plusminus INT, PRIMARY KEY(game_id, player_id));
"""

LINE_COLS = {"ou": "REAL", "spread_home": "REAL", "ml_home": "REAL", "ml_away": "REAL"}

def ensure_columns(con):
    have = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    for c, t in LINE_COLS.items():
        if c not in have: con.execute(f"ALTER TABLE games ADD COLUMN {c} {t}")

def market_lines(js):
    """ESPN's pickcenter (DraftKings): total, home spread (negative = home favoured), moneylines. Empty dict if absent."""
    pc = (js.get("pickcenter") or [None])[0]
    if not pc: return {}
    num = lambda x: float(x) if isinstance(x, (int, float)) else None
    return dict(ou=num(pc.get("overUnder")), spread_home=num(pc.get("spread")),
                ml_home=num((pc.get("homeTeamOdds") or {}).get("moneyLine")), ml_away=num((pc.get("awayTeamOdds") or {}).get("moneyLine")))

def get(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))

# the 30 real NBA teams (ESPN abbreviations); All-Star weekend games use teams like STARS/STRIPES/WORLD and must be ignored
REAL_TEAMS = {"ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GS","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NO","NY","OKC","ORL","PHI","PHX","POR","SAC","SA","TOR","UTAH","WSH"}

def day_games(d):
    js = get(SB, {"dates": f"{d:%Y%m%d}", "limit": 100})
    out = []
    for e in js["events"]:
        if e["status"]["type"]["name"] != "STATUS_FINAL":
            continue
        c = e["competitions"][0]["competitors"]
        home = next(x for x in c if x["homeAway"] == "home")
        away = next(x for x in c if x["homeAway"] == "away")
        if home["team"]["abbreviation"] not in REAL_TEAMS or away["team"]["abbreviation"] not in REAL_TEAMS: continue
        out.append(dict(game_id=e["id"], date=e["date"][:10], season=str(e["season"]["year"]),
                        season_type=e["season"]["type"], home_id=home["team"]["id"], home=home["team"]["abbreviation"],
                        away_id=away["team"]["id"], away=away["team"]["abbreviation"],
                        home_pts=int(home["score"]), away_pts=int(away["score"])))
    return out

def final_games(start, end):
    """Completed games between two dates (inclusive); ESPN's scoreboard only takes one day per call."""
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    with ThreadPoolExecutor(6) as ex:
        for games in ex.map(day_games, days):
            yield from games

def parse_box(g):
    js = get(SUM, {"event": g["game_id"]})
    rows = []
    for t in js["boxscore"]["players"]:
        tid = t["team"]["id"]; is_home = tid == g["home_id"]
        team = g["home"] if is_home else g["away"]; opp = g["away"] if is_home else g["home"]
        for grp in t["statistics"]:
            keys = grp["keys"]
            for a in grp["athletes"]:
                if a.get("didNotPlay") or not a["stats"]:
                    continue
                s = dict(zip(keys, a["stats"]))
                try:
                    mins = float(s["minutes"])
                except ValueError:
                    continue
                if mins <= 0:
                    continue
                pm = s.get("plusMinus", "0").replace("+", "")
                rows.append((g["game_id"], a["athlete"]["id"], a["athlete"]["displayName"], tid, team, opp,
                             int(is_home), g["date"], g["season"], int(a.get("starter", False)), mins,
                             int(s["points"]), int(s["rebounds"]), int(s["assists"]), int(s["steals"]),
                             int(s["blocks"]), int(s["turnovers"]),
                             int(s["threePointFieldGoalsMade-threePointFieldGoalsAttempted"].split("-")[0]),
                             int(pm) if pm.lstrip("-").isdigit() else 0))
    return rows, market_lines(js)

def run(start, end):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB); con.executescript(SCHEMA); ensure_columns(con)
    have = {r[0] for r in con.execute("SELECT game_id FROM games")}
    todo = [g for g in final_games(start, end) if g["game_id"] not in have]
    print(f"{len(todo)} new games {start}..{end}", flush=True)
    def work(g):
        try: return g, parse_box(g)
        except Exception as ex: print("fail", g["game_id"], ex, flush=True); return g, None
    done = 0
    with ThreadPoolExecutor(6) as ex:
        for g, res in ex.map(work, todo):
            if res is None: continue
            rows, lines = res
            con.execute("INSERT OR REPLACE INTO games VALUES(:game_id,:date,:season,:season_type,:home_id,:home,:away_id,:away,:home_pts,:away_pts)", g)
            con.executemany("INSERT OR REPLACE INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            if lines: con.execute("UPDATE games SET ou=:ou, spread_home=:spread_home, ml_home=:ml_home, ml_away=:ml_away WHERE game_id=:gid", dict(lines, gid=g["game_id"]))
            done += 1
            if done % 100 == 0: con.commit(); print(done, flush=True)
    con.commit()

if __name__ == "__main__":
    s = dt.date.fromisoformat(sys.argv[1]); e = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date.today()
    run(s, e)
