"""Bookmaker odds from The Odds API (au region; Sportsbet preferred). Needs ODDS_API_KEY; without it the app just shows model prices.

Credit cost on the API = markets x regions per event call, so we fetch each upcoming game once (not on every page view)
and stop when the remaining credits get low.
"""
import os, re, sqlite3, time, unicodedata, datetime as dt
import requests

BASE = "https://api.the-odds-api.com/v4"
MARKETS = os.environ.get("ODDS_MARKETS", "player_points,player_rebounds,player_assists,player_double_double,player_triple_double")
STAT_OF = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
           "player_double_double": "dd", "player_triple_double": "td"}
PREFERRED = "sportsbet"
DB = os.path.join(os.environ.get("DATA_DIR", "data"), "nba.db")
MIN_CREDITS = int(os.environ.get("ODDS_MIN_CREDITS", "20"))
GAMES_PER_RUN = int(os.environ.get("ODDS_GAMES_PER_DAY", "10"))
REFETCH_HOURS = 10  # a game's odds are refreshed at most this often

def available():
    return bool(os.environ.get("ODDS_API_KEY"))

SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
def norm_name(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    toks = [t for t in re.sub(r"[^a-z ]", " ", n).split() if t not in SUFFIX]
    return " ".join(toks)

def init(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS book_odds(game_id TEXT, player_id TEXT, stat TEXT, line REAL, side TEXT, price REAL, book TEXT, fetched_at TEXT,
        PRIMARY KEY(game_id, player_id, stat, line, side, book));
    CREATE TABLE IF NOT EXISTS odds_status(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS odds_fetch(game_id TEXT PRIMARY KEY, fetched_at TEXT);""")

def _get(path, **params):
    r = requests.get(f"{BASE}{path}", params={"apiKey": os.environ["ODDS_API_KEY"], **params}, timeout=30)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")

def parse_event(js, name_to_pid):
    """Turn one event-odds response into rows (player_id, stat, line, side, price, book). Unknown players are skipped."""
    rows = []
    for b in js.get("bookmakers", []):
        for m in b.get("markets", []):
            stat = STAT_OF.get(m["key"])
            if not stat: continue
            for o in m.get("outcomes", []):
                pid = name_to_pid.get(norm_name(o.get("description") or ""))
                side = (o.get("name") or "").lower()
                if not pid or side not in ("over", "under", "yes"): continue
                if stat in ("dd", "td"): line = 0.0
                else:
                    if o.get("point") is None: continue
                    line = float(o["point"])
                rows.append((pid, stat, line, "over" if side == "yes" else side, float(o["price"]), b["key"]))
    return rows

def refresh(games, players_by_game):
    """games: upcoming ESPN games [{game_id,start,home,away}] ; players_by_game: {game_id: [(player_id, player_name)]}."""
    import slate as sl
    if not available(): return dict(ok=False, msg="No ODDS_API_KEY set")
    con = sqlite3.connect(DB); init(con)
    now = dt.datetime.now(dt.timezone.utc)
    try:
        events, remaining = _get("/sports/basketball_nba/events")
    except Exception as e:
        return dict(ok=False, msg=f"events call failed: {e}")
    nick = {k: v.lower() for k, v in sl.TEAM_NICK.items()}
    def abbr(full):
        f = full.lower()
        for a, n in nick.items():
            if f.endswith(n): return a
    ev_by = {(abbr(e["home_team"]), abbr(e["away_team"])): e for e in events}
    fetched = 0
    for g in sorted(games, key=lambda x: x["start"]):
        start = dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00"))
        if start - now > dt.timedelta(hours=30) or start < now: continue
        row = con.execute("SELECT fetched_at FROM odds_fetch WHERE game_id=?", (g["game_id"],)).fetchone(); last = row[0] if row else None  # remembered even when a fetch returned no prices
        if last and now - dt.datetime.fromisoformat(last) < dt.timedelta(hours=REFETCH_HOURS): continue
        ev = ev_by.get((g["home"], g["away"]))
        if not ev or fetched >= GAMES_PER_RUN: continue
        if remaining is not None and int(float(remaining)) < MIN_CREDITS: break
        try:
            js, remaining = _get(f"/sports/basketball_nba/events/{ev['id']}/odds", regions="au", markets=MARKETS, oddsFormat="decimal")
        except Exception as e:
            con.execute("INSERT OR REPLACE INTO odds_status VALUES('error',?)", (str(e)[:200],)); con.commit(); continue
        n2p = {norm_name(nm): pid for pid, nm in players_by_game.get(g["game_id"], [])}
        rows = parse_event(js, n2p)
        con.execute("DELETE FROM book_odds WHERE game_id=?", (g["game_id"],))
        con.execute("INSERT OR REPLACE INTO odds_fetch VALUES(?,?)", (g["game_id"], now.isoformat()))
        con.executemany("INSERT OR REPLACE INTO book_odds VALUES(?,?,?,?,?,?,?,?)",
                        [(g["game_id"], pid, st, ln, sd, pr, bk, now.isoformat()) for pid, st, ln, sd, pr, bk in rows])
        fetched += 1
    for k, v in (("remaining", remaining), ("last_run", now.isoformat()), ("error", "")):
        con.execute("INSERT OR REPLACE INTO odds_status VALUES(?,?)", (k, str(v)))
    con.commit()
    return dict(ok=True, fetched=fetched, remaining=remaining)

def load(con):
    """{player_id: {stat: {line: {over:(price,book), under:(price,book)}}}} preferring Sportsbet, else best price."""
    try: rows = con.execute("SELECT game_id,player_id,stat,line,side,price,book FROM book_odds").fetchall()
    except sqlite3.OperationalError: return {}
    out = {}
    for gid, pid, stat, line, side, price, book in rows:
        slot = out.setdefault((gid, pid), {}).setdefault(stat, {}).setdefault(line, {})
        cur = slot.get(side)
        better = cur is None or (book == PREFERRED and cur[1] != PREFERRED) or (cur[1] != PREFERRED and book != PREFERRED and price > cur[0])
        if better: slot[side] = (price, book)
    return out

def status(con):
    try: return dict(con.execute("SELECT k,v FROM odds_status").fetchall())
    except sqlite3.OperationalError: return {}
