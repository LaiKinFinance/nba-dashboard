"""Bookmaker odds from The Odds API (au region; Sportsbet preferred). Needs ODDS_API_KEY; without it the app just shows model prices.

Credit cost on the API = markets x regions per event call, so we fetch each upcoming game once (not on every page view)
and stop when the remaining credits get low.
"""
import os, re, sqlite3, time, unicodedata, datetime as dt, math
import requests

BASE = "https://api.the-odds-api.com/v4"
MARKETS = os.environ.get("ODDS_MARKETS", "player_points,player_rebounds,player_assists,player_double_double,player_triple_double")
STAT_OF = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
           "player_points_alternate": "pts", "player_rebounds_alternate": "reb", "player_assists_alternate": "ast",
           "player_points_rebounds_assists": "pra", "player_points_rebounds": "pr", "player_points_assists": "pa",
           "player_threes": "fg3m", "player_threes_alternate": "fg3m", "player_steals": "stl", "player_blocks": "blk",
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
    CREATE TABLE IF NOT EXISTS odds_fetch(game_id TEXT PRIMARY KEY, fetched_at TEXT);
    CREATE TABLE IF NOT EXISTS odds_archive(fetched_at TEXT, game_id TEXT, player_id TEXT, stat TEXT, line REAL, side TEXT, price REAL, book TEXT);
    CREATE INDEX IF NOT EXISTS odds_archive_k ON odds_archive(game_id, player_id, stat, fetched_at);
    CREATE TABLE IF NOT EXISTS lines_archive(fetched_at TEXT, game_id TEXT, market TEXT, side TEXT, line REAL, price REAL, book TEXT);
    CREATE INDEX IF NOT EXISTS lines_archive_k ON lines_archive(game_id, market, fetched_at);
    CREATE TABLE IF NOT EXISTS book_lines(game_id TEXT, market TEXT, side TEXT, line REAL, price REAL, book TEXT, fetched_at TEXT,
        PRIMARY KEY(game_id, market, side, line, book));""")

_SECRET = re.compile(r"(?i)\b(api[_-]?key|apikey|key|token)=[^&\s\"']+")

def scrub(text):
    """Remove API keys / tokens from any text before it is logged, stored or returned to a browser."""
    return _SECRET.sub(lambda m: m.group(0).split("=")[0] + "=***", str(text))

def _get(path, **params):
    try:
        r = requests.get(f"{BASE}{path}", params={"apiKey": os.environ["ODDS_API_KEY"], **params}, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        # requests puts the full URL (including ?apiKey=...) in its error text, so never let the original exception escape
        code = getattr(getattr(e, "response", None), "status_code", None)
        raise RuntimeError(f"Odds API request failed{f' (HTTP {code})' if code else ''}: {type(e).__name__}") from None
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
                if not pid: continue
                binary = stat in ("dd", "td")
                if side not in (("yes", "no") if binary else ("over", "under")): continue
                try:
                    line = 0.0 if binary else float(o["point"])
                    price = float(o["price"])
                except (KeyError, TypeError, ValueError): continue
                if not math.isfinite(line) or line < 0 or not math.isfinite(price) or price <= 1: continue
                rows.append((pid, stat, line, {"yes": "over", "no": "under"}.get(side, side), price, b["key"]))
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
        return dict(ok=False, msg=scrub(f"events call failed: {e}"))
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
            con.execute("INSERT OR REPLACE INTO odds_status VALUES('error',?)", (scrub(e)[:200],)); con.commit(); continue
        n2p = {norm_name(nm): pid for pid, nm in players_by_game.get(g["game_id"], [])}
        rows = parse_event(js, n2p)
        con.execute("INSERT OR REPLACE INTO odds_fetch VALUES(?,?)", (g["game_id"], now.isoformat()))
        if rows:      # an empty answer must not erase the last good prices
            data = [(g["game_id"], pid, st, ln, sd, pr, bk, now.isoformat()) for pid, st, ln, sd, pr, bk in rows]
            con.execute("DELETE FROM book_odds WHERE game_id=?", (g["game_id"],))
            con.executemany("INSERT OR REPLACE INTO book_odds VALUES(?,?,?,?,?,?,?,?)", data)
            # append-only history of every price seen before tip-off: the only way to later check closing prices and real edge
            con.executemany("INSERT INTO odds_archive VALUES(?,?,?,?,?,?,?,?)", [(d[7], d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in data])
        fetched += 1
    for k, v in (("remaining", remaining), ("last_run", now.isoformat()), ("error", "")):
        con.execute("INSERT OR REPLACE INTO odds_status VALUES(?,?)", (k, str(v)))
    con.commit()
    return dict(ok=True, fetched=fetched, remaining=remaining)

def _better(cur, price, book):
    """Preferred book (Sportsbet) wins; otherwise the best price."""
    return cur is None or (book == PREFERRED and cur[1] != PREFERRED) or (cur[1] != PREFERRED and book != PREFERRED and price > cur[0])

def load(con):
    """{(game_id, player_id): {stat: {line: {over:(price,book), under:(..), over_best:(price,book)}}}}.
    over/under = Sportsbet if it offers the line, else the best other bookmaker; *_best = highest price across all bookmakers
    (used to flag when TAB, Ladbrokes etc. pay more)."""
    try: rows = con.execute("SELECT game_id,player_id,stat,line,side,price,book FROM book_odds").fetchall()
    except sqlite3.OperationalError: return {}
    out = {}
    for gid, pid, stat, line, side, price, book in rows:
        slot = out.setdefault((gid, pid), {}).setdefault(stat, {}).setdefault(line, {})
        if _better(slot.get(side), price, book): slot[side] = (price, book)
        bk = slot.get(side + "_best")
        if bk is None or price > bk[0]: slot[side + "_best"] = (price, book)
    return out

def status(con):
    try: return {k: scrub(v) for k, v in con.execute("SELECT k,v FROM odds_status").fetchall()}   # also cleans anything stored before this fix
    except sqlite3.OperationalError: return {}


LINES_REFETCH_HOURS = float(os.environ.get("ODDS_LINES_HOURS", "8"))

def refresh_lines(games):
    """Winner / handicap / total prices for every upcoming game in ONE request (3 credits), at most every LINES_REFETCH_HOURS."""
    import slate as sl
    if not available(): return dict(ok=False, msg="No ODDS_API_KEY set")
    con = sqlite3.connect(DB); init(con); now = dt.datetime.now(dt.timezone.utc)
    row = con.execute("SELECT v FROM odds_status WHERE k='lines_at'").fetchone()
    if row and now - dt.datetime.fromisoformat(row[0]) < dt.timedelta(hours=LINES_REFETCH_HOURS): return dict(ok=True, skipped=True)
    try: js, remaining = _get("/sports/basketball_nba/odds", regions="au", markets="h2h,spreads,totals", oddsFormat="decimal")
    except Exception as e: return dict(ok=False, msg=scrub(e))
    nick = {k: v.lower() for k, v in sl.TEAM_NICK.items()}
    abbr = lambda full: next((a for a, n in nick.items() if full.lower().endswith(n)), None)
    by = {(g["home"], g["away"]): g["game_id"] for g in games}; n = 0
    for ev in js:
        h, a = abbr(ev["home_team"]), abbr(ev["away_team"]); gid = by.get((h, a))
        if not gid: continue
        if not ev.get("bookmakers"): continue          # keep last good prices if the answer is empty
        con.execute("DELETE FROM book_lines WHERE game_id=?", (gid,))
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                for oc in m.get("outcomes", []):
                    nm = oc["name"]; side = "home" if abbr(nm) == h else "away" if abbr(nm) == a else nm.lower()
                    if m["key"] == "h2h" and side in ("home", "away"): mk, line = "h2h", 0.0
                    elif m["key"] == "spreads" and side in ("home", "away") and oc.get("point") is not None:
                        mk, line = "spread", float(oc["point"]) if side == "home" else -float(oc["point"])  # always stored as the HOME handicap
                    elif m["key"] == "totals" and side in ("over", "under") and oc.get("point") is not None: mk, line = "total", float(oc["point"])
                    else: continue
                    con.execute("INSERT OR REPLACE INTO book_lines VALUES(?,?,?,?,?,?,?)", (gid, mk, side, line, float(oc["price"]), b["key"], now.isoformat()))
                    con.execute("INSERT INTO lines_archive VALUES(?,?,?,?,?,?,?)", (now.isoformat(), gid, mk, side, line, float(oc["price"]), b["key"]))
        n += 1
    for k, v in (("lines_at", now.isoformat()), ("remaining", remaining)): con.execute("INSERT OR REPLACE INTO odds_status VALUES(?,?)", (k, str(v)))
    con.commit(); return dict(ok=True, games=n, remaining=remaining)

def load_lines(con):
    """{game_id: {h2h:{home:(p,b),away:..,home_best:..}, spread:{line:{...}}, total:{line:{...}}}} (Sportsbet preferred, plus *_best)."""
    try: rows = con.execute("SELECT game_id,market,side,line,price,book FROM book_lines").fetchall()
    except sqlite3.OperationalError: return {}
    out = {}
    for gid, mk, side, line, price, book in rows:
        slot = out.setdefault(gid, {}).setdefault(mk, {})
        if mk != "h2h": slot = slot.setdefault(line, {})
        if _better(slot.get(side), price, book): slot[side] = (price, book)
        bk = slot.get(side + "_best")
        if bk is None or price > bk[0]: slot[side + "_best"] = (price, book)
    return out


def fetch_ages(con):
    """{game_id: hours since a real price was last stored for it}. Taken from the stored prices themselves, so a fetch that
    came back empty (old prices kept) does not make them look fresh."""
    now = dt.datetime.now(dt.timezone.utc)
    try: rows = con.execute("SELECT game_id, max(fetched_at) FROM book_odds GROUP BY game_id").fetchall()
    except sqlite3.OperationalError: return {}
    return {g: (now - dt.datetime.fromisoformat(t)).total_seconds() / 3600 for g, t in rows if t}

def line_ages(con):
    """{game_id: hours since a real winner/handicap/total price was last stored for it} (same idea as fetch_ages, for team bets)."""
    now = dt.datetime.now(dt.timezone.utc)
    try: rows = con.execute("SELECT game_id, max(fetched_at) FROM book_lines GROUP BY game_id").fetchall()
    except sqlite3.OperationalError: return {}
    return {g: (now - dt.datetime.fromisoformat(t)).total_seconds() / 3600 for g, t in rows if t}

def closing_price(con, game_id, player_id, stat, line, side, book, tip_utc):
    """Last archived price for this exact bet before tip-off: (price, book, minutes_before_tip) or None.
    With no book given, the best price at the latest snapshot. The 'closing' price is only as late as our last fetch, so the gap is returned."""
    tip_utc = dt.datetime.fromisoformat(tip_utc.replace("Z", "+00:00")).isoformat()
    try:
        q = "SELECT fetched_at FROM odds_archive WHERE game_id=? AND player_id=? AND stat=? AND line=? AND side=? AND fetched_at<=?" + (" AND book=?" if book else "")
        a = [game_id, player_id, stat, line, side, tip_utc] + ([book] if book else [])
        t = con.execute(q + " ORDER BY fetched_at DESC LIMIT 1", a).fetchone()
        if not t: return None
        r = con.execute(q.replace("SELECT fetched_at", "SELECT price, book") + " AND fetched_at=? ORDER BY price DESC LIMIT 1", a + [t[0]]).fetchone()
    except sqlite3.OperationalError:
        return None
    mins = (dt.datetime.fromisoformat(tip_utc.replace("Z", "+00:00")) - dt.datetime.fromisoformat(t[0])).total_seconds() / 60
    return r[0], r[1], mins
