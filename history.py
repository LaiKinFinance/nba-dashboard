"""Past bookmaker prices from The Odds API (for studies only; the app never shows them).

Step 1 — one-game cost check: fetch ONE past game's player prices (points, rebounds, assists, pts+reb+ast) as they stood about an hour
before tip-off, record exactly what it cost and how much came back, and stop. Nothing more is downloaded until the owner approves a
monthly budget (HISTORY_MONTHLY_BUDGET, 0 = off)."""
import datetime as dt, json, os, sqlite3
import requests
import odds

HISTORY_MARKETS = "player_points,player_rebounds,player_assists,player_points_rebounds_assists"
HISTORY_REGIONS = "au"
HISTORY_MONTHLY_BUDGET = 0          # the download is OFF until the cost check has been seen and approved
CHECK_MIN_CREDITS = 3000            # never run the check with fewer credits than this (the app's normal fetches come first)


def _get(path, **params):
    """Like odds._get, but also returns the credit headers, and never lets the key appear in an error."""
    try:
        r = requests.get(f"{odds.BASE}{path}", params={"apiKey": os.environ["ODDS_API_KEY"], **params}, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        raise RuntimeError(f"Odds API request failed{f' (HTTP {code})' if code else ''}: {type(e).__name__}") from None
    h = r.headers
    return r.json(), dict(remaining=h.get("x-requests-remaining"), used=h.get("x-requests-used"), last=h.get("x-requests-last"))


def init(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS hist_odds(game_id TEXT, snapshot TEXT, player TEXT, stat TEXT, line REAL, side TEXT, price REAL, book TEXT,
        PRIMARY KEY(game_id, snapshot, player, stat, line, side, book));
    """)


def pick_game(con):
    """A finished regular-season game from last season, mid-March (plenty of player markets back then)."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    tip = "tip_utc" if "tip_utc" in cols else "NULL"
    seasons = [r[0] for r in con.execute("SELECT DISTINCT season FROM games WHERE home_pts IS NOT NULL ORDER BY season DESC LIMIT 2")]
    if len(seasons) < 2: return None
    r = con.execute(f"SELECT game_id, date, home, away, {tip} FROM games WHERE season=? AND home_pts IS NOT NULL AND date LIKE '%-03-1%' ORDER BY date LIMIT 1",
                    (seasons[1],)).fetchone()
    if not r: return None
    start = r[4] or f"{r[1]}T23:30Z"
    return dict(game_id=r[0], date=r[1], home=r[2], away=r[3], start=start)


def cost_check(con):
    """Run the one-game check once. Returns (and stores in odds_status 'history_check') what it cost and what came back."""
    init(con)
    done = con.execute("SELECT v FROM odds_status WHERE k='history_check'").fetchone()
    if done: return json.loads(done[0])
    bal = odds.valid_balance((odds.status(con) or {}).get("remaining"))
    if bal is None or bal < CHECK_MIN_CREDITS: return dict(ran=False, reason=f"needs a known balance of at least {CHECK_MIN_CREDITS} credits")
    g = pick_game(con)
    if not g: return dict(ran=False, reason="no past game found")
    snap = (dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00")) - dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = dict(ran=True, game=f"{g['away']} at {g['home']}", date=g["date"], snapshot=snap, markets=HISTORY_MARKETS, regions=HISTORY_REGIONS,
               at=dt.datetime.now(dt.timezone.utc).isoformat())
    con.execute("INSERT OR REPLACE INTO odds_status VALUES('history_check', ?)", (json.dumps(dict(out, state="sent")),)); con.commit()   # saved before sending: never repeated
    try:
        ev_js, h1 = _get("/historical/sports/basketball_nba/events", date=snap)
        nick = {k: v.lower() for k, v in __import__("slate").TEAM_NICK.items()}
        abbr = lambda full: next((a for a, n in nick.items() if full.lower().endswith(n)), None)
        ev = odds.match_event(ev_js.get("data") or [], g["home"], g["away"], g["start"], abbr)
        out.update(events_cost=h1.get("last"), balance_before=h1.get("remaining"))
        if not ev:
            out.update(state="no matching past event", found=False)
        else:
            js, h2 = _get(f"/historical/sports/basketball_nba/events/{ev['id']}/odds", date=snap, regions=HISTORY_REGIONS, markets=HISTORY_MARKETS, oddsFormat="decimal")
            data = js.get("data") or {}
            rows = []
            for b in data.get("bookmakers", []):
                for m in b.get("markets", []):
                    stat = odds.STAT_OF.get(m["key"])
                    for o in m.get("outcomes", []):
                        try: rows.append((g["game_id"], js.get("timestamp") or snap, o.get("description") or "", stat, float(o["point"]), (o.get("name") or "").lower(), float(o["price"]), b["key"]))
                        except (KeyError, TypeError, ValueError): pass
            con.executemany("INSERT OR IGNORE INTO hist_odds VALUES(?,?,?,?,?,?,?,?)", rows)
            out.update(state="done", found=True, odds_cost=h2.get("last"), balance_after=h2.get("remaining"), bookmakers=len(data.get("bookmakers", [])),
                       prices=len(rows), players=len({r[2] for r in rows}), books=sorted({r[7] for r in rows}))
    except Exception as e:
        out.update(state="failed", error=odds.scrub(e)[:200])
    con.execute("INSERT OR REPLACE INTO odds_status VALUES('history_check', ?)", (json.dumps(out),)); con.commit()
    return out
