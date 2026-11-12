"""Past bookmaker prices from The Odds API (for studies only; the app never shows them).

Step 1 — one-game cost check: fetch ONE past game's player prices (points, rebounds, assists, pts+reb+ast) as they stood about an hour
before tip-off, record exactly what it cost and how much came back, and stop. Nothing more is downloaded until the owner approves a
monthly budget (HISTORY_MONTHLY_BUDGET, 0 = off).

Step 2 (owner chose option B: points+rebounds+assists only, 5,000 credits a month): download_step() buys one past game day per run —
the list of that day's games (1 credit) then each game's PRA prices about an hour before its tip-off (about 10 credits each) — and stops
for the month at the budget. Every request is written down before it is sent, so nothing is ever bought twice."""
import datetime as dt, json, os, random, sqlite3
import requests
import odds

HISTORY_MARKETS = "player_points,player_rebounds,player_assists,player_points_rebounds_assists"
HISTORY_REGIONS = "au"
HISTORY_MONTHLY_BUDGET = 5000       # credits a month for past prices (owner approved option B, 5,000 a month)
HISTORY_EXTRA = {"2026-11": 5000}   # one-off extra the owner approved for November 2026 (spare credits, more past games sooner)

def month_budget(month=None):
    return HISTORY_MONTHLY_BUDGET + HISTORY_EXTRA.get(month or _month(), 0) if HISTORY_MONTHLY_BUDGET > 0 else 0
DOWNLOAD_MARKETS = "player_points_rebounds_assists"   # option B: the market the top player calls use
DOWNLOAD_MIN_CREDITS = 3000         # never buy past prices with fewer credits than this left (the app's normal fetches come first)
EST_EVENTS, EST_GAME = 1, 10        # usual cost of a day's game list and of one game's PRA prices, for requests whose cost was never recorded
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
    CREATE TABLE IF NOT EXISTS hist_fetch(k TEXT PRIMARY KEY, month TEXT, kind TEXT, state TEXT, cost INT, prices INT, at TEXT);
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


# ---------- step 2: the monthly download (option B) ----------
def _month(now=None): return (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m")

def spent(con, month=None):
    """Credits used on past prices this month: recorded costs, plus the usual cost for any request whose reply never came back."""
    init(con); month = month or _month(); tot = 0
    for kind, cost in con.execute("SELECT kind, cost FROM hist_fetch WHERE month=?", (month,)):
        tot += cost if cost is not None else (EST_EVENTS if kind == "events" else EST_GAME)
    return tot

def _days(con):
    """Last season's finished game days, in a fixed shuffled order (spread across the whole season), skipping days already done."""
    seasons = [r[0] for r in con.execute("SELECT DISTINCT season FROM games WHERE home_pts IS NOT NULL ORDER BY season DESC LIMIT 2")]
    if len(seasons) < 2: return []
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM games WHERE season=? AND home_pts IS NOT NULL ORDER BY date", (seasons[1],))]
    random.Random(2026).shuffle(days)
    tried = {}
    for (k,) in con.execute("SELECT k FROM hist_fetch WHERE kind='events'"): d = k.split(":")[1]; tried[d] = tried.get(d, 0) + 1
    have = {r[0] for r in con.execute("SELECT substr(k, 6) FROM hist_fetch WHERE kind='game'")}
    open_day = lambda d: any(g not in have for (g,) in con.execute("SELECT game_id FROM games WHERE date=? AND home_pts IS NOT NULL", (d,)))
    return [d for d in days if tried.get(d, 0) < 3 and open_day(d)]      # a day is finished once every game on it is recorded (at most 3 tries)

def _note(con, k, kind, state, cost=None, prices=None, month=None):
    con.execute("INSERT OR REPLACE INTO hist_fetch VALUES(?,?,?,?,?,?,?)", (k, month or _month(), kind, state, cost, prices, dt.datetime.now(dt.timezone.utc).isoformat())); con.commit()

def _cost(h):
    try: return int(float(h.get("last")))
    except (TypeError, ValueError): return None

def download_step(con, budget=None):
    """Buy one past game day (if the month's budget and the credit balance allow). Returns what it did."""
    init(con); budget = month_budget() if budget is None else budget
    if budget <= 0: return dict(ran=False, reason="off")
    bal = odds.valid_balance((odds.status(con) or {}).get("remaining"))
    used = spent(con)
    def room(cost): return bal is not None and bal - cost >= DOWNLOAD_MIN_CREDITS and used + cost <= budget
    if not room(EST_EVENTS + EST_GAME): return dict(ran=False, reason="budget or balance", spent=used, balance=bal)
    days = _days(con)
    if not days: return dict(ran=False, reason="all past game days done")
    day = days[0]
    games = con.execute("SELECT game_id, home, away FROM games WHERE date=? AND home_pts IS NOT NULL", (day,)).fetchall()
    nick = {k: v.lower() for k, v in __import__("slate").TEAM_NICK.items()}
    abbr = lambda full: next((a for a, n in nick.items() if full.lower().endswith(n)), None)
    ek = f"events:{day}:{con.execute('SELECT count(*) FROM hist_fetch WHERE k LIKE ?', (f'events:{day}:%',)).fetchone()[0] + 1}"
    _note(con, ek, "events", "sent")                                    # written down before sending: never bought twice
    try:
        ev_js, h = _get("/historical/sports/basketball_nba/events", date=f"{day}T15:00:00Z")    # 11am US Eastern: every game that day still to come
    except Exception as e:
        _note(con, ek, "events", "failed", cost=0); return dict(ran=True, day=day, state="failed", error=odds.scrub(e)[:200])
    c = _cost(h); bal = odds.valid_balance(h.get("remaining")); used += c if c is not None else EST_EVENTS
    _note(con, ek, "events", "done", cost=c)
    events = ev_js.get("data") or []; got = 0; prices = 0
    for gid, home, away in games:
        if con.execute("SELECT 1 FROM hist_fetch WHERE k=?", (f"game:{gid}",)).fetchone(): continue
        ev = odds.match_event(events, home, away, f"{day}T23:30Z", abbr)
        if not ev: _note(con, f"game:{gid}", "game", "no match", cost=0); continue
        if not room(EST_GAME): break
        snap = (dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) - dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _note(con, f"game:{gid}", "game", "sent")
        try:
            js, h = _get(f"/historical/sports/basketball_nba/events/{ev['id']}/odds", date=snap, regions=HISTORY_REGIONS, markets=DOWNLOAD_MARKETS, oddsFormat="decimal")
        except Exception as e:
            _note(con, f"game:{gid}", "game", "failed", cost=0); break                         # try again next run, from the next day
        c = _cost(h); b = odds.valid_balance(h.get("remaining")); bal = b if b is not None else None; used += c if c is not None else EST_GAME
        rows = []
        for bk in (js.get("data") or {}).get("bookmakers", []):
            for m in bk.get("markets", []):
                stat = odds.STAT_OF.get(m["key"])
                for o in m.get("outcomes", []):
                    try: rows.append((gid, js.get("timestamp") or snap, o.get("description") or "", stat, float(o["point"]), (o.get("name") or "").lower(), float(o["price"]), bk["key"]))
                    except (KeyError, TypeError, ValueError): pass
        con.executemany("INSERT OR IGNORE INTO hist_odds VALUES(?,?,?,?,?,?,?,?)", rows)
        _note(con, f"game:{gid}", "game", "done", cost=c, prices=len(rows)); got += 1; prices += len(rows)
    if bal is not None: con.execute("INSERT OR REPLACE INTO odds_status VALUES('remaining', ?)", (str(int(bal)),)); con.commit()   # keep the app's credit count right
    return dict(ran=True, day=day, games=got, prices=prices, spent=used, balance=bal)

def summary(con):
    """For /api/health: this month's spend and what has been stored so far."""
    init(con)
    g = con.execute("SELECT count(*), coalesce(sum(prices),0) FROM hist_fetch WHERE kind='game' AND state='done'").fetchone()
    return dict(budget=month_budget(), month=_month(), spent_this_month=spent(con), games_stored=g[0], prices_stored=g[1],
                days_tried=con.execute("SELECT count(DISTINCT substr(k, 8, 10)) FROM hist_fetch WHERE kind='events'").fetchone()[0],
                failed=con.execute("SELECT count(*) FROM hist_fetch WHERE state='failed'").fetchone()[0])
