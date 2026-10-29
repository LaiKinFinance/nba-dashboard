"""Live scores and box scores from ESPN's public JSON, with a short server-side cache so many viewers cost few requests."""
from zoneinfo import ZoneInfo
import time, datetime as dt
import requests

H = {"User-Agent": "Mozilla/5.0"}
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
_cache = {}

def _get(url, **params):
    """ESPN occasionally returns an empty body; retry a few times before giving up."""
    for i in range(3):
        try:
            r = requests.get(url, params=params, headers=H, timeout=20); r.raise_for_status(); return r.json()
        except Exception:
            if i == 2: raise
            time.sleep(0.6 * (i + 1))

def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl: return hit[1]
    val = fn(); _cache[key] = (time.time(), val); return val

def _et_dates():
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    return [(now + dt.timedelta(days=d)).strftime("%Y%m%d") for d in (-1, 0)]

def _scoreboard():
    games, seen = [], set()
    for d in _et_dates():
        js = _get(SB, dates=d)
        for e in js.get("events", []):
            if e["id"] in seen: continue
            seen.add(e["id"]); s = e["status"]; t = s["type"]
            c = e["competitions"][0]["competitors"]
            side = lambda x: dict(team=x["team"]["abbreviation"], score=int(x.get("score") or 0),
                                  record=(x.get("records") or [{}])[0].get("summary"),
                                  lines=[int(float(q.get("value") or 0)) for q in x.get("linescores", [])])
            home = side(next(x for x in c if x["homeAway"] == "home")); away = side(next(x for x in c if x["homeAway"] == "away"))
            games.append(dict(game_id=e["id"], state=t["state"], name=t["name"], detail=t.get("shortDetail", ""), period=s.get("period", 0),
                              clock=s.get("displayClock", ""), start=e["date"], home=home, away=away))
    return games

def scoreboard():
    """Live now, then not-yet-started, then finished (finished games older than ~16h are dropped)."""
    games = cached("sb", 10, _scoreboard)
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=16)).strftime("%Y-%m-%dT%H:%MZ")
    games = [g for g in games if g["state"] != "post" or g["start"] >= cutoff]
    order = {"in": 0, "pre": 1, "post": 2}
    return sorted(games, key=lambda g: (order.get(g["state"], 3), g["start"]))

def _num(s):
    try: return int(float(s))
    except (TypeError, ValueError): return 0

def _box(game_id):
    js = _get(SUM, event=game_id)
    teams = []
    for t in js.get("boxscore", {}).get("players", []):
        rows = []
        for grp in t.get("statistics", []):
            keys = grp["keys"]
            for a in grp["athletes"]:
                st = dict(zip(keys, a.get("stats") or []))
                if not st or a.get("didNotPlay"): continue
                made3 = (st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted") or "0-0").split("-")[0]
                pm = (st.get("plusMinus") or "0").replace("+", "")
                rows.append(dict(player_id=a["athlete"]["id"], player=a["athlete"]["displayName"], starter=bool(a.get("starter")),
                                 min=_num(st.get("minutes")), pts=_num(st.get("points")), reb=_num(st.get("rebounds")), ast=_num(st.get("assists")),
                                 stl=_num(st.get("steals")), blk=_num(st.get("blocks")), tov=_num(st.get("turnovers")), fg3m=_num(made3),
                                 fg=st.get("fieldGoalsMade-fieldGoalsAttempted", ""), pm=_num(pm)))
        rows.sort(key=lambda r: (-r["min"], -r["pts"]))
        teams.append(dict(team=t["team"]["abbreviation"], players=rows))
    return dict(game_id=game_id, teams=teams)

def boxscore(game_id):
    return cached("box" + game_id, 10, lambda: _box(game_id))
