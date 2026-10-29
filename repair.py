"""Additive data repairs, safe to run any number of times (nothing is deleted; old values are kept in new columns).

1. Game dates: ESPN's `date` is a UTC timestamp, so late tip-offs landed on the next calendar day. We keep the old value in `date_utc`,
   store the real tip-off in `tip_utc`, and make `date` the US Eastern date (the date basketball schedules use). Rest days, ordering
   and the train/test cut then all use real game days.
2. Current rosters: who is on which team *now* (ESPN roster endpoint), instead of "the last team he played for".
3. Missing games: any final game ESPN lists that we never ingested is fetched and stored.
Run: python3 repair.py [--dry]"""
import os, sys, sqlite3, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
import ingest

ET = ZoneInfo("America/New_York")
TEAMS = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
MIN_ROSTER = 8     # every real team lists at least this many players; fewer means the answer was incomplete

def et_date(tip_utc):
    return dt.datetime.fromisoformat(tip_utc.replace("Z", "+00:00")).astimezone(ET).date().isoformat()

def ensure_schema(con):
    ingest.ensure_columns(con)
    for table, cols in (("games", {"date_utc": "TEXT", "tip_utc": "TEXT"}), ("player_games", {"date_utc": "TEXT"})):
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for c, t in cols.items():
            if c not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {c} {t}")
    # keep the original value exactly once, before `date` is ever rewritten
    con.execute("UPDATE games SET date_utc=date WHERE date_utc IS NULL")
    con.execute("UPDATE player_games SET date_utc=date WHERE date_utc IS NULL")
    con.execute("""CREATE TABLE IF NOT EXISTS rosters(player_id TEXT PRIMARY KEY, team TEXT, name TEXT, fetched_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS roster_log(fetched_at TEXT, player_id TEXT, name TEXT, old_team TEXT, new_team TEXT)""")
    con.commit()

def _tip_times(day):
    js = ingest.get(ingest.SB, {"dates": f"{day:%Y%m%d}", "limit": 100})
    return {e["id"]: e["date"] for e in js["events"]}

def backfill_tip_times(con):
    """Fetch the real tip-off for every stored game that lacks one; returns (filled, still_missing)."""
    need = {r[0] for r in con.execute("SELECT game_id FROM games WHERE tip_utc IS NULL")}
    if not need: return 0, 0
    lo, hi = con.execute("SELECT min(date_utc), max(date_utc) FROM games WHERE tip_utc IS NULL").fetchone()
    lo = dt.date.fromisoformat(lo) - dt.timedelta(days=1); hi = dt.date.fromisoformat(hi) + dt.timedelta(days=1)
    days = [lo + dt.timedelta(days=i) for i in range((hi - lo).days + 1)]
    filled = 0
    with ThreadPoolExecutor(6) as ex:
        for ev in ex.map(_tip_times, days):
            for gid, tip in ev.items():
                if gid in need:
                    con.execute("UPDATE games SET tip_utc=? WHERE game_id=?", (tip, gid)); need.discard(gid); filled += 1
    con.commit()
    return filled, len(need)

def apply_et_dates(con):
    """date := Eastern date of the tip-off, for every game with a known tip-off; player rows follow their game."""
    n = 0
    for gid, tip in con.execute("SELECT game_id, tip_utc FROM games WHERE tip_utc IS NOT NULL").fetchall():
        d = et_date(tip)
        n += con.execute("UPDATE games SET date=? WHERE game_id=? AND date<>?", (d, gid, d)).rowcount
        con.execute("UPDATE player_games SET date=? WHERE game_id=? AND date<>?", (d, gid, d))
    con.commit(); return n

def refresh_rosters(con):
    """Current NBA rosters from ESPN. Changes of team are logged. Returns number of players on a roster (0 = fetch failed, nothing touched)."""
    teams = ingest.get(TEAMS, {})["sports"][0]["leagues"][0]["teams"]
    def one(t):
        js = ingest.get(f"{TEAMS}/{t['team']['id']}/roster", {})
        return t["team"]["abbreviation"], [(a["id"], a["displayName"]) for a in js.get("athletes", [])]
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(one, teams))
    cur = {pid: (ab, nm) for ab, ps in res if ab in ingest.REAL_TEAMS for pid, nm in ps}
    small = [ab for ab, ps in res if ab in ingest.REAL_TEAMS and len(ps) < MIN_ROSTER]
    if len(cur) < 300 or len({ab for ab, _ in res if ab in ingest.REAL_TEAMS}) < 30 or small:
        return 0                                   # a partial answer (even one empty team) must never change the table
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    old = {r[0]: r[1] for r in con.execute("SELECT player_id, team FROM rosters")}
    for pid, (ab, nm) in cur.items():
        if old.get(pid) != ab:
            con.execute("INSERT INTO roster_log VALUES(?,?,?,?,?)", (now, pid, nm, old.get(pid), ab))
        con.execute("INSERT INTO rosters VALUES(?,?,?,?) ON CONFLICT(player_id) DO UPDATE SET team=excluded.team, name=excluded.name, fetched_at=excluded.fetched_at", (pid, ab, nm, now))
    for pid in set(old) - set(cur):                # left the league's rosters (waived, released): keep the row, mark team empty
        con.execute("UPDATE rosters SET team=NULL, fetched_at=? WHERE player_id=?", (now, pid))
    con.commit(); return len(cur)

def missing_games(con, since=None):
    """Final games ESPN lists that are not stored. Only counts; ingest.run stores them."""
    since = since or dt.date.fromisoformat(con.execute("SELECT min(date_utc) FROM games").fetchone()[0])
    have = {r[0] for r in con.execute("SELECT game_id FROM games")}
    return [g["game_id"] for g in ingest.final_games(since, dt.date.today()) if g["game_id"] not in have]

def run_all(full_reconcile=False):
    con = sqlite3.connect(ingest.DB, timeout=30); ensure_schema(con)
    out = {}
    out["tips_filled"], out["tips_missing"] = backfill_tip_times(con)
    out["dates_changed"] = apply_et_dates(con)
    out["roster_players"] = refresh_rosters(con)
    if full_reconcile:
        miss = missing_games(con)
        out["missing_before"] = len(miss)
        if miss:
            ingest.run(dt.date.fromisoformat(con.execute('SELECT min(date_utc) FROM games').fetchone()[0]), dt.date.today())
            out["missing_after"] = len(missing_games(con))
    problems = []
    if out["tips_missing"]: problems.append(f"{out['tips_missing']} games have no tip-off time")
    if not out["roster_players"]: problems.append("current rosters could not be refreshed (the last good list is kept)")
    if out.get("missing_after"): problems.append(f"{out['missing_after']} finished games are still missing")
    out["problem"] = "; ".join(problems) or None
    return out

def dry_run():
    """Look only: report what run_all would change. Opens the database read-only and writes nothing."""
    con = sqlite3.connect(f"file:{ingest.DB}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    out = dict(games=con.execute("SELECT count(*) FROM games").fetchone()[0])
    if "tip_utc" in cols:
        out["games_without_tip_time"] = con.execute("SELECT count(*) FROM games WHERE tip_utc IS NULL").fetchone()[0]
        out["games_with_utc_date_not_eastern"] = sum(1 for g, t, d in con.execute("SELECT game_id, tip_utc, date FROM games WHERE tip_utc IS NOT NULL") if et_date(t) != d)
    else:
        out["games_without_tip_time"] = out["games"]; out["note"] = "database has not been through the repair yet"
    return out

if __name__ == "__main__":
    print(dry_run() if "--dry" in sys.argv else run_all(full_reconcile="--full" in sys.argv))
