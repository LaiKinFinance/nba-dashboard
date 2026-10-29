"""Data-integrity audit for the NBA dashboard. Run: python tools/audit_data.py  (uses DATA_DIR, default ./data)
Prints PASS/WARN/FAIL lines and a summary; exit code 1 if any FAIL."""
import json, os, pickle, sqlite3, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # so models.pkl can import the project modules
import numpy as np, pandas as pd

D = os.environ.get("DATA_DIR", "data"); con = sqlite3.connect(os.path.join(D, "nba.db")); res = []
def chk(level, name, detail=""): res.append((level, name, detail)); print(f"{level:4s} {name}" + (f" — {detail}" if detail else ""))

g = pd.read_sql("SELECT * FROM games", con); p = pd.read_sql("SELECT * FROM player_games", con)
chk("PASS", f"loaded {len(g)} games, {len(p)} player-game rows, {p.player_id.nunique()} players, {g.season.nunique()} seasons")

# --- only real NBA teams (All-Star weekend exhibition teams must not be in the data)
from ingest import REAL_TEAMS
fake = sorted((set(p.team) | set(p.opp) | set(g.home) | set(g.away)) - REAL_TEAMS)
chk("PASS" if not fake else "FAIL", "all teams are real NBA teams", f"unexpected: {fake}" if fake else "")
# --- uniqueness / nulls
d = p.duplicated(["game_id", "player_id"]).sum(); chk("PASS" if d == 0 else "FAIL", "no duplicate (game, player) rows", f"{d} duplicates" if d else "")
d = g.game_id.duplicated().sum(); chk("PASS" if d == 0 else "FAIL", "no duplicate games", f"{d}" if d else "")
for col in ("pts", "reb", "ast", "min", "team", "opp", "date"):
    n = p[col].isna().sum(); chk("PASS" if n == 0 else "FAIL", f"player_games.{col} has no nulls", f"{n} nulls" if n else "")
# --- value ranges
bad = p[(p.pts < 0) | (p.reb < 0) | (p.ast < 0) | (p.pts > 90) | (p.reb > 40) | (p.ast > 30) | (p["min"] > 65) | (p["min"] <= 0)]
chk("PASS" if bad.empty else "FAIL", "player stat ranges plausible", f"{len(bad)} rows outside range, e.g. {bad.head(2)[['player','date','pts','reb','ast','min']].values.tolist()}" if len(bad) else "")
# --- team minutes (~240 + 25 per OT) and points vs box score
tm = p.groupby(["game_id", "team"]).agg(m=("min", "sum"), pts=("pts", "sum")).reset_index()
off = tm[(tm.m < 235) | (tm.m > 330)]; chk("PASS" if len(off) < 0.01 * len(tm) else "WARN", "team minutes ≈ 240 (+OT)", f"{len(off)} of {len(tm)} team-games outside 235–330")
gg = g.set_index("game_id"); tm = tm.join(gg[["home", "away", "home_pts", "away_pts"]], on="game_id")
tm["score"] = np.where(tm.team == tm.home, tm.home_pts, tm.away_pts); mis = tm[(tm.pts - tm.score).abs() > 2]
chk("PASS" if len(mis) < 0.01 * len(tm) else "FAIL", "player points sum to the team score", f"{len(mis)} of {len(tm)} team-games differ by >2")
# --- schedule completeness
reg = g[g.season_type == 2]
for s, x in reg.groupby("season"):
    per = pd.concat([x.home, x.away]).value_counts(); ok = (per >= 80).sum() if s != str(g.season.max()) else 30
    chk("PASS" if len(per) == 30 else "WARN", f"season {s}: {len(per)} teams, regular-season games per team {int(per.min())}-{int(per.max())}")
# --- market lines
if "ou" in g:
    cov = g.ou.notna().mean(); chk("PASS" if cov > 0.8 else "WARN", f"market lines present for {cov:.0%} of games")
    bad = g[(g.ou < 170) | (g.ou > 290) | (g.spread_home.abs() > 30)]; chk("PASS" if bad.empty else "FAIL", "market totals/spreads plausible", f"{len(bad)} implausible" if len(bad) else "")
# --- projections for upcoming games
try:
    pr = [json.loads(r[0]) for r in con.execute("SELECT payload FROM projections")]; chk("PASS", f"{len(pr)} stored projections")
    probs = []
    for x in pr:
        if x["p_td"] > x["p_dd"] + 1e-9: probs.append(f"{x['player']}: TD>DD")
        for s in ("pts", "reb", "ast"):
            ps = [l["p_over"] for l in x["ladders"][s]]
            if any(b > a + 1e-9 for a, b in zip(ps, ps[1:])): probs.append(f"{x['player']} {s} ladder not decreasing")
        if not (0 <= x["proj"]["pts"] <= 60 and 0 <= x["proj"]["reb"] <= 25 and 0 <= x["proj"]["ast"] <= 18): probs.append(f"{x['player']}: implausible projection {x['proj']}")
    chk("PASS" if not probs else "FAIL", "projection sanity (TD≤DD, monotone ladders, plausible ranges)", "; ".join(probs[:4]))
    gp = [json.loads(r[0]) for r in con.execute("SELECT payload FROM game_preds")]
    bad = [x["game_id"] for x in gp if not (0.02 < x["p_home"] < 0.98) or not (190 < x["total"] < 270)]; chk("PASS" if not bad else "WARN", f"{len(gp)} game predictions plausible", f"{bad[:3]}" if bad else "")
except sqlite3.OperationalError as e: chk("WARN", "projection tables missing", str(e))
# --- model artefact
try:
    b = pickle.load(open(os.path.join(D, "models.pkl"), "rb")); a = b["alphas"]
    chk("PASS" if all(v > 0 for v in a.values()) else "FAIL", "dispersion parameters positive", str({k: round(v, 3) for k, v in a.items()}))
    chk("PASS", f"model trained through {b['trained_through']}; report keys {sorted(b['report'])}")
except Exception as e: chk("FAIL", "models.pkl loads", str(e))
fails = sum(1 for r in res if r[0] == "FAIL"); warns = sum(1 for r in res if r[0] == "WARN")
print(f"\nSUMMARY: {len(res)-fails-warns} pass, {warns} warn, {fails} fail"); sys.exit(1 if fails else 0)
