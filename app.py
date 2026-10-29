"""NBA top-3 dashboard API + static UI.  Run: uvicorn app:app --host 0.0.0.0 --port 8000"""
import json, os, sqlite3, datetime as dt
from contextlib import closing
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import predict, parlay as pl, slate as sl

DB = os.path.join(os.environ.get("DATA_DIR", "data"), "nba.db")
ACCESS_CODE = os.environ.get("APP_ACCESS_CODE", "")  # shared secret for friends; empty = open
import threading, time, logging

def refresh_odds():
    import odds
    if not odds.available(): return
    preds = projections_all(); games = {}; pbg = {}
    for p in preds:
        games.setdefault(p["game_id"], dict(game_id=p["game_id"], start=p["start"], home=p["team"] if p["home"] else p["opp"], away=p["opp"] if p["home"] else p["team"]))
        pbg.setdefault(p["game_id"], []).append((p["player_id"], p["player"]))
    gl = list(games.values())
    return dict(props=odds.refresh(gl, pbg), lines=odds.refresh_lines(gl))

def _refresh_loop():
    """Background refresh every 4h: new box scores, re-project upcoming games, fetch bookmaker odds. Retrains weekly."""
    import ingest
    last_train = 0
    while True:
        try:
            t = dt.date.today(); ingest.run(t - dt.timedelta(days=4), t)
            if time.time() - last_train > 7 * 86400: predict.train(); last_train = time.time()
            predict.save(predict.project())
            logging.info("odds refresh: %s", refresh_odds())
        except Exception:
            logging.exception("refresh failed")
        time.sleep(4 * 3600)

app = FastAPI(title="NBA Top-3 Dashboard")
if os.environ.get("AUTO_REFRESH") == "1":
    threading.Thread(target=_refresh_loop, daemon=True).start()

def con():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

def init():
    with closing(con()) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS picks(id INTEGER PRIMARY KEY, user TEXT, created TEXT, game_id TEXT, legs TEXT, note TEXT, odds REAL, model_prob REAL);
        CREATE TABLE IF NOT EXISTS comments(id INTEGER PRIMARY KEY, pick_id INTEGER, user TEXT, created TEXT, body TEXT);
        CREATE TABLE IF NOT EXISTS votes(pick_id INTEGER, user TEXT, v INTEGER, PRIMARY KEY(pick_id, user));
        CREATE TABLE IF NOT EXISTS bets(id INTEGER PRIMARY KEY, user TEXT, created TEXT, legs TEXT, stake REAL, odds REAL, model_prob REAL, result TEXT DEFAULT 'open');
        CREATE TABLE IF NOT EXISTS manual_odds(game_id TEXT, player_id TEXT, stat TEXT, line REAL, side TEXT, odds REAL, PRIMARY KEY(game_id, player_id, stat, line, side));
        """); c.commit()
init()

def who(x_user, x_code):
    if ACCESS_CODE and x_code != ACCESS_CODE: raise HTTPException(401, "Wrong access code")
    if not x_user or len(x_user.strip()) < 2: raise HTTPException(400, "Enter your name first")
    return x_user.strip()[:30]

def check(x_code):
    if ACCESS_CODE and x_code != ACCESS_CODE: raise HTTPException(401, "Wrong access code")

def projections_all():
    with closing(con()) as c:
        return [json.loads(r["payload"]) for r in c.execute("SELECT payload FROM projections")]

def projections():
    rows = projections_all()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return [r for r in rows if r["start"] >= (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")]

def odds_status():
    import odds
    with closing(con()) as c: return dict(enabled=odds.available(), **odds.status(c))

@app.get("/api/meta")
def meta(x_code: str = Header("")):
    check(x_code)
    b = predict.load_models()
    with closing(con()) as c:
        last = c.execute("SELECT max(date) d FROM games").fetchone()["d"]
        made = c.execute("SELECT max(made_at) m FROM projections").fetchone()["m"]
    return dict(data_through=last, projections_made=made, trained_through=b["trained_through"],
                odds_api=__import__("odds").available(), open_access=not ACCESS_CODE, odds=__import__("odds").status(c) if False else odds_status())

def _dd(r): return int((r["pts"] >= 10) + (r["reb"] >= 10) + (r["ast"] >= 10) >= 2)
def _td(r): return int((r["pts"] >= 10) + (r["reb"] >= 10) + (r["ast"] >= 10) >= 3)

STRIPS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "pra", "pr", "pa")
COMBO_PARTS = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast")}
EXTRA_MIN = {"fg3m": 0.7, "stl": 0.8, "blk": 0.6}   # only show a market when the player realistically produces it

def enriched():
    """Projections + last-10 games, consistency counts per line, and bookmaker prices/edges where available."""
    import odds
    from model import nb_over
    preds = projections()
    ids = list({p["player_id"] for p in preds}); rows = {}
    with closing(con()) as c:
        if ids:
            q = ",".join("?" * len(ids))
            for r in c.execute(f"SELECT player_id,date,opp,home,min,pts,reb,ast,fg3m,stl,blk FROM player_games WHERE player_id IN ({q}) ORDER BY date DESC", ids):
                rows.setdefault(r["player_id"], [])
                if len(rows[r["player_id"]]) < 20: rows[r["player_id"]].append(dict(r))
        book = odds.load(c)
    alphas = predict.load_models()["alphas"]
    # who realistically gets double-/triple-doubles: full stored history (this and last season) plus recent near-misses
    hist = {}
    if ids:
        with closing(con()) as c:
            q = ",".join("?" * len(ids))
            full = pd.read_sql(f"SELECT player_id, date, pts, reb, ast FROM player_games WHERE player_id IN ({q}) ORDER BY date DESC", c, params=ids)
        for pid, d in full.groupby("player_id"):
            cats = (d.pts >= 10).astype(int) + (d.reb >= 10).astype(int) + (d.ast >= 10).astype(int)
            top3 = d[["pts", "reb", "ast"]].apply(lambda r: sorted(r, reverse=True), axis=1)
            near = [(c >= 3) or (c == 2 and t[2] >= 7) for c, t in zip(cats, top3)]
            hist[pid] = dict(games=len(d), td=int((cats >= 3).sum()), dd=int((cats >= 2).sum()), dd20=int((cats.head(20) >= 2).sum()), near20=int(sum(near[:20])))
    for p in preds:
        h = hist.get(p["player_id"], dict(games=0, td=0, dd=0, dd20=0, near20=0)); p["dd_td_hist"] = h
        p["show_td"] = bool(h["td"] >= 1 or h["near20"] >= 2 or p["p_td"] >= 0.03)
        p["show_dd"] = bool(h["dd20"] >= 2 or p["p_dd"] >= 0.15)
        g = rows.get(p["player_id"], []); l10 = g[:10]; n = len(l10)
        p["last10"] = l10
        for s in ("pts", "reb", "ast"):
            for l in p["ladders"][s]: l["hit"] = sum(1 for r in l10 if r[s] > l["line"]); l["n"] = n
        p["dd_hit"] = dict(k=sum(_dd(r) for r in l10), n=n); p["td_hit"] = dict(k=h["td"], n=h["games"])
        b = book.get((p["game_id"], p["player_id"]), {}); p["book"] = {}; p["strip"] = {}
        for s in STRIPS:
            combo = s in COMBO_PARTS
            mu = None if combo else p["proj"].get(s)
            cm = {x["line"]: x["p_over"] for x in (p.get("combo") or {}).get(s, [])} if combo else {}
            if not combo and (mu is None or mu < EXTRA_MIN.get(s, 0)): p["strip"][s] = []; continue
            if combo and not cm: p["strip"][s] = []; continue
            pmodel = (lambda L, s=s, mu=mu: float(nb_over(mu, alphas[s], L))) if not combo else (lambda L, cm=cm: cm.get(L))
            val = (lambda r, s=s: sum(r[c] for c in COMBO_PARTS[s])) if combo else (lambda r, s=s: r[s])
            bl = {}
            for line, sides in sorted(b.get(s, {}).items()):
                if "over" not in sides: continue
                po = pmodel(line)
                if po is None: continue
                price, bk = sides["over"]; best = sides.get("over_best"); alt = None
                if best and best[1] != bk and best[0] >= price * 1.03: alt = [best[0], best[1]]
                bl[line] = dict(line=line, price=price, book=bk, p_over=po, edge=po - 1 / price, hit=sum(1 for r in l10 if val(r) > line), n=n,
                                under=(sides.get("under") or [None])[0], alt=alt)
            if bl: p["book"][s] = list(bl.values())
            lines = {}
            if combo:
                lines = {L: q for L, q in cm.items() if 0.10 <= q <= 0.92}
            else:
                for k in range(-14, 15):
                    L = float(np.floor(mu) + k + 0.5)
                    if L < 0.5: continue
                    q = pmodel(L)
                    if 0.10 <= q <= 0.92: lines[L] = q
            for L in bl:
                q = pmodel(L)
                if q is not None and 0.04 <= q <= 0.97: lines[L] = q
            ordered = sorted(lines)
            if len(ordered) > 16:  # keep the 16 lines closest to a coin flip
                ordered = sorted(sorted(ordered, key=lambda L: abs(lines[L] - 0.5))[:16])
            core = s in ("pts", "reb", "ast")
            pf, pb = (p.get("proj_full", {}).get(s), p.get("proj_base", {}).get(s)) if core else (None, None)
            p["strip"][s] = [dict(line=L, p_over=lines[L], hit=sum(1 for r in l10 if val(r) > L), n=n,
                                  boost=(lines[L] - float(nb_over(pf, alphas[s], L))) if pf is not None and p.get("absent") else 0.0,
                                  fade=(lines[L] - float(nb_over(pb, alphas[s], L))) if pb is not None else 0.0,
                                  price=(bl[L]["price"] if L in bl else None), book=(bl[L]["book"] if L in bl else None),
                                  alt=(bl[L]["alt"] if L in bl else None),
                                  edge=((lines[L] - 1 / bl[L]["price"]) if L in bl else None)) for L in ordered]
        for s, pk in (("dd", "p_dd"), ("td", "p_td")):
            sides = b.get(s, {}).get(0.0, {})
            if "over" in sides: p["book"][s] = dict(price=sides["over"][0], book=sides["over"][1], edge=p[pk] - 1 / sides["over"][0])
    return preds

def game_preds():
    with closing(con()) as c:
        try: return {r["game_id"]: json.loads(r["payload"]) for r in c.execute("SELECT game_id, payload FROM game_preds")}
        except sqlite3.OperationalError: return {}

def _alt(sides, side):
    """[price, book] when another bookmaker pays at least 3% more than the one shown (Sportsbet preferred)."""
    cur, best = sides.get(side), sides.get(side + "_best")
    return [best[0], best[1]] if cur and best and best[1] != cur[1] and best[0] >= cur[0] * 1.03 else None

def match_payload(gp, bl):
    """Market-based prices for winner/handicap/total/team totals/margin bands, with bookmaker prices merged in where the Odds API supplied them."""
    if not gp: return None
    m = {k: gp[k] for k in ("home", "away", "margin", "total", "home_pts", "away_pts", "p_home", "p_away", "handicaps", "totals", "team_totals", "bands") if k in gp}
    m["source"] = gp.get("source", "model")
    bl = bl or {}
    h2 = bl.get("h2h", {}); m["book_h2h"] = {s: list(h2[s]) for s in ("home", "away") if s in h2}
    m["alt_h2h"] = {s: _alt(h2, s) for s in ("home", "away") if _alt(h2, s)}
    hand = {l["line"]: dict(l) for l in m["handicaps"]}
    for line, sides in (bl.get("spread") or {}).items():
        e = hand.setdefault(line, dict(line=line, p_home=float(stats_norm_cdf((gp["margin"] + line) / gp["sigma_margin"]))))
        e["p_away"] = 1 - e["p_home"]
        for s in ("home", "away"):
            if s in sides: e[f"price_{s}"] = sides[s][0]; e["book"] = sides[s][1]; e[f"alt_{s}"] = _alt(sides, s)
    m["handicaps"] = [hand[k] for k in sorted(hand)]
    tot = {l["line"]: dict(l) for l in m["totals"]}
    for line, sides in (bl.get("total") or {}).items():
        e = tot.setdefault(line, dict(line=line, p_over=float(stats_norm_cdf((gp["total"] - line) / gp["sigma_total"]))))
        e["p_under"] = 1 - e["p_over"]
        for s in ("over", "under"):
            if s in sides: e[f"price_{s}"] = sides[s][0]; e["book"] = sides[s][1]; e[f"alt_{s}"] = _alt(sides, s)
    m["totals"] = [tot[k] for k in sorted(tot)]
    return m

def stats_norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)

@app.get("/api/slate")
def get_slate(x_code: str = Header("")):
    check(x_code)
    import odds
    games = {}; gp = game_preds(); with_c = closing(con())
    with with_c as c: bl = odds.load_lines(c)
    for p in enriched():
        g = games.setdefault(p["game_id"], dict(game_id=p["game_id"], start=p["start"], teams={}, match=match_payload(gp.get(p["game_id"]), bl.get(p["game_id"]))))
        g["teams"].setdefault(p["team"], dict(team=p["team"], opp=p["opp"], home=p["home"], lineup=p["lineup_status"], players=[]))["players"].append(p)
    return sorted(games.values(), key=lambda g: g["start"])

def bkName_(b):
    return "Sportsbet" if b == "sportsbet" else (b or "the bookmaker")

def _conf(p, s, line, hit, n, edge, with_evidence):
    """Simple, transparent confidence points (each one is shown to the user as evidence)."""
    sd = (p.get("sd") or {}).get(s); m = p.get("min"); pts = 0; why = []
    if n and hit / n >= 0.7: pts += 1
    if sd and abs(p["proj"][s] - line) >= 0.6 * sd: pts += 1
    if p["starting"] and m and m >= 28 and ((p.get("sd") or {}).get("min") or 9) <= 5: pts += 1
    if with_evidence: pts += 1
    if (p.get("injury") or {}).get("status") in ("Questionable", "Day-To-Day", "Doubtful"): pts -= 1
    if edge >= 0.10: pts += 1
    return "Strong" if pts >= 4 else "Medium" if pts >= 3 else "Speculative"

def _evidence(p, s, line, prob, price, edge, adj):
    parts = COMBO_PARTS.get(s, (s,)); pv = sum(p["proj"][c] for c in parts); lab = {"pra": "pts+reb+ast", "pr": "pts+reb", "pa": "pts+ast", "fg3m": "3PM"}.get(s, s)
    ev = [f"Model chance {prob*100:.0f}% vs price implies {100/price:.0f}% (edge {edge*100:+.0f} pts; {adj*100:.0f}% after leaning 30% toward the market to stay cautious)"]
    ev.append(f"Projection {pv:.1f} {lab} vs line {line}")
    l10 = [sum(r[c] for c in parts) for r in p.get("last10", [])]
    if l10: ev.append(f"Cleared {line} in {sum(1 for v in l10 if v > line)} of his last {len(l10)} games (avg {sum(l10)/len(l10):.1f})")
    if p.get("min"): ev.append(f"Averaging {p['min']:.0f} minutes" + (", starting tonight" if p["starting"] else ", not a confirmed starter"))
    if s in ("pts", "reb", "ast"):
        of = (p.get("opp_factor") or {}).get(s)
        if of and abs(of - 1) >= 0.03: ev.append(f"{p['opp']} has allowed {'more' if of > 1 else 'less'} {s} than average lately ({(of-1)*100:+.0f}%)")
    for a in (p.get("absent") or [])[:3]:
        ev.append(f"Teammate {a['name']} ({a['pts']:.1f} ppg) is {'out' if a['weight'] >= 0.85 else 'questionable'}")
    for e in (p.get("evidence") or [])[:2]:
        if s in ("pts", "reb", "ast"):
            ev.append(f"Without {e['who']}: {e['without'][s]:.1f} {s} in {e['without']['n']} games vs {e['with_'][s]:.1f} in {e['with_']['n']} with him")
    inj = (p.get("injury") or {}).get("status")
    if inj and inj != "Active": ev.append(f"Caution: listed {inj}")
    return ev

@app.get("/api/value")
def value_picks(x_code: str = Header("")):
    """Legs where the model's chance clearly beats the bookmaker's price, each with the evidence behind it. Needs bookmaker prices."""
    check(x_code)
    preds = enriched(); out = []
    for p in preds:
        if (p.get("injury") or {}).get("status") in ("Out",): continue
        for s in ("pts", "reb", "ast", "pra", "pr", "pa", "fg3m"):
            for l in p["strip"].get(s, []):
                if not l["price"]: continue
                edge = l["p_over"] - 1 / l["price"]; adj = 0.7 * l["p_over"] + 0.3 / l["price"]; aedge = adj - 1 / l["price"]
                if edge < 0.06 or l["n"] < 5: continue
                evd = bool(p.get("evidence")) and l.get("boost", 0) >= 0.03
                out.append(dict(kind="player", game_id=p["game_id"], player_id=p["player_id"], player=p["player"], team=p["team"], stat=s, line=l["line"], side="over",
                                price=l["price"], book=l["book"], p=l["p_over"], edge=edge, adj_edge=aedge, hit=l["hit"], n=l["n"], boost=l.get("boost", 0),
                                confidence=_conf(p, s, l["line"], l["hit"], l["n"], edge, evd), evidence=_evidence(p, s, l["line"], l["p_over"], l["price"], edge, adj)))
        for s, pk in (("dd", "p_dd"), ("td", "p_td")):
            b = p["book"].get(s)
            if not b or not p.get("show_" + s): continue
            edge = p[pk] - 1 / b["price"]; adj = 0.7 * p[pk] + 0.3 / b["price"]
            if edge < 0.06: continue
            hk = p[f"{s}_hit"]
            out.append(dict(kind="player", game_id=p["game_id"], player_id=p["player_id"], player=p["player"], team=p["team"], stat=s, line=0, side="over", price=b["price"], book=b["book"],
                            p=p[pk], edge=edge, adj_edge=adj - 1 / b["price"], hit=hk["k"], n=hk["n"], boost=0,
                            confidence="Medium" if edge >= 0.1 and hk["k"] / max(hk["n"], 1) >= 0.25 else "Speculative",
                            evidence=[f"Model chance {p[pk]*100:.1f}% vs price implies {100/b['price']:.1f}%", f"Did it in {hk['k']} of his last {hk['n']} games"] +
                                     [f"Teammate {a['name']} ({a['pts']:.1f} ppg) is out" for a in (p.get('absent') or [])[:2]]))
    # one best line per player/stat, ranked by cautious edge
    best = {}
    for o in out:
        k = (o["player_id"], o["stat"])
        if k not in best or o["adj_edge"] > best[k]["adj_edge"]: best[k] = o
    ranked = sorted(best.values(), key=lambda o: -o["adj_edge"])[:12]
    # team markets from bookmaker prices
    gp = game_preds(); import odds as _o
    with closing(con()) as c: bl = _o.load_lines(c)
    for gid, g in gp.items():
        b = bl.get(gid) or {}
        if g.get("source") != "market": continue   # without a market line our own team model has no proven edge, so no team value claims
        for side, pk, nmx in (("home", "p_home", g["home"]), ("away", "p_away", g["away"])):
            pr = (b.get("h2h") or {}).get(side)
            if pr and g[pk] - 1 / pr[0] >= 0.04:
                ranked.append(dict(kind="team", game_id=gid, stat="ml", side=side, line=0, text=f"{nmx} to win", price=pr[0], book=pr[1], p=g[pk], edge=g[pk] - 1 / pr[0],
                                   adj_edge=g[pk] - 1 / pr[0], confidence="Medium" if g[pk] - 1 / pr[0] >= 0.06 else "Speculative",
                                   evidence=[f"Betting-market consensus (DraftKings) puts {nmx} at {g[pk]*100:.0f}% to win; {bkName_(pr[1])}'s price implies only {100/pr[0]:.0f}%",
                                             f"Expected score {g['away']} {g['away_pts']:.0f} – {g['home']} {g['home_pts']:.0f} (market spread and total)",
                                             "In testing, market lines predicted winners better than our own team model, so this is a price comparison, not a model opinion"]))
    ranked.sort(key=lambda o: -o["adj_edge"])
    return dict(picks=ranked[:10], odds_connected=bool(_o.available()))

@app.get("/api/best")
def best(x_code: str = Header("")):
    """Safest legs of the day plus a suggested 2-3 leg combo aiming for fair odds of ~2.0+ while staying likely."""
    check(x_code)
    preds = enriched(); cands = []
    for p in preds:
        if p["injury"] and p["injury"].get("status") in ("Out", "Doubtful"): continue
        for s in ("pts", "reb", "ast"):
            opts = [dict(line=b["line"], p=b["p_over"], hit=b["hit"], n=b["n"], price=b["price"]) for b in p["book"].get(s, [])] or \
                   [dict(line=l["line"], p=l["p_over"], hit=l["hit"], n=l["n"], price=None) for l in p["ladders"][s]]
            ok = [o for o in opts if o["p"] >= 0.62 and o["n"] >= 5 and o["hit"] / o["n"] >= 0.7]
            if not ok: continue
            o = max(ok, key=lambda x: x["line"])  # highest line that is still safe = best payout among safe legs
            if o["p"] > 0.86: continue  # too short to be worth a leg (pays almost nothing)
            cands.append(dict(game_id=p["game_id"], player_id=p["player_id"], player=p["player"], team=p["team"], stat=s, line=o["line"], side="over",
                              p=o["p"], hit=o["hit"], n=o["n"], price=o["price"], starting=p["starting"], score=0.6 * o["p"] + 0.4 * o["hit"] / o["n"]))
    cands.sort(key=lambda x: -x["score"])
    combo, used, games = [], set(), set()
    pmap = {(p["game_id"], p["player_id"]): p for p in preds}
    alph = predict.load_models(); res = None
    for c in cands:
        if c["player_id"] in used or c["game_id"] in games: continue  # spread across players and games to limit correlation
        combo.append(c); used.add(c["player_id"]); games.add(c["game_id"])
        legs = [dict(game_id=x["game_id"], player_id=x["player_id"], stat=x["stat"], line=x["line"], side="over") for x in combo]
        res = pl.parlay(legs, pmap, alph["alphas"], alph["corr"], n=60000)
        if (res["fair_odds"] or 0) >= 2.0 or len(combo) == 3: break
    return dict(legs=cands[:8], combo=dict(legs=combo, joint=res["joint"] if res else None, fair_odds=res["fair_odds"] if res else None))

_status_cache = dict(t=0, data=None)
@app.get("/api/status")
def live_status(x_code: str = Header("")):
    """Cheap live check of lineup/injury changes for players on tonight's slate (cached 2 min)."""
    check(x_code)
    if time.time() - _status_cache["t"] > 120:
        try: _status_cache.update(t=time.time(), data=(sl.lineups(), sl.injuries()))
        except Exception: logging.exception("status fetch failed"); return {}
    lu, inj = _status_cache["data"] or ({}, {}); out = {}
    for p in projections():
        t = lu.get(p["team"], {}); out_names = [n for n, _ in t.get("out", [])]
        ij = next((v for n, v in inj.items() if sl.name_match(p["player"], n) and v["team"] == p["team"]), None)
        out[p["player_id"]] = dict(player=p["player"], starting=any(sl.name_match(p["player"], s) for s in t.get("starters", [])),
                                   listed_out=any(sl.name_match(p["player"], n) for n in out_names), injury=ij["status"] if ij else None,
                                   lineup=t.get("status", ""))
    return out

_lu_cache = dict(t=0, data=None, starts={})
@app.get("/api/lineups")
def lineups_tab(x_code: str = Header("")):
    """Ad-free starting lineups (Rotowire data) with photos, our top-3 picks flagged, and starters who differ from the team's last game."""
    check(x_code)
    if time.time() - _lu_cache["t"] > 300:
        try:
            _lu_cache.update(t=time.time(), data=sl.lineups_full(),
                             starts={(g["home"], g["away"]): g["start"] for g in sl.upcoming(2)})
        except Exception:
            logging.exception("lineups fetch failed")
            if not _lu_cache["data"]: raise HTTPException(502, "Lineups are temporarily unavailable")
    with closing(con()) as c:
        roster = pd.read_sql("SELECT player_id, player, team, max(date) d FROM player_games GROUP BY player_id", c)
        pg = pd.read_sql("SELECT team, game_id, date, player, starter FROM player_games WHERE date >= date('now','-45 day')", c)
    last = pd.DataFrame(columns=["team", "player"])
    if not pg.empty:
        latest = pg.sort_values(["date", "game_id"]).groupby("team").tail(1)[["team", "game_id"]]
        last = pg.merge(latest, on=["team", "game_id"]); last = last[last.starter == 1]
    picks = {}
    for p in projections_all(): picks.setdefault(p["team"], set()).add(p["player_id"])
    last_starters = {t: list(d.player) for t, d in last.groupby("team")}
    ros = {t: list(zip(d.player_id, d.player)) for t, d in roster.groupby("team")}
    import copy
    games = copy.deepcopy(_lu_cache["data"] or [])
    for g in games:
        g["start"] = _lu_cache["starts"].get((g["home"]["team"], g["away"]["team"]))
        for side in (g["away"], g["home"]):
            for lst, is_start in ((side["starters"], True), (side["out"], False)):
                for r in lst:
                    r["player_id"] = next((pid for pid, nm in ros.get(side["team"], []) if sl.name_match(nm, r["name"])), None)
                    r["top3"] = r["player_id"] in picks.get(side["team"], set())
                    if is_start and last_starters.get(side["team"]):
                        r["new"] = not any(sl.name_match(nm, r["name"]) for nm in last_starters[side["team"]])
    return games

@app.get("/api/live")
def live_scores(x_code: str = Header("")):
    check(x_code)
    import live
    try: return live.scoreboard()
    except Exception: logging.exception("scoreboard failed"); raise HTTPException(502, "Live scores are temporarily unavailable")

@app.get("/api/live/{game_id}")
def live_box(game_id: str, x_code: str = Header("")):
    """Live box score plus which players are our tracked top-3 for this game, with their pre-game projections."""
    check(x_code)
    import live
    try: box = live.boxscore(game_id)
    except Exception: logging.exception("boxscore failed"); raise HTTPException(502, "Live stats are temporarily unavailable")
    box["tracked"] = {p["player_id"]: p["proj"] for p in projections_all() if p["game_id"] == game_id}
    return box

@app.get("/api/players")
def players(q: str = "", x_code: str = Header("")):
    check(x_code)
    if len(q.strip()) < 2: return []
    with closing(con()) as c:
        rows = c.execute("""SELECT player_id, player, team, max(date) d FROM player_games WHERE player LIKE ? GROUP BY player_id ORDER BY d DESC, player LIMIT 12""", (f"%{q.strip()}%",)).fetchall()
    return [dict(player_id=r["player_id"], player=r["player"], team=r["team"]) for r in rows]

@app.get("/api/player/{pid}")
def player(pid: str, x_code: str = Header("")):
    """Plain data for the player page: last 20 games (newest first), double/triple-double counts, tonight's projection if any."""
    check(x_code)
    with closing(con()) as c:
        d = pd.read_sql("SELECT date,opp,home,min,pts,reb,ast,season,player,team FROM player_games WHERE player_id=? ORDER BY date DESC", c, params=(pid,))
    if d.empty: raise HTTPException(404)
    d["dd"] = ((d.pts >= 10).astype(int) + (d.reb >= 10) + (d.ast >= 10) >= 2).astype(int)
    d["td"] = ((d.pts >= 10).astype(int) + (d.reb >= 10) + (d.ast >= 10) >= 3).astype(int)
    cur = d[d.season == d.season.iloc[0]]
    tonight = next((p for p in enriched() if p["player_id"] == pid), None)
    return dict(player=d.player.iloc[0], team=d.team.iloc[0], games=d.head(20)[["date", "opp", "home", "min", "pts", "reb", "ast", "dd", "td"]].to_dict("records"),
                season=dict(games=len(cur), dd=int(cur.dd.sum()), td=int(cur.td.sum()), pts=float(cur.pts.mean()), reb=float(cur.reb.mean()), ast=float(cur.ast.mean()), min=float(cur["min"].mean())),
                last20=dict(dd=int(d.head(20).dd.sum()), td=int(d.head(20).td.sum())),
                tonight=dict(game_id=tonight["game_id"], opp=tonight["opp"], home=tonight["home"], start=tonight["start"], proj=tonight["proj"], p_dd=tonight["p_dd"], p_td=tonight["p_td"]) if tonight else None)

TEAM_STATS = ("ml", "spread", "total", "tt", "band")
PLAYER_STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "pra", "pr", "pa", "dd", "td")
SIDE_GROUP, TOTAL_GROUP = ("ml", "spread", "band"), ("total", "tt")

def team_leg_prob(l, gp):
    k = l["stat"]
    if k == "ml": return gp["p_home"] if l["side"] == "home" else gp["p_away"]
    if k == "spread":
        q = float(stats_norm_cdf((gp["margin"] + l["line"]) / gp["sigma_margin"])); return q if l["side"] == "home" else 1 - q
    if k == "total":
        q = float(stats_norm_cdf((gp["total"] - l["line"]) / gp["sigma_total"])); return q if l["side"] == "over" else 1 - q
    if k == "tt":
        mu = gp["home_pts"] if l["team"] == "home" else gp["away_pts"]; sg = float(np.sqrt((gp["sigma_total"] ** 2 + gp["sigma_margin"] ** 2) / 4))
        q = float(1 - stats_norm_cdf((l["line"] - mu) / sg)); return q if l["side"] == "over" else 1 - q
    b = next((x for x in gp["bands"] if x["team"] == l["team"] and x["lo"] == l["lo"] and x["hi"] == l.get("hi")), None)
    if not b: raise HTTPException(400, "Unknown margin band")
    return float(b["p"])

@app.post("/api/parlay")
def parlay_calc(body: dict = Body(...), x_code: str = Header("")):
    check(x_code)
    preds = {(p["game_id"], p["player_id"]): p for p in projections()}
    gps = game_preds(); legs = body["legs"]
    if not 1 <= len(legs) <= 6: raise HTTPException(400, "1-6 legs")
    pl_legs = [l for l in legs if l["stat"] not in TEAM_STATS]; tm_legs = [l for l in legs if l["stat"] in TEAM_STATS]
    for l in pl_legs:
        if l["stat"] not in PLAYER_STATS: raise HTTPException(400, "Unknown market")
        if (l["game_id"], l["player_id"]) not in preds: raise HTTPException(400, "Unknown player/game")
    for l in tm_legs:
        if l["game_id"] not in gps: raise HTTPException(400, "Unknown game")
    side_games = [l["game_id"] for l in tm_legs if l["stat"] in SIDE_GROUP]
    if len(side_games) != len(set(side_games)): raise HTTPException(400, "Pick only one winner, handicap or margin leg per game")
    tot_games = [l["game_id"] for l in tm_legs if l["stat"] in TOTAL_GROUP]
    if len(tot_games) != len(set(tot_games)): raise HTTPException(400, "Pick only one game total or team total per game")
    b = predict.load_models()
    rp = pl.parlay(pl_legs, preds, b["alphas"], b["corr"]) if pl_legs else dict(leg_probs=[], independent=1.0, joint=1.0)
    tp = [team_leg_prob(l, gps[l["game_id"]]) for l in tm_legs]
    pi, ti = iter(rp["leg_probs"]), iter(tp)
    probs = [next(ti) if l["stat"] in TEAM_STATS else next(pi) for l in legs]
    joint = rp["joint"] * float(np.prod(tp) if tp else 1.0); indep = rp["independent"] * float(np.prod(tp) if tp else 1.0)
    r = dict(leg_probs=probs, independent=indep, joint=joint, fair_odds=(1 / joint if joint > 0 else None))
    mixed = {l["game_id"] for l in tm_legs} & {l["game_id"] for l in pl_legs}
    r["warning"] = ("Team and player legs from the same game are linked (for example, a big scoring night helps both the player and the total), "
                    "so this combined chance is approximate.") if mixed else None
    odds = body.get("odds")
    if odds: r["value"] = pl.value(r["joint"], float(odds))
    r["legs"] = [dict(l, p=p) for l, p in zip(legs, probs)]
    return r

@app.get("/api/scorecard")
def scorecard(x_code: str = Header("")):
    check(x_code)
    rep = predict.load_models()["report"]
    with closing(con()) as c:
        bets = [dict(r) for r in c.execute("SELECT * FROM bets ORDER BY id DESC")]
    users = {}
    for b in bets:
        u = users.setdefault(b["user"], dict(user=b["user"], bets=0, settled=0, staked=0.0, returned=0.0, wins=0, model_prob_sum=0.0, all_prob_sum=0.0, open=0))
        u["bets"] += 1; u["all_prob_sum"] += b["model_prob"] or 0; u["open"] += b["result"] == "open"
        if b["result"] in ("won", "lost"):
            u["settled"] += 1; u["staked"] += b["stake"]; u["wins"] += b["result"] == "won"
            u["returned"] += b["stake"] * b["odds"] if b["result"] == "won" else 0; u["model_prob_sum"] += b["model_prob"] or 0
    for u in users.values():
        u["profit"] = u["returned"] - u["staked"]; u["avg_model_prob"] = u["all_prob_sum"] / u["bets"]
        u["roi"] = (u["returned"] - u["staked"]) / u["staked"] if u["staked"] else None
        u["hit_rate"] = u["wins"] / u["settled"] if u["settled"] else None
        u["expected_hit_rate"] = u["model_prob_sum"] / u["settled"] if u["settled"] else None
    return dict(model=rep, users=sorted(users.values(), key=lambda u: -(u["profit"])), bets=bets[:100])

@app.post("/api/bets")
def add_bet(body: dict = Body(...), x_user: str = Header(""), x_code: str = Header("")):
    user = who(x_user, x_code)
    with closing(con()) as c:
        c.execute("INSERT INTO bets(user,created,legs,stake,odds,model_prob) VALUES(?,?,?,?,?,?)",
                  (user, dt.datetime.now(dt.timezone.utc).isoformat(), json.dumps(body["legs"]), float(body["stake"]), float(body["odds"]), body.get("model_prob")))
        c.commit()
    return {"ok": True}

@app.post("/api/bets/{bid}/result")
def bet_result(bid: int, body: dict = Body(...), x_user: str = Header(""), x_code: str = Header("")):
    user = who(x_user, x_code)
    if body["result"] not in ("won", "lost", "void", "open"): raise HTTPException(400)
    with closing(con()) as c:
        r = c.execute("UPDATE bets SET result=? WHERE id=? AND user=?", (body["result"], bid, user)); c.commit()
        if r.rowcount == 0: raise HTTPException(404, "Not your bet")
    return {"ok": True}

@app.get("/api/picks")
def picks(x_code: str = Header("")):
    check(x_code)
    with closing(con()) as c:
        ps = [dict(r) for r in c.execute("SELECT * FROM picks ORDER BY id DESC LIMIT 100")]
        for p in ps:
            p["legs"] = json.loads(p["legs"])
            p["comments"] = [dict(r) for r in c.execute("SELECT * FROM comments WHERE pick_id=? ORDER BY id", (p["id"],))]
            v = c.execute("SELECT coalesce(sum(v),0) s, count(*) n FROM votes WHERE pick_id=?", (p["id"],)).fetchone(); p["score"], p["voters"] = v["s"], v["n"]
    return ps

@app.post("/api/picks")
def add_pick(body: dict = Body(...), x_user: str = Header(""), x_code: str = Header("")):
    user = who(x_user, x_code)
    with closing(con()) as c:
        c.execute("INSERT INTO picks(user,created,game_id,legs,note,odds,model_prob) VALUES(?,?,?,?,?,?,?)",
                  (user, dt.datetime.now(dt.timezone.utc).isoformat(), "", json.dumps(body["legs"]), body.get("note", "")[:500], body.get("odds"), body.get("model_prob"))); c.commit()
    return {"ok": True}

@app.post("/api/picks/{pid}/comment")
def comment(pid: int, body: dict = Body(...), x_user: str = Header(""), x_code: str = Header("")):
    user = who(x_user, x_code)
    with closing(con()) as c:
        c.execute("INSERT INTO comments(pick_id,user,created,body) VALUES(?,?,?,?)", (pid, user, dt.datetime.now(dt.timezone.utc).isoformat(), body["body"][:500])); c.commit()
    return {"ok": True}

@app.post("/api/picks/{pid}/vote")
def vote(pid: int, body: dict = Body(...), x_user: str = Header(""), x_code: str = Header("")):
    user = who(x_user, x_code); v = 1 if body["v"] > 0 else -1
    with closing(con()) as c:
        c.execute("INSERT OR REPLACE INTO votes VALUES(?,?,?)", (pid, user, v)); c.commit()
    return {"ok": True}

@app.post("/api/refresh")
def refresh(x_code: str = Header("")):
    check(x_code)
    import ingest
    ingest.run(dt.date.today() - dt.timedelta(days=4), dt.date.today())
    preds = predict.project(); predict.save(preds)
    return dict(projections=len(preds))

app.mount("/", StaticFiles(directory="static", html=True), name="static")
