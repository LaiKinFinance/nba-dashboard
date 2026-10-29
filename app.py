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
    return odds.refresh(list(games.values()), pbg)

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

def enriched():
    """Projections + last-10 games, consistency counts per line, and bookmaker prices/edges where available."""
    import odds
    from model import nb_over
    preds = projections()
    ids = list({p["player_id"] for p in preds}); rows = {}
    with closing(con()) as c:
        if ids:
            q = ",".join("?" * len(ids))
            for r in c.execute(f"SELECT player_id,date,opp,home,min,pts,reb,ast FROM player_games WHERE player_id IN ({q}) ORDER BY date DESC", ids):
                rows.setdefault(r["player_id"], [])
                if len(rows[r["player_id"]]) < 20: rows[r["player_id"]].append(dict(r))
        book = odds.load(c)
    alphas = predict.load_models()["alphas"]
    for p in preds:
        g = rows.get(p["player_id"], []); l10 = g[:10]; n = len(l10)
        p["last10"] = l10
        for s in ("pts", "reb", "ast"):
            for l in p["ladders"][s]: l["hit"] = sum(1 for r in l10 if r[s] > l["line"]); l["n"] = n
        p["dd_hit"] = dict(k=sum(_dd(r) for r in l10), n=n); p["td_hit"] = dict(k=sum(_td(r) for r in g), n=len(g))
        b = book.get((p["game_id"], p["player_id"]), {}); p["book"] = {}
        for s in ("pts", "reb", "ast"):
            out = []
            for line, sides in sorted(b.get(s, {}).items()):
                if "over" not in sides: continue
                po = float(nb_over(p["proj"][s], alphas[s], line)); price, bk = sides["over"]
                out.append(dict(line=line, price=price, book=bk, p_over=po, edge=po - 1 / price,
                                hit=sum(1 for r in l10 if r[s] > line), n=n, under=(sides.get("under") or [None])[0]))
            if out: p["book"][s] = out
        for s, pk in (("dd", "p_dd"), ("td", "p_td")):
            sides = b.get(s, {}).get(0.0, {})
            if "over" in sides: p["book"][s] = dict(price=sides["over"][0], book=sides["over"][1], edge=p[pk] - 1 / sides["over"][0])
    return preds

@app.get("/api/slate")
def get_slate(x_code: str = Header("")):
    check(x_code)
    games = {}
    for p in enriched():
        g = games.setdefault(p["game_id"], dict(game_id=p["game_id"], start=p["start"], teams={}))
        g["teams"].setdefault(p["team"], dict(team=p["team"], opp=p["opp"], home=p["home"], lineup=p["lineup_status"], players=[]))["players"].append(p)
    return sorted(games.values(), key=lambda g: g["start"])

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

@app.post("/api/parlay")
def parlay_calc(body: dict = Body(...), x_code: str = Header("")):
    check(x_code)
    preds = {(p["game_id"], p["player_id"]): p for p in projections()}
    legs = body["legs"]
    if not 1 <= len(legs) <= 6: raise HTTPException(400, "1-6 legs")
    for l in legs:
        if (l["game_id"], l["player_id"]) not in preds: raise HTTPException(400, "Unknown player/game")
    b = predict.load_models()
    r = pl.parlay(legs, preds, b["alphas"], b["corr"])
    odds = body.get("odds")
    if odds: r["value"] = pl.value(r["joint"], float(odds))
    r["legs"] = [dict(l, player=preds[(l["game_id"], l["player_id"])]["player"], p=p) for l, p in zip(legs, r["leg_probs"])]
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
