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

def _refresh_loop():
    """Background refresh every 4h: new box scores, then re-project upcoming games. Retrains weekly."""
    import ingest
    last_train = 0
    while True:
        try:
            t = dt.date.today(); ingest.run(t - dt.timedelta(days=4), t)
            if time.time() - last_train > 7 * 86400: predict.train(); last_train = time.time()
            predict.save(predict.project())
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

def projections():
    with closing(con()) as c:
        rows = [json.loads(r["payload"]) for r in c.execute("SELECT payload FROM projections")]
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return [r for r in rows if r["start"] >= (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")]

@app.get("/api/meta")
def meta(x_code: str = Header("")):
    check(x_code)
    b = predict.load_models()
    with closing(con()) as c:
        last = c.execute("SELECT max(date) d FROM games").fetchone()["d"]
        made = c.execute("SELECT max(made_at) m FROM projections").fetchone()["m"]
    return dict(data_through=last, projections_made=made, trained_through=b["trained_through"],
                odds_api=__import__("odds").available(), open_access=not ACCESS_CODE)

@app.get("/api/slate")
def get_slate(x_code: str = Header("")):
    check(x_code)
    preds = projections(); games = {}
    for p in preds:
        g = games.setdefault(p["game_id"], dict(game_id=p["game_id"], start=p["start"], teams={}))
        g["teams"].setdefault(p["team"], dict(team=p["team"], opp=p["opp"], home=p["home"], lineup=p["lineup_status"], players=[]))["players"].append(p)
    ids = list({p["player_id"] for p in preds})
    last5 = {}
    if ids:
        with closing(con()) as c:
            q = ",".join("?" * len(ids))
            for r in c.execute(f"SELECT player_id,date,opp,home,min,pts,reb,ast FROM player_games WHERE player_id IN ({q}) ORDER BY date DESC", ids):
                last5.setdefault(r["player_id"], [])
                if len(last5[r["player_id"]]) < 5: last5[r["player_id"]].append(dict(r))
    for p in preds: p["last5"] = last5.get(p["player_id"], [])
    return sorted(games.values(), key=lambda g: g["start"])

@app.get("/api/player/{pid}")
def player(pid: str, x_code: str = Header("")):
    """Consistency card: hit rates over recent windows at a ladder of lines, plus double/triple-double history."""
    check(x_code)
    with closing(con()) as c:
        d = pd.read_sql("SELECT * FROM player_games WHERE player_id=? ORDER BY date", c, params=(pid,))
    if d.empty: raise HTTPException(404)
    d["td"] = ((d.pts >= 10).astype(int) + (d.reb >= 10) + (d.ast >= 10) >= 3).astype(int)
    d["dd"] = ((d.pts >= 10).astype(int) + (d.reb >= 10) + (d.ast >= 10) >= 2).astype(int)
    out = dict(player=d.player.iloc[-1], team=d.team.iloc[-1], games=len(d), hit={}, last=[], td=dict(l10=int(d.td.tail(10).sum()), l20=int(d.td.tail(20).sum()), l40=int(d.td.tail(40).sum()), season=int(d.td.sum()), n=len(d)),
               dd=dict(l10=int(d.dd.tail(10).sum()), l20=int(d.dd.tail(20).sum()), l40=int(d.dd.tail(40).sum())))
    for s in ("pts", "reb", "ast"):
        mean = d[s].tail(20).mean(); lines = sorted({max(0.5, np.floor(mean) + k + 0.5) for k in range(-5, 6)})
        out["hit"][s] = dict(mean20=float(mean), sd20=float(d[s].tail(20).std()), rows=[
            dict(line=float(l), **{f"l{w}": float((d[s].tail(w) > l).mean()) for w in (5, 10, 20, 40)}) for l in lines])
    out["last"] = d.tail(10)[["date", "opp", "home", "min", "pts", "reb", "ast", "td"]].iloc[::-1].to_dict("records")
    return out

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
        u = users.setdefault(b["user"], dict(user=b["user"], bets=0, settled=0, staked=0.0, returned=0.0, wins=0, model_prob_sum=0.0))
        u["bets"] += 1
        if b["result"] in ("won", "lost"):
            u["settled"] += 1; u["staked"] += b["stake"]; u["wins"] += b["result"] == "won"
            u["returned"] += b["stake"] * b["odds"] if b["result"] == "won" else 0; u["model_prob_sum"] += b["model_prob"] or 0
    for u in users.values():
        u["roi"] = (u["returned"] - u["staked"]) / u["staked"] if u["staked"] else None
        u["hit_rate"] = u["wins"] / u["settled"] if u["settled"] else None
        u["expected_hit_rate"] = u["model_prob_sum"] / u["settled"] if u["settled"] else None
    return dict(model=rep, users=list(users.values()), bets=bets[:100])

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
