"""Train final models, project upcoming games for each team's top-3 players, write results to SQLite + JSON."""
import json, pickle, sqlite3, datetime as dt, os
import numpy as np, pandas as pd
from scipy import stats
from features import build, FEATURES
from model import rotation, fit, dispersion, nb_over, TARGETS
from top3 import top3
import slate as sl
import team_model

DB = os.path.join(os.environ.get("DATA_DIR", "data"), "nba.db"); MODELS = os.path.join(os.environ.get("DATA_DIR", "data"), "models.pkl")

def residual_corr(df, m, alphas, cut="2026-02-01"):
    """Gaussian-copula correlations of PIT normal scores: same player across stats, teammates, opponents."""
    te = df[df.date >= cut].copy()
    for t in TARGETS:
        mu = m[t].predict(te[FEATURES]); a = alphas[t]; n = 1 / a; p = n / (n + mu)
        lo = stats.nbinom.cdf(te[t] - 1, n, p); hi = stats.nbinom.cdf(te[t], n, p)
        te[f"z_{t}"] = stats.norm.ppf(np.clip((lo + hi) / 2, 1e-3, 1 - 1e-3))
    same = te[[f"z_{t}" for t in TARGETS]].corr().values
    # teammate / opponent correlations: pair players within a game among high-minute players
    tm, op = {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    for _, g in te[te.min_m10 >= 28].groupby("game_id"):
        for i, r in g.iterrows():
            for j, q in g.iterrows():
                if r.player_id >= q.player_id: continue
                (tm if r.team == q.team else op)  # noqa
                for t in TARGETS:
                    (tm if r.team == q.team else op)[t].append((r[f"z_{t}"], q[f"z_{t}"]))
    cross = lambda d: {t: float(np.corrcoef(np.array(v).T)[0, 1]) for t, v in d.items() if len(v) > 30}
    return dict(same=same.tolist(), teammate=cross(tm), opponent=cross(op))

def train(report=True):
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    cut = "2026-02-01"; tr = full[full.date < cut]
    m_eval = fit(tr); alphas = {t: dispersion(m_eval[t], tr[FEATURES], tr[t]) for t in TARGETS}
    corr = residual_corr(full, m_eval, alphas)
    m = fit(full); alphas = {t: dispersion(m[t], full[FEATURES], full[t]) for t in TARGETS}
    from model import evaluate
    pickle.dump(dict(models=m, alphas=alphas, corr=corr, trained_through=str(full.date.max()), report=evaluate(full)), open(MODELS, "wb"))
    return m, alphas, corr

def load_models():
    if not os.path.exists(MODELS): train()
    return pickle.load(open(MODELS, "rb"))

def pick_top3(raw, team, out_names, lineup_names):
    """Top-3 by recent production, skipping anyone ruled out (injury report or Rotowire may-not-play)."""
    # pull more than 3 candidates so injured players can be replaced
    cand = top3_candidates(raw, team)
    keep = []
    for _, r in cand.iterrows():
        if any(sl.name_match(r.player, o) for o in out_names): continue
        keep.append(r)
        if len(keep) == 3: break
    return keep

def top3_candidates(raw, team, window=15):
    d = raw[raw.cur_team == team].assign(pra=lambda x: x.pts + x.reb + x.ast)
    d = d.sort_values("date").groupby("player_id").tail(window)
    g = d.groupby(["player_id", "player"]).agg(games=("pra", "size"), min=("min", "mean"), pra=("pra", "mean")).reset_index()
    g = g[g.games >= 2]; g["score"] = g.pra + 0.5 * g["min"]
    return g.sort_values("score", ascending=False)

def project(days=2):
    bundle = load_models(); m, alphas = bundle["models"], bundle["alphas"]
    games = [g for g in sl.upcoming(days) if g["status"] == "STATUS_SCHEDULED"]
    inj = sl.injuries(); lu = sl.lineups()
    raw = pd.read_sql("SELECT * FROM player_games", sqlite3.connect(DB))
    latest_team = raw.sort_values("date").groupby("player_id").tail(1)[["player_id", "team"]]
    raw = raw.merge(latest_team.rename(columns={"team": "cur_team"}), on="player_id")
    rows, meta = [], []
    for g in games:
        for team, opp, home in ((g["home"], g["away"], 1), (g["away"], g["home"], 0)):
            outs = [n for n, s in (lu.get(team, {}).get("out", []))] + \
                   [n for n, v in inj.items() if v["team"] == team and v["status"] in ("Out", "Doubtful")]
            picks = pick_top3(raw, team, outs, lu.get(team, {}).get("starters", []))
            for r in picks:
                pg = raw[raw.player_id == r.player_id].iloc[-1]
                rows.append(dict(game_id=g["game_id"], player_id=r.player_id, player=r.player, team=team, opp=opp, home=home,
                                 date=g["date"], season=pg.season, starter=0, min=0., pts=0, reb=0, ast=0, stl=0, blk=0, tov=0,
                                 fg3m=0, plusminus=0, team_id="", _upcoming=1))
                meta.append(dict(game=g, player_id=r.player_id,
                                 starting=any(sl.name_match(r.player, s) for s in lu.get(team, {}).get("starters", [])),
                                 lineup_status=lu.get(team, {}).get("status", "No lineup posted yet"),
                                 injury=next((v for n, v in inj.items() if sl.name_match(r.player, n) and v["team"] == team), None)))
    try:
        gp = team_model.predict([dict(game_id=g["game_id"], date=g["date"], home=g["home"], away=g["away"]) for g in games])
        project.game_preds = {gid: dict(team_model.price(v), game_id=gid, start=next(x["start"] for x in games if x["game_id"] == gid),
                                        home=next(x["home"] for x in games if x["game_id"] == gid), away=next(x["away"] for x in games if x["game_id"] == gid))
                              for gid, v in gp.items()}
    except Exception as e:
        print("team model failed:", e); project.game_preds = {}
    if not rows: return []
    ph = pd.DataFrame(rows)
    base = raw.drop(columns=["cur_team"]).assign(_upcoming=0)
    feats = build(pd.concat([base, ph], ignore_index=True))  # placeholder rows only read earlier games
    feats = feats[feats._upcoming == 1] if "_upcoming" in feats else feats
    out = []
    for (_, f), mt in zip(feats.sort_values(["game_id", "team"]).iterrows(), sorted(meta, key=lambda x: (x["game"]["game_id"], 0)), ) if False else []:
        pass
    # align metadata by (game, player)
    mk = {(x["game"]["game_id"], x["player_id"]): x for x in meta}
    X = feats[FEATURES]
    mus = {t: m[t].predict(X) for t in TARGETS}
    pdd = m["dd"].predict_proba(X)[:, 1]; ptd = m["td"].predict_proba(feats)
    for k, (_, f) in enumerate(feats.iterrows()):
        mt = mk[(f.game_id, f.player_id)]; g = mt["game"]
        rec = dict(game_id=f.game_id, start=g["start"], date=g["date"], player_id=f.player_id, player=f.player, team=f.team, opp=f.opp,
                   home=int(f.home), starting=bool(mt["starting"]), lineup_status=mt["lineup_status"], injury=mt["injury"],
                   min=float(f.min_m10) if pd.notna(f.min_m10) else None, rest=float(f.rest) if pd.notna(f.rest) else None,
                   p_dd=float(pdd[k]), p_td=float(min(ptd[k], pdd[k])), proj={}, ladders={})
        for t in TARGETS:
            mu = float(mus[t][k]); rec["proj"][t] = mu
            lines = sorted({max(0.5, np.floor(mu) + d + 0.5) for d in range(-6 if t == "pts" else -4, 7 if t == "pts" else 5)})
            rec["ladders"][t] = [dict(line=float(l), p_over=float(nb_over(mu, alphas[t], l))) for l in lines]
        out.append(rec)
    return out

def save(preds):
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS projections(game_id TEXT, player_id TEXT, payload TEXT, made_at TEXT, PRIMARY KEY(game_id, player_id))")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    # replace rows for re-projected games; keep rows for games that started in the last 12h (the Live tab marks our picks);
    # drop anything older
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%MZ")
    for gid, payload in con.execute("SELECT game_id, payload FROM projections").fetchall():
        if json.loads(payload)["start"] < cutoff: con.execute("DELETE FROM projections WHERE game_id=?", (gid,))
    con.executemany("DELETE FROM projections WHERE game_id=?", [(g,) for g in {x["game_id"] for x in preds}])
    for p in preds:
        con.execute("INSERT OR REPLACE INTO projections VALUES(?,?,?,?)", (p["game_id"], p["player_id"], json.dumps(p), now))
    con.execute("CREATE TABLE IF NOT EXISTS game_preds(game_id TEXT PRIMARY KEY, payload TEXT, made_at TEXT)")
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%MZ")
    for gid, payload in con.execute("SELECT game_id, payload FROM game_preds").fetchall():
        if json.loads(payload)["start"] < cutoff: con.execute("DELETE FROM game_preds WHERE game_id=?", (gid,))
    for gid, v in getattr(project, "game_preds", {}).items():
        con.execute("INSERT OR REPLACE INTO game_preds VALUES(?,?,?)", (gid, json.dumps(v, default=float), now))
    con.commit()

if __name__ == "__main__":
    import sys
    if "--train" in sys.argv: train()
    preds = project(); save(preds)
    print(len(preds), "projections")
    for p in preds[:6]:
        print(p["team"], "vs", p["opp"], p["player"], {k: round(v, 1) for k, v in p["proj"].items()}, "DD", round(p["p_dd"], 2), "TD", round(p["p_td"], 3), "start" if p["starting"] else "")
