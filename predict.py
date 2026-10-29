"""Train final models, project upcoming games for each team's top-3 players, write results to SQLite + JSON."""
import json, pickle, sqlite3, datetime as dt, os
import numpy as np, pandas as pd
from scipy import stats
from features import build, FEATURES
from model import rotation, fit, dispersion, nb_over, TARGETS, ALL, feats_for
import combos
from top3 import top3
import slate as sl
import team_model
import absence

DB = os.path.join(os.environ.get("DATA_DIR", "data"), "nba.db"); MODELS = os.path.join(os.environ.get("DATA_DIR", "data"), "models.pkl")

def residual_corr(df, m, alphas, cut="2026-02-01"):
    """Gaussian-copula correlations of PIT normal scores: same player across all six stats, plus teammates / opponents (core stats)."""
    te = df[df.date >= cut].copy()
    for t in ALL:
        mu = m[t].predict(te[feats_for(t)]); a = alphas[t]; n = 1 / a; p = n / (n + np.maximum(mu, 0.05))
        lo = stats.nbinom.cdf(te[t] - 1, n, p); hi = stats.nbinom.cdf(te[t], n, p)
        te[f"z_{t}"] = stats.norm.ppf(np.clip((lo + hi) / 2, 1e-3, 1 - 1e-3))
    same6 = te[[f"z_{t}" for t in ALL]].corr().values
    same = same6[:3, :3]
    tm, op = {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    for _, g in te[te.min_m10 >= 28].groupby("game_id"):
        rows = list(g.itertuples())
        for i, r in enumerate(rows):
            for q in rows[i + 1:]:
                for t in TARGETS:
                    (tm if r.team == q.team else op)[t].append((getattr(r, f"z_{t}"), getattr(q, f"z_{t}")))
    cross = lambda d: {t: float(np.corrcoef(np.array(v).T)[0, 1]) for t, v in d.items() if len(v) > 30}
    return dict(same=same.tolist(), same6=same6.tolist(), order=list(ALL), teammate=cross(tm), opponent=cross(op))

def _team_report():
    try:
        g = pd.read_sql("SELECT * FROM games", sqlite3.connect(DB)); return team_model.evaluate(team_model.build(g))
    except Exception as e:
        print("team report failed:", e); return None

def train(report=True):
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    cut = "2026-02-01"; tr = full[full.date < cut]
    m_eval = fit(tr); alphas = {t: dispersion(m_eval[t], tr[feats_for(t)], tr[t]) for t in ALL}
    corr = residual_corr(full, m_eval, alphas)
    m = fit(full); alphas = {t: dispersion(m[t], full[feats_for(t)], full[t]) for t in ALL}
    from model import evaluate
    pickle.dump(dict(models=m, alphas=alphas, corr=corr, trained_through=str(full.date.max()), report=dict(evaluate(full), team=_team_report())), open(MODELS, "wb"))
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

STATUS_W = {"out": 1.0, "ofs": 1.0, "doubtful": 0.85, "questionable": 0.5, "game-time decision": 0.5, "day-to-day": 0.5, "probable": 0.1}

def status_weight(s):
    return STATUS_W.get((s or "").strip().lower(), 0.5 if s else 1.0)

def tonight_absent(team, lu, inj, state):
    """Regulars of `team` who are out/doubtful/questionable tonight, weighted by how likely they are to miss, and the summed absence features."""
    entries = [(n, status_weight(s)) for n, s in lu.get(team, {}).get("out_status", [])] + \
              [(n, status_weight(v["status"])) for n, v in inj.items() if v["team"] == team]
    found, feats = [], dict(abs_min=0.0, abs_pts=0.0, abs_reb=0.0, abs_ast=0.0, abs_n=0.0)
    for pid, s in absence.regulars(state.get(team, {})).items():
        w = max([wt for n, wt in entries if sl.name_match(s["name"], n)] or [0.0])
        if w < 0.3: continue
        found.append(dict(player_id=pid, name=s["name"], weight=w, min=s["min"], pts=s["pts"], reb=s["reb"], ast=s["ast"]))
        for c in absence.COLS: feats["abs_" + c] += w * s[c]
        feats["abs_n"] += w
    found.sort(key=lambda x: -x["pts"] * x["weight"])
    return feats, found

def with_without(real, team, pid, apid):
    """This player's own stats in games the teammate played vs games he missed (while still on the team that season)."""
    mine = real[(real.team == team) & (real.player_id == pid)]
    t = real[(real.team == team)]
    in_game = t[t.player_id == apid].game_id
    seasons = set(t[t.player_id == apid].season)
    has = mine.game_id.isin(set(in_game)); without = (~has) & mine.season.isin(seasons)
    def agg(x): return dict(n=int(len(x)), pts=float(x.pts.mean()), reb=float(x.reb.mean()), ast=float(x.ast.mean()), min=float(x["min"].mean())) if len(x) else None
    return agg(mine[has]), agg(mine[without])

def current_teams():
    """player_id -> team from the stored ESPN rosters (repair.refresh_rosters); None if the table is missing or looks incomplete."""
    try:
        r = pd.read_sql("SELECT player_id, team AS cur_team FROM rosters WHERE team IS NOT NULL", sqlite3.connect(DB))
    except Exception:
        return None
    return r if len(r) >= 300 else None

def project(days=2):
    bundle = load_models(); m, alphas = bundle["models"], bundle["alphas"]
    games = [g for g in sl.upcoming(days) if g["status"] == "STATUS_SCHEDULED"]
    inj = sl.injuries(); lu = sl.lineups()
    raw = pd.read_sql("SELECT * FROM player_games", sqlite3.connect(DB))
    cur = current_teams()
    if cur is None:                                       # no roster feed yet: fall back to the team he last played for
        cur = raw.sort_values("date").groupby("player_id").tail(1)[["player_id", "team"]].rename(columns={"team": "cur_team"})
    raw = raw.merge(cur, on="player_id")                  # players on no current roster drop out (waived, retired, overseas)
    real = raw.drop(columns=["cur_team"])
    af, state = absence.track(real)
    recent = {t: af.xs(t, level="team").tail(10).mean() for t in set(af.index.get_level_values("team"))}
    absmap, rows, meta = {}, [], []
    for g in games:
        for team, opp, home in ((g["home"], g["away"], 1), (g["away"], g["home"], 0)):
            outs = [n for n, s in lu.get(team, {}).get("out_status", []) if status_weight(s) >= 0.85] + \
                   [n for n, v in inj.items() if v["team"] == team and status_weight(v["status"]) >= 0.85]  # questionable players stay in, flagged
            absmap[(g["game_id"], team)] = tonight_absent(team, lu, inj, state)
            picks = pick_top3(raw, team, outs, lu.get(team, {}).get("starters", []))
            for r in picks:
                pg = raw[raw.player_id == r.player_id].iloc[-1]
                starting_now = any(sl.name_match(r.player, s) for s in lu.get(team, {}).get("starters", []))
                rows.append(dict(game_id=g["game_id"], player_id=r.player_id, player=r.player, team=team, opp=opp, home=home,
                                 date=g["date"], season=pg.season, starter=int(starting_now), min=0., pts=0, reb=0, ast=0, stl=0, blk=0, tov=0,
                                 fg3m=0, plusminus=0, team_id="", _upcoming=1, ou=g.get("ou"), spread_home=g.get("spread_home")))
                meta.append(dict(game=g, player_id=r.player_id,
                                 starting=any(sl.name_match(r.player, s) for s in lu.get(team, {}).get("starters", [])),
                                 lineup_status=lu.get(team, {}).get("status", "No lineup posted yet"),
                                 injury=next((v for n, v in inj.items() if sl.name_match(r.player, n) and v["team"] == team), None)))
    try:
        gp = team_model.predict([dict(game_id=g["game_id"], date=g["date"], home=g["home"], away=g["away"], ou=g.get("ou"), spread_home=g.get("spread_home"), ml_home=g.get("ml_home"), ml_away=g.get("ml_away")) for g in games])
        project.game_preds = {gid: dict(team_model.price(v), game_id=gid, start=next(x["start"] for x in games if x["game_id"] == gid),
                                        home=next(x["home"] for x in games if x["game_id"] == gid), away=next(x["away"] for x in games if x["game_id"] == gid))
                              for gid, v in gp.items()}
    except Exception as e:
        print("team model failed:", e); project.game_preds = {}
    if not rows: return []
    ph = pd.DataFrame(rows)
    base = raw.drop(columns=["cur_team"]).assign(_upcoming=0)
    # A player with two games in the window must not read his first (still empty) placeholder as history for the second,
    # so each "wave" holds at most one upcoming game per player and is built separately against real games only.
    ph["wave"] = ph.sort_values(["date", "game_id"]).groupby("player_id").cumcount()
    parts = []
    for w in sorted(ph.wave.unique()):
        f = build(pd.concat([base, ph[ph.wave == w].drop(columns="wave")], ignore_index=True))
        parts.append(f[f._upcoming == 1])
    feats = pd.concat(parts, ignore_index=True)
    feats = feats.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    prev = feats.groupby("player_id").date.shift(1); later = prev.notna()     # rest before his 2nd game counts from his 1st upcoming game
    feats.loc[later, "rest"] = (pd.to_datetime(feats.date[later]) - pd.to_datetime(prev[later])).dt.days.clip(upper=10)
    feats["b2b"] = (feats.rest == 1).astype(int)
    out = []
    for (_, f), mt in zip(feats.sort_values(["game_id", "team"]).iterrows(), sorted(meta, key=lambda x: (x["game"]["game_id"], 0)), ) if False else []:
        pass
    # align metadata by (game, player)
    mk = {(x["game"]["game_id"], x["player_id"]): x for x in meta}
    feats = feats.copy(); Xb_rows = []
    for idx, f in feats.iterrows():
        a = absmap[(f.game_id, f.team)][0]; r = recent.get(f.team)
        for c in absence.ABS: feats.at[idx, c] = a[c]
    X = feats[FEATURES]
    Xb = X.copy()   # "recent norm" counterfactual: the absence level that was already baked into his last 10 games
    for idx, f in feats.iterrows():
        r = recent.get(f.team)
        for c in absence.ABS: Xb.at[idx, c] = float(r[c]) if r is not None else 0.0
    Xf = X.copy()   # "full strength" counterfactual: nobody missing
    for c in absence.ABS: Xf[c] = 0.0
    mus = {t: m[t].predict(feats[feats_for(t)]) for t in ALL}
    mus_base = {t: m[t].predict(Xb) for t in TARGETS}
    mus_full = {t: m[t].predict(Xf) for t in TARGETS}
    pdd = m["dd"].predict_proba(X)[:, 1]; ptd = m["td"].predict_proba(feats)
    for k, (_, f) in enumerate(feats.iterrows()):
        mt = mk[(f.game_id, f.player_id)]; g = mt["game"]
        rec = dict(game_id=f.game_id, start=g["start"], date=g["date"], player_id=f.player_id, player=f.player, team=f.team, opp=f.opp,
                   home=int(f.home), starting=bool(mt["starting"]), lineup_status=mt["lineup_status"], injury=mt["injury"],
                   min=float(f.min_m10) if pd.notna(f.min_m10) else None, rest=float(f.rest) if pd.notna(f.rest) else None,
                   p_dd=float(pdd[k]), p_td=float(min(ptd[k], pdd[k])), proj={}, ladders={})
        absent = absmap[(f.game_id, f.team)][1]
        rec["absent"] = [dict(a, weight=float(a["weight"])) for a in absent]
        rec["proj_base"] = {t: float(mus_base[t][k]) for t in TARGETS}
        rec["proj_full"] = {t: float(mus_full[t][k]) for t in TARGETS}
        rec["opp_factor"] = {t: (float(f[f"opp_allow_{t}"]) if pd.notna(f[f"opp_allow_{t}"]) else 1.0) for t in TARGETS}
        rec["sd"] = {t: (float(f[f"{t}_sd10"]) if pd.notna(f[f"{t}_sd10"]) else None) for t in ("pts", "reb", "ast", "min")}
        rec["evidence"] = []
        for a in absent[:4]:
            if a["player_id"] == f.player_id: continue
            w, wo = with_without(real, f.team, f.player_id, a["player_id"])
            if wo and wo["n"] >= 3 and w and w["n"] >= 3: rec["evidence"].append(dict(who=a["name"], who_pts=a["pts"], weight=a["weight"], with_=w, without=wo))
        for t in ALL:
            rec["proj"][t] = float(mus[t][k])
        rec["combo"] = combos.combo_lines({t: rec["proj"][t] for t in TARGETS}, alphas, bundle["corr"]["same"])
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
