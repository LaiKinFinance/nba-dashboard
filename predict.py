"""Train final models, project upcoming games for each team's top-3 players, write results to SQLite + JSON."""
import json, pickle, sqlite3, datetime as dt, os
import numpy as np, pandas as pd
from scipy import stats
from features import build, FEATURES, FEATURE_SCHEMA
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

class ModelMissing(Exception):
    """There is no saved model file. Loading never trains one (training takes minutes and must not happen inside a web request)."""

class StaleModel(Exception):
    """The saved model was trained under different feature definitions (or has no version) and must be retrained first."""

ALPHA_DAYS = 42      # newest weeks held out to measure dispersion on games the model has not seen

def model_stale():
    try: b = pickle.load(open(MODELS, "rb"))
    except Exception: return True
    return b.get("schema") != FEATURE_SCHEMA

MIN_TEST_ROWS = 1000

def _finite(x): return isinstance(x, (int, float)) and np.isfinite(x)

def validate(report, alphas=None):
    """Gate before a new model replaces the old one. Returns (ok, reasons). It must have been tested on enough games, every number must
    be a real number, and no core stat may be clearly worse than a plain 10-game average. (This checks accuracy only; probability honesty
    is checked separately by served_check on games played after training.)"""
    why = []
    if not report.get("n_test") or report["n_test"] < MIN_TEST_ROWS: why.append("too few test games")
    for t in TARGETS:
        st = (report.get("stats") or {}).get(t) or {}
        m, b = st.get("mae_model"), st.get("mae_10game_avg")
        if not (_finite(m) and _finite(b) and m > 0 and b > 0): why.append(f"{t}: error is missing, zero or not a number")
        elif m > 1.02 * b: why.append(f"{t}: worse than a 10-game average")
    for t in ("dd", "td"):
        br = ((report.get("stats") or {}).get(t) or {}).get("brier")
        if not _finite(br): why.append(f"{t}: score is missing or not a number")
    if alphas is not None and not all(_finite(a) and a > 0 for a in alphas.values()): why.append("spread values are not valid numbers")
    return (not why), why

def calibration_summary(p, y, high=0.75):
    """How honest were predicted chances? p: predicted chances, y: 1 if it happened. Returns overall and high-chance gaps (predicted minus actual)."""
    p, y = np.asarray(p, float), np.asarray(y, float); hi = p >= high
    return dict(n=int(len(p)), mean_pred=float(p.mean()) if len(p) else None, mean_actual=float(y.mean()) if len(y) else None,
                gap=float(p.mean() - y.mean()) if len(p) else None, brier=float(np.mean((p - y) ** 2)) if len(p) else None,
                high_n=int(hi.sum()), high_pred=float(p[hi].mean()) if hi.any() else None, high_actual=float(y[hi].mean()) if hi.any() else None)

def served_check(min_rows=300, offsets=(-3, -2, -1, 0, 1, 2, 3)):
    """Test the model that is actually being served on games played AFTER it was trained. Honest answer to 'do 80% chances come true 80% of the time?'.
    Until enough later games exist it says so instead of guessing."""
    b = load_models(strict=False)
    n_new = sqlite3.connect(DB).execute("SELECT count(*) FROM player_games WHERE date > ?", (b["trained_through"],)).fetchone()[0]
    if n_new < min_rows: return dict(status="waiting", later_rows=int(n_new), trained_through=b["trained_through"])    # cheap exit: nothing to test yet
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    later = full[full.date > b["trained_through"]]
    if len(later) < min_rows: return dict(status="waiting", later_rows=int(len(later)), trained_through=b["trained_through"])
    out = {}
    for t in ALL:
        mu = b["models"][t].predict(later[feats_for(t)]); ps, ys = [], []
        for o in offsets:
            line = np.maximum(np.floor(mu) + 0.5 + o, 0.5); ps.append(nb_over(mu, b["alphas"][t], line)); ys.append((later[t].values > line).astype(int))
        out[t] = calibration_summary(np.concatenate(ps), np.concatenate(ys))
    worst = max(((v["high_pred"] - v["high_actual"]) for v in out.values() if v["high_n"] >= 100), default=None)
    return dict(status="ok", later_rows=int(len(later)), trained_through=b["trained_through"], by_stat=out, worst_high_chance_gap=worst)

def train(report=True):
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    cut = "2026-02-01"; tr = full[full.date < cut]
    m_eval = fit(tr); alphas_eval = {t: dispersion(m_eval[t], tr[feats_for(t)], tr[t]) for t in ALL}
    corr = residual_corr(full, m_eval, alphas_eval)
    # dispersion measured out of sample: fit on everything except the newest ALPHA_DAYS, measure the spread of what happened after
    end = pd.Timestamp(full.date.max()); w0 = str((end - pd.Timedelta(days=ALPHA_DAYS)).date())
    A, W = full[full.date < w0], full[full.date >= w0]
    if len(W) < 1500:                                    # off-season or very short window: widen to the last 1500 rows
        w0 = full.date.sort_values().iloc[-1500]; A, W = full[full.date < w0], full[full.date >= w0]
    mA = fit(A); alphas = {t: dispersion(mA[t], W[feats_for(t)], W[t]) for t in ALL}
    m = fit(full)
    from model import evaluate
    rep = dict(evaluate(full), team=_team_report(), alpha_window=[str(w0), str(full.date.max())], alpha_train_method="out-of-sample")
    ok, bad = validate(rep, alphas)
    import hashlib
    mid = hashlib.sha1((FEATURE_SCHEMA + str(full.date.max()) + json.dumps(alphas, sort_keys=True)).encode()).hexdigest()[:12]   # stamped on every projection
    bundle = dict(models=m, alphas=alphas, corr=corr, trained_through=str(full.date.max()), report=rep, schema=FEATURE_SCHEMA, model_id=mid)
    target = MODELS if ok else MODELS + ".rejected"      # a model that fails validation is kept for inspection but never used
    tmp = target + ".tmp"                                # write beside the live file, then swap: a crash never leaves a half-written model
    with open(tmp, "wb") as f: pickle.dump(bundle, f)
    os.replace(tmp, target)
    if not ok: raise RuntimeError(f"new model rejected: {'; '.join(bad)}")
    return m, alphas, corr

def bundle_id(b):
    return b.get("model_id")

def load_models(strict=True):
    if not os.path.exists(MODELS): raise ModelMissing("no saved model")
    b = pickle.load(open(MODELS, "rb"))
    if strict and b.get("schema") != FEATURE_SCHEMA: raise StaleModel("saved model predates the current feature definitions; retrain first")
    return b

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

def attach_current_team(raw, cur):
    """Add each player's current team. EVERY historical game is kept (training saw them all); the current roster only decides who may
    be picked as a top-3 player, so players who have since left still count in team history."""
    return raw.merge(cur, on="player_id", how="left")

def apply_scheduled_rest(feats):
    """A player's second upcoming game counts rest from his first upcoming game (not from his last played one)."""
    feats = feats.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    prev = feats.groupby("player_id").date.shift(1); later = prev.notna()
    feats.loc[later, "rest"] = (pd.to_datetime(feats.date[later]) - pd.to_datetime(prev[later])).dt.days.clip(upper=10)
    feats["b2b"] = (feats.rest == 1).astype(int)
    return feats

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
    raw = attach_current_team(raw, cur)
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
    feats = apply_scheduled_rest(pd.concat(parts, ignore_index=True))
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
        rec = dict(model_id=bundle_id(bundle), game_id=f.game_id, start=g["start"], date=g["date"], player_id=f.player_id, player=f.player, team=f.team, opp=f.opp,
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
