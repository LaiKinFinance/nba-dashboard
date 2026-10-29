"""Team-level model: predicts home margin and game total from recent scoring form, then prices winner / handicap / total lines.

Features use only games before the one being predicted. Ratings are exponentially weighted points for/against per team, carried
across seasons (early-season ratings therefore lean on last season)."""
import os, sqlite3
import numpy as np, pandas as pd
from scipy import stats

DB = os.path.join(os.environ.get("DATA_DIR", "data"), "nba.db")
HALF = 10

def _long(g):
    h = g.rename(columns={"home": "team", "away": "opp", "home_pts": "pf", "away_pts": "pa"}).assign(home=1)[["game_id", "date", "team", "opp", "pf", "pa", "home"]]
    a = g.rename(columns={"away": "team", "home": "opp", "away_pts": "pf", "home_pts": "pa"}).assign(home=0)[["game_id", "date", "team", "opp", "pf", "pa", "home"]]
    return pd.concat([h, a]).sort_values(["date", "game_id"]).reset_index(drop=True)

def _rolling(L):
    L = L.copy(); gb = L.groupby("team")
    for c, n in (("pf", "off"), ("pa", "def")):
        L[n] = gb[c].transform(lambda x: x.shift(1).ewm(halflife=HALF, min_periods=3).mean())
    L["rest"] = (pd.to_datetime(L.date) - pd.to_datetime(gb["date"].shift(1))).dt.days.clip(upper=6)
    return L

K_ELO, CARRY, HCA = 0.06, 0.5, 2.6  # tuned on pre-cut seasons; ratings are "points above average"

def elo(g):
    """Margin-based Elo, regressed toward average between seasons. Returns (pre-game expectations per game, final ratings)."""
    g = g.sort_values(["date", "game_id"]); R, rows, season = {}, [], None
    for r in g.itertuples():
        if season is not None and r.season != season:
            for t in R: R[t] *= CARRY
        season = r.season
        rh, ra = R.get(r.home, 0.0), R.get(r.away, 0.0); exp = rh - ra + HCA
        rows.append((r.game_id, exp)); err = (r.home_pts - r.away_pts) - exp
        R[r.home] = rh + K_ELO * err; R[r.away] = ra - K_ELO * err
    return pd.DataFrame(rows, columns=["game_id", "margin_pred"]).set_index("game_id"), R

def build(g):
    """One row per finished game: Elo margin prediction and scoring-form total prediction (both pre-game)."""
    L = _rolling(_long(g)); h = L[L.home == 1].set_index("game_id"); a = L[L.home == 0].set_index("game_id")
    d = pd.DataFrame(index=h.index); d["date"] = h.date
    d["total_pred"] = (h.off + a["def"] + a.off + h["def"]) / 2
    d["margin"] = h.pf - h.pa; d["total"] = h.pf + h.pa
    return d.join(elo(g)[0]).dropna(subset=["total_pred", "margin_pred"])

def fit(d):
    return dict(sigma=dict(margin=float(np.std(d.margin - d.margin_pred)), total=float(np.std(d.total - d.total_pred))))

def evaluate(d, cut="2026-02-01"):
    tr, te = d[d.date < cut], d[d.date >= cut]; sig = fit(tr)["sigma"]
    win = (te.margin > 0).astype(int); p = stats.norm.cdf(te.margin_pred / sig["margin"]); brier = lambda y, q: float(np.mean((y - q) ** 2))
    mk = None
    try:
        gg = pd.read_sql("SELECT game_id, ou, spread_home, ml_home, ml_away FROM games", sqlite3.connect(DB)).set_index("game_id")
        x = te.join(gg).dropna(subset=["ou", "spread_home"]); w = (x.margin > 0).astype(int)
        pm = np.array([_nv(a, b) if _nv(a, b) is not None else 0.5 for a, b in zip(x.ml_home, x.ml_away)])
        mk = dict(n=int(len(x)), margin_mae=float(np.mean(np.abs(x.margin + x.spread_home))), margin_mae_elo=float(np.mean(np.abs(x.margin - x.margin_pred))),
                  total_mae=float(np.mean(np.abs(x.total - x.ou))), total_mae_form=float(np.mean(np.abs(x.total - x.total_pred))),
                  win_brier=float(np.mean((w - pm) ** 2)), win_brier_elo=float(np.mean((w - stats.norm.cdf(x.margin_pred / sig["margin"])) ** 2)))
    except Exception: pass
    return dict(cut=cut, n_train=len(tr), n_test=len(te), market=mk,
                margin_mae=float(np.mean(np.abs(te.margin - te.margin_pred))), margin_mae_constant=float(np.mean(np.abs(te.margin - tr.margin.mean()))),
                total_mae=float(np.mean(np.abs(te.total - te.total_pred))), total_mae_constant=float(np.mean(np.abs(te.total - tr.total.mean()))),
                win_acc=float(np.mean((te.margin_pred > 0) == (te.margin > 0))), home_win_rate=float(win.mean()),
                win_brier=brier(win, p), win_brier_constant=brier(win, np.full(len(win), win.mean())), sigma=sig)

def _nv(ml_h, ml_a):
    """No-vig win probability for the home team from American moneylines."""
    imp = lambda x: (-x / (-x + 100)) if x < 0 else (100 / (x + 100))
    try: ph, pa = imp(float(ml_h)), imp(float(ml_a)); return ph / (ph + pa)
    except (TypeError, ValueError, ZeroDivisionError): return None

def predict(upcoming):
    """upcoming: [{game_id, date, home, away, ou, spread_home, ml_home, ml_away}] ->
    {game_id: {margin, total, sigma_margin, sigma_total, source, p_home_ml}}.
    In backtests the betting market's margin/total/moneyline beat our own Elo/form model, so the market is the central estimate when a
    line exists; our own model is only the fallback."""
    con = sqlite3.connect(DB); g = pd.read_sql("SELECT * FROM games", con)
    d = build(g); sig = fit(d)["sigma"]; R = elo(g)[1]
    if "spread_home" in g and g.spread_home.notna().any():
        lg = g.dropna(subset=["ou", "spread_home"]); sig_mk = dict(margin=float(np.std((lg.home_pts - lg.away_pts) + lg.spread_home)), total=float(np.std(lg.home_pts + lg.away_pts - lg.ou)))
    else: sig_mk = None
    L = _long(g); ew = {}
    for team, x in L.groupby("team"):
        ew[team] = dict(off=x.pf.ewm(halflife=HALF, min_periods=3).mean().iloc[-1], dfn=x.pa.ewm(halflife=HALF, min_periods=3).mean().iloc[-1])
    out = {}
    for u in upcoming:
        mkt = u.get("ou") is not None and u.get("spread_home") is not None and sig_mk is not None
        if mkt:
            out[u["game_id"]] = dict(margin=-float(u["spread_home"]), total=float(u["ou"]), sigma_margin=sig_mk["margin"], sigma_total=sig_mk["total"],
                                     source="market", p_home_ml=_nv(u.get("ml_home"), u.get("ml_away")))
            continue
        h, a = ew.get(u["home"]), ew.get(u["away"])
        if not h or not a or u["home"] not in R or u["away"] not in R: continue
        out[u["game_id"]] = dict(margin=R[u["home"]] - R[u["away"]] + HCA, total=(h["off"] + a["dfn"] + a["off"] + h["dfn"]) / 2,
                                 sigma_margin=sig["margin"], sigma_total=sig["total"], source="model", p_home_ml=None)
    return out

def price(p):
    """Winner chances plus handicap and total ladders (half-point lines kept to a realistic 15%-85% range)."""
    sm, st, m, t = p["sigma_margin"], p["sigma_total"], p["margin"], p["total"]
    ph = float(p.get("p_home_ml") if p.get("p_home_ml") is not None else stats.norm.cdf(m / sm))
    hand = []
    for k in np.arange(-30.5, 30.6, 1.0):  # home handicap k: home wins the bet if margin + k > 0
        q = float(stats.norm.cdf((m + k) / sm))
        if 0.2 <= q <= 0.8: hand.append(dict(line=float(k), p_home=q, p_away=1 - q))
    tot = []
    for L in np.arange(180.5, 280.6, 2.0):
        q = float(stats.norm.cdf((t - L) / st))
        if 0.2 <= q <= 0.8: tot.append(dict(line=float(L), p_over=q, p_under=1 - q))
    # team totals: each side's expected points; its spread is the margin/total uncertainty split between the two teams
    hp, ap = (t + m) / 2, (t - m) / 2; sg = float(np.sqrt((st ** 2 + sm ** 2) / 4)); tteam = {}
    for side, mu in (("home", hp), ("away", ap)):
        tteam[side] = [dict(line=float(L), p_over=float(1 - stats.norm.cdf((L - mu) / sg))) for L in np.arange(60.5, 170.6, 2.0)
                       if 0.2 <= 1 - stats.norm.cdf((L - mu) / sg) <= 0.8]
    # winning-margin bands (integer margins, so use continuity at x.5)
    bands = []
    for side, mm in (("home", m), ("away", -m)):
        for lo, hi in ((1, 6), (7, 12), (13, 18), (19, None)):
            q = float((1 - stats.norm.cdf((lo - 0.5 - mm) / sm)) - ((1 - stats.norm.cdf((hi + 0.5 - mm) / sm)) if hi else 0.0))
            bands.append(dict(team=side, lo=lo, hi=hi, p=q))
    tot_b = sum(b["p"] for b in bands) or 1.0
    for b in bands: b["p"] = b["p"] / tot_b          # margins of 0 are impossible, so the bands must add to 1
    return dict(margin=m, total=t, home_pts=hp, away_pts=ap, p_home=ph, p_away=1 - ph, handicaps=hand, totals=tot, team_totals=tteam, bands=bands,
                sigma_margin=sm, sigma_total=st, source=p.get("source", "model"))

if __name__ == "__main__":
    con = sqlite3.connect(DB); g = pd.read_sql("SELECT * FROM games", con); d = build(g)
    import json; print(json.dumps(evaluate(d), indent=1))
