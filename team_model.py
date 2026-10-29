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
    return dict(cut=cut, n_train=len(tr), n_test=len(te),
                margin_mae=float(np.mean(np.abs(te.margin - te.margin_pred))), margin_mae_constant=float(np.mean(np.abs(te.margin - tr.margin.mean()))),
                total_mae=float(np.mean(np.abs(te.total - te.total_pred))), total_mae_constant=float(np.mean(np.abs(te.total - tr.total.mean()))),
                win_acc=float(np.mean((te.margin_pred > 0) == (te.margin > 0))), home_win_rate=float(win.mean()),
                win_brier=brier(win, p), win_brier_constant=brier(win, np.full(len(win), win.mean())), sigma=sig)

def predict(upcoming):
    """upcoming: [{game_id, date, home, away}] -> {game_id: {margin, total, sigma_margin, sigma_total}} from the full history."""
    con = sqlite3.connect(DB); g = pd.read_sql("SELECT * FROM games", con)
    d = build(g); sig = fit(d)["sigma"]; R = elo(g)[1]
    L = _long(g); ew = {}
    for team, x in L.groupby("team"):
        ew[team] = dict(off=x.pf.ewm(halflife=HALF, min_periods=3).mean().iloc[-1], dfn=x.pa.ewm(halflife=HALF, min_periods=3).mean().iloc[-1])
    out = {}
    for u in upcoming:
        h, a = ew.get(u["home"]), ew.get(u["away"])
        if not h or not a or u["home"] not in R or u["away"] not in R: continue
        out[u["game_id"]] = dict(margin=R[u["home"]] - R[u["away"]] + HCA, total=(h["off"] + a["dfn"] + a["off"] + h["dfn"]) / 2,
                                 sigma_margin=sig["margin"], sigma_total=sig["total"])
    return out

def price(p):
    """Winner chances plus handicap and total ladders (half-point lines kept to a realistic 15%-85% range)."""
    sm, st, m, t = p["sigma_margin"], p["sigma_total"], p["margin"], p["total"]
    ph = float(stats.norm.cdf(m / sm))
    hand = []
    for k in np.arange(-30.5, 30.6, 1.0):  # home handicap k: home wins the bet if margin + k > 0
        q = float(stats.norm.cdf((m + k) / sm))
        if 0.2 <= q <= 0.8: hand.append(dict(line=float(k), p_home=q, p_away=1 - q))
    tot = []
    for L in np.arange(180.5, 280.6, 2.0):
        q = float(stats.norm.cdf((t - L) / st))
        if 0.2 <= q <= 0.8: tot.append(dict(line=float(L), p_over=q, p_under=1 - q))
    return dict(margin=m, total=t, home_pts=(t + m) / 2, away_pts=(t - m) / 2, p_home=ph, p_away=1 - ph, handicaps=hand, totals=tot,
                sigma_margin=sm, sigma_total=st)

if __name__ == "__main__":
    con = sqlite3.connect(DB); g = pd.read_sql("SELECT * FROM games", con); d = build(g)
    import json; print(json.dumps(evaluate(d), indent=1))
