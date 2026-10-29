"""Do referee crews systematically change game outcomes? Out-of-sample test with a permutation null.

Outcomes (per game):
  total_resid  = actual total points - market total          (scoring vs what the market already expected)
  margin_resid = actual home margin + home spread             (home-court / favourite bias vs the market)
  fta          = both teams' free-throw attempts              (foul-calling volume)
  fouls        = both teams' personal fouls
For each outcome we ask: does knowing the crew predict the outcome on games it has not seen (5-fold CV, ridge on referee indicators),
and is that better than chance? The null is built by shuffling crews between games, repeated many times.
Run: python tools/referee_analysis.py"""
import os, sqlite3, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

D = os.environ.get("DATA_DIR", "data"); con = sqlite3.connect(os.path.join(D, "nba.db"))
g = pd.read_sql("SELECT * FROM games WHERE home_fta IS NOT NULL", con).set_index("game_id")
o = pd.read_sql("SELECT game_id, name FROM officials", con)
crew = o.groupby("game_id").name.apply(list); g = g.join(crew, how="inner"); g = g[g.name.map(len) >= 3]
g["total_resid"] = (g.home_pts + g.away_pts) - g.ou; g["margin_resid"] = (g.home_pts - g.away_pts) + g.spread_home
g["fta"] = g.home_fta + g.away_fta; g["fouls"] = g.home_pf + g.away_pf
# remove season-level differences (league-wide trends) so referees are not credited for them
for c in ("fta", "fouls"): g[c] = g[c] - g.groupby("season")[c].transform("mean")
refs = sorted({r for c in g.name for r in c}); idx = {r: i for i, r in enumerate(refs)}
X = np.zeros((len(g), len(refs)))
for i, c in enumerate(g.name):
    for r in c: X[i, idx[r]] = 1.0
n_per = X.sum(0); print(f"{len(g)} games, {len(refs)} referees, games per referee: median {int(np.median(n_per))}, min {int(n_per.min())}, max {int(n_per.max())}")

def cv_gain(X, y, alpha, seed=0):
    """Out-of-sample improvement over predicting the mean (1 - MSE_model / MSE_mean); >0 means the crew carries information."""
    se_m = se_0 = 0.0
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr]); se_m += ((y[te] - m.predict(X[te])) ** 2).sum(); se_0 += ((y[te] - y[tr].mean()) ** 2).sum()
    return 1 - se_m / se_0

rng = np.random.default_rng(1); out = []
for col in ("total_resid", "margin_resid", "fta", "fouls"):
    d = g[[col]].dropna(); ok = g.index.isin(d.index); Xc = X[ok]; y = d[col].values.astype(float); y = y - y.mean()
    best = max(((cv_gain(Xc, y, a), a) for a in (10, 30, 100, 300)), key=lambda t: t[0]); gain, a = best
    null = []
    for _ in range(100):
        Xp = Xc[rng.permutation(len(Xc))]; null.append(cv_gain(Xp, y, a))
    p = float((np.array(null) >= gain).mean())
    m = Ridge(alpha=a).fit(Xc, y); eff = pd.Series(m.coef_, index=refs); n = pd.Series(Xc.sum(0), index=refs)
    top = eff.reindex(eff.abs().sort_values(ascending=False).index)[:3]
    out.append(dict(outcome=col, n=len(y), sd=float(y.std()), cv_gain=gain, p_value=p, top=[(r, round(float(eff[r]), 2), int(n[r])) for r in top.index]))
    print(f"\n{col}: games {len(y)}, outcome SD {y.std():.2f}\n  out-of-sample gain from knowing the crew: {gain*100:+.2f}%  (shuffled-crew null: mean {np.mean(null)*100:+.2f}%, 95th pct {np.percentile(null,95)*100:+.2f}%)  p = {p:.2f}")
    print("  largest shrunken referee effects (name, effect, games):", out[-1]["top"])
