"""Train per-player projections (pts/reb/ast) and double/triple-double classifiers; time-split evaluation."""
import pickle, numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, brier_score_loss, log_loss
from features import build, FEATURES

TARGETS = ["pts", "reb", "ast"]
EXTRA = ["fg3m", "stl", "blk"]          # three-pointers made, steals, blocks
ALL = TARGETS + EXTRA
XF = {t: [f"{t}_{s}" for s in ("m5", "m10", "m20", "exp")] for t in EXTRA}   # their own recent form

def feats_for(t):
    return list(FEATURES) + XF.get(t, [])

def rotation(df):
    return df[(df.min_m10 >= 22) & (df.prior_games >= 5)]

TD_FEATS = ["td_m40", "dd_m20", "pts_m20", "reb_m20", "ast_m20", "min_m20", "pra_m10"]

class TDModel:
    """Logistic regression on rolling rates; beat the boosted classifier on rare triple-doubles in backtests."""
    def __init__(self, C=0.5): self.C = C
    def _x(self, d):
        x = d[TD_FEATS].copy(); x["td_m40"] = x.td_m40.fillna(d.td_exp).fillna(0); x["dd_m20"] = x.dd_m20.fillna(0)
        return x.fillna(self.med)
    def fit(self, d):
        d = d[d.pra_m10 >= 20]
        self.med = d[TD_FEATS].median(); x = self._x(d); self.mu, self.sd = x.mean(), x.std().replace(0, 1)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit((x - self.mu) / self.sd, d.td); return self
    def predict_proba(self, d):
        return self.lr.predict_proba((self._x(d) - self.mu) / self.sd)[:, 1]

def fit(train):
    m = {}
    for t in ALL:
        m[t] = HGR(loss="poisson", max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=60,
                   l2_regularization=2.0, random_state=0).fit(train[feats_for(t)], train[t])
    m["dd"] = HGC(max_iter=200, learning_rate=0.04, max_depth=3, min_samples_leaf=80, l2_regularization=3.0,
                  random_state=0).fit(train[FEATURES], train["dd"])
    m["td"] = TDModel().fit(train)
    return m

def dispersion(model, X, y):
    """Negative-binomial alpha by moment matching: Var = mu + alpha*mu^2."""
    mu = model.predict(X); r = (y - mu) ** 2 - mu
    return float(max(np.sum(r) / np.sum(mu ** 2), 0.001))

def nb_over(mu, alpha, line):
    """P(X > line) for half-point lines, X ~ NegBin(mean mu, dispersion alpha)."""
    n = 1.0 / alpha; p = n / (n + mu)
    return stats.nbinom.sf(np.floor(line), n, p)

def evaluate(df, cut="2026-02-01"):
    """Backtest metrics for the scorecard page (trained before `cut`, scored after)."""
    tr, te = df[df.date < cut], df[df.date >= cut]; m = fit(tr); rep = dict(cut=cut, n_train=len(tr), n_test=len(te), stats={})
    for t in ALL:
        pred = m[t].predict(te[feats_for(t)]); base = te[f"{t}_m10"].fillna(te[f"{t}_exp"])
        rep["stats"][t] = dict(mae_model=float(mean_absolute_error(te[t], pred)), mae_10game_avg=float(mean_absolute_error(te[t], base)))
    for t in ("dd", "td"):
        p = m[t].predict_proba(te) if t == "td" else m[t].predict_proba(te[FEATURES])[:, 1]; y = te[t]
        bins = pd.cut(p, [0, .02, .05, .1, .2, .35, .6, 1.0])
        cal = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean")).reset_index()
        rep["stats"][t] = dict(base_rate=float(y.mean()), events=int(y.sum()), brier=float(brier_score_loss(y, p)),
                               brier_constant=float(brier_score_loss(y, np.full(len(y), y.mean()))),
                               calibration=[dict(bin=str(r.b), n=int(r.n), pred=float(r.pred), actual=float(r.actual)) for r in cal.itertuples()])
    rep["absence"] = absence_report(df, cut)
    return rep

def absence_report(df, cut):
    """Does knowing which teammates are missing help? Same models with and without the absence features, scored on later games."""
    import absence, model as me
    tr, te = df[df.date < cut], df[df.date >= cut]; hi = (te.abs_pts >= 15).values
    full = list(FEATURES); base = [f for f in full if f not in absence.ABS]; out = dict(n_high=int(hi.sum()))
    res = {}
    try:
        for name, feats in (("without", base), ("with", full)):
            me.FEATURES = feats; mm = fit(tr); res[name] = {t: mm[t].predict(te[feats]) for t in TARGETS}
    finally:
        me.FEATURES = full
    for t in TARGETS:
        y = te[t].values
        out[t] = dict(mae_without=float(mean_absolute_error(y, res["without"][t])), mae_with=float(mean_absolute_error(y, res["with"][t])),
                      bias_without=float(np.mean(y[hi] - res["without"][t][hi])), bias_with=float(np.mean(y[hi] - res["with"][t][hi])))
    return out

if __name__ == "__main__":
    df = build(); df = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    cut = "2026-02-01"
    tr, te = df[df.date < cut], df[(df.date >= cut)]
    print("train", len(tr), "test", len(te), "(test includes playoffs + new season)")
    m = fit(tr)
    for t in TARGETS:
        pred = m[t].predict(te[FEATURES]); base = te[f"{t}_m10"].fillna(te[f"{t}_exp"])
        a = dispersion(m[t], tr[FEATURES], tr[t])
        # hit-rate calibration on a typical line (round player's prediction to nearest .5)
        line = np.round(pred * 2) / 2 - 0.5 + 0.0
        line = np.floor(pred) + 0.5
        p_over = nb_over(pred, a, line); actual = (te[t] > line).astype(int)
        print(f"{t}: MAE model {mean_absolute_error(te[t], pred):.2f} vs 10-game avg {mean_absolute_error(te[t], base):.2f}"
              f" | alpha {a:.3f} | over-prob Brier {brier_score_loss(actual, p_over):.4f} "
              f"(constant {brier_score_loss(actual, np.full(len(actual), actual.mean())):.4f}), mean pred {p_over.mean():.3f} vs actual {actual.mean():.3f}")
    for t in ("dd", "td"):
        p = (m[t].predict_proba(te) if t == "td" else m[t].predict_proba(te[FEATURES])[:, 1]); base = te["td_m40" if t == "td" else "dd_m20"].fillna(te[f"{t}_exp"]).fillna(0)
        y = te[t]
        print(f"{t}: base rate {y.mean():.3f} | Brier model {brier_score_loss(y, p):.4f} vs player-rate {brier_score_loss(y, base.clip(0,1)):.4f} vs constant {brier_score_loss(y, np.full(len(y), y.mean())):.4f}")
        bins = pd.cut(p, [0, .02, .05, .1, .2, .35, .6, 1.0]); print(pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean")).round(3).to_string())
