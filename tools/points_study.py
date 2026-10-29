"""Why are points chances too confident, and which fix works on games the fix never saw?
Periods (all by game date):  TRAIN < T1 <= CALIBRATION < T2 <= TEST.  The base model sees TRAIN only; the correction is chosen on
CALIBRATION only; TEST (everything after T2) is looked at once, at the end.  python3 tools/points_study.py [T1 T2]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats
from features import build
from model import rotation, fit, feats_for

T1, T2 = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2026-01-15", "2026-03-15")

def main():
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    tr, cal, te = full[full.date < T1], full[(full.date >= T1) & (full.date < T2)], full[full.date >= T2]
    m = fit(tr)["pts"]
    out = {}
    for name, d in (("train", tr), ("calibration", cal), ("test", te)):
        mu = m.predict(d[feats_for("pts")]); y = d["pts"].values
        out[name] = dict(n=len(d), mean_pred=float(mu.mean()), mean_actual=float(y.mean()), bias=float((y - mu).mean()),
                         var_ratio=float(((y - mu) ** 2).mean() / (mu + 0.1 * mu ** 2).mean()))
        # left-tail check: how often does a player score less than 60% of his projection, versus what an honest spread (alpha .12) says
        r = 1 / 0.12; p = r / (r + mu)
        out[name]["share_below_60pct"] = float((y < 0.6 * mu).mean()); out[name]["nb_expected_below_60pct"] = float(stats.nbinom.cdf(np.ceil(0.6 * mu) - 1, r, p).mean())
    print(json.dumps(out, indent=1))
    # by projection size
    mu = m.predict(te[feats_for("pts")]); y = te["pts"].values
    b = pd.cut(mu, [0, 8, 12, 16, 20, 25, 60])
    print(pd.DataFrame({"b": b, "mu": mu, "y": y}).groupby("b", observed=True).agg(n=("y", "size"), mu=("mu", "mean"), y=("y", "mean")).round(2).to_string())

if __name__ == "__main__": main()

# ---- candidate corrections, chosen on CALIBRATION only, then scored once on TEST ----
from scipy.optimize import minimize
from model import nb_over

def nb_loglik(params, mu, y):
    b, la = params; a = np.exp(la); r = 1 / a; m = np.maximum(mu * np.exp(b), 0.05); p = r / (r + m)
    return -stats.nbinom.logpmf(y, r, p).sum()

def score(mu, alpha, d, offsets=(-4, -3, -2, -1, 0, 1, 2, 3, 4)):
    ps, ys = [], []
    for o in offsets:
        line = np.maximum(np.floor(mu) + 0.5 + o, 0.5); ps.append(nb_over(mu, alpha, line)); ys.append((d["pts"].values > line).astype(int))
    p, y = np.concatenate(ps), np.concatenate(ys); hi = p >= 0.75
    return dict(brier=float(np.mean((p - y) ** 2)), logloss=float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))),
                mean_gap=float(p.mean() - y.mean()), high_n=int(hi.sum()), high_pred=float(p[hi].mean()), high_actual=float(y[hi].mean()),
                high_gap=float(p[hi].mean() - y[hi].mean()))

def candidates():
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    tr, cal, te = full[full.date < T1], full[(full.date >= T1) & (full.date < T2)], full[full.date >= T2]
    m = fit(tr)["pts"]; mu_tr, mu_c, mu_t = (m.predict(d[feats_for("pts")]) for d in (tr, cal, te))
    res = {}
    a_train = float(max(np.sum((tr.pts.values - mu_tr) ** 2 - mu_tr) / np.sum(mu_tr ** 2), 0.001))                   # what the app did originally
    a_cal = float(max(np.sum((cal.pts.values - mu_c) ** 2 - mu_c) / np.sum(mu_c ** 2), 0.001))                        # spread measured on unseen games
    res["0 original (spread from training games)"] = score(mu_t, a_train, te)
    res["1 spread from calibration games"] = score(mu_t, a_cal, te)
    fitres = minimize(nb_loglik, [0.0, np.log(0.1)], args=(mu_c, cal.pts.values), method="Nelder-Mead")
    b, a_mle = float(fitres.x[0]), float(np.exp(fitres.x[1]))
    res["2 bias + spread, maximum likelihood on calibration"] = score(mu_t * np.exp(b), a_mle, te)
    bias_only = float(np.mean(cal.pts.values) / np.mean(mu_c)) - 1
    res["3 bias only (scale), spread from calibration"] = score(mu_t * (1 + bias_only), a_cal, te)
    res["4 bias + spread MLE, applied back to calibration (sanity: should look good)"] = score(mu_c * np.exp(b), a_mle, cal)
    print("\nfitted on calibration: bias factor exp(b) = %.4f, spread alpha(MLE) = %.4f, alpha(moments) = %.4f, alpha(train) = %.4f" % (np.exp(b), a_mle, a_cal, a_train))
    for k, v in res.items(): print(f"{k:78s} brier {v['brier']:.4f} logloss {v['logloss']:.4f} mean gap {v['mean_gap']:+.3f} | >=75%: n={v['high_n']:5d} pred {v['high_pred']:.3f} actual {v['high_actual']:.3f} gap {v['high_gap']:+.3f}")
    return res

if __name__ == "__main__" and "--candidates" in sys.argv: candidates()

def by_month():
    """Is the leftover over-confidence spread evenly, or concentrated in some part of the year? Also: does re-fitting the correction on the
    6 weeks BEFORE each month (never including the month itself) help?"""
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    tr = full[full.date < T1]; m = fit(tr)["pts"]
    full = full.assign(mu=m.predict(full[feats_for("pts")])); cal = full[(full.date >= T1) & (full.date < T2)]
    fr = minimize(nb_loglik, [0.0, np.log(0.1)], args=(cal.mu.values, cal.pts.values), method="Nelder-Mead"); b0, a0 = float(fr.x[0]), float(np.exp(fr.x[1]))
    rows = []
    for mon, d in full[full.date >= T2].groupby(full.date.str[:7]):
        start = pd.Timestamp(mon + "-01"); w = full[(full.date >= str((start - pd.Timedelta(days=42)).date())) & (full.date < str(start.date()))]
        fixed = score(d.mu.values * np.exp(b0), a0, d)
        if len(w) >= 800:
            fw = minimize(nb_loglik, [0.0, np.log(0.1)], args=(w.mu.values, w.pts.values), method="Nelder-Mead"); bw, aw = float(fw.x[0]), float(np.exp(fw.x[1]))
            roll = score(d.mu.values * np.exp(bw), aw, d)
        else: roll = None
        rows.append((mon, len(d), fixed, roll, float(d.pts.mean() - d.mu.mean())))
    print("\nmonth   rows | actual-minus-projected pts | fixed correction: >=75% pred/actual gap | rolling 6-week correction: pred/actual gap")
    for mon, n, f, r, bias in rows:
        print(f"{mon}  {n:5d} | {bias:+.2f} | {f['high_pred']:.3f}/{f['high_actual']:.3f} {f['high_gap']:+.3f} (n={f['high_n']}) | " + (f"{r['high_pred']:.3f}/{r['high_actual']:.3f} {r['high_gap']:+.3f} (n={r['high_n']})" if r else "not enough prior games"))

if __name__ == "__main__" and "--months" in sys.argv: by_month()

# ---- shape fix: a small share of games where the player plays far less than expected ("bad night" component) ----
def mix_over(mu, alpha, b, q, kappa, line):
    m = np.maximum(mu * np.exp(b), 0.05); r = 1 / alpha; k = np.floor(line)
    full = stats.nbinom.sf(k, r, r / (r + m)); low = stats.nbinom.sf(k, r, r / (r + kappa * m))
    return (1 - q) * full + q * low

def mix_nll(params, mu, y):
    b, la, lq, lk = params; q = 1 / (1 + np.exp(-lq)); kappa = 1 / (1 + np.exp(-lk)); r = 1 / np.exp(la); m = np.maximum(mu * np.exp(b), 0.05)
    pm = (1 - q) * stats.nbinom.pmf(y, r, r / (r + m)) + q * stats.nbinom.pmf(y, r, r / (r + kappa * m))
    return -np.log(np.maximum(pm, 1e-300)).sum()

def score_fn(overf, d, mu, offsets=(-4, -3, -2, -1, 0, 1, 2, 3, 4)):
    ps, ys = [], []
    for o in offsets:
        line = np.maximum(np.floor(mu) + 0.5 + o, 0.5); ps.append(overf(mu, line)); ys.append((d["pts"].values > line).astype(int))
    p, y = np.concatenate(ps), np.concatenate(ys); hi = p >= 0.75
    return dict(brier=float(np.mean((p - y) ** 2)), logloss=float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))), mean_gap=float(p.mean() - y.mean()),
                high_n=int(hi.sum()), high_pred=float(p[hi].mean()), high_actual=float(y[hi].mean()), high_gap=float(p[hi].mean() - y[hi].mean()))

def mixture_study():
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    tr = full[full.date < T1]; m = fit(tr)["pts"]
    full = full.assign(mu=m.predict(full[feats_for("pts")])); cal = full[(full.date >= T1) & (full.date < T2)]; te = full[full.date >= T2]
    x0 = [0.03, np.log(0.12), -2.0, 0.0]
    fr = minimize(mix_nll, x0, args=(cal.mu.values, cal.pts.values), method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-4, fatol=1e-4))
    b, la, lq, lk = fr.x; q, kappa, alpha = 1 / (1 + np.exp(-lq)), 1 / (1 + np.exp(-lk)), float(np.exp(la))
    print(f"\nmixture fitted on calibration only: scale exp(b)={np.exp(b):.3f}  spread alpha={alpha:.3f}  bad-night share q={q:.3f}  bad-night level kappa={kappa:.2f}")
    base = minimize(nb_loglik, [0.0, np.log(0.1)], args=(cal.mu.values, cal.pts.values), method="Nelder-Mead"); b2, a2 = float(base.x[0]), float(np.exp(base.x[1]))
    nll_nb, nll_mix = base.fun, fr.fun; print(f"calibration fit quality (lower is better): plain {nll_nb:.1f}  mixture {nll_mix:.1f}   difference {nll_nb - nll_mix:.1f} (about 4 parameters)")
    f_nb = lambda mu, L: nb_over(mu * np.exp(b2), a2, L); f_mix = lambda mu, L: mix_over(mu, alpha, b, q, kappa, L)
    for name, f in (("plain NB (bias+spread)", f_nb), ("mixture (bias+spread+bad nights)", f_mix)):
        t = score_fn(f, te, te.mu.values); print(f"TEST {name:36s} brier {t['brier']:.4f} logloss {t['logloss']:.4f} mean gap {t['mean_gap']:+.3f} | >=75% n={t['high_n']} pred {t['high_pred']:.3f} actual {t['high_actual']:.3f} gap {t['high_gap']:+.3f}")
    print("month  rows | plain >=75% gap | mixture >=75% gap")
    for mon, d in te.groupby(te.date.str[:7]):
        a = score_fn(f_nb, d, d.mu.values); c = score_fn(f_mix, d, d.mu.values); print(f"{mon} {len(d):5d} | {a['high_gap']:+.3f} (n={a['high_n']}) | {c['high_gap']:+.3f} (n={c['high_n']})")
    return dict(b=float(b), alpha=alpha, q=float(q), kappa=float(kappa))

if __name__ == "__main__" and "--mixture" in sys.argv: mixture_study()
