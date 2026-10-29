"""Is the model's 'chance of going over' honest? Compares dispersion fitted on training games (current method) with dispersion
measured on later, unseen games, using only data before each test period.  python3 tools/calibration_study.py [--json out.json]
Reported per stat, on lines the app actually offers (floor(projection)+0.5 shifted by -3..+3):
  mean predicted vs actual over-rate, Brier score, log loss, and a reliability table (predicted bucket vs what happened).
A better score here means better calibrated probabilities. It is NOT evidence of betting profit: that needs bookmaker prices
recorded before tip-off, which the app has only just started archiving."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import brier_score_loss, log_loss
from features import build
from model import rotation, fit, dispersion, nb_over, ALL, feats_for

D1, D2 = "2025-12-20", "2026-02-01"      # alpha-estimation window is [D1, D2); final test is >= D2

def over_table(m, alphas, te, offsets=(-3, -2, -1, 0, 1, 2, 3)):
    rows = []
    for t in ALL:
        mu = m[t].predict(te[feats_for(t)])
        for o in offsets:
            line = np.maximum(np.floor(mu) + 0.5 + o, 0.5)
            p = np.clip(nb_over(mu, alphas[t], line), 1e-4, 1 - 1e-4); y = (te[t].values > line).astype(int)
            rows.append(dict(stat=t, offset=o, p=p, y=y))
    return rows

def _logit(p): return np.log(p / (1 - p))

def fit_platt(rows):
    """Per-stat logistic recalibration p' = sigmoid(a*logit(p)+b), fitted on the unseen window's over-probabilities."""
    from sklearn.linear_model import LogisticRegression
    cal = {}
    for t in ALL:
        p = np.concatenate([r["p"] for r in rows if r["stat"] == t]); y = np.concatenate([r["y"] for r in rows if r["stat"] == t])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y); cal[t] = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
    return cal

def apply_platt(rows, cal):
    return [dict(r, p=np.clip(1 / (1 + np.exp(-(cal[r["stat"]][0] * _logit(r["p"]) + cal[r["stat"]][1]))), 1e-4, 1 - 1e-4)) for r in rows]

def summarise(rows, label):
    out = {}
    for t in ALL:
        p = np.concatenate([r["p"] for r in rows if r["stat"] == t]); y = np.concatenate([r["y"] for r in rows if r["stat"] == t])
        b = pd.cut(p, [0, .1, .25, .4, .6, .75, .9, 1.0])
        rel = pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
        out[t] = dict(label=label, n=int(len(p)), mean_pred=float(p.mean()), mean_actual=float(y.mean()), brier=float(brier_score_loss(y, p)),
                      logloss=float(log_loss(y, p)), reliability=[dict(bucket=str(i), n=int(r.n), pred=float(r.pred), actual=float(r.actual)) for i, r in rel.iterrows()])
    return out

def main():
    df = build(); full = rotation(df).dropna(subset=["min_m10", "pts_m10"])
    A = full[full.date < D1]; W = full[(full.date >= D1) & (full.date < D2)]; tr = full[full.date < D2]; te = full[full.date >= D2]
    mA = fit(A); alpha_oos = {t: dispersion(mA[t], W[feats_for(t)], W[t]) for t in ALL}      # measured on games the model had not seen
    mT = fit(tr); alpha_train = {t: dispersion(mT[t], tr[feats_for(t)], tr[t]) for t in ALL}   # what the app does today
    res = {"windows": dict(train_to=D2, alpha_window=[D1, D2], test_from=D2, n_test=int(len(te))), "alpha_train": alpha_train, "alpha_oos": alpha_oos}
    res["current"] = summarise(over_table(mT, alpha_train, te), "alpha from training games")
    res["oos"] = summarise(over_table(mT, alpha_oos, te), "alpha from unseen games")
    cal = fit_platt(over_table(mA, alpha_oos, W))                                  # curve learned on the unseen window only
    res["platt"] = summarise(apply_platt(over_table(mT, alpha_oos, te), cal), "alpha from unseen games + Platt curve"); res["platt_params"] = cal
    print(f"test games: {len(te)}  (alpha window {D1}..{D2})")
    print(f"{'stat':5} {'alpha train':>11} {'alpha oos':>10} | {'Brier cur':>9} {'Brier oos':>9} | {'LL cur':>7} {'LL oos':>7} | pred/actual cur  pred/actual oos | Brier+Platt")
    for t in ALL:
        c, o = res["current"][t], res["oos"][t]; q = res["platt"][t]
        print(f"{t:5} {alpha_train[t]:11.3f} {alpha_oos[t]:10.3f} | {c['brier']:9.4f} {o['brier']:9.4f} | {c['logloss']:7.4f} {o['logloss']:7.4f} | {c['mean_pred']:.3f}/{c['mean_actual']:.3f}      {o['mean_pred']:.3f}/{o['mean_actual']:.3f}   | {q['brier']:.4f}")
    if "--json" in sys.argv: json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    return res

if __name__ == "__main__": main()
