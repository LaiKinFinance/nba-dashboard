"""Offline diagnostics on the fixed time split; never writes trained artifacts."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features
import model
import team_model


def main():
    d = model.rotation(features.build()).dropna(subset=['min_m10', 'pts_m10'])
    tr, te = d[d.date < '2026-02-01'], d[d.date >= '2026-02-01']
    fitted = model.fit(tr)
    report = dict(n_train=len(tr), n_test=len(te), td_events=int(te.td.sum()), dd_events=int(te.dd.sum()), stats={})
    for target in model.ALL:
        f = model.feats_for(target)
        pred = fitted[target].predict(te[f])
        alpha = model.dispersion(fitted[target], tr[f], tr[target])
        heldout_alpha = model.dispersion(fitted[target], te[f], te[target])
        line = np.floor(pred)+.5
        p = model.nb_over(pred, alpha, line)
        y = (te[target].to_numpy() > line).astype(int)
        bins = pd.DataFrame(dict(p=p, y=y)).assign(bin=lambda x: pd.cut(x.p, [0, .2, .4, .6, .8, 1], include_lowest=True))
        cal = bins.groupby('bin', observed=True).agg(n=('y', 'size'), predicted=('p', 'mean'), actual=('y', 'mean')).reset_index()
        cal['bin'] = cal['bin'].astype(str)
        report['stats'][target] = dict(mae=float(np.abs(te[target]-pred).mean()), baseline_mae=float(np.abs(te[target]-te[target+'_m10']).mean()),
                                       alpha_train=alpha, alpha_test_diagnostic_only=heldout_alpha,
                                       brier=float(np.mean((p-y)**2)), brier_constant=float(np.var(y)), mean_pred=float(p.mean()), mean_actual=float(y.mean()), calibration=cal.to_dict('records'))
    td = fitted['td'].predict_proba(te)
    actual = int(te.td.sum()); n = len(te); z = 1.96
    ph = actual/n; den = 1+z*z/n; mid = (ph+z*z/(2*n))/den; half = z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    report['td'] = dict(mean_pred=float(td.mean()), rate=ph, wilson95_naive=[mid-half, mid+half], note='Players and games are dependent; this interval is optimistic.')
    import sqlite3
    with sqlite3.connect(team_model.DB) as con:
        g = pd.read_sql('SELECT * FROM games', con)
    mk = g.dropna(subset=['ou', 'spread_home'])
    em = mk.home_pts-mk.away_pts+mk.spread_home; et = mk.home_pts+mk.away_pts-mk.ou
    report['team_residuals'] = dict(n=len(mk), margin_bias=float(em.mean()), total_bias=float(et.mean()), margin_total_correlation=float(np.corrcoef(em, et)[0, 1]),
                                    sigma_margin=float(em.std(ddof=0)), sigma_total=float(et.std(ddof=0)))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
