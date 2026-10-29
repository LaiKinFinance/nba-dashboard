import numpy as np
import pytest
from scipy import stats

import combos
from model import nb_over
import parlay
import team_model


@pytest.mark.parametrize("mu,alpha", [(2, .1), (25, .15)])
def test_nb_tail_matches_distribution_and_decreases(mu, alpha):
    lines = np.arange(.5, 40, 1)
    p = nb_over(mu, alpha, lines)
    assert np.all(np.diff(p) <= 0)
    assert np.allclose(p, stats.nbinom.sf(np.floor(lines), 1/alpha, 1/(1+alpha*mu)))


def test_copula_orientation_marginals_and_combo_sums():
    mu = dict(pts=20, reb=8, ast=6)
    alpha = dict.fromkeys(mu, .1)
    same = [[1, .5, .3], [.5, 1, .2], [.3, .2, 1]]
    sim = combos.simulate(mu, alpha, same, n=60000)
    assert np.corrcoef(sim['pts'], sim['reb'])[0, 1] > .4
    for s in mu:
        assert abs(sim[s].mean()-mu[s]) < .15
    assert np.all(sim['pts']+sim['reb'] >= sim['pts'])


def test_parlay_duplicate_and_contradictory_half_point_legs():
    pred = dict(proj=dict(pts=20), team='BOS')
    leg = dict(game_id='g', player_id='p', stat='pts', line=20.5, side='over')
    args = ({('g', 'p'): pred}, {'pts': .1}, {'same': np.eye(3).tolist()})
    r = parlay.parlay([leg, leg], *args)
    assert abs(r['joint']-r['leg_probs'][0]) < .01
    r = parlay.parlay([leg, dict(leg, side='under')], *args)
    assert r['joint'] == 0


@pytest.mark.parametrize("margin", [-8, 0, 8])
def test_team_price_complements_signs_and_margin_partition(margin):
    p = team_model.price(dict(margin=margin, total=225, sigma_margin=14, sigma_total=18))
    assert p['p_home']+p['p_away'] == 1
    assert sum(x['p'] for x in p['bands']) == pytest.approx(1)
    assert p['home_pts']-p['away_pts'] == margin
    hand = p['handicaps']
    assert all(a['p_home'] < b['p_home'] for a, b in zip(hand, hand[1:]))
    assert all(x['p_home']+x['p_away'] == 1 for x in hand)
    assert all(x['p_over']+x['p_under'] == 1 for x in p['totals'])


@pytest.mark.parametrize("home,away,expected", [(-150, 150, .6), (100, -100, .5), (None, 100, None)])
def test_no_vig_moneyline(home, away, expected):
    assert team_model._nv(home, away) == expected
