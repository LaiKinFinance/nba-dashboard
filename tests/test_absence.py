import pandas as pd
import pandas.testing as pdt

import absence


def history():
    rows = []
    for i in range(8):
        for pid in (['p', 'q'] if i < 5 else ['q']):
            rows.append(dict(game_id=str(i), date=f'2026-01-{i+1:02}', season='2026',
                             team='BOS', player_id=pid, player=pid, min=30, pts=20, reb=5, ast=5))
    return pd.DataFrame(rows)


def test_future_trade_cannot_rewrite_past_absences():
    before = history()
    future = pd.DataFrame([dict(before.iloc[0], game_id='9', date='2026-02-01', team='NY')])
    a, _ = absence.track(before)
    b, state = absence.track(pd.concat([before, future]))
    pdt.assert_frame_equal(a, b.loc[a.index])
    assert a.loc[('5', 'BOS'), 'abs_pts'] == 20
    assert 'p' not in state['BOS']


def test_current_game_stats_only_update_future_features():
    before = history()
    changed = before.copy()
    changed.loc[changed.game_id == '5', 'pts'] = 200
    a, _ = absence.track(before)
    b, _ = absence.track(changed)
    pdt.assert_series_equal(a.loc[('5', 'BOS')], b.loc[('5', 'BOS')])


def test_regulars_expire_and_seasons_reset():
    d = history()
    extras = [dict(d.iloc[-1], game_id=str(i), date=f'2026-02-{i-7:02}') for i in range(8, 34)]
    _, state = absence.track(pd.concat([d, pd.DataFrame(extras)]))
    assert 'p' not in absence.regulars(state['BOS'])
    new = pd.DataFrame([dict(d.iloc[-1], game_id='99', date='2026-10-20', season='2027')])
    f, _ = absence.track(pd.concat([d, new]))
    assert f.loc[('99', 'BOS'), 'abs_n'] == 0
