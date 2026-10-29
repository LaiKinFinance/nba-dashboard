import numpy as np
import pandas as pd
import pandas.testing as pdt

import features
import predict


def history():
    rows = []
    for i in range(12):
        for team, opp in [('BOS', 'NY'), ('NY', 'BOS')]:
            rows.append(dict(game_id=str(i), player_id=team, player=team, team=team, opp=opp,
                             home=int(team == 'BOS'), date=f'2026-01-{i+1:02}', season='2026',
                             min=30., pts=20+i, reb=5, ast=5, starter=1, fg3m=2, stl=1, blk=1, _upcoming=0))
    return pd.DataFrame(rows)


def test_current_outcomes_do_not_change_rolling_or_opponent_features(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    d = history()
    changed = d.copy()
    changed.loc[changed.game_id == '11', ['pts', 'reb', 'ast', 'min']] = 90
    a, b = features.build(d), features.build(changed)
    cols = [x for x in features.FEATURES if x != 'starter_today']
    pdt.assert_frame_equal(a.loc[a.game_id == '11', cols], b.loc[b.game_id == '11', cols])


def test_later_upcoming_game_never_uses_earlier_placeholder_zeros(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    d = history()
    first = d.tail(2).assign(game_id='20', date='2026-01-20', _upcoming=1, pts=0, min=0)
    second = first.assign(game_id='21', date='2026-01-21')
    alone = predict.upcoming_features(d, second)
    together = predict.upcoming_features(d, pd.concat([first, second]))
    cols = features.FEATURES
    pdt.assert_frame_equal(alone[cols].reset_index(drop=True), together[together.game_id == '21'][cols].reset_index(drop=True))
    assert np.all(alone.min_m10 == 30)
