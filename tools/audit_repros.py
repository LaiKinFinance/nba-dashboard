"""Synthetic, offline demonstrations of pricing limitations left for redesign."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parlay
import team_model


if __name__ == '__main__':
    p = dict(proj=dict(pts=20, reb=8, ast=6), team='BOS', p_dd=.3, p_td=.1)
    legs = [dict(game_id='g', player_id='p', stat='td', side='over', line=0),
            dict(game_id='g', player_id='p', stat='dd', side='under', line=0)]
    r = parlay.parlay(legs, {('g', 'p'): p}, dict.fromkeys(parlay.CORE, .1), {'same': np.eye(3).tolist()}, n=200000)
    print(json.dumps(dict(impossible_td_yes_dd_no_joint=r['joint'], must_equal=0)))
    p = team_model.price(dict(margin=8, total=225, sigma_margin=14, sigma_total=18, p_home_ml=.8))
    home_bands = sum(x['p'] for x in p['bands'] if x['team'] == 'home')
    print(json.dumps(dict(moneyline=p['p_home'], home_margin_bands=home_bands, should_agree=True)))
