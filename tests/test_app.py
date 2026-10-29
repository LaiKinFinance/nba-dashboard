import importlib

import pytest

import team_model


@pytest.fixture
def api_module(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AUTO_REFRESH', '0')
    module = importlib.import_module('app')
    monkeypatch.setattr(module, 'DB', str(tmp_path / 'nba.db'))
    module.init()
    return module


def game():
    return dict(team_model.price(dict(margin=5, total=225, sigma_margin=14, sigma_total=18)), home='BOS', away='NY')


def test_match_books_belong_to_each_side(api_module):
    p = api_module.match_payload(game(), {'spread': {-4.5: {'home': (1.9, 'tab'), 'away': (2., 'neds')}},
                                         'total': {224.5: {'over': (1.8, 'sportsbet'), 'under': (2.1, 'tab')}}})
    h = next(x for x in p['handicaps'] if x['line'] == -4.5)
    t = next(x for x in p['totals'] if x['line'] == 224.5)
    assert h['book_home'] == 'tab' and h['book_away'] == 'neds'
    assert t['book_over'] == 'sportsbet' and t['book_under'] == 'tab'


@pytest.mark.parametrize('stat,side,line', [('ml', 'home', 0), ('spread', 'away', -4.5), ('total', 'under', 224.5), ('tt', 'over', 112.5)])
def test_team_api_pricing_in_range(api_module, stat, side, line):
    p = api_module.team_leg_prob(dict(stat=stat, side=side, line=line, team='home'), game())
    assert 0 < p < 1
