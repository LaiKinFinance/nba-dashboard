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


def test_invalid_saved_data_rejected(api_module, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api_module, 'ACCESS_CODE', '')
    client = TestClient(api_module.app)
    leg = dict(game_id='g', player_id='p', stat='pts', line=20.5, side='over')
    bad = dict(leg, line='<img src=x onerror=alert(1)>')
    assert client.post('/api/picks', json={'legs': [bad]}, headers={'X-User': 'Audit'}).status_code == 422
    assert client.post('/api/bets', json={'legs': [leg], 'stake': -10, 'odds': 2}, headers={'X-User': 'Audit'}).status_code == 422
    assert client.post('/api/picks', json={'legs': [leg], 'odds': '<img>'}, headers={'X-User': 'Audit'}).status_code == 422


def test_numeric_inputs_are_finite():
    from pydantic import ValidationError
    from inputs import BetInput
    with pytest.raises(ValidationError):
        BetInput(legs=[dict(game_id='g', player_id='p', stat='pts')], odds=2, stake=float('inf'))


@pytest.mark.parametrize('leg', [dict(game_id='g', stat='pts'), dict(game_id='g', stat='ml', side='over'),
                                dict(game_id='g', stat='tt', line=110.5), dict(game_id='g', stat='band', side='home', team='home', lo=10, hi=5)])
def test_incomplete_or_inconsistent_markets_are_validation_errors(leg):
    from pydantic import ValidationError
    from inputs import Leg
    with pytest.raises(ValidationError):
        Leg(**leg)


def test_shared_code_rejects_missing_or_wrong_header(api_module, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api_module, 'ACCESS_CODE', 'synthetic-auth-test')
    client = TestClient(api_module.app)
    assert client.get('/api/picks').status_code == 401
    assert client.get('/api/picks', headers={'X-Code': 'wrong'}).status_code == 401
    assert client.get('/api/picks', headers={'X-Code': 'synthetic-auth-test'}).status_code == 200


def test_player_search_treats_sql_payload_as_a_value(api_module, monkeypatch):
    from contextlib import closing
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api_module, 'ACCESS_CODE', '')
    with closing(api_module.con()) as con:
        con.execute('CREATE TABLE player_games(player_id TEXT, player TEXT, team TEXT, date TEXT)')
        con.execute('INSERT INTO player_games VALUES(?,?,?,?)', ('p', 'Desmond Bane', 'ORL', '2026-10-28'))
        con.commit()
    client = TestClient(api_module.app)
    assert client.get('/api/players', params={'q': 'Bane'}).json()[0]['player_id'] == 'p'
    assert client.get('/api/players', params={'q': "x%' OR 1=1 --"}).json() == []
