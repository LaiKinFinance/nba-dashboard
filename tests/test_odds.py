import itertools
import sqlite3

import pytest
import requests

import odds


def test_parse_skips_malformed_odds_and_maps_binary_sides():
    event = {'bookmakers': [{'key': 'sportsbet', 'markets': [
        {'key': 'player_points', 'outcomes': [
            dict(description='Nikola Jokić', name='Over', point=20.5, price=1.9),
            dict(description='Unknown Player', name='Over', point=20.5, price=1.9),
            dict(description='Nikola Jokic', name='Yes', price=1.9),
            *[dict(description='Nikola Jokic', name='Over', point=20.5, price=x) for x in [0, 1, 'bad', float('nan'), float('inf')]]]},
        {'key': 'player_double_double', 'outcomes': [dict(description='Nikola Jokic', name=s, price=2) for s in ['Yes', 'No']]}]}]}
    rows = odds.parse_event(event, {'nikola jokic': 'p'})
    assert rows == [('p', 'pts', 20.5, 'over', 1.9, 'sportsbet'), ('p', 'dd', 0., 'over', 2., 'sportsbet'), ('p', 'dd', 0., 'under', 2., 'sportsbet')]


@pytest.mark.parametrize('order', list(itertools.permutations([(1.8, 'sportsbet'), (2.1, 'tab'), (2., 'neds')])))
def test_preferred_book_and_best_price_are_order_independent(order):
    with sqlite3.connect(':memory:') as con:
        odds.init(con)
        for price, book in order:
            con.execute('INSERT INTO book_odds VALUES(?,?,?,?,?,?,?,?)', ('g', 'p', 'pts', 20.5, 'over', price, book, 'now'))
            con.execute('INSERT INTO book_lines VALUES(?,?,?,?,?,?,?)', ('g', 'spread', 'home', -4.5, price, book, 'now'))
        for slot in [odds.load(con)[('g', 'p')]['pts'][20.5], odds.load_lines(con)['g']['spread'][-4.5]]:
            side = 'over' if 'over' in slot else 'home'
            assert slot[side] == (1.8, 'sportsbet')
            assert slot[side+'_best'] == (2.1, 'tab')


def test_provider_failure_never_returns_request_url(monkeypatch):
    # Synthetic request marker; no credential is read or used by this test.
    marker = 'redaction-test-marker'
    monkeypatch.setenv('ODDS_API_KEY', marker)
    response = requests.Response(); response.status_code = 401
    def fail(*args, **kwargs):
        raise requests.HTTPError('URL includes '+marker, response=response)
    monkeypatch.setattr(odds.requests, 'get', fail)
    with pytest.raises(RuntimeError) as err:
        odds._get('/test')
    assert marker not in str(err.value)
    assert '401' in str(err.value)
