import datetime as dt
import sqlite3
from contextlib import closing

import pytest

import ingest


@pytest.mark.parametrize('extended_schema', [False, True], ids=['14-columns', '20-columns'])
def test_ingest_inserts_game_after_market_columns_added(tmp_path, monkeypatch, extended_schema):
    db = tmp_path / 'nba.db'
    if extended_schema:
        with closing(sqlite3.connect(db)) as con:
            con.executescript(ingest.SCHEMA)
            ingest.ensure_columns(con)
            for field in ['home_fta', 'away_fta', 'home_pf', 'away_pf', 'home_fga', 'away_fga']:
                con.execute(f'ALTER TABLE games ADD COLUMN {field} REAL')
            con.commit()
            assert len(con.execute('PRAGMA table_info(games)').fetchall()) == 20
    monkeypatch.setattr(ingest, 'DB', str(db))
    game = dict(game_id='g', date='2026-10-28', season='2027', season_type=2,
                home_id='1', home='BOS', away_id='2', away='NY', home_pts=110, away_pts=100)
    monkeypatch.setattr(ingest, 'final_games', lambda *_: [game])
    lines = dict(ou=220.5, spread_home=-4.5, ml_home=-150, ml_away=130)
    monkeypatch.setattr(ingest, 'parse_box', lambda _: ([], lines))
    ingest.run(dt.date(2026, 10, 28), dt.date(2026, 10, 28))
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT home,ou FROM games').fetchone() == ('BOS', 220.5)
