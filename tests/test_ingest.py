import datetime as dt
import sqlite3

import ingest


def test_ingest_inserts_game_after_market_columns_added(tmp_path, monkeypatch):
    db = tmp_path / 'nba.db'
    monkeypatch.setattr(ingest, 'DB', str(db))
    game = dict(game_id='g', date='2026-10-28', season='2027', season_type=2,
                home_id='1', home='BOS', away_id='2', away='NY', home_pts=110, away_pts=100)
    monkeypatch.setattr(ingest, 'final_games', lambda *_: [game])
    lines = dict(ou=220.5, spread_home=-4.5, ml_home=-150, ml_away=130)
    monkeypatch.setattr(ingest, 'parse_box', lambda _: ([], lines))
    ingest.run(dt.date(2026, 10, 28), dt.date(2026, 10, 28))
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT home,ou FROM games').fetchone() == ('BOS', 220.5)
