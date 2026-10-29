"""Regression test: ingestion must work on the CURRENT (migrated) games schema and must not wipe extra columns on re-ingest."""
import os, shutil, sqlite3, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ingest

def _db():
    d = tempfile.mkdtemp(); p = os.path.join(d, "nba.db"); shutil.copy(os.path.join(os.path.dirname(__file__), "..", "data", "nba.db"), p); return p

def test_insert_new_game_on_migrated_schema(monkeypatch=None):
    con = sqlite3.connect(_db()); ingest.ensure_columns(con)
    assert len([r for r in con.execute("PRAGMA table_info(games)")]) > 10          # schema really has the extra columns
    g = dict(game_id="TEST-NEW", date="2099-01-01", date_utc="2099-01-02", tip_utc="2099-01-02T00:30Z", season="2099", season_type=2, home_id="1", home="ATL", away_id="2", away="BOS", home_pts=101, away_pts=99)
    sql = open(os.path.join(os.path.dirname(__file__), "..", "ingest.py")).read()
    assert "INSERT OR REPLACE INTO games VALUES" not in sql                          # the old positional insert is gone
    # run the exact statement used by ingest.run
    stmt = sql[sql.index('con.execute("""INSERT INTO games('):]; stmt = stmt[stmt.index('"""') + 3: stmt.index('""", g)')]
    con.execute(stmt, g); assert con.execute("SELECT home_pts FROM games WHERE game_id='TEST-NEW'").fetchone()[0] == 101

def test_reingest_keeps_market_lines():
    con = sqlite3.connect(_db()); ingest.ensure_columns(con)
    gid, = con.execute("SELECT game_id FROM games WHERE ou IS NOT NULL LIMIT 1").fetchone(); before = con.execute("SELECT ou, spread_home FROM games WHERE game_id=?", (gid,)).fetchone()
    row = dict(zip(["game_id", "date", "season", "season_type", "home_id", "home", "away_id", "away", "home_pts", "away_pts"], con.execute(
        "SELECT game_id,date,season,season_type,home_id,home,away_id,away,home_pts,away_pts FROM games WHERE game_id=?", (gid,)).fetchone()))
    sql = open(os.path.join(os.path.dirname(__file__), "..", "ingest.py")).read(); stmt = sql[sql.index('con.execute("""INSERT INTO games('):]; stmt = stmt[stmt.index('"""') + 3: stmt.index('""", g)')]
    row.update(date_utc=row["date"], tip_utc=None); con.execute(stmt, row); assert con.execute("SELECT ou, spread_home FROM games WHERE game_id=?", (gid,)).fetchone() == before

if __name__ == "__main__":
    test_insert_new_game_on_migrated_schema(); test_reingest_keeps_market_lines(); print("ingest hotfix tests passed")
