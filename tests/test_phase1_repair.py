"""Needs a database that has been through repair.run_all(). Set REPAIRED_DATA_DIR to a folder holding that nba.db and a current models.pkl;
otherwise these checks are skipped (the bundled database is deliberately left unrepaired)."""
import os, sqlite3
import pytest
D = os.environ.get("REPAIRED_DATA_DIR")
pytestmark = pytest.mark.skipif(not D, reason="set REPAIRED_DATA_DIR to a repaired copy of the data to run these")

def test_dates_and_rosters():
    con = sqlite3.connect(D + "/nba.db")
    assert con.execute("select count(*) from (select team,date from player_games group by team,date having count(distinct game_id)>1)").fetchone()[0] == 0
    assert con.execute("select count(*) from games where date_utc is null or tip_utc is null").fetchone()[0] == 0
    assert con.execute("select count(*) from games where date<>date_utc").fetchone()[0] > 500
    assert con.execute("select count(*) from rosters where team is not null").fetchone()[0] >= 400
