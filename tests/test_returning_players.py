"""Players in their first 2 games back after missing 3+ team games get no Best pick and are flagged."""
import sqlite3
import app
from tests.test_best_picks_real_lines import player, entry

def _db():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.executescript("""CREATE TABLE games(game_id TEXT, date TEXT, season TEXT, home TEXT, away TEXT, home_pts INT, away_pts INT);
                       CREATE TABLE player_games(game_id TEXT, player_id TEXT, date TEXT, team TEXT, season TEXT);""")
    for i in range(1, 11): c.execute("INSERT INTO games VALUES(?,?,?,?,?,100,90)", (f"g{i}", f"2026-11-{i:02d}", "2027", "A", "B"))
    def played(pid, games):
        for i in games: c.execute("INSERT INTO player_games VALUES(?,?,?,?,?)", (f"g{i}", pid, f"2026-11-{i:02d}", "A", "2027"))
    played("fresh", range(1, 11)); played("back1", range(1, 6)); played("back2", [1, 2, 3, 4, 5, 10]); played("short", [1, 2, 3, 4, 5, 6, 7, 9, 10]); played("short2", range(1, 8))
    return c

def test_flags_first_and_second_game_back_only():
    out = app.returning_status(_db(), ["fresh", "back1", "back2", "short", "short2"])
    assert out["back1"] == dict(game_back=1, missed=5)
    assert out["back2"] == dict(game_back=2, missed=4)
    assert "fresh" not in out and "short" not in out          # one missed game is not an absence
    assert out["short2"] == dict(game_back=1, missed=3)

def test_returning_player_gets_no_best_pick(monkeypatch):
    mk = lambda pid, ret: dict(player("p" + pid, "g1", book_reb=[entry(3.5, .70, 1.8)], ladder_reb=[entry(3.5, .70)]), returning=ret)
    got = app._best_candidates([mk("a", None), mk("b", dict(game_back=1, missed=4))])
    assert {c["player_id"] for c in got} == {"pa"}
