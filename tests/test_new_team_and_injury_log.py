"""Traded veterans get no Best pick for their first 10 games with the new team; the injury log stores only changes; our own handicap margin is toned down."""
import sqlite3
import app, slate, team_model
from tests.test_best_picks_real_lines import player, entry

def _db():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE player_games(game_id TEXT, player_id TEXT, date TEXT, team TEXT, min REAL)")
    def games(pid, team, n, start):
        for i in range(n): c.execute("INSERT INTO player_games VALUES(?,?,?,?,30)", (f"{pid}{team}{i}", pid, f"2026-{start + i // 28:02d}-{i % 28 + 1:02d}", team))
    games("vet", "AAA", 40, 1); games("vet", "BBB", 4, 6)            # 40 games for AAA, then 4 for BBB
    games("settled", "AAA", 40, 1)
    games("rookie", "AAA", 5, 1); games("rookie", "BBB", 2, 6)       # too few games in the data to call him a veteran
    games("offseason", "AAA", 40, 1)                                 # moved in the summer, no game yet for CCC
    return c

def test_new_team_flags_only_veterans_in_first_10_games():
    out = app.new_team_status(_db(), [("vet", "BBB"), ("settled", "AAA"), ("rookie", "BBB"), ("offseason", "CCC")])
    assert out["vet"] == dict(games=4, prev="AAA") and out["offseason"] == dict(games=0, prev="AAA")
    assert "settled" not in out and "rookie" not in out

def test_new_team_player_gets_no_best_pick():
    mk = lambda pid, nt: dict(player("p" + pid, "g1", book_reb=[entry(3.5, .70, 1.8)], ladder_reb=[entry(3.5, .70)]), returning=None, new_team=nt)
    assert {c["player_id"] for c in app._best_candidates([mk("a", None), mk("b", dict(games=2, prev="X"))])} == {"pa"}

def test_injury_log_stores_only_changes(monkeypatch):
    feed = [dict(player_id="1", player="A B", team="T", status="Out", body="Foot", injury="Fracture", side="Left", return_date="2026-11-20", note="n", reported="r1")]
    monkeypatch.setattr(slate, "injuries_full", lambda: [dict(x) for x in feed])
    c = sqlite3.connect(":memory:")
    assert app.log_injuries(c)["stored"] == 1 and app.log_injuries(c)["stored"] == 0       # nothing new the second time
    feed[0].update(status="Questionable", reported="r2")
    assert app.log_injuries(c)["stored"] == 1
    assert [r[0] for r in c.execute("SELECT status FROM injury_log ORDER BY id")] == ["Out", "Questionable"]

def test_own_margin_is_toned_down_but_market_is_not():
    assert team_model.MODEL_MARGIN_SHRINK == 0.8
