"""My bets are marked won / lost / refunded by the app from real results, with no clicking."""
import json, sqlite3
import pytest
from fastapi.testclient import TestClient
import app, settle

GAME = dict(home_pts=112, away_pts=106)             # home wins by 6, 218 total
BOX = dict(pts=27, reb=8, ast=5, fg3m=3, stl=1, blk=0)

def L(stat, line=0, side="over", **kw): return dict(game_id="g", player_id="p", stat=stat, line=line, side=side, **kw)

@pytest.mark.parametrize("leg,expect", [
    (L("pts", 24.5), "won"), (L("pts", 27.5), "lost"), (L("pts", 24.5, "under"), "lost"), (L("pts", 27.5, "under"), "won"),
    (L("pts", 27), "push"), (L("pts", 27, "under"), "push"),
    (L("pra", 39.5), "won"), (L("pra", 40), "push"), (L("pr", 36.5), "lost"), (L("pa", 31.5), "won"),
    (L("dd"), "lost"), (L("td"), "lost"), (L("td", side="under"), "won"), (L("dd", side="under"), "won"),      # 27 pts only: one category at 10+
    (dict(game_id="g", stat="ml", side="home"), "won"), (dict(game_id="g", stat="ml", side="away"), "lost"),
    (dict(game_id="g", stat="spread", side="home", line=-5.5), "won"), (dict(game_id="g", stat="spread", side="home", line=-6), "push"),
    (dict(game_id="g", stat="spread", side="away", line=-6.5), "won"), (dict(game_id="g", stat="spread", side="away", line=-5.5), "lost"),
    (dict(game_id="g", stat="total", side="over", line=217.5), "won"), (dict(game_id="g", stat="total", side="under", line=218), "push"),
    (dict(game_id="g", stat="tt", side="over", team="home", line=111.5), "won"), (dict(game_id="g", stat="tt", side="under", team="away", line=106), "push"),
    (dict(game_id="g", stat="band", team="home", lo=1, hi=5), "lost"), (dict(game_id="g", stat="band", team="home", lo=6, hi=10), "won"),
    (dict(game_id="g", stat="band", team="away", lo=1), "lost"), (dict(game_id="g", stat="band", team="home", lo=5), "won"),
])
def test_each_kind_of_leg(leg, expect):
    assert settle.leg_result(leg, GAME, BOX) == expect

def test_five_category_double_double_counts_steals_and_blocks():
    box = dict(pts=12, reb=4, ast=3, fg3m=0, stl=10, blk=1)
    assert settle.leg_result(L("dd"), GAME, box) == "won" and settle.leg_result(L("td"), GAME, box) == "lost"

def test_game_not_finished_or_box_not_stored_is_pending():
    assert settle.leg_result(L("pts", 20.5), None, None) is None and settle.leg_result(L("pts", 20.5), GAME, None) is None
    assert settle.leg_result(dict(game_id="g", stat="ml", side="home"), None, None) is None

def test_player_who_did_not_play_is_cancelled_not_lost():
    assert settle.leg_result(L("pts", 10.5), GAME, "missing") == "void"

@pytest.mark.parametrize("results,expect", [
    (["won", "won"], "won"), (["won", "lost"], "lost"), (["won", None], "open"), (["lost", None], "lost"),     # one lost leg decides it at once
    (["won", "push"], "won"), (["push", "void"], "void"), (["void"], "void"), (["push", None], "open"),
])
def test_whole_bet(results, expect):
    assert settle.settle_bet([{}] * len(results), results) == expect

def test_reduced_payout_needs_every_leg_price():
    legs = [dict(price=1.5), dict(price=2.0), dict(price=1.8)]
    assert settle.reduced_odds(legs, ["won", "push", "won"], 5.4) == (pytest.approx(2.7), True)
    legs[0].pop("price"); assert settle.reduced_odds(legs, ["won", "push", "won"], 5.4) == (5.4, False)
    assert settle.reduced_odds(legs, ["won", "won", "won"], 5.4) == (5.4, True)

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "b.db")); monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "model_status", lambda preds=None: ({"report": {"stats": {}}}, None))
    app.init()
    c = sqlite3.connect(app.DB)
    c.executescript("""CREATE TABLE IF NOT EXISTS games(game_id TEXT PRIMARY KEY, home_pts INT, away_pts INT);
                       CREATE TABLE IF NOT EXISTS player_games(game_id TEXT, player_id TEXT, pts INT, reb INT, ast INT, fg3m INT, stl INT, blk INT);""")
    c.execute("INSERT INTO games VALUES('g',112,106)"); c.execute("INSERT INTO player_games VALUES('g','p',27,8,5,3,1,0)")
    c.execute("INSERT INTO player_games VALUES('g','q',10,2,2,0,0,0)"); c.commit(); return c

def _bet(c, legs, odds=3.0, stake=10.0, **kw):
    c.execute("INSERT INTO bets(user,created,legs,stake,odds,model_prob) VALUES('mark','t',?,?,?,0.3)", (json.dumps(legs), stake, odds)); c.commit()
    return c.execute("SELECT max(id) FROM bets").fetchone()[0]

def _row(c, i): return dict(zip([d[0] for d in c.execute("SELECT * FROM bets LIMIT 0").description], c.execute("SELECT * FROM bets WHERE id=?", (i,)).fetchone()))

def test_open_bets_are_marked_by_the_app(db):
    won = _bet(db, [L("pts", 24.5), dict(game_id="g", stat="ml", side="home")]); lost = _bet(db, [L("pts", 30.5)])
    not_played = _bet(db, [dict(game_id="g", player_id="zzz", stat="pts", line=10.5, side="over")])
    pending = _bet(db, [dict(game_id="future", player_id="p", stat="pts", line=10.5, side="over")])
    assert app.grade_bets()["graded"] == 3
    assert _row(db, won)["result"] == "won" and _row(db, lost)["result"] == "lost" and _row(db, not_played)["result"] == "void" and _row(db, pending)["result"] == "open"
    assert json.loads(_row(db, won)["leg_results"]) == ["won", "won"]

def test_one_lost_leg_settles_the_bet_before_the_other_game_finishes(db):
    b = _bet(db, [L("pts", 30.5), dict(game_id="future", player_id="p", stat="pts", line=10.5, side="over")])
    app.grade_bets(); assert _row(db, b)["result"] == "lost"

def test_refunded_leg_reduces_the_payout_when_prices_are_known(db):
    b = _bet(db, [dict(L("pts", 24.5), price=1.5), dict(L("pts", 27), price=2.0), dict(game_id="g", stat="ml", side="home", price=1.8)], odds=5.4)
    app.grade_bets(); r = _row(db, b)
    assert r["result"] == "won" and r["payout_odds"] == pytest.approx(1.5 * 1.8) and "refunded" in r["settle_note"]

def test_refund_without_leg_prices_keeps_full_odds_and_says_so(db):
    b = _bet(db, [L("pts", 24.5), L("pts", 27)], odds=3.0); app.grade_bets(); r = _row(db, b)
    assert r["result"] == "won" and r["payout_odds"] == 3.0 and "Check the exact amount" in r["settle_note"]

def test_bets_marked_by_hand_are_left_alone_and_undo_hands_back(db):
    b = _bet(db, [L("pts", 30.5)]); c = TestClient(app.app)
    assert c.post(f"/api/bets/{b}/result", json={"result": "won"}, headers={"X-User": "mark"}).status_code == 200
    app.grade_bets(); r = _row(db, b); assert r["result"] == "won" and r["manual"] == 1                     # the app did not override him
    c.post(f"/api/bets/{b}/result", json={"result": "open"}, headers={"X-User": "mark"}); app.grade_bets()
    assert _row(db, b)["result"] == "lost" and _row(db, b)["manual"] == 0

def test_scorecard_profit_uses_the_reduced_payout(db):
    _bet(db, [dict(L("pts", 24.5), price=1.5), dict(L("pts", 27), price=2.0)], odds=3.0, stake=10.0)
    s = TestClient(app.app).get("/api/scorecard").json()
    me = next(u for u in s["users"] if u["user"] == "mark")
    assert me["profit"] == pytest.approx(10 * 1.5 - 10)

def test_grading_twice_changes_nothing(db):
    b = _bet(db, [L("pts", 24.5)]); app.grade_bets(); first = _row(db, b); assert app.grade_bets()["graded"] == 0 and _row(db, b) == first
