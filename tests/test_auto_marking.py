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
    (["won", "push"], "review"), (["push", "void"], "review"), (["void"], "review"), (["push", None], "open"), (["lost", "push"], "lost"),
])
def test_whole_bet(results, expect):
    assert settle.settle_bet([{}] * len(results), results) == expect

def test_the_app_no_longer_works_out_reduced_payouts():
    assert not hasattr(settle, "reduced_odds")

def test_twenty_or_more_is_judged_as_nineteen_and_a_half():
    box = dict(pts=20, reb=0, ast=0, fg3m=0, stl=0, blk=0)
    assert settle.leg_result(L("pts", 20, kind="atleast"), GAME, box) == "won"
    assert settle.leg_result(L("pts", 20, kind="atleast"), GAME, dict(box, pts=19)) == "lost"
    assert settle.leg_result(L("pts", 20), GAME, box) == "push"                  # "over 20" on exactly 20 finishes level; "20 or more" wins

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
    assert _row(db, won)["result"] == "won" and _row(db, lost)["result"] == "lost" and _row(db, pending)["result"] == "open"
    assert json.loads(_row(db, won)["leg_results"]) == ["won", "won"]
    r = _row(db, not_played); assert r["result"] == "open" and r["needs_amount"] == 1 and "Enter the amount" in r["settle_note"]      # not guessed

def test_one_lost_leg_settles_the_bet_before_the_other_game_finishes(db):
    b = _bet(db, [L("pts", 30.5), dict(game_id="future", player_id="p", stat="pts", line=10.5, side="over")])
    app.grade_bets(); assert _row(db, b)["result"] == "lost"

def test_a_level_leg_waits_for_the_amount_sportsbet_returned(db):
    b = _bet(db, [dict(L("pts", 24.5), price=1.5), L("pts", 27)], odds=3.0, stake=10.0)
    app.grade_bets(); r = _row(db, b)
    assert r["result"] == "open" and r["needs_amount"] == 1 and r["payout_odds"] is None and json.loads(r["leg_results"]) == ["won", "push"]
    assert app.grade_bets()["graded"] == 0                                                       # not re-processed, not guessed later either

def test_entering_the_amount_decides_the_result_and_the_profit(db):
    c = TestClient(app.app); H = {"X-User": "mark"}
    a = _bet(db, [L("pts", 24.5), L("pts", 27)], stake=10.0); b = _bet(db, [L("pts", 24.5), L("pts", 27)], stake=10.0); d = _bet(db, [L("pts", 24.5), L("pts", 27)], stake=10.0)
    app.grade_bets()
    assert c.post(f"/api/bets/{a}/returned", json={"amount": 18.5}, headers=H).json()["result"] == "won"
    assert c.post(f"/api/bets/{b}/returned", json={"amount": 10}, headers=H).json()["result"] == "void"
    assert c.post(f"/api/bets/{d}/returned", json={"amount": 0}, headers=H).json()["result"] == "lost"
    me = next(u for u in c.get("/api/scorecard").json()["users"] if u["user"] == "mark")
    assert me["profit"] == pytest.approx((18.5 - 10) + 0 + (0 - 10)) and me["awaiting_amount"] == 0           # only the amounts he typed in

def test_bets_waiting_for_an_amount_are_left_out_of_profit_and_counted(db):
    _bet(db, [L("pts", 24.5), L("pts", 27)], stake=10.0); app.grade_bets()
    me = next(u for u in TestClient(app.app).get("/api/scorecard").json()["users"] if u["user"] == "mark")
    assert me["profit"] == 0 and me["settled"] == 0 and me["awaiting_amount"] == 1

@pytest.mark.parametrize("bad", ["abc", -1, "nan", "inf", None])
def test_bad_amounts_are_refused(db, bad):
    b = _bet(db, [L("pts", 24.5)])
    r = TestClient(app.app, raise_server_exceptions=False).post(f"/api/bets/{b}/returned", json={"amount": bad}, headers={"X-User": "mark"}); assert r.status_code in (400, 422)

def test_someone_elses_bet_cannot_be_given_an_amount(db):
    b = _bet(db, [L("pts", 24.5)])
    assert TestClient(app.app).post(f"/api/bets/{b}/returned", json={"amount": 5}, headers={"X-User": "other"}).status_code == 404

def test_old_records_with_refund_columns_still_load(db):
    db.execute("INSERT INTO bets(user,created,legs,stake,odds,model_prob,result,leg_results,settle_note,payout_odds,manual) VALUES('mark','2026-01-01T00:00Z','[]',10,3,.3,'won','[\"won\",\"push\"]','old note',1.5,0)"); db.commit()
    s = TestClient(app.app).get("/api/scorecard").json(); assert any(b["payout_odds"] == 1.5 for b in s["bets"]); assert app.grade_bets()["graded"] == 0

def test_bets_marked_by_hand_are_left_alone_and_undo_hands_back(db):
    b = _bet(db, [L("pts", 30.5)]); c = TestClient(app.app)
    assert c.post(f"/api/bets/{b}/result", json={"result": "won"}, headers={"X-User": "mark"}).status_code == 200
    app.grade_bets(); r = _row(db, b); assert r["result"] == "won" and r["manual"] == 1                     # the app did not override him
    c.post(f"/api/bets/{b}/result", json={"result": "open"}, headers={"X-User": "mark"}); app.grade_bets()
    assert _row(db, b)["result"] == "lost" and _row(db, b)["manual"] == 0

def test_grading_twice_changes_nothing(db):
    b = _bet(db, [L("pts", 24.5)]); app.grade_bets(); first = _row(db, b); assert app.grade_bets()["graded"] == 0 and _row(db, b) == first
