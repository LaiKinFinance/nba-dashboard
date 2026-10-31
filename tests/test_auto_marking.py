"""Personal picks are scored from game results. No money is entered or calculated."""
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
    (["won", "won"], "won"), (["won", "lost"], "lost"), (["won", None], "open"), (["lost", None], "open"),     # wait for the complete pick before scoring it
    (["won", "push"], "void"), (["push", "void"], "void"), (["void"], "void"), (["push", None], "open"), (["lost", "push"], "lost"), (["void", "lost"], "lost"),
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

def _bet(c, legs):
    c.execute("INSERT INTO bets(user,created,legs,model_prob) VALUES('mark','t',?,0.3)", (json.dumps(legs),)); c.commit()
    return c.execute("SELECT max(id) FROM bets").fetchone()[0]


def _row(c, i):
    c.row_factory = sqlite3.Row
    return dict(c.execute("SELECT * FROM bets WHERE id=?", (i,)).fetchone())


def test_open_picks_are_scored_from_game_results(db):
    won = _bet(db, [L("pts", 24.5), dict(game_id="g", stat="ml", side="home")]); lost = _bet(db, [L("pts", 30.5)])
    absent = _bet(db, [dict(game_id="g", player_id="zzz", stat="pts", line=10.5, side="over")])
    pending = _bet(db, [dict(game_id="future", player_id="p", stat="pts", line=10.5, side="over")])
    assert app.grade_bets()["graded"] == 3
    assert _row(db, won)["result"] == "won" and _row(db, lost)["result"] == "lost"
    assert _row(db, absent)["result"] == "void" and _row(db, pending)["result"] == "open"
    assert json.loads(_row(db, won)["leg_results"]) == ["won", "won"]


def test_incomplete_combination_waits_even_if_one_pick_is_wrong(db):
    b = _bet(db, [L("pts", 30.5), dict(game_id="future", player_id="p", stat="pts", line=10.5, side="over")])
    app.grade_bets(); assert _row(db, b)["result"] == "open"


@pytest.mark.parametrize("legs", [
    [L("pts", 24.5), L("pts", 27)],
    [L("pts", 24.5), dict(game_id="g", player_id="absent", stat="reb", line=4.5, side="over")],
])
def test_unscorable_picks_are_left_out_of_win_loss_count(db, legs):
    b = _bet(db, legs); app.grade_bets()
    assert _row(db, b)["result"] == "void"
    me = TestClient(app.app).get("/api/scorecard").json()["users"][0]
    assert me["not_counted"] == 1 and me["settled"] == 0 and me["hit_rate"] is None
    assert app.grade_bets()["graded"] == 0


def test_saving_a_pick_needs_no_money_or_bookmaker_price(db):
    client = TestClient(app.app)
    r = client.post("/api/bets", json={"legs": [L("reb", 7.5)], "model_prob": .4}, headers={"X-User": "mark"})
    assert r.status_code == 200
    saved = _row(db, 1)
    assert saved["stake"] is None and saved["odds"] is None and saved["model_prob"] is None     # the chance comes from the server, not the phone
    score = client.get("/api/scorecard").json()
    assert score["users"][0]["wins"] == 1
    for financial in ("profit", "roi", "returned", "staked", "returned_amount", "payout_odds", "stake", "needs_amount"):
        assert financial not in score["users"][0] and financial not in score["bets"][0]


def test_recording_money_is_no_longer_a_route(db):
    client = TestClient(app.app)
    assert "/api/bets/{bid}/returned" not in client.get("/openapi.json").json()["paths"]
    b = _bet(db, [L("pts", 24.5)])
    assert client.post(f"/api/bets/{b}/returned", json={"amount": 15}, headers={"X-User":"mark"}).status_code in (404, 405)


def test_existing_money_fields_and_records_are_preserved_but_unused(db):
    # Simulate a database upgraded by the old release, with its retired fields.
    for name, kind in (("payout_odds", "REAL"), ("returned_amount", "REAL"), ("needs_amount", "INT")):
        db.execute(f"ALTER TABLE bets ADD COLUMN {name} {kind}")
    db.execute("INSERT INTO bets(user,created,legs,stake,odds,model_prob,result,payout_odds,returned_amount,needs_amount,manual) VALUES('mark','old','[]',10,3,.3,'won',1.5,15,0,1)"); db.commit()
    before = _row(db, 1)
    app.init(); app.init()
    client = TestClient(app.app); score = client.get("/api/scorecard").json()
    assert _row(db, 1) == before
    assert score["users"][0]["wins"] == 1
    assert client.post('/api/bets/1/result', json={"result":"lost"}, headers={"X-User":"mark"}).status_code in (404, 405)
    assert _row(db, 1) == before


def test_old_pending_records_do_not_wait_for_money(db):
    db.execute("ALTER TABLE bets ADD COLUMN needs_amount INT")
    b = _bet(db, [L("pts", 27)]); db.execute("UPDATE bets SET needs_amount=1 WHERE id=?",(b,)); db.commit()
    app.grade_bets()
    assert _row(db,b)["result"] == "void"


def test_results_page_does_not_need_a_prediction_model(db, monkeypatch):
    monkeypatch.setattr(app, "model_status", lambda *args: (_ for _ in ()).throw(AssertionError("No model needed to read results")))
    assert TestClient(app.app).get('/api/scorecard').status_code == 200


def test_won_and_lost_counts_ignore_unscored_picks(db):
    for legs in ([L("pts",24.5)], [L("pts",30.5)], [L("pts",27)], [dict(game_id="future",player_id="p",stat="reb",line=3.5,side="over")]):
        _bet(db, legs)
    user = TestClient(app.app).get('/api/scorecard').json()["users"][0]
    assert user == dict(user="mark",bets=4,settled=2,wins=1,losses=1,open=1,not_counted=1,hit_rate=.5)


def test_results_cannot_be_set_by_hand(db):
    b = _bet(db, [L("pts", 24.5)])
    assert TestClient(app.app).post(f"/api/bets/{b}/result", json={"result": "lost"}, headers={"X-User": "mark"}).status_code in (404, 405)
    app.grade_bets(); assert _row(db, b)["result"] == "won"


def test_grading_twice_changes_nothing(db):
    b = _bet(db, [L("pts", 24.5)]); app.grade_bets(); first = _row(db, b)
    assert app.grade_bets()["graded"] == 0 and _row(db, b) == first


def test_a_lost_leg_is_not_hidden_by_a_cancelled_leg(db):
    b = _bet(db, [L("pts", 30.5), dict(game_id="g", player_id="absent", stat="reb", line=4.5, side="over")]); app.grade_bets()
    assert _row(db, b)["result"] == "lost"
    me = TestClient(app.app).get("/api/scorecard").json()["users"][0]
    assert me["losses"] == 1 and me["not_counted"] == 0


def test_one_time_clean_start_clears_saved_picks_once_and_keeps_a_copy(db):
    import os
    db.execute("DELETE FROM one_time_jobs"); b = _bet(db, [L("pts", 24.5)])
    db.execute("INSERT INTO picks(user,created,game_id,legs) VALUES('mark','x','g','[]')"); db.commit()
    app.one_time_reset()
    assert db.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0 and db.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 0
    kept = os.path.join(os.path.dirname(app.DB), "backups", "before-reset-" + app.RESET_ID + ".sqlite")
    assert os.path.exists(kept)
    _bet(db, [L("pts", 24.5)]); app.one_time_reset()
    assert db.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 1


def test_a_fresh_start_on_the_bundled_database_sets_itself_up():
    import subprocess, sys, shutil, tempfile, os
    d = tempfile.mkdtemp(); shutil.copy(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nba.db"), d)
    code = ("import app; from fastapi.testclient import TestClient; r = TestClient(app.app).get('/api/scorecard'); "
            "assert r.status_code == 200, r.text; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "DATA_DIR": d, "AUTO_REFRESH": "0"},
                         cwd=os.path.dirname(os.path.dirname(__file__)))
    assert "ok" in out.stdout, out.stderr[-600:]


# ---- delete a waiting pick, leaderboard, saving after tip-off ----
def _api(headers=None): return TestClient(app.app)

def test_owner_can_delete_a_waiting_pick_only(db):
    b = _bet(db, [dict(game_id="future", player_id="p", stat="pts", line=10.5, side="over")]); c = TestClient(app.app)
    assert c.delete(f"/api/bets/{b}", headers={"X-User": "other"}).status_code == 404
    assert c.delete(f"/api/bets/{b}", headers={"X-User": "mark"}).status_code == 200
    assert db.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0

def test_a_pick_with_a_result_cannot_be_deleted(db):
    b = _bet(db, [L("pts", 24.5)]); app.grade_bets()
    assert TestClient(app.app).delete(f"/api/bets/{b}", headers={"X-User": "mark"}).status_code == 409
    assert db.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 1

def test_a_pick_cannot_be_saved_or_deleted_once_its_game_has_started(db):
    db.execute("CREATE TABLE IF NOT EXISTS projections(game_id TEXT, player_id TEXT, payload TEXT, made_at TEXT, PRIMARY KEY(game_id, player_id))")
    db.execute("INSERT INTO projections VALUES('started','p','{\"start\":\"2020-01-01T00:00Z\"}','t')")
    db.execute("INSERT INTO projections VALUES('later','p','{\"start\":\"2999-01-01T00:00Z\"}','t')"); db.commit()
    c = TestClient(app.app); leg = lambda g: dict(game_id=g, player_id="p", stat="pts", line=10.5, side="over")
    assert c.post("/api/bets", json=dict(legs=[leg("started")]), headers={"X-User": "mark"}).status_code == 400
    assert c.post("/api/bets", json=dict(legs=[leg("later")]), headers={"X-User": "mark"}).status_code == 200
    b = _bet(db, [leg("started")])
    assert c.delete(f"/api/bets/{b}", headers={"X-User": "mark"}).status_code == 409

def test_leaderboard_boards(db):
    for user, legs in (("ann", [L("pts", 24.5)]), ("ann", [L("pts", 23.5)]), ("bob", [L("pts", 24.5)]), ("bob", [L("pts", 30.5)])):
        i = _bet(db, legs); db.execute("UPDATE bets SET user=?, model_prob=? WHERE id=?", (user, .2 if user == "ann" else .5, i)); db.commit()
    lb = TestClient(app.app).get("/api/leaderboard").json()
    assert [r["user"] for r in lb["most_won"]] == ["ann", "bob"] and lb["most_won"][0]["won"] == 2
    assert lb["hardest"][0] == dict(user="ann", chance=.2)
    assert lb["rate"] == []          # nobody has 5 finished picks yet


def test_bookmaker_price_is_the_saved_leg_prices_multiplied_and_needs_every_leg():
    assert app.bookmaker_price([dict(price=1.5), dict(price=2.0)]) == 3.0
    assert app.bookmaker_price([dict(price=1.5), dict()]) is None and app.bookmaker_price([]) is None

def test_leaderboard_biggest_price_won(db):
    i = _bet(db, [dict(L("pts", 24.5), price=2.0), dict(L("pts", 24.5), price=1.5)]); db.execute("UPDATE bets SET user='ann' WHERE id=?", (i,))
    j = _bet(db, [L("pts", 24.5)]); db.execute("UPDATE bets SET user='bob' WHERE id=?", (j,)); db.commit()     # no saved price: not on this board
    assert TestClient(app.app).get("/api/leaderboard").json()["biggest_price"] == [dict(user="ann", price=3.0)]


def test_the_same_pick_cannot_be_saved_twice_and_never_counts_twice(db):
    c = TestClient(app.app); body = {"legs": [dict(L("pts", 24.5), game_id="future")]}
    assert c.post("/api/bets", json=body, headers={"X-User": "mark"}).status_code == 200
    assert c.post("/api/bets", json=body, headers={"X-User": "mark"}).status_code == 409
    assert c.post("/api/bets", json=body, headers={"X-User": "other"}).status_code == 200        # another friend may save the same pick
    a = _bet(db, [L("pts", 24.5)]); b = _bet(db, [L("pts", 24.5)]); app.grade_bets()               # older duplicates already in the database
    assert TestClient(app.app).get("/api/leaderboard").json()["most_won"] == [dict(user="mark", won=1, picks=1)]

def test_the_chance_and_price_sent_by_the_phone_are_not_trusted(db):
    c = TestClient(app.app)
    leg = dict(L("reb", 7.5), game_id="future", price=500.0)
    assert c.post("/api/bets", json={"legs": [leg], "model_prob": 0.0001}, headers={"X-User": "mark"}).status_code == 200
    saved = _row(db, 1); assert saved["model_prob"] is None and "price" not in json.loads(saved["legs"])[0]


def test_a_skipped_check_does_not_restart_its_timer(db):
    app._status_set("reconcile", "ok"); before = db.execute("SELECT ran_at FROM refresh_status WHERE stage='reconcile'").fetchone()[0]
    import time; time.sleep(0.01)
    assert app._stage("reconcile", lambda: dict(skipped=True, problem=None)) == "ok"
    assert db.execute("SELECT ran_at FROM refresh_status WHERE stage='reconcile'").fetchone()[0] == before

def test_no_retrain_when_the_missing_game_check_could_not_run(db, monkeypatch):
    import repair, ingest
    monkeypatch.setattr(repair, "run_all", lambda *a, **k: {}); monkeypatch.setattr(ingest, "run", lambda *a, **k: {})
    monkeypatch.setattr(app, "weekly_reconcile", lambda: (_ for _ in ()).throw(RuntimeError("ESPN unreachable")))
    monkeypatch.setattr(app, "_odds_on", lambda: False)
    for name in ("grade_picks", "grade_bets", "log_picks", "store_served_check", "backup_db"): monkeypatch.setattr(app, name, lambda *a, **k: None)
    trained = []; monkeypatch.setattr(app, "train_quietly", lambda: trained.append(1)); monkeypatch.setattr(app, "project_quietly", lambda: trained.append(2))
    app.refresh_once()
    assert trained == [] and db.execute("SELECT state FROM refresh_status WHERE stage='train'").fetchone()[0] == "skipped"

def test_players_and_lineup_pages_use_the_current_roster_team(db):
    db.executescript("""CREATE TABLE IF NOT EXISTS rosters(player_id TEXT PRIMARY KEY, team TEXT, name TEXT, fetched_at TEXT);
        INSERT INTO rosters VALUES('p1','DAL','Santi Aldama','t'); INSERT INTO rosters VALUES('p2',NULL,'Waived Guy','t');""")
    db.execute("ALTER TABLE player_games ADD COLUMN player TEXT"); db.execute("ALTER TABLE player_games ADD COLUMN team TEXT"); db.execute("ALTER TABLE player_games ADD COLUMN date TEXT")
    for pid, nm in (("p1", "Santi Aldama"), ("p2", "Waived Guy"), ("p3", "Santi Other")): db.execute("INSERT INTO player_games(game_id,player_id,player,team,date) VALUES('g9',?,?,?,'2026-10-01')", (pid, nm, "MEM"))
    db.commit(); c = TestClient(app.app)
    got = {r["player"]: r["team"] for r in c.get("/api/players", params={"q": "Santi"}).json()}
    assert got == {"Santi Aldama": "DAL", "Santi Other": "MEM"}                      # traded player shows the new team, an unlisted player keeps his last team
    assert {r["player"]: r["team"] for r in c.get("/api/players", params={"q": "Waived"}).json()} == {"Waived Guy": ""}
    import pandas as pd
    df = pd.DataFrame([dict(player_id="p1", player="A", team="MEM"), dict(player_id="p2", player="B", team="MEM"), dict(player_id="p3", player="C", team="MEM")])
    out = app._with_current_team(df, db)                                          # lineups use this: traded -> new team, off every roster -> left out, unlisted -> last team
    assert dict(zip(out.player_id, out.team)) == {"p1": "DAL", "p3": "MEM"}


def test_the_same_pick_cannot_be_shared_twice_but_can_be_shared_and_saved(db):
    c = TestClient(app.app); body = {"legs": [dict(L("pts", 24.5), game_id="future")]}
    assert c.post("/api/picks", json=body, headers={"X-User": "mark"}).status_code == 200
    assert c.post("/api/picks", json=body, headers={"X-User": "mark"}).status_code == 409
    assert c.post("/api/picks", json=body, headers={"X-User": "other"}).status_code == 200
    assert c.post("/api/bets", json=body, headers={"X-User": "mark"}).status_code == 200      # saving the same pick as well is allowed
    assert db.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 2


def test_only_the_owner_can_delete_a_shared_pick_and_its_comments_and_thumbs_go_with_it(db):
    c = TestClient(app.app); body = {"legs": [dict(L("pts", 24.5), game_id="future")]}
    c.post("/api/picks", json=body, headers={"X-User": "mark"})
    c.post("/api/picks/1/comment", json={"body": "nice"}, headers={"X-User": "other"}); c.post("/api/picks/1/vote", json={"v": 1}, headers={"X-User": "other"})
    assert c.delete("/api/picks/1", headers={"X-User": "other"}).status_code == 404                 # someone else cannot remove it
    assert db.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 1
    assert c.delete("/api/picks/1", headers={"X-User": "mark"}).status_code == 200
    assert [db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("picks", "comments", "votes")] == [0, 0, 0]
    assert c.delete("/api/picks/1", headers={"X-User": "mark"}).status_code == 404                 # already gone


def test_shared_picks_get_a_result_from_the_game(db):
    c = TestClient(app.app); lost = {"legs": [dict(L("pts", 30.5))]}; won = {"legs": [dict(L("pts", 24.5))]}
    app.init()
    db.execute("INSERT INTO picks(user,created,game_id,legs) VALUES('mark','t','',?)", (json.dumps(won["legs"]),)); db.execute("INSERT INTO picks(user,created,game_id,legs) VALUES('mark','t','',?)", (json.dumps(lost["legs"]),))
    db.execute("INSERT INTO picks(user,created,game_id,legs) VALUES('mark','t','',?)", (json.dumps([dict(game_id='future', player_id='p', stat='pts', line=10.5, side='over')]),)); db.commit()
    got = {p["id"]: p for p in c.get("/api/picks").json()}
    assert [got[i]["result"] for i in (1, 2, 3)] == ["won", "lost", "open"] and got[1]["leg_results"] == ["won"]

def test_a_pick_cannot_be_shared_once_its_game_has_started(db):
    db.execute("CREATE TABLE IF NOT EXISTS projections(game_id TEXT, player_id TEXT, payload TEXT, made_at TEXT, PRIMARY KEY(game_id, player_id))")
    db.execute("INSERT INTO projections VALUES('started','p','{\"start\":\"2020-01-01T00:00Z\"}','t')"); db.commit()
    r = TestClient(app.app).post("/api/picks", json={"legs": [dict(game_id="started", player_id="p", stat="pts", line=10.5, side="over")]}, headers={"X-User": "mark"})
    assert r.status_code == 400

def test_team_page_lists_the_current_roster_with_season_averages(db):
    db.executescript("""CREATE TABLE IF NOT EXISTS rosters(player_id TEXT PRIMARY KEY, team TEXT, name TEXT, fetched_at TEXT);
        INSERT INTO rosters VALUES('a','DAL','Aldama','t'); INSERT INTO rosters VALUES('b','DAL','Bench','t'); INSERT INTO rosters VALUES('z','MEM','Other','t');""")
    for col in ("player TEXT", "team TEXT", "season TEXT", "min REAL"): db.execute(f"ALTER TABLE player_games ADD COLUMN {col}")
    for pid, nm, mn, pts in (("a", "Aldama", 30, 14), ("a", "Aldama", 34, 18), ("b", "Bench", 10, 2), ("z", "Other", 35, 25)):
        db.execute("INSERT INTO player_games(game_id,player_id,player,team,season,min,pts,reb,ast) VALUES('g',?,?,'DAL','2026',?,?,5,3)", (pid, nm, mn, pts))
    db.commit()
    out = TestClient(app.app).get("/api/team/dal").json()
    assert [p["player"] for p in out] == ["Aldama", "Bench"] and out[0]["pts"] == 16.0 and out[0]["games"] == 2 and out[0]["min"] == 32.0


def test_steadiness_board_icons_crowns_and_top_three(db):
    import random
    for col in ("player TEXT", "team TEXT", "season TEXT", "min REAL"): db.execute(f"ALTER TABLE player_games ADD COLUMN {col}")
    random.seed(1)
    def add(pid, team, avg, swing, n=14):
        for i in range(n): db.execute("INSERT INTO player_games(game_id,player_id,player,team,season,min,pts,reb,ast) VALUES(?,?,?,?,'2027',30,?,0,0)", (f"g{pid}{i}", pid, pid, team, round(avg + swing * (1 if i % 2 else -1))))
    add("steady_a", "AAA", 30, 3); add("streaky_a", "AAA", 30, 12); add("steady_b", "BBB", 28, 4); add("lonely_c", "CCC", 30, 2); add("tiny_d", "DDD", 8, 1); add("short_e", "EEE", 30, 1, n=5)
    db.commit(); app._steady_cache.update(t=0.0, data=None)
    from contextlib import closing
    with closing(app.con()) as c: b = app.steady_board(c)
    assert not b["fallback"] and b["qualified"] == 4                                          # the 8-point player and the 5-game player are not ranked
    assert [t["player"] for t in b["top3"]] == ["lonely_c", "steady_a", "steady_b"]            # smallest swing first
    assert b["players"]["steady_a"]["crown"] and not b["players"]["streaky_a"]["crown"] and not b["players"]["lonely_c"]["crown"]    # a crown needs 2+ ranked players on the team
