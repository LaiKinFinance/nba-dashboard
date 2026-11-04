"""Codex audit (main 7532940): the five ranked findings, each tested with the example that failed."""
import datetime as dt, json, sqlite3
from contextlib import closing
import numpy as np
from fastapi.testclient import TestClient
import app, odds, parlay

def test_1_prices_never_attach_to_the_other_game_of_a_home_and_home():
    ab = {"New York Knicks": "NY", "Boston Celtics": "BOS"}.get
    today = dict(id="today", home_team="New York Knicks", away_team="Boston Celtics", commence_time="2026-11-05T00:00:00Z")
    tomorrow = dict(id="tomorrow", home_team="New York Knicks", away_team="Boston Celtics", commence_time="2026-11-06T00:30:00Z")
    assert odds.match_event([tomorrow, today], "NY", "BOS", "2026-11-05T00:00Z", ab)["id"] == "today"
    assert odds.match_event([tomorrow], "NY", "BOS", "2026-11-05T00:00Z", ab) is None          # only tomorrow's prices exist: today's game gets none

def test_2_a_combined_chance_never_beats_its_least_likely_part():
    pred = dict(player_id="p", game_id="g", team="A", home=1, proj={"pts": 20.0, "reb": 8.0, "ast": 5.0, "fg3m": 2.0, "stl": 1.0, "blk": 1.0}, p_dd=.3, p_td=.05)
    pred2 = dict(pred, player_id="q")
    corr = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}; al = {k: 0.1 for k in parlay.ALL}
    legs = [dict(game_id="g", player_id="p", stat="pra", line=30.5, side="over"), dict(game_id="g", player_id="q", stat="reb", line=6.5, side="over")]
    r = parlay.parlay(legs, {("g", "p"): pred, ("g", "q"): pred2}, al, corr, fixed={0: 0.25})
    assert abs(r["leg_probs"][0] - 0.25) < 1e-9 and r["joint"] <= 0.25 + 1e-9 and r["joint"] <= min(r["leg_probs"])

def test_3_a_combo_leg_is_graded_at_its_own_line(monkeypatch):
    monkeypatch.setattr(app, "record_closing_prices", lambda c: None)      # not part of this check (needs a column the bundled test data lacks)
    with closing(app.con()) as c:
        c.execute("INSERT OR REPLACE INTO games(game_id, date, season, home, away, home_pts, away_pts) VALUES('aud1','2026-11-01','2027','NY','BOS',100,90)")
        c.execute("INSERT OR REPLACE INTO player_games(game_id, player_id, player, team, opp, date, season, min, pts, reb, ast, fg3m, stl, blk) VALUES('aud1','pp','P','NY','BOS','2026-11-01','2027',30,10,8,2,0,0,0)")
        c.execute("DELETE FROM pick_log WHERE game_id='aud1'"); c.execute("DELETE FROM combo_log WHERE day='2026-11-01-aud'")
        c.execute("INSERT INTO pick_log(day,game_id,player_id,player,team,kind,stat,line,side,model_p) VALUES('2026-11-01','aud1','pp','P','NY','leg','reb',7.5,'over',.7)")   # 8 > 7.5: won
        c.execute("INSERT INTO combo_log(day,legs,joint) VALUES('2026-11-01-aud',?,.5)", (json.dumps([dict(game_id="aud1", player_id="pp", stat="reb", line=8.5)]),)); c.commit()
    app.grade_picks()
    with closing(app.con()) as c:
        assert c.execute("SELECT result FROM combo_log WHERE day='2026-11-01-aud'").fetchone()[0] == "lost"     # 8 rebounds is NOT over 8.5
        assert c.execute("SELECT result FROM pick_log WHERE game_id='aud1'").fetchone()[0] == "won"

def test_4_started_games_and_old_copies_are_never_served(monkeypatch):
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%MZ")
    fut = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    res = app._not_started_best(dict(legs=[dict(start=past), dict(start=fut)], combo=dict(legs=[dict(start=past)], joint=.5, fair_odds=2)))
    assert len(res["legs"]) == 1 and res["combo"]["legs"] == []
    monkeypatch.setitem(app._warm, "on", True)
    assert app._serve_last(app.time.time() - 10) and not app._serve_last(app.time.time() - 600)   # a copy older than 3 minutes is worked out again

def test_5_bad_timing_data_cannot_break_the_health_page(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); app._hits.clear(); app.PHONE_TIMES.clear()
    c = TestClient(app.app)
    c.post("/api/timing", json={"t": [{"p": "slate", "ms": 300}, {"p": "<script>", "ms": 5}, {"p": "best", "ms": "NaN"}, {"p": "best", "ms": 1e12}, {"p": "x" * 500, "ms": 1}]})
    c.post("/api/timing", json={"t": "not a list"})
    assert [p for p, _, _ in app.PHONE_TIMES] == ["slate"]
    assert c.get("/api/health").status_code == 200

def test_2b_over_and_under_the_same_line_can_never_both_win():
    pred = dict(player_id="p", game_id="g", team="A", home=1, proj={"pts": 15.0, "reb": 7.0, "ast": 4.0, "fg3m": 1.0, "stl": 1.0, "blk": 1.0}, p_dd=.2, p_td=.02)
    corr = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}; al = {k: 0.1 for k in parlay.ALL}
    o = dict(game_id="g", player_id="p", stat="pra", line=26.5, side="over"); u = dict(o, side="under")
    for fixed in ({0: 0.55}, {}, {0: 0.3}):                         # pinned (consistent player) or not
        r = parlay.parlay([o, u], {("g", "p"): pred}, al, corr, fixed=fixed)
        assert r["joint"] == 0.0 and abs(sum(r["leg_probs"]) - 1) < 0.01
    o2 = dict(o, line=24.5); u2 = dict(o, line=28.5, side="under")     # mixed lines stay consistent too
    r = parlay.parlay([o, o2, u2], {("g", "p"): pred}, al, corr, fixed={0: 0.55})
    assert r["joint"] <= min(r["leg_probs"]) + 1e-9

def test_4b_expired_prices_and_a_new_model_are_never_served_from_a_saved_copy(monkeypatch):
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=8, minutes=5)).isoformat()
    ok = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=7, minutes=55)).isoformat()
    fut = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    res = app._not_started_best(dict(legs=[dict(start=fut, price_time=old), dict(start=fut, price_time=ok)], combo=dict(legs=[dict(start=fut, price_time=old)])))
    assert len(res["legs"]) == 1 and res["combo"]["legs"] == []          # the price crossed 8 hours since the copy was made
    assert app._not_started_value(dict(picks=[dict(start=fut, price_time=old)]))["picks"] == []
    monkeypatch.setitem(app._warm, "on", True); monkeypatch.setattr(app, "_calc", {})
    keys = iter([(("db", "wal", "model-A", 1), None), (("db2", "wal", "model-B", 1), None)])
    monkeypatch.setattr(app, "_enriched_key", lambda: next(keys)[0])
    calls = []
    app.cached_calc("best", lambda: calls.append(1) or "A")              # made with model A
    assert app.cached_calc("best", lambda: calls.append(1) or "B") == "B" and len(calls) == 2     # 90 s later the model changed: worked out again, not served
