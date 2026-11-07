"""Best picks: once a game starts, its saved picks stay on the page, locked (no Add), until about the next morning."""
import json, sqlite3, datetime as dt
import pytest
import app

NOW = dt.datetime(2026, 11, 8, 2, 0, tzinfo=dt.timezone.utc)
iso = lambda h: (NOW + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ")

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "t.db")); app.init()
    starts = {"live": iso(-1), "soon": iso(+2), "old": iso(-30)}
    monkeypatch.setattr(app, "_game_start", lambda c, gid: starts.get(gid))
    c = sqlite3.connect(app.DB)
    rows = [("live", "p1", "leg", "pts", 1.9, "over"), ("live", "p2", "leg", "reb", None, "over"),        # second one was never shown (no price)
            ("soon", "p3", "leg", "ast", 1.8, "over"), ("old", "p4", "leg", "pts", 1.9, "over"),
            ("live", "p5", "call", "pra", 1.87, "under"), ("live", "p6", "value", "reb", 2.1, "over")]
    for g, p, k, st, pr, side in rows:
        c.execute("INSERT INTO pick_log(day,game_id,player_id,player,team,kind,stat,line,side,model_p,price,in_combo,created) VALUES('2026-11-07',?,?,?,?,?,?,?,?,?,?,?,?)",
                  (g, p, p.upper(), "LAL", k, st, 10.5, side, .7, pr, 0, "t"))
    c.execute("INSERT INTO combo_log(day,legs,joint,fair_odds,created) VALUES('2026-11-07',?,?,?,?)",
              (json.dumps([dict(game_id="live", player_id="p1", player="P1", team="LAL", stat="pts", line=10.5, p=.8), dict(game_id="soon", player_id="p3", player="P3", team="LAL", stat="ast", line=4.5, p=.7)]), .5, 2.0, "t"))
    c.commit(); return c

def test_started_picks_stay_locked_and_unstarted_ones_do_not_show(db):
    L = app.locked_picks(NOW)
    assert [x["player_id"] for x in L["legs"]] == ["p1"]                    # live and shown; not the unpriced one, not the one still to start, not 30 hours ago
    assert [x["player_id"] for x in L["calls"]] == ["p5"] and L["calls"][0]["side"] == "under"
    assert [x["player_id"] for x in L["value"]] == ["p6"]
    assert L["combo"] and len(L["combo"]["legs"]) == 2 and L["combo"]["joint"] == .5     # a combo is locked once its first leg tips off

def test_nothing_locked_before_any_game_starts(db):
    L = app.locked_picks(NOW - dt.timedelta(hours=3))
    assert not L["legs"] and not L["calls"] and not L["value"] and L["combo"] is None

def test_locked_picks_leave_after_18_hours(db):
    L = app.locked_picks(NOW + dt.timedelta(hours=18))
    assert [x["player_id"] for x in L["legs"]] == ["p3"] and L["combo"] is not None     # the 19-hour-old game has gone; the later one (16 hours) stays
    L = app.locked_picks(NOW + dt.timedelta(hours=21))
    assert L["combo"] is None
