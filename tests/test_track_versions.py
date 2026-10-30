"""The Track record counts every published pick, old and new, as Mark asked."""
import pytest
from fastapi.testclient import TestClient
import app

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DB", str(tmp_path / "t.db")); monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "grade_picks", lambda: None); monkeypatch.setattr(app, "_current_model_id", lambda: "NEW")
    app.init(); return TestClient(app.app)

def _pick(c, model, result, p=0.8, kind="leg", i=[0]):
    i[0] += 1
    c.execute("INSERT INTO pick_log(day,game_id,player_id,player,team,kind,stat,line,side,model_p,created,result,model_id) VALUES('2026-10-28',?,?,?,?,?,?,?,?,?,?,?,?)",
              ("g", f"p{i[0]}", "X", "A", kind, "reb", 5.5, "over", p, "t", result, model))

def test_track_record_counts_every_published_pick(client):
    import sqlite3
    c = sqlite3.connect(app.DB)
    for r in ("won", "won", "lost"): _pick(c, "NEW", r)
    for r in ("lost",) * 10: _pick(c, "OLD", r)
    _pick(c, None, "lost")
    c.commit(); t = client.get("/api/track").json()
    assert t["legs"]["n"] == 14 and t["legs"]["won"] == 2
    assert "earlier_versions" not in t


def test_each_day_carries_the_first_start_time_so_the_page_can_use_the_phone_clock(client):
    import sqlite3, json
    c = sqlite3.connect(app.DB)
    c.execute("CREATE TABLE IF NOT EXISTS projections(game_id TEXT, player_id TEXT, payload TEXT, made_at TEXT, PRIMARY KEY(game_id, player_id))")
    c.execute("INSERT INTO projections VALUES('g','x',?,?)", (json.dumps({"start": "2026-10-30T00:00Z"}), "t"))
    _pick(c, "NEW", None); c.commit()
    day = client.get("/api/track").json()["days"][0]
    assert day["day"] == "2026-10-28" and day["start"] == "2026-10-30T00:00Z"


def test_todays_picks_cover_only_the_next_game_day():
    rows = [dict(start="2026-10-30T00:00Z", id=1), dict(start="2026-10-30T02:00Z", id=2), dict(start="2026-10-30T23:30Z", id=3), dict(start="2026-10-31T02:00Z", id=4)]
    assert [r["id"] for r in app.next_game_day(rows)] == [1, 2]
    assert app.next_game_day([]) == []


def test_coming_days_come_first_soonest_at_the_top(client):
    import sqlite3, json
    c = sqlite3.connect(app.DB)
    c.execute("CREATE TABLE IF NOT EXISTS projections(game_id TEXT, player_id TEXT, payload TEXT, made_at TEXT, PRIMARY KEY(game_id, player_id))")
    for g, start in (("gPast", "2020-01-02T00:00Z"), ("gSoon", "2998-01-01T00:00Z"), ("gLater", "2999-01-01T00:00Z")):
        c.execute("INSERT INTO projections VALUES(?,?,?,?)", (g, "x", json.dumps({"start": start}), "t"))
    for day, g, res in (("2020-01-01", "gPast", "won"), ("2999-01-01", "gLater", None), ("2998-01-01", "gSoon", None)):
        c.execute("INSERT INTO pick_log(day,game_id,player_id,player,team,kind,stat,line,side,model_p,created,result,model_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (day, g, "p" + g, "A", "T", "leg", "reb", 5.5, "over", .8, "t", res, "NEW"))
    c.commit()
    assert [d["day"] for d in client.get("/api/track").json()["days"]] == ["2998-01-01", "2999-01-01", "2020-01-01"]


def _row(i, day, result, price, stat="reb", line=3.5, game="g", pid="p", kind="leg"):
    return dict(id=i, day=day, game_id=game, player_id=pid + str(i), stat=stat, line=line, side="over", result=result, price=price, kind=kind)

def test_hypothetical_bankroll_flat_ten_dollars_on_priced_picks():
    legs = [_row(1, "2026-10-01", "won", 2.0), _row(2, "2026-10-01", "lost", 1.5), _row(3, "2026-10-02", "won", 1.8),
            _row(4, "2026-10-02", "void", 3.0), _row(5, "2026-10-02", None, 2.0), _row(6, "2026-10-02", "won", None)]
    b = app.bankroll(legs)
    assert (b["won"], b["lost"], b["void"], b["waiting"], b["no_price"]) == (2, 1, 1, 1, 1)
    assert round(b["balance"], 2) == round(1000 + 10 * 1.0 - 10 + 10 * 0.8, 2) and round(b["profit"], 2) == 8.0
    assert [round(x["balance"], 2) for x in b["series"]] == [1000.0, 1008.0] and round(b["roi"], 4) == round(8 / 30, 4)

def test_a_pick_published_twice_counts_once_and_prefers_the_priced_copy():
    a = _row(1, "2026-10-01", "won", None, pid="same"); b = dict(_row(2, "2026-10-01", "won", 2.0, kind="value"), player_id=a["player_id"])
    out = app.bankroll([a, b]); assert out["won"] == 1 and out["no_price"] == 0 and round(out["profit"], 2) == 10.0

def test_bankroll_with_nothing_settled_is_flat():
    out = app.bankroll([_row(1, "2026-10-01", None, 2.0)])
    assert out["balance"] == 1000.0 and out["roi"] is None and out["series"] == [] and out["waiting"] == 1
