"""Odds archive, closing-price lookup and the stricter edge rule."""
import datetime as dt, sqlite3
import odds, ingest

def _db(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "DB", str(tmp_path / "o.db")); monkeypatch.setenv("ODDS_API_KEY", "k")
    con = sqlite3.connect(odds.DB); odds.init(con); return con

def _t(h): return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=h)).isoformat()

def test_closing_price_uses_last_snapshot_before_tip(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    rows = [("g1", "p1", "pts", 24.5, "over", 1.90, "sportsbet", _t(-5)), ("g1", "p1", "pts", 24.5, "over", 1.80, "sportsbet", _t(-1)), ("g1", "p1", "pts", 24.5, "over", 1.95, "tab", _t(-1))]
    con.executemany("INSERT INTO odds_archive VALUES(?,?,?,?,?,?,?,?)", [(r[7], r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]); con.commit()
    c = odds.closing_price(con, "g1", "p1", "pts", 24.5, "over", "sportsbet", _t(0)); assert c[0] == 1.80 and abs(c[2] - 60) < 1
    c = odds.closing_price(con, "g1", "p1", "pts", 24.5, "over", None, _t(0)); assert c[0] == 1.95 and c[1] == "tab"
    assert odds.closing_price(con, "g1", "p1", "pts", 24.5, "over", "sportsbet", _t(-6)) is None

def test_min_edge_is_stricter_when_status_unsure():
    import app
    assert app.min_edge({}) == app.MIN_EDGE
    assert app.min_edge({"injury": {"status": "Questionable"}}) == app.MIN_EDGE_UNSURE
