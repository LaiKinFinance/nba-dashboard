"""Finished games with no box score are recorded openly (never invented, never deleted); unexplained gaps are flagged."""
import sqlite3
import pytest, requests
import ingest, repair

def http_error(code):
    r = requests.Response(); r.status_code = code; return requests.HTTPError(f"{code} Client Error", response=r)

def game(gid, d="2026-05-07"):
    return dict(game_id=gid, date=d, date_utc=d, tip_utc=d + "T23:00Z", season="2026", season_type=3, home_id="1", home="DET", away_id="2", away="CLE", home_pts=107, away_pts=97)

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DB", str(tmp_path / "m.db")); return ingest.DB

def test_404_box_score_is_recorded_not_invented(db, monkeypatch):
    monkeypatch.setattr(ingest, "final_games", lambda s, e: iter([game("A")]))
    monkeypatch.setattr(ingest, "parse_box", lambda g: (_ for _ in ()).throw(http_error(404)))
    out = ingest.run(__import__("datetime").date(2026, 5, 1), __import__("datetime").date(2026, 5, 31))
    con = sqlite3.connect(db)
    assert out["known_unavailable"] == 1 and out["failed"] == 0 and out["problem"] is None
    assert con.execute("SELECT count(*) FROM games").fetchone()[0] == 0 and con.execute("SELECT count(*) FROM player_games").fetchone()[0] == 0     # nothing invented
    assert con.execute("SELECT game_id, home_pts, away_pts, attempts FROM unavailable_games").fetchall() == [("A", 107, 97, 1)]
    ingest.run(__import__("datetime").date(2026, 5, 1), __import__("datetime").date(2026, 5, 31))
    assert con.execute("SELECT attempts FROM unavailable_games").fetchone()[0] == 2                                                              # re-checked each time

def test_other_errors_are_failures_not_known_gaps(db, monkeypatch):
    monkeypatch.setattr(ingest, "final_games", lambda s, e: iter([game("B")]))
    monkeypatch.setattr(ingest, "parse_box", lambda g: (_ for _ in ()).throw(http_error(500)))
    import datetime as dt
    out = ingest.run(dt.date(2026, 5, 1), dt.date(2026, 5, 31)); con = sqlite3.connect(db)
    assert out["failed"] == 1 and out["problem"] and con.execute("SELECT count(*) FROM unavailable_games").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM ingest_issues").fetchone()[0] == 1

def test_game_that_turns_up_later_is_stored_and_the_note_is_closed_not_deleted(db, monkeypatch):
    import datetime as dt
    monkeypatch.setattr(ingest, "final_games", lambda s, e: iter([game("C")]))
    monkeypatch.setattr(ingest, "parse_box", lambda g: (_ for _ in ()).throw(http_error(404))); ingest.run(dt.date(2026, 5, 1), dt.date(2026, 5, 31))
    row = ("C", "p1", "P One", "1", "DET", "CLE", 1, "2026-05-07", "2026", 1, 30.0, 20, 5, 5, 1, 0, 2, 2, 5)
    monkeypatch.setattr(ingest, "parse_box", lambda g: ([row], {}))
    monkeypatch.setattr(ingest, "final_games", lambda s, e: iter([game("C")])); ingest.run(dt.date(2026, 5, 1), dt.date(2026, 5, 31)); con = sqlite3.connect(db)
    assert con.execute("SELECT count(*) FROM games").fetchone()[0] == 1 and con.execute("SELECT count(*) FROM player_games").fetchone()[0] == 1
    assert con.execute("SELECT resolved_at IS NOT NULL FROM unavailable_games WHERE game_id='C'").fetchone()[0] == 1          # kept as history, marked resolved

def test_report_separates_stored_known_gaps_and_unexplained(db, monkeypatch):
    import datetime as dt
    con = sqlite3.connect(db); con.executescript(ingest.SCHEMA); ingest.ensure_columns(con)
    con.execute("INSERT INTO games(game_id,date,date_utc,home,away) VALUES('S','2026-05-01','2026-05-01','A','B')")
    con.execute("INSERT INTO unavailable_games(game_id,date,home,away,home_pts,away_pts,reason,first_seen,last_checked) VALUES('K','2026-05-02','C','D',100,90,'no box score','t','t')"); con.commit()
    monkeypatch.setattr(ingest, "final_games", lambda s, e: iter([game("S"), game("K"), game("U")]))
    rep = repair.missing_games_report(con, dt.date(2026, 5, 1))
    assert rep["stored"] == 1 and [g["game_id"] for g in rep["no_box_score_on_espn"]] == ["K"] and rep["unexplained_missing"] == ["U"]

def test_unexplained_gap_is_a_problem_known_gap_is_not(db, monkeypatch):
    import datetime as dt
    con = sqlite3.connect(db); con.executescript(ingest.SCHEMA); ingest.ensure_columns(con); repair.ensure_schema(con)
    con.execute("INSERT INTO games(game_id,date,date_utc,tip_utc,home,away) VALUES('S','2026-05-01','2026-05-01','2026-05-01T23:00Z','A','B')")
    con.execute("INSERT INTO unavailable_games(game_id,date,home,away,home_pts,away_pts,reason,first_seen,last_checked) VALUES('K','2026-05-02','C','D',100,90,'no box score','t','t')"); con.commit(); con.close()
    monkeypatch.setattr(repair, "backfill_tip_times", lambda c: (0, 0)); monkeypatch.setattr(repair, "apply_et_dates", lambda c: 0); monkeypatch.setattr(repair, "refresh_rosters", lambda c: 500)
    monkeypatch.setattr(ingest, "run", lambda s, e: {}); 
    monkeypatch.setattr(ingest, "final_games", lambda s, e: iter([game("S"), game("K")]))
    assert repair.run_all(full_reconcile=True)["problem"] is None
    monkeypatch.setattr(ingest, "final_games", lambda s, e: iter([game("S"), game("K"), game("U")]))
    out = repair.run_all(full_reconcile=True); assert out["missing_after"] == 1 and "missing and unexplained" in out["problem"]
