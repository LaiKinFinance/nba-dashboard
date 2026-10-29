"""Odds archive, no-erase on empty answers, closing price lookup, CLV in track record, stricter value threshold. Run: python3 tests/test_archive_and_clv.py"""
import os, sys, tempfile, sqlite3, datetime as dt
d = tempfile.mkdtemp(); os.environ["DATA_DIR"] = d; os.environ["ODDS_API_KEY"] = "k"; sys.path.insert(0, ".")
import odds, ingest
con = sqlite3.connect(d + "/nba.db"); ingest.SCHEMA and con.executescript(ingest.SCHEMA); ingest.ensure_columns(con); odds.init(con)
now = dt.datetime.now(dt.timezone.utc); t = lambda h: (now + dt.timedelta(hours=h)).isoformat()
rows = [("g1", "p1", "pts", 24.5, "over", 1.90, "sportsbet", t(-5)), ("g1", "p1", "pts", 24.5, "over", 1.80, "sportsbet", t(-1)), ("g1", "p1", "pts", 24.5, "over", 1.95, "tab", t(-1))]
con.executemany("INSERT INTO odds_archive VALUES(?,?,?,?,?,?,?,?)", [(r[7], r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]); con.commit()
tip = t(0)
c = odds.closing_price(con, "g1", "p1", "pts", 24.5, "over", "sportsbet", tip); assert c[0] == 1.80 and abs(c[2] - 60) < 1, c      # last Sportsbet price before tip
c = odds.closing_price(con, "g1", "p1", "pts", 24.5, "over", None, tip); assert c[0] == 1.95 and c[1] == "tab", c               # no book: best at latest snapshot
assert odds.closing_price(con, "g1", "p1", "pts", 24.5, "over", "sportsbet", t(-6)) is None                                       # nothing before an earlier tip
# empty answer must not erase prices
con.execute("INSERT INTO book_odds VALUES('g1','p1','pts',24.5,'over',1.8,'sportsbet',?)", (t(-1),)); con.commit()
odds._get = lambda path, **kw: ([{"id": "e1", "home_team": "New York Knicks", "away_team": "Boston Celtics"}], 100) if path.endswith("/events") else ({"bookmakers": []}, 99)
odds.refresh([dict(game_id="g1", start=t(5), home="NY", away="BOS")], {"g1": [("p1", "Some Player")]})
assert con.execute("SELECT count(*) FROM book_odds WHERE game_id='g1'").fetchone()[0] == 1
# thresholds
import app
fresh = {"odds_age_h": 1.0}; old = {"odds_age_h": 8.0}; q = {"odds_age_h": 1.0, "injury": {"status": "Questionable"}}
assert app.min_edge(fresh) == app.MIN_EDGE and app.min_edge(old) == app.MIN_EDGE_UNSURE and app.min_edge(q) == app.MIN_EDGE_UNSURE
print("archive/CLV tests passed")
