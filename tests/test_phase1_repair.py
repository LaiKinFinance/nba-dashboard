"""Phase 1 checks on a copy of the repaired data: Eastern dates, back-to-back placeholders, current rosters. Needs DATA_DIR with repaired nba.db + models.pkl."""
import os, sys, sqlite3
sys.path.insert(0, ".")
import pandas as pd
D = os.environ["DATA_DIR"]; con = sqlite3.connect(D + "/nba.db")
# 1 dates: no team plays two games on one Eastern date; old UTC value preserved
dup = con.execute("select count(*) from (select team,date from player_games group by team,date having count(distinct game_id)>1)").fetchone()[0]
assert dup == 0, dup
assert con.execute("select count(*) from games where date_utc is null or tip_utc is null").fetchone()[0] == 0
assert con.execute("select count(*) from games where date<>date_utc").fetchone()[0] > 500   # evening games moved back a day
# 2 rosters
assert con.execute("select count(*) from rosters where team is not null").fetchone()[0] >= 400
# 3 back-to-back placeholders do not contaminate each other
import slate as sl, predict
t1, t2 = "2026-10-30", "2026-10-31"
fake = [dict(game_id="x1", start=t1 + "T23:00Z", status="STATUS_SCHEDULED", date=t1, ou=220., spread_home=-3., ml_home=-150., ml_away=130., home="NY", away="BOS", home_record=None, away_record=None),
        dict(game_id="x2", start=t2 + "T23:00Z", status="STATUS_SCHEDULED", date=t2, ou=220., spread_home=-3., ml_home=-150., ml_away=130., home="MIA", away="NY", home_record=None, away_record=None)]
sl.upcoming = lambda days=2: fake; sl.injuries = lambda: {}; sl.lineups = lambda: {}
preds = predict.project(days=2)
ny = {}
for p in preds:
    if p["team"] == "NY": ny.setdefault(p["player_id"], {})[p["game_id"]] = p
two = [v for v in ny.values() if len(v) == 2]
assert two, "no NY player projected in both games"
for v in two:
    a, b = v["x1"], v["x2"]
    assert abs(a["min"] - b["min"]) < 1e-9, (a["player"], a["min"], b["min"])      # same real history, so same recent minutes
    assert a["rest"] >= 1 and b["rest"] == 1, (a["rest"], b["rest"])               # second game is a back-to-back
print("phase 1 tests passed")
