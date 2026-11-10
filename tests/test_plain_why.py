"""'Why this pick' in plain words: built only from facts the app holds, one or two sentences, never an error."""
import app

P = dict(absent=[dict(name="Joel Embiid", weight=0.9, pts=30), dict(name="Bench Guy", weight=0.4, pts=3)],
         last10=[dict(pts=10, reb=r, ast=2) for r in (8, 9, 12, 7, 10, 11, 6, 9, 13, 8)], proj=dict(pts=12, reb=9.4, ast=3), opp="BOS", opp_factor=dict(reb=1.08))

def test_teammate_out_and_recent_record():
    assert app.plain_why(P, "reb", 7.5) == "Joel Embiid is out tonight, which leaves more rebounds for him. Went over 7.5 rebounds in 8 of his last 10 games."

def test_under_uses_under_wording_and_skips_teammate_boost():
    w = app.plain_why(P, "pra", 30.5, "under")
    assert w.startswith("Stayed under 30.5 points + rebounds + assists in 10 of his last 10 games.") and "Embiid" not in w

def test_missing_facts_never_break_the_page():
    assert app.plain_why({}, "reb", 7.5) == "" and app.plain_why(None, "pts", 20.5) == ""
