"""Top player calls: each team's top player, over/under the bookmaker's main pts+reb+ast line, only the 3 clearest (60%+), same safety rules as Best picks."""
import datetime as dt
import app

FUT = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%MZ")
def P(pid, team, opp, pra, p_over, injury=None, **k):
    b = [dict(line=30.5, price=1.9, under=1.9, p_over=p_over, book="sportsbet"), dict(line=35.5, price=3.0, under=1.35, p_over=0.2, book="sportsbet")]
    return dict(player_id=pid, player=pid, game_id="g-" + "".join(sorted([team, opp])), team=team, opp=opp, home=1, start=FUT, proj=dict(pts=pra - 10, reb=6, ast=4),
                injury=injury, book=dict(pra=b), odds_age_h=1.0, last10=[], absent=[], **k)

def test_top_three_clearest_calls_only(monkeypatch):
    preds = [P("a", "AAA", "BBB", 30, 0.30), P("a2", "AAA", "BBB", 20, 0.90),        # a2 is not AAA's top player
             P("b", "BBB", "AAA", 31, 0.72), P("c", "CCC", "DDD", 30, 0.55),           # c: coin flip, left out
             P("d", "DDD", "CCC", 30, 0.20, injury=dict(status="Day-To-Day")),         # may not play: left out
             P("e", "EEE", "FFF", 30, 0.35), P("f", "FFF", "EEE", 30, 0.38, new_team=dict(games=2, prev="X")),
             P("g", "GGG", "HHH", 30, 0.66)]
    monkeypatch.setattr(app, "enriched", lambda: preds); monkeypatch.setattr(app, "next_game_day", lambda p: p)
    monkeypatch.setattr(app, "model_status", lambda p=None: (None, None)); monkeypatch.setattr(app, "game_preds", lambda: {})
    monkeypatch.setattr(app, "_team_history", lambda c, t, o: None)
    calls = app.compute_calls()["calls"]
    assert [(x["player_id"], x["side"]) for x in calls] == [("b", "over"), ("a", "under"), ("g", "over")]
    assert calls[1]["line"] == 30.5 and round(calls[1]["p"], 2) == 0.70 and calls[1]["price"] == 1.9     # the main (evenly priced) line, under price
