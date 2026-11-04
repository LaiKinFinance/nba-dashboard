"""Prices are refreshed for each game about 2 hours and about 30 minutes before tip-off (once each), on top of 8pm/2am/8am, with a credit brake."""
import datetime as dt
import app

NOW = dt.datetime(2026, 11, 5, 0, 0, tzinfo=dt.timezone.utc)
at = lambda mins: (NOW + dt.timedelta(minutes=mins)).strftime("%Y-%m-%dT%H:%MZ")
GAMES = [dict(game_id="g2h", start=at(110)), dict(game_id="g30", start=at(28)), dict(game_id="later", start=at(300)), dict(game_id="soon", start=at(5))]

def test_each_game_gets_its_two_pregame_refreshes_once():
    assert sorted(app.pregame_due(GAMES, NOW, set(), "19000")) == [("2h", "g2h"), ("30m", "g30")]
    assert app.pregame_due(GAMES, NOW, {"pre 2h g2h", "pre 30m g30"}, "19000") == []          # already done: not again

def test_brake_keeps_only_the_three_main_fetches_when_credits_run_low():
    assert app.pregame_due(GAMES, NOW, set(), "900") == []
