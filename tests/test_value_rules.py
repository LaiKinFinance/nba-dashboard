"""Value bets after their poor start (won 41% when the app said 54%): lean halfway to the bookmaker, and no long shots over 3.00."""
import inspect, re
import app

def test_lean_is_halfway_to_the_bookmaker():
    assert app.VALUE_LEAN == 0.5 and abs(app.value_gap(0.60, 2.0) - 0.05) < 1e-9     # 10 points ahead becomes 5

def test_a_gap_must_survive_the_lean():
    assert app.value_gap(0.61, 2.0) < app.MIN_EDGE <= app.value_gap(0.63, 2.0)       # needs about 12+ points over the bookmaker now (was 6)

def test_no_long_shots_and_same_rule_on_the_page():
    src = inspect.getsource(app.compute_value)
    assert "VALUE_MAX_PRICE" in src and "value_gap" in src and app.VALUE_MAX_PRICE == 3.0
    html = open("static/index.html").read()
    assert "const isVal=(edge,price)=>!!price&&price<=3&&0.5*edge>=0.06" in html and "Experimental" in html
