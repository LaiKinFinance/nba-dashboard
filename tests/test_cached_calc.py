"""Best picks and Value bets are worked out once per data change and shared, not recomputed by every tab and pick log."""
import app

def test_same_data_is_worked_out_once(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_calc", {}); monkeypatch.setattr(app, "_enriched_key", lambda: "k1")
    f = lambda: calls.append(1) or {"x": 1}
    assert app.cached_calc("best", f) == {"x": 1} and app.cached_calc("best", f) == {"x": 1}
    assert len(calls) == 1
    monkeypatch.setattr(app, "_enriched_key", lambda: "k2")          # new data or a new minute: worked out again
    app.cached_calc("best", f); assert len(calls) == 2
