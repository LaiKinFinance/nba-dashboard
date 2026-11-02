"""The Teams tab is worked out again only when new game results or roster moves are saved."""
from fastapi.testclient import TestClient
import app

def test_team_page_reused_until_new_results(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); app._hits.clear(); app._team_cache.clear()
    calls = []; stamp = ["s1"]
    monkeypatch.setattr(app, "results_stamp", lambda c: stamp[0])
    monkeypatch.setattr(app, "_team_rows", lambda c, abbr: calls.append(abbr) or [{"player": "x"}])
    cl = TestClient(app.app)
    assert cl.get("/api/team/BOS").json() == [{"player": "x"}] and cl.get("/api/team/BOS").json() == [{"player": "x"}]
    assert calls == ["BOS"]                                  # second visit: nothing worked out again
    stamp[0] = "s2"; cl.get("/api/team/BOS"); assert calls == ["BOS", "BOS"]     # new results: rebuilt once
