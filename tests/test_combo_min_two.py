"""The Daily combo needs 2+ picks: a one-pick day shows that pick as the day's best pick and saves no combo."""
import app

def _setup(monkeypatch, cands):
    monkeypatch.setattr(app, "enriched", lambda: [dict(start="2099-01-01T00:00Z", game_id=c["game_id"], player_id=c["player_id"]) for c in cands])
    monkeypatch.setattr(app, "model_status", lambda preds: (None, None))
    monkeypatch.setattr(app, "_best_candidates", lambda preds: [dict(c) for c in cands])
    monkeypatch.setattr(app, "require_current_model", lambda preds: dict(alphas={}, corr={}))
    monkeypatch.setattr(app.pl, "parlay", lambda legs, *a, **k: dict(joint=0.6, fair_odds=1.7 if len(legs) < 3 else 2.4))
    monkeypatch.setattr(app, "playing_picks", lambda: [])
    monkeypatch.setattr(app, "plain_why", lambda *a, **k: "")

C = lambda g, p: dict(game_id=g, player_id=p, stat="ast", line=1.5, real=True, score=0.8, start="2099-01-01T00:00Z")

def test_one_pick_is_not_a_combo(monkeypatch):
    _setup(monkeypatch, [C("g1", "p1")])
    r = app.compute_best()
    assert r["combo"]["legs"] == [] and r["combo"]["joint"] is None and len(r["legs"]) == 1

def test_two_or_more_picks_make_a_combo(monkeypatch):
    _setup(monkeypatch, [C("g1", "p1"), C("g2", "p2"), C("g3", "p3")])
    r = app.compute_best()
    assert len(r["combo"]["legs"]) == 3 and r["combo"]["joint"] == 0.6
