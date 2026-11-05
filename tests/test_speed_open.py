"""Speed options: live scores and lineups are fetched at server start; photos/logos use smaller, still-sharp copies; the phone opens from its saved copy."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import app, live

HTML = open(os.path.join(os.path.dirname(__file__), "..", "static", "index.html")).read()


def test_startup_prefetch_fills_scores_and_lineups(monkeypatch):
    calls = []
    monkeypatch.setattr(live, "scoreboard", lambda: calls.append("scores") or [])
    monkeypatch.setattr(app.sl, "lineups_full", lambda: calls.append("lineups") or [{"home": {"team": "LAL", "starters": [], "out": []}, "away": {"team": "BOS", "starters": [], "out": []}}])
    monkeypatch.setattr(app.sl, "upcoming", lambda n: [])
    monkeypatch.setattr(app, "_lineups_build", lambda: calls.append("build") or [])
    app._lu_cache.update(t=0, data=None, starts={}); app._calc.pop("lineups", None)
    app._startup_prefetch()
    assert calls == ["scores", "lineups", "build"] and app._lu_cache["data"] and "startup_prefetch_s" in app.SPEED


def test_startup_prefetch_survives_a_failed_source(monkeypatch):
    def boom(): raise RuntimeError("down")
    monkeypatch.setattr(live, "scoreboard", boom)
    monkeypatch.setattr(app.sl, "lineups_full", boom)
    app._lu_cache.update(t=0, data=None, starts={})
    app._startup_prefetch()          # no exception: the app still starts


def test_images_use_sharp_smaller_copies():
    sizes = {k: (int(w), int(h)) for k, w, h in re.findall(r'(\w*|""):\[(\d+),(\d+)\]', re.search(r"const FACE_SZ=\{([^}]*)\}", HTML).group(1))}
    shown = {"s": 22, '""': 30, "l": 48, "xl": 76}
    for k, px in shown.items():
        assert sizes[k][1] >= 3 * px * 0.99, k      # photo height at least 3x the circle shown (sharp on 3x phone screens)
    assert "teamlogos/nba/500/${a.toLowerCase()}.png&w=144&h=144" in HTML      # logos shown at most 48px
    assert 'onerror="imgBack(this)"' in HTML and "data-full=" in HTML           # a failed small copy falls back to the full image
