import re
from fastapi.testclient import TestClient
import app

def test_page_never_reads_the_fields_the_slate_no_longer_sends():
    html = open("static/index.html", encoding="utf-8").read()
    for k in app._SLATE_DROP:
        assert not re.search(r"[.\[\'\"]" + k + r"\b", html), k

def test_lean_rounds_numbers_without_touching_the_original():
    src = dict(p=0.8808422703731392, n=10, xs=[0.123456789, None], nested=dict(v=1.0000004))
    out = app._lean(src)
    assert out == dict(p=0.880842, n=10, xs=[0.123457, None], nested=dict(v=1.0)) and src["p"] == 0.8808422703731392

def test_lean_player_drops_unused_fields_and_keeps_the_original():
    p = dict(player_id="1", ladders={"pts": []}, alphas={}, strip={"pts": [dict(p_over=0.123456789)]}, team="NO")
    out = app._lean_player(p)
    assert "ladders" not in out and "alphas" not in out and out["strip"]["pts"][0]["p_over"] == 0.123457 and "ladders" in p

def test_ping_needs_nothing():
    assert TestClient(app.app).get("/api/ping").json() == {"ok": True}

def test_watchdog_waits_for_five_misses_and_never_restarts_twice_in_an_hour():
    now = 10_000_000.0
    assert not app.watchdog_should_restart(4, 0, now)
    assert app.watchdog_should_restart(5, 0, now)
    assert not app.watchdog_should_restart(9, now - 600, now)           # restarted ten minutes ago: leave it alone
    assert app.watchdog_should_restart(5, now - 3601, now)


def test_new_style_names_do_not_reuse_the_player_row_class():
    css = open("static/index.html", encoding="utf-8").read()
    assert css.count(".pl{") == 1 and ".pl.hit" not in css and ".pl.tg" not in css      # .pl is the Games tab player row; the number pill is .npl


def test_slate_keeps_no_copies_in_memory_between_requests():
    assert not hasattr(app, "_lean_memo")       # a memo of lean copies grew by about 140 KB per player per rebuild and could fill the 1 GB server


def test_the_page_is_always_rechecked_so_a_phone_never_keeps_an_old_version():
    c = TestClient(app.app)
    r = c.get("/"); assert r.status_code == 200 and r.headers["cache-control"] == "no-cache" and r.headers.get("etag")
    assert c.get("/api/ping").headers["cache-control"] == "no-store"


def test_tabs_are_best_games_picks_players_only():
    html = open("static/index.html", encoding="utf-8").read()
    assert 'const TABS=[["best","Best picks"],["today","Games"],["pk","Picks"],["players","Players"]];' in html
    assert "async function liveTab" not in html and "async function lineupsTab" not in html          # Live and Lineups now live inside each game


def test_choosing_a_pick_never_pops_the_pick_list_open():
    html = open("static/index.html", encoding="utf-8").read()
    for name in ("function addG3(", "function addBest(", "function addBestLeg(", "function pickTeam("):
        line = html[html.index(name):].split("\n", 1)[0]
        assert "slipMin=false" not in line, name
