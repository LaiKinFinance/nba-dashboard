"""Live scores watched by a phone are refreshed in the background, so a phone gets a current copy, not the one from its previous visit."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import live


def setup_function(_):
    live._cache.clear(); live._wanted.clear()


def test_watched_scores_are_refreshed_before_the_next_visit():
    n = {"v": 0}
    def fetch(): n["v"] += 1; return n["v"]
    assert live.cached("sb", 10, fetch) == 1
    live._cache["sb"] = (time.time() - 11, 1)          # copy is now older than 10 s
    live.keep_fresh_once()
    assert live.cached("sb", 10, fetch) == 2           # the next visit gets the new copy straight away


def test_fresh_copies_are_not_fetched_again():
    n = {"v": 0}
    def fetch(): n["v"] += 1; return n["v"]
    live.cached("sb", 10, fetch); live.keep_fresh_once()
    assert n["v"] == 1


def test_nobody_watching_for_two_minutes_stops_the_refreshes():
    n = {"v": 0}
    def fetch(): n["v"] += 1; return n["v"]
    live.cached("box1", 10, fetch)
    live.keep_fresh_once(now=time.time() + live.KEEP_FOR + 1)
    assert n["v"] == 1 and "box1" not in live._wanted


def test_a_failed_refresh_keeps_the_last_good_copy():
    live.cached("sb", 10, lambda: "good")
    live._cache["sb"] = (time.time() - 11, "good")
    live._wanted["sb"] = (time.time(), 10, lambda: 1 / 0)
    live.keep_fresh_once()
    assert live._cache["sb"][1] == "good"
