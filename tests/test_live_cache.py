"""Live scores: an old copy is returned at once and refreshed in the background, so a page never waits for ESPN."""
import time, live

def test_old_copy_returned_at_once_and_refreshed_in_background():
    live._cache.clear(); calls = []
    def slow(): calls.append(1); time.sleep(0.3); return len(calls)
    assert live.cached("k", 10, slow) == 1                      # first time: fetched now
    live._cache["k"] = (time.time() - 60, 1)                     # make it old
    t = time.time(); assert live.cached("k", 10, slow) == 1; assert time.time() - t < 0.1     # old copy straight away
    assert live.cached("k", 10, slow) == 1 and len(calls) == 2   # only one refresh running at a time
    time.sleep(0.5); assert live.cached("k", 10, slow) == 2      # fresh copy in place
