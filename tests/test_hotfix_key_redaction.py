"""Regression test: the Odds API key must never appear in raised errors, stored status text, or API responses."""
import os, sqlite3, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import requests, odds

KEY = "SUPERSECRET-123"

class _Resp:
    status_code = 401; headers = {}
    def raise_for_status(self): raise requests.exceptions.HTTPError(f"401 Client Error: Unauthorized for url: https://api.the-odds-api.com/v4/x?apiKey={KEY}&regions=au", response=self)

def test_get_never_leaks_key():
    os.environ["ODDS_API_KEY"] = KEY; orig = requests.get; requests.get = lambda *a, **k: _Resp()
    try:
        try: odds._get("/sports/basketball_nba/events")
        except Exception as e:
            assert KEY not in str(e) and KEY not in repr(e) and "HTTP 401" in str(e); assert e.__cause__ is None and e.__suppress_context__
        else: raise AssertionError("expected an error")
    finally: requests.get = orig

def test_scrub_and_status_clean_old_leaks():
    assert KEY not in odds.scrub(f"failed for url https://h/v4/e?foo=1&apiKey={KEY}&x=2") and "apiKey=***" in odds.scrub(f"?apiKey={KEY}")
    con = sqlite3.connect(os.path.join(tempfile.mkdtemp(), "t.db")); odds.init(con)
    con.execute("INSERT INTO odds_status VALUES('error', ?)", (f"401 for url https://api/x?apiKey={KEY}",)); con.commit()
    assert KEY not in str(odds.status(con))          # text stored before the fix is also cleaned when read

if __name__ == "__main__":
    test_get_never_leaks_key(); test_scrub_and_status_clean_old_leaks(); print("key redaction tests passed")
