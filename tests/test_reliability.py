"""Health page, refresh stage isolation, backup, fail-closed access, rate limit. Run: python3 tests/test_reliability.py"""
import os, sys, tempfile, shutil, sqlite3
d = tempfile.mkdtemp(); os.environ["DATA_DIR"] = d
shutil.copy("data/nba.db", d + "/nba.db"); shutil.copy("data/models.pkl", d + "/models.pkl")
sys.path.insert(0, ".")
import app
from fastapi.testclient import TestClient
c = TestClient(app.app)
# open server on Fly must refuse
os.environ["FLY_APP_NAME"] = "x"; app.ACCESS_CODE = ""
assert c.get("/api/meta").status_code == 503
del os.environ["FLY_APP_NAME"]
# one failing stage is recorded and does not stop the next
calls = []
app._stage("a", lambda: 1 / 0); app._stage("b", lambda: calls.append(1))
h = c.get("/api/health").json()
assert h["stages"]["a"]["fail_at"] and "ZeroDivisionError" in h["stages"]["a"]["error"] and h["stages"]["b"]["ok_at"] and calls == [1], h
# key text never shown
app._stage("k", lambda: (_ for _ in ()).throw(RuntimeError("https://x/y?apiKey=abcdef123456&z=1")))
assert "abcdef123456" not in str(c.get("/api/health").json())
# backup is a readable copy, not duplicated per day
b = app.backup_db(); assert sqlite3.connect(b).execute("select count(*) from games").fetchone()[0] > 1000 and app.backup_db() == b
# rate limit
app.RATE = (3, 60); app._hits.clear()
codes = [c.get("/api/health").status_code for _ in range(5)]
assert codes[:3] == [200] * 3 and codes[3] == 429, codes
print("reliability tests passed")
