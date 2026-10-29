"""One-off/occasional: fill market lines (total, spread, moneylines) for games that don't have them yet."""
import sqlite3, time
from concurrent.futures import ThreadPoolExecutor
import ingest

def run(limit=None):
    con = sqlite3.connect(ingest.DB); ingest.ensure_columns(con)
    todo = [r[0] for r in con.execute("SELECT game_id FROM games WHERE ou IS NULL ORDER BY date DESC")][:limit]
    print(len(todo), "games to fill", flush=True)
    def work(gid):
        try: return gid, ingest.market_lines(ingest.get(ingest.SUM, {"event": gid}))
        except Exception as e: return gid, None
    n = 0
    with ThreadPoolExecutor(6) as ex:
        for gid, ln in ex.map(work, todo):
            if ln and ln.get("ou") is not None:
                con.execute("UPDATE games SET ou=?, spread_home=?, ml_home=?, ml_away=? WHERE game_id=?", (ln["ou"], ln["spread_home"], ln["ml_home"], ln["ml_away"], gid)); n += 1
            if n and n % 100 == 0: con.commit(); print(n, flush=True)
    con.commit(); print("filled", n)

if __name__ == "__main__":
    run()
