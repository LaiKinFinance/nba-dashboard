"""Backup, check and safe restore of the app's data.  Everything works on COPIES: nothing is ever overwritten or deleted.

  python3 tools/backup_restore.py backup  <data_dir> <out_dir>            make a bundle: database copy + prediction file + manifest with checksums
  python3 tools/backup_restore.py verify  <bundle_dir>                    re-check the bundle's checksums and that the database opens cleanly
  python3 tools/backup_restore.py restore <bundle_dir> <live_db> <new_db>  write a NEW database = the bundle's database + every newer bet,
                                                                           pick, comment, vote, published prediction and price record found in <live_db>
The live database is only read.  <new_db> must not exist yet."""
import gzip, hashlib, json, os, shutil, sqlite3, sys, datetime as dt
from contextlib import closing

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def make_bundle(data_dir, out_dir, code_commit=None):
    """Consistent copy of nba.db (SQLite online backup, safe while the app runs) + models.pkl + manifest."""
    os.makedirs(out_dir, exist_ok=False)
    db_out = os.path.join(out_dir, "nba.db")
    with closing(sqlite3.connect(os.path.join(data_dir, "nba.db"))) as src, closing(sqlite3.connect(db_out)) as dst: src.backup(dst)
    files = {"nba.db": db_out}
    model = os.path.join(data_dir, "models.pkl")
    if os.path.exists(model): shutil.copy2(model, os.path.join(out_dir, "models.pkl")); files["models.pkl"] = os.path.join(out_dir, "models.pkl")
    manifest = dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(), code_commit=code_commit,
                    files={n: dict(sha256=sha256(p), bytes=os.path.getsize(p)) for n, p in files.items()})
    with closing(sqlite3.connect(db_out)) as c: manifest["tables"] = {t: c.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return manifest

def unpack_download(gz_path, out_dir, expected_sha=None):
    """Turn a downloaded nba-YYYYMMDD.db.gz (from /api/backup) into a bundle folder; checks its checksum if given."""
    os.makedirs(out_dir, exist_ok=False); db = os.path.join(out_dir, "nba.db")
    with gzip.open(gz_path, "rb") as f, open(db, "wb") as o: shutil.copyfileobj(f, o)
    if expected_sha and sha256(db) != expected_sha: raise ValueError("downloaded backup does not match its checksum")
    with closing(sqlite3.connect(db)) as c: assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    json.dump(dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(), files={"nba.db": dict(sha256=sha256(db), bytes=os.path.getsize(db))}), open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)

def verify_bundle(bundle_dir):
    """Returns a list of problems (empty = the bundle is intact)."""
    problems = []; m = json.load(open(os.path.join(bundle_dir, "manifest.json")))
    for name, info in m["files"].items():
        p = os.path.join(bundle_dir, name)
        if not os.path.exists(p): problems.append(f"{name} is missing"); continue
        if sha256(p) != info["sha256"]: problems.append(f"{name} does not match its checksum")
    try:
        with closing(sqlite3.connect(os.path.join(bundle_dir, "nba.db"))) as c:
            r = c.execute("PRAGMA integrity_check").fetchone()[0]
            if r != "ok": problems.append(f"database check says: {r}")
    except Exception as e: problems.append(f"database cannot be opened: {e}")
    return problems

# tables holding things people or the app recorded over time; rows newer than the backup are carried over on restore
def _cols(c, table): return [r[1] for r in c.execute(f'PRAGMA table_info("{table}")')]
def _exists(c, table): return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None

def _copy_new(src, dst, table, key_cols, skip_cols=("id",)):
    """Insert rows from src that dst does not have (matched on key_cols). Returns how many were added."""
    if not _exists(src, table) or not _exists(dst, table): return 0
    cols = [c for c in _cols(src, table) if c in _cols(dst, table) and c not in skip_cols]; added = 0
    have = {tuple(r) for r in dst.execute(f'SELECT {",".join(key_cols)} FROM "{table}"')}
    for row in src.execute(f'SELECT {",".join(cols)} FROM "{table}"').fetchall():
        d = dict(zip(cols, row))
        if tuple(d.get(k) for k in key_cols) in have: continue
        dst.execute(f'INSERT INTO "{table}"({",".join(cols)}) VALUES({",".join("?" * len(cols))})', row); added += 1
    return added

def restore(bundle_dir, live_db, new_db):
    """Write NEW database file = the bundle's database + all newer records from live_db. Neither live_db nor the bundle is changed."""
    if os.path.exists(new_db): raise FileExistsError(f"{new_db} already exists; refusing to overwrite anything")
    bad = verify_bundle(bundle_dir)
    if bad: raise ValueError("backup bundle is not intact: " + "; ".join(bad))
    shutil.copy2(os.path.join(bundle_dir, "nba.db"), new_db); added = {}
    with closing(sqlite3.connect(f"file:{live_db}?mode=ro", uri=True)) as live, closing(sqlite3.connect(new_db)) as dst:
        added["bets"] = _copy_new(live, dst, "bets", ("user", "created", "legs", "stake", "odds"))
        # a bet that already existed in the backup but was marked (won/lost/refunded, by the app or by hand) afterwards keeps that newer mark
        n = 0
        if _exists(live, "bets") and _exists(dst, "bets"):
            have_cols = [c for c in ("result", "manual", "settled_at", "leg_results", "settle_note", "payout_odds") if c in _cols(live, "bets") and c in _cols(dst, "bets")]
            for row in live.execute(f'SELECT user,created,legs,stake,odds,{",".join(have_cols)} FROM bets WHERE result IS NOT NULL AND result<>\'open\'').fetchall():
                sets = ",".join(f"{c}=?" for c in have_cols)
                n += dst.execute(f"UPDATE bets SET {sets} WHERE user=? AND created=? AND legs=? AND stake=? AND odds=? AND result='open'", (*row[5:], *row[:5])).rowcount
        added["bets_marked_since"] = n
        added["picks"] = _copy_new(live, dst, "picks", ("user", "created", "legs"))
        # comments and votes point at a pick by number; numbers can differ between the two databases, so match picks by who/when/what
        if _exists(live, "comments") and _exists(dst, "comments"):
            old_key = {r[0]: (r[1], r[2], r[3]) for r in live.execute("SELECT id,user,created,legs FROM picks")}
            new_id = {(r[1], r[2], r[3]): r[0] for r in dst.execute("SELECT id,user,created,legs FROM picks")}
            have = {(r[0], r[1], r[2]) for r in dst.execute("SELECT pick_id,user,created FROM comments")}; n = 0
            for pid, user, created, body in live.execute("SELECT pick_id,user,created,body FROM comments").fetchall():
                nid = new_id.get(old_key.get(pid))
                if nid is None or (nid, user, created) in have: continue
                dst.execute("INSERT INTO comments(pick_id,user,created,body) VALUES(?,?,?,?)", (nid, user, created, body)); n += 1
            added["comments"] = n
            n = 0
            for pid, user, v in live.execute("SELECT pick_id,user,v FROM votes").fetchall():
                nid = new_id.get(old_key.get(pid))
                if nid is not None: n += dst.execute("INSERT OR IGNORE INTO votes(pick_id,user,v) VALUES(?,?,?)", (nid, user, v)).rowcount
            added["votes"] = n
        added["pick_log"] = _copy_new(live, dst, "pick_log", ("game_id", "player_id", "stat", "kind"))
        added["combo_log"] = _copy_new(live, dst, "combo_log", ("day",))
        added["prediction_log"] = _copy_new(live, dst, "prediction_log", ("game_id", "player_id", "stat", "line"), skip_cols=())
        added["manual_odds"] = _copy_new(live, dst, "manual_odds", ("game_id", "player_id", "stat", "line", "side"), skip_cols=())
        added["odds_archive"] = _copy_new(live, dst, "odds_archive", ("fetched_at", "game_id", "player_id", "stat", "line", "side", "book"), skip_cols=())
        added["lines_archive"] = _copy_new(live, dst, "lines_archive", ("fetched_at", "game_id", "market", "side", "line", "book"), skip_cols=())
        added["unavailable_games"] = _copy_new(live, dst, "unavailable_games", ("game_id",), skip_cols=())
        dst.commit()
        r = dst.execute("PRAGMA integrity_check").fetchone()[0]
    if r != "ok": raise RuntimeError("restored database failed its check: " + r)
    return added

if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "backup": print(json.dumps(make_bundle(a[1], a[2]), indent=1))
    elif len(a) == 2 and a[0] == "verify": p = verify_bundle(a[1]); print("INTACT" if not p else "PROBLEMS: " + "; ".join(p)); sys.exit(1 if p else 0)
    elif len(a) == 4 and a[0] == "restore": print(json.dumps(restore(a[1], a[2], a[3]), indent=1))
    else: print(__doc__); sys.exit(2)
