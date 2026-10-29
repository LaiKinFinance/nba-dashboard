"""One private copy of the bundled data for the whole test run, so no test touches the real files and tests cannot
overwrite each other's DATA_DIR. Modules read DATA_DIR when first imported, so it is set here, before any test file loads."""
import os, shutil, tempfile
_root = os.path.join(os.path.dirname(__file__), "..")
_tmp = tempfile.mkdtemp(prefix="nba-tests-")
for _f in ("nba.db", "models.pkl"):
    if os.path.exists(os.path.join(_root, "data", _f)): shutil.copy(os.path.join(_root, "data", _f), _tmp)
os.environ["DATA_DIR"] = _tmp
os.environ["AUTO_REFRESH"] = "0"                # no background refresh may start while tests import the app
os.environ.pop("ODDS_API_KEY", None); os.environ.pop("FLY_APP_NAME", None); os.environ.pop("APP_ACCESS_CODE", None)
