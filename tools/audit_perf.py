"""Offline endpoint timings on a temporary copy; no changes to bundled data."""
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as target:
        for name in ['nba.db', 'models.pkl']:
            shutil.copy2(Path('data') / name, Path(target) / name)
        os.environ['DATA_DIR'] = target
        os.environ['AUTO_REFRESH'] = '0'
        app = importlib.import_module('app')
        # Ignore age cutoff to measure the complete bundled 60-player workload.
        app.projections = app.projections_all
        measurements = {}
        for name, fn in [('enriched', app.enriched), ('slate', lambda: app.get_slate(x_code=app.ACCESS_CODE)),
                         ('best', lambda: app.best(x_code=app.ACCESS_CODE)), ('value', lambda: app.value_picks(x_code=app.ACCESS_CODE))]:
            times = []
            for _ in range(3):
                start = time.perf_counter(); result = fn(); times.append(round(time.perf_counter()-start, 3))
            measurements[name] = dict(seconds=times, payload_bytes=len(json.dumps(result).encode()))
        print(json.dumps(measurements, indent=2))
