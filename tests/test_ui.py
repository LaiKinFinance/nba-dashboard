"""Run the browser-independent JavaScript checks when Node.js is installed."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js is needed for the JavaScript display checks')
def test_live_progress_javascript():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([shutil.which('node'), '--test', 'tests/ui_progress.test.cjs'], cwd=root,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
