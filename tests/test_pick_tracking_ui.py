"""Exercise the real page's saved-pick flow without money inputs."""
import shutil
import subprocess
import pytest


def test_pick_tracking_page():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js needed for page checks')
    result = subprocess.run([node, '--test', 'tests/ui_pick_tracking.test.cjs'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
