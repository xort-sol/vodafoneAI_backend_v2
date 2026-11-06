import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo.sh"
# Stands in for uv: the dialler exits at once with an error, the API runs until it is stopped
FAKE_UV = """#!/bin/bash
case "$*" in
  *uvicorn*) echo $$ > "$API_PID"; exec sleep 60 ;;
  *) sleep 1; exit 3 ;;
esac
"""


# /bin/bash on macOS is 3.2, the oldest shell the script must run on
def test_when_one_process_dies_the_other_stops_and_the_script_fails(tmp_path):
    uv = tmp_path / "uv"
    uv.write_text(FAKE_UV)
    uv.chmod(0o755)
    api_pid = tmp_path / "api.pid"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "DIAL_ALLOWLIST": "+353870000001", "API_PID": str(api_pid)}
    # its own process group: the script stops everything in its group when it exits
    done = subprocess.run(["/bin/bash", SCRIPT], env=env, start_new_session=True, capture_output=True, text=True, timeout=20)
    assert done.returncode == 1, done.stderr
    with pytest.raises(ProcessLookupError):
        os.kill(int(api_pid.read_text()), 0)


def test_script_refuses_without_an_allowlist():
    done = subprocess.run(["/bin/bash", SCRIPT], env={**os.environ, "DIAL_ALLOWLIST": " * "}, capture_output=True, text=True)
    assert done.returncode == 1 and "DIAL_ALLOWLIST" in done.stderr
