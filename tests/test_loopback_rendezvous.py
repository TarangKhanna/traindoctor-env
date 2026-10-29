"""The actual rendezvous listener and Gloo peers must bind only loopback."""
import os
import subprocess
import sys
import time

import pytest

from jobs.build import base_config, materialize
from jobs.launcher import run_job


@pytest.mark.executed
def test_live_job_and_rendezvous_listeners_are_loopback(tmp_path, needs_sandbox):
    if sys.platform != "darwin":
        pytest.skip("this live lsof check is for the supported Mac backend")
    observed = []

    def listeners(pid, port=None):
        query = "-iTCP" + (f":{port}" if port else "")
        result = subprocess.run(["lsof", "-a", "-p", str(pid), query, "-sTCP:LISTEN", "-P", "-n"],
                                capture_output=True, text=True, timeout=5)
        return [line.split()[-2] for line in result.stdout.splitlines()[1:]]

    def inspect(pid, port):
        parent = listeners(os.getpid(), port)
        child = []
        for _ in range(20):
            child = listeners(pid)
            if child:
                break
            time.sleep(.2)
        observed.extend(parent + child)
        assert f"127.0.0.1:{port}" in parent and child
        assert all(a.startswith("127.0.0.1:") or a.startswith("[::ffff:127.0.0.1]:")
                   or a.startswith("[::1]:") for a in observed), observed
        print("Observed loopback listeners:", observed)

    jd = tmp_path / "job"
    materialize(jd, base_config(2, 777, 5, 5, 0))
    result = run_job(jd, on_spawn=inspect, sandbox_jobs=True, salt=b"loopback-regression")
    assert result.status == "completed" and observed
