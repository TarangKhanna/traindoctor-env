"""The job and its scheduler: completes and checkpoints, config parser, lock liveness."""
import fcntl
import json
import os
import random

import pytest
from hypothesis import given, strategies as st

from jobs import manifest
from jobs.build import build, materialize
from jobs.config import dumps, loads
from jobs.resubmit import clean_stale_state, resubmit


def pristine(tmp_path, world=3):
    spec = build("B_OK", 0, world, 5, random.Random(1))
    materialize(tmp_path, spec.config)
    return tmp_path


@pytest.mark.executed
@pytest.mark.parametrize("world", [3, 4])
def test_pristine_job_completes_and_checkpoints(tmp_path, world):
    jd = pristine(tmp_path, world)
    r = resubmit(jd)
    assert r.ok and r.job.last_step == 11 and len({x.digest for x in r.job.ranks}) == 1
    entries = manifest.load(jd)
    assert [e["step"] for e in entries] == [4, 8, 12]
    manifest.validate(jd, entries)


@given(st.dictionaries(st.from_regex(r"[a-z][a-z_]{0,6}", fullmatch=True),
                       st.one_of(st.integers(-5, 99), st.floats(0.01, 9, allow_nan=False).map(lambda x: round(x, 3)),
                                 st.booleans(), st.from_regex(r"part-[0-9]{1,2}", fullmatch=True),
                                 st.dictionaries(st.from_regex(r"[0-9]", fullmatch=True), st.integers(0, 9), max_size=3)),
                       max_size=6))
def test_config_roundtrip(d):
    assert loads(dumps(d)) == d


def test_stale_lock_is_decided_by_who_holds_it_not_by_pid(tmp_path):
    """Rule: a lock is live only while its owner holds it. A copied lock whose recorded pid now belongs to a
    live process is stale; a held lock is live whatever pid it records (spec M1_v2 6.4)."""
    (tmp_path / ".lock").write_text(json.dumps(dict(pid=os.getpid(), run_id="x")))
    assert clean_stale_state(tmp_path) and not (tmp_path / ".lock").exists()
    (tmp_path / ".lock").write_text(json.dumps(dict(pid=0, run_id="x")))
    owner = os.open(tmp_path / ".lock", os.O_RDWR)
    fcntl.flock(owner, fcntl.LOCK_EX)
    try:
        assert not clean_stale_state(tmp_path) and (tmp_path / ".lock").exists()
        assert resubmit(tmp_path).status == "refused"
    finally:
        os.close(owner)
