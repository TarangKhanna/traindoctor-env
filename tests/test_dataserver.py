"""The trusted data server and the process boundary around job code: protocol abuse, salt secrecy, environment
allowlist. Every probe below is either REJECTED by the server or HARMLESS to grading; the docstrings say which and why.
Unit tests need no sockets; the live probes run a real job whose train.py is the abuser (no Gloo)."""
import json
import os
import random
import textwrap

import pytest

from jobs import dataserver, trusted
from jobs.build import build, materialize
from jobs.dataserver import MAX_LINE, MAX_SERVED, DataServer, DataSpec, batch_tensors, salt_for
from jobs.launcher import ENV_PASSTHROUGH, job_env
from jobs.resubmit import resubmit

SPEC = DataSpec(world=2, num_samples=96, batch_size=4, seed=7)
SALT = salt_for("unit")


def req(**kw):
    return json.dumps(dict(op="batch", epoch=0, index=0) | kw).encode()


def test_replayed_and_out_of_order_requests_are_served_and_logged_so_they_stay_visible():
    """HARMLESS: a position is job state (restore replays it, B_SKIP/B_DUP are exactly wrong positions), so the
    server serves any in-range position. Grading never takes the log at face value: committed checkpoint states
    and the logical sample accounting (which compares the log against the reference schedule) expose a job that
    trains on the wrong positions; the log keeps every replay, so duplicates stay countable."""
    s = DataServer(SPEC, SALT)
    for e, i in [(0, 3), (0, 3), (0, 1), (1, 0)]:
        kind, msg = s.parse(0, req(epoch=e, index=i))
        assert kind == "batch" and "ids" in s.serve(0, msg)
    assert [(e, i) for e, i, _, _ in s.served[0]] == [(0, 3), (0, 3), (0, 1), (1, 0)] and not s.served[1]


@pytest.mark.parametrize("extra", [dict(rank=1), dict(step=3), dict(rank=0, world=9), dict(salt="x")])
def test_forged_claims_about_rank_or_step_are_rejected(extra):
    """REJECTED: a request is exactly {op, epoch, index}. Who is asking is the socket the line arrived on, never
    the text, so a job cannot speak for another rank, and cannot claim a step."""
    kind, msg = DataServer(SPEC, SALT).parse(0, req(**extra))
    assert kind == "error" and "unexpected" in msg["error"]


def test_the_asking_rank_is_the_socket_not_the_message():
    """HARMLESS: whatever rank 0's process sends is served and logged as rank 0's batch of rank 0's shard."""
    s = DataServer(SPEC, SALT)
    _, msg = s.parse(0, req())
    ids = s.serve(0, msg)["ids"]
    assert all(i % 2 == 0 for i in ids) and s.served[0] and not s.served[1]


@pytest.mark.parametrize("raw, kind", [
    (b"not json", "error"), (b"[1, 2]", "error"), (b"null", "error"), (b'{"op": "nope"}', "error"),
    (req(epoch=True), "error"), (req(epoch=1.5), "error"), (req(epoch="0"), "error"), (req(epoch=-1), "error"),
    (req(index=-1), "error"), (req(index=SPEC.steps_per_epoch), "error"), (req(epoch=10**9), "error"),
    (b"x" * (MAX_LINE + 1), "drop"),
])
def test_malformed_out_of_range_and_oversized_requests_are_rejected(raw, kind):
    """REJECTED: bad JSON, wrong types (bool is not an int), positions outside the dataset and epoch bound get an
    error and are not served or logged; an oversized line cuts the connection (the launcher kills that rank)."""
    s = DataServer(SPEC, SALT)
    assert s.parse(0, raw)[0] == kind and not s.served[0]


def test_requests_after_the_run_is_closed_are_refused():
    """REJECTED: the launcher closes the server when the run ends (and closes the sockets); a straggler process
    that outlives the run gets an error / EOF and nothing is logged."""
    s = DataServer(SPEC, SALT)
    _, msg = s.parse(0, req())
    s.close()
    assert s.serve(0, msg) == dict(error="run closed") and not s.served[0]


def test_flooding_is_capped():
    """REJECTED: a rank gets at most MAX_SERVED batches per run, so a job cannot grow the launcher's memory/log."""
    s = DataServer(SPEC, SALT)
    _, msg = s.parse(0, req())
    s.served[0] = [(0, 0, [], 0)] * MAX_SERVED
    assert s.serve(0, msg) == dict(error="too many requests")


def test_step_acks_are_evidence_only_and_out_of_order_acks_are_ignored():
    """HARMLESS: acks come from inside the job's process, so no verdict reads them (they only fill the
    agent-readable run records). Replayed or skipped counters are ignored; a job that forges them gains nothing."""
    s = DataServer(SPEC, SALT)
    for n in (1, 1, 5, 2):
        s.note_step(0, dict(op="step", n=n, digest="d"))
    assert s.acked[0] == 2
    s.note_step(0, dict(op="step", n=3, digest="x" * 100))
    assert s.acked[0] == 3 and s.digests[0] is None


def test_the_salt_decides_the_samples_but_not_the_ids():
    """The data a job trains on cannot be produced without the salt; ids (positions) are public by design."""
    ids = list(range(8))
    (xa, _), (xb, _) = batch_tensors(salt_for("a"), 7, ids), batch_tensors(salt_for("b"), 7, ids)
    assert not (xa == xb).any() and (batch_tensors(salt_for("a"), 7, ids)[0] == xa).all()
    assert dataserver.schedule_ids(SPEC, 0, 0, 0) == dataserver.schedule_ids(SPEC, 0, 0, 0)


def test_sampler_schedule_semantics():
    """Documented in evidence/archive/specs/M1_NOTES.md: rank r owns ids with id % world == r, shuffled per epoch; position
    (epoch, index) is all a checkpoint needs to resume the same stream; an epoch covers the dataset once."""
    spec = DataSpec(3, 96, 4, 7)
    epoch = {r: [i for idx in range(spec.steps_per_epoch) for i in dataserver.schedule_ids(spec, r, 0, idx)] for r in range(3)}
    assert all(all(i % 3 == r for i in ids) and len(set(ids)) == len(ids) == 96 // 3 for r, ids in epoch.items())
    assert sorted(i for ids in epoch.values() for i in ids) == list(range(96))
    assert dataserver.schedule_ids(DataSpec(3, 96, 4, 8), 0, 0, 0) != epoch[0][:4]
    assert dataserver.schedule_ids(spec, 0, 1, 0) != epoch[0][:4]
    from jobs.template import data
    a, b = data.Sampler(96, 3, 1, 4, 7), data.Sampler(96, 3, 1, 4, 7)
    a.epoch, a.index = 0, 5
    b.load_state(a.state())
    assert (b.epoch, b.index) == (0, 5)


def test_job_environment_is_an_explicit_allowlist(monkeypatch):
    """No ambient variable (tokens, keys, ssh agent, PYTHONPATH tricks) reaches a job process."""
    for k in ("TD_KEY", "AWS_SECRET_ACCESS_KEY", "SSH_AUTH_SOCK", "PYTHONPATH", "HF_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(k, "leak")
    env = job_env(1, 4, 12345, "run3", "ck0004", 4, 9)
    assert set(env) <= set(ENV_PASSTHROUGH) | {"MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "RUN_ID", "RESUME_FROM",
                                                "HARNESS_START", "TD_FD", "GLOO_SOCKET_IFNAME", "OMP_NUM_THREADS",
                                                "MKL_NUM_THREADS", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE"}
    assert env["MASTER_ADDR"] == "127.0.0.1" and "leak" not in env.values()


def test_copy_job_never_carries_private_material(tmp_path):
    """Item 11: a working copy a job process can run in never contains _private/ (reference, metadata)."""
    src = tmp_path / "fixture"
    (src / "_private").mkdir(parents=True)
    (src / "_private" / "reference.json").write_text("{}")
    (src / "job").mkdir()
    (src / "job" / "x.py").write_text("")
    trusted.append_run(src, dict(run_id="run0"))
    trusted.copy_job(src, tmp_path / "copy")
    assert not list((tmp_path / "copy").rglob("_private")) and (tmp_path / "copy" / "job" / "x.py").exists()
    assert trusted.run_ids(tmp_path / "copy") == ["run0"]


# --- live probes: a real job whose train.py is the abuser -------------------------------------------------------
ABUSER = textwrap.dedent('''
    import json, os, subprocess, sys, time
    import td_client
    rank, run = int(os.environ["RANK"]), os.environ["RUN_ID"]
    out = {}
    if rank == 1:
        time.sleep(1.0)
        sys.exit(0)
    s = td_client._sock()
    def rpc(msg):
        s.sendall((msg if isinstance(msg, bytes) else json.dumps(msg).encode()) + b"\\n")
        return td_client._recv()
    out["replay"] = [td_client.fetch(0, 2)[0], td_client.fetch(0, 2)[0]]
    out["out_of_order"] = [td_client.fetch(0, 5)[0], td_client.fetch(0, 1)[0]]
    out["forged_rank"] = rpc(dict(op="batch", epoch=0, index=0, rank=1))
    out["garbage"] = rpc(b"not json")
    out["range"] = rpc(dict(op="batch", epoch=0, index=-1))
    s.sendall(json.dumps(dict(op="step", n=9, digest="forged")).encode() + b"\\n")  # an out-of-order ack: no reply
    out["env_keys"] = sorted(os.environ)
    out["env_values"] = "\\n".join(os.environ.values())
    out["files"] = "".join(open(os.path.join(d, f), errors="ignore").read() for d, _, fs in os.walk(".") for f in fs
                           if os.path.getsize(os.path.join(d, f)) < 100000)
    try:
        out["ps"] = subprocess.run(["ps", "eww", "-p", str(os.getppid())], capture_output=True, text=True).stdout
    except OSError as e:
        out["ps"] = ""
    json.dump(out, open(f"runs/{run}/probe.json", "w"))
    s.sendall(b"x" * 10000)   # an oversized request: the connection is cut and the rank is killed
    time.sleep(30)
''')


@pytest.mark.executed
def test_live_protocol_abuse_is_rejected_or_harmless_and_the_salt_is_not_reachable(tmp_path, monkeypatch):
    """A job that replays, reorders, forges a rank, sends garbage and out-of-range positions, forges acks, hunts for
    the salt in its environment, files and the launcher's process environment, and finally sends an oversized
    request. The server answers errors for the bad ones, serves (and logs) the odd-but-legal ones, kills the
    oversized sender, and the run ends: no hang, no salt in sight."""
    key = tmp_path / "key"
    key.write_text("live-probe-key")
    monkeypatch.setattr(dataserver, "KEY_FILE", key)
    monkeypatch.delenv("TD_KEY", raising=False)
    salt = salt_for("live")
    jd = tmp_path / "jd"
    materialize(jd, build("B_OK", 0, 2, 5, random.Random(1)).config)
    (jd / "job" / "train.py").write_text(ABUSER)
    res = resubmit(jd, salt=salt, allow_unsandboxed=True)
    assert res.status == "failed"                      # rank 0 was killed for the oversized request
    probe = json.loads((jd / "runs" / "run0" / "probe.json").read_text())
    assert probe["replay"][0] == probe["replay"][1] and probe["out_of_order"][0] != probe["out_of_order"][1]
    assert "unexpected fields" in probe["forged_rank"]["error"] and "malformed" in probe["garbage"]["error"]
    assert "out of range" in probe["range"]["error"]
    assert set(probe["env_keys"]) <= set(ENV_PASSTHROUGH) | {"MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "RUN_ID", "RESUME_FROM",
                                                             "HARNESS_START", "TD_FD", "GLOO_SOCKET_IFNAME", "OMP_NUM_THREADS",
                                                             "MKL_NUM_THREADS", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "LC_CTYPE",
                                                             "__CF_USER_TEXT_ENCODING"}  # the last two: added by macOS / Python
    for hay in (probe["env_values"], probe["files"], probe["ps"]):
        assert salt.hex() not in hay and "live-probe-key" not in hay
    rec = trusted.ledger(jd)[0]
    served0 = rec["served"]["0"]
    assert [(e, i) for e, i, *_ in served0] == [(0, 2), (0, 2), (0, 5), (0, 1)] and rec["served"]["1"] == []
    assert rec["acked"] == {"0": 0, "1": 0}            # the forged ack counted for nothing
