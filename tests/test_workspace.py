"""Tier B: the agent works in an isolated copy of the job directory with real file edits and relaunch."""
import dataclasses
import hashlib
import re

import pytest

from jobs.build import LABELS
from tests.conftest import NAMES
from traindoctor_env.oracle import reference_solution
from traindoctor_env.snapshot import load_truth
from traindoctor_env.tools import ACK, VISIBLE, Episode

pytestmark = pytest.mark.executed
ID = re.compile(r"\bE[0-9a-f]{6}\b")
W = lambda path, content: {"tool": "write_file", "args": {"path": path, "content": content}}  # noqa: E731
R = lambda **a: {"tool": "relaunch", "args": a}  # noqa: E731
READS = [("job_status", {}), ("job_config", {}), ("ckpt_manifest", {}), ("scheduler_events", {}), ("list_files", {}),
         ("rank_timeline", {}), ("grep_logs", {"pattern": ""}), ("process_state", {"rank": 0}),
         *[("collective_trace", {"step": s}) for s in range(4, 9)], *[("sample_ids", {"step": s}) for s in range(3, 11)]]
BANNED = ["failure_class", "truth", "fixture", "oracle", "_private", "reference.json", "root cause", "recommend", "hint",
          "bug", "inject", "fault", "chaos", "hack", "workaround", "todo", "fixme", "jobs/template", "m2_candidates"]


# What an agent must never be able to read, specified here and not derived from tools.VISIBLE (which is what is
# under test). Patterns match real paths under the workspace; PROBES are concrete paths tried through the tools.
FORBIDDEN_PATTERNS = [r"(^|/)_private(/|$)", r"truth\.json", r"reference\.json", r"meta\.json", r"(^|/)\.lock$", r"\.trusted",
                      r"ledger\.jsonl", r"(^|/)captures(/|$)", r"(^|/)\.harness(/|$)", r"runs/run\d+/job\.json",
                      r"runs/run\d+/start\.json", r"runs/run\d+/stderr/", r"ckpt/ck\d+/", r"\.pt$", r"(^|/)public(/|$)"]
PROBES = ["_private/reference.json", "_private/meta.json", "_private", "../truth.json", "truth.json", "../public/records.jsonl",
          "runs/run0/job.json", "runs/run0/start.json", "runs/run0/stderr/rank0.txt", "ckpt/ck0004/state.pt", ".lock",
          ".harness/run0/rank0.jsonl", "../fixture.trusted/ledger.jsonl", "../workspace.trusted/ledger.jsonl",
          "ledger.jsonl", "captures", "job/../_private/meta.json", "/etc/passwd"]
EXISTS_ON_DISK = ["runs/run0/start.json", "runs/run0/stderr/rank0.txt", "ckpt/ck0004/state.pt"]


def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + p.read_bytes())
    return h.hexdigest()


def apply_text(ep, edits):
    """The agent's file writes for a set of edits (read the file, change it, write it back)."""
    for path in sorted({e.path for e in edits}):
        text = ep.step({"tool": "read_file", "args": {"path": path}})[0]
        text = "\n".join(re.sub(r"^\d+: ", "", ln) for ln in text.splitlines()) + "\n"
        for e in (e for e in edits if e.path == path):
            assert text.count(e.old) == 1, e
            text = text.replace(e.old, e.new)
        assert ep.step(W(path, text)) == (ACK, False)


def outputs(ep):
    return [ep.step({"tool": t, "args": a})[0] for t, a in READS]


def test_edit_relaunch_serves_fresh_artifacts_and_matches_the_batch_verdict(snapshot, menu_table):
    """Rule (grading semantics, docs/VERIFICATION.md): `relaunches` counts EVERY recovery attempt since the incident,
    the agent's workspace relaunches and the verification protocol's own run. The batch table has only the protocol
    run, so after one explicit relaunch the live verdict must equal it in everything except relaunches == table + 1."""
    snap = snapshot("A1_src")
    menu, _, table = menu_table("A1_src")
    before = tree_hash(snap.fixture_dir)
    _, _, edits = reference_solution(snap, load_truth(snap))
    ep = Episode()
    ep.reset(snap)
    pre_status = ep.step({"tool": "job_status"})[0]
    pre_ids = set(ID.findall("\n".join(outputs(ep))))
    # the source is readable and editable; an edit alone produces no new artifacts
    assert "def sync_grads" in ep.step({"tool": "read_file", "args": {"path": "job/train.py"}})[0]
    apply_text(ep, edits)
    assert ep.step({"tool": "job_status"})[0] == pre_status
    assert ep.outcome() == table[next(c.id for c in menu if c.role == "correct")]  # same verdict as the batch table
    # relaunch: everything read afterwards comes from the fresh run, with fresh ids
    assert ep.step(R()) == (ACK, False)
    post = "\n".join(outputs(ep))
    assert not (set(ID.findall(post)) & {r["id"] for r in snap.records}) and pre_ids
    assert "-> completed" in ep.step({"tool": "job_status"})[0].splitlines()[-1]
    assert ep.step({"tool": "ckpt_manifest"})[0].count("ck0012") == 1
    out, batch = ep.outcome(), table[next(c.id for c in menu if c.role == "correct")]
    assert out.ok and out.relaunches == batch.relaunches + 1
    assert dataclasses.replace(out, relaunches=batch.relaunches) == batch
    assert tree_hash(snap.fixture_dir) == before         # the fixture is never touched
    ep.close()


def test_claimed_outcome_is_recorded_beside_the_verified_one(snapshot, needs_sandbox):
    snap = snapshot("A2")
    _, _, edits = reference_solution(snap, load_truth(snap))
    diag = lambda claimed: {"tool": "submit_diagnosis", "args": dict(label="x", evidence_ids=[], claimed_fixed=claimed, fix_summary="s")}  # noqa: E731
    for fixed, claimed, expect in ((True, True, "match"), (False, True, "overclaim")):
        ep = Episode()
        ep.reset(snap)
        if fixed:
            apply_text(ep, edits)
        assert ep.step({"tool": "submit_diagnosis", "args": {"label": "x", "evidence_ids": [], "fix_summary": "s"}})[0].startswith("error")
        assert ep.step(diag(claimed))[0] == ACK
        ep.step({"tool": "finish"})
        res = ep.state()["result"]
        assert res["claimed_vs_verified"] == expect and res["claimed_fixed"] is claimed and res["verified_fixed"] is fixed
        ep.close()


def test_agent_cannot_write_outside_job_or_read_harness_files(snapshot):
    ep = Episode()
    ep.reset(snapshot("A2"))
    for bad in ("ckpt/MANIFEST.json", "job/../scheduler.jsonl", "job/sub/x.py", "/etc/passwd", "data/part-0/shard0.idx"):
        assert ep.step(W(bad, "x"))[0].startswith("error: invalid arguments"), bad
    for hidden in ("_private/reference.json", "_private/meta.json", "runs/run0/job.json", "runs/run0/stderr/rank0.txt",
                   "../truth.json", "job/../_private/meta.json"):
        assert ep.step({"tool": "read_file", "args": {"path": hidden}})[0].startswith("error: no such file"), hidden
    listing = ep.step({"tool": "list_files", "args": {"path": "."}})[0]
    assert "job/train.py" in listing and "data/part-0/shard0.idx" in listing and "_private" not in listing
    ep.close()


def test_task_b_agent_repairs_by_editing_and_rerunning_with_the_incident(snapshot, needs_sandbox):
    snap = snapshot("B_DUP")
    _, _, edits = reference_solution(snap, load_truth(snap))
    ep = Episode()
    ep.reset(snap)
    assert "-> completed" in ep.step({"tool": "job_status"})[0].splitlines()[-1]   # the restarted run looks fine
    apply_text(ep, edits)                              # fix save/restore, then re-run from scratch with the incident
    assert ep.step(R(resume_from="none", chaos=True))[0] == ACK
    assert "-> completed" in ep.step({"tool": "job_status"})[0].splitlines()[-1]
    out = ep.outcome()
    assert out.ok and out.samples_match_reference and out.sample_gaps == out.sample_dups == 0
    ep.close()


def test_workspace_leaks_nothing(snapshot):
    """No ground truth, no hints about the injected change, no patch history, in anything the agent can read."""
    inventory = None
    for name in NAMES:
        snap = snapshot(name)
        ep = Episode()
        intro = ep.reset(snap)
        assert all(label in intro for label in set(LABELS.values()))   # the answer vocabulary is shown as a whole
        ep.step({"tool": "list_files"})
        files = ep._visible_files()
        assert not [f for f in files if any(re.search(p, f) for p in FORBIDDEN_PATTERNS)], files   # independent of VISIBLE
        assert not any(f.endswith((".orig", ".rej", ".bak", ".patch", ".diff")) for f in files)
        assert not any(".git" in p.parts for p in ep.root.rglob("*"))
        job_files = sorted(f for f in files if f.startswith("job/"))
        assert inventory in (None, job_files)               # the same files whatever the scenario
        inventory = job_files
        text = "\n".join(outputs(ep) + [(ep.root / f).read_text() for f in files]).lower()
        banned = [*BANNED, *(label.lower() for label in set(LABELS.values())), str(snap.root).lower(), str(ep.root).lower()]
        assert not [w for w in banned if w in text], (name, [w for w in banned if w in text])
        ep.close()


def test_forbidden_files_exist_on_disk_but_cannot_be_read_or_listed(snapshot):
    """Probe the tools with a hard-coded list of forbidden paths (not the filtered inventory): files that really are in
    the workspace tree (run records, stderr, checkpoints) and private material beside it."""
    snap = snapshot("A2")
    ep = Episode()
    ep.reset(snap)
    for p in EXISTS_ON_DISK:
        assert (ep.root / p).exists(), p                       # the probe is not vacuous
    assert (ep.root / "_private" / "reference.json").exists() and (ep.root.parent / "truth.json").exists()
    listing = ep.step({"tool": "list_files", "args": {"path": "."}})[0]
    for p in PROBES:
        assert ep.step({"tool": "read_file", "args": {"path": p}})[0].startswith("error: no such file"), p
    assert not [x for x in listing.splitlines() if any(re.search(f, x) for f in FORBIDDEN_PATTERNS)], listing
    for sub in ("runs", "runs/run0", "ckpt", "_private"):
        out = ep.step({"tool": "list_files", "args": {"path": sub}})[0]
        assert not [x for x in out.splitlines() if any(re.search(f, x) for f in FORBIDDEN_PATTERNS)], (sub, out)
    ep.close()


@pytest.mark.executed
def test_private_material_is_never_copied_into_the_workspace_or_the_protocol_dir(snapshot, tmp_path):
    """Item 11: no _private/ directory, no reference or metadata content, in the agent's workspace (before and after a
    relaunch) or in the dir the verifier re-executes in."""
    from traindoctor_env.reexec import Replay, _fresh_protocol_dir
    marker = (snapshot("A2").fixture_dir / "_private" / "reference.json").read_text()[:140].encode()
    for name in ("A2", "B_DUP"):
        snap = snapshot(name)
        rp = Replay(snap)
        try:
            _, _, edits = reference_solution(snap, load_truth(snap))
            for e in edits:
                rp.edit(e)
            rp.relaunch(None)
            (tmp_path / name).mkdir()
            jd, *_ = _fresh_protocol_dir(snap, rp.jd / "job", tmp_path / name)
            for root in (rp.jd, jd, rp.jd.with_name(rp.jd.name + ".trusted").parent):
                files = [p for p in root.rglob("*")]
                assert not [p for p in files if "_private" in p.parts], (name, root)
                assert not [p for p in files if p.is_file() and p.stat().st_size and marker[60:140] in p.read_bytes()], (name, root)
        finally:
            rp.close()


def test_fix_tools_ack_identically_whether_or_not_they_help(snapshot):
    snap = snapshot("A2")
    _, _, edits = reference_solution(snap, load_truth(snap))
    acks = set()
    for right in (True, False):
        ep = Episode()
        ep.reset(snap)
        if right:
            apply_text(ep, edits)
        else:
            acks.add(ep.step(W("job/notes.txt", "x"))[0])
        acks |= {ep.step({"tool": "kill_ranks", "args": {"ranks": [0]}})[0], ep.step({"tool": "escalate", "args": {"reason": "x"}})[0]}
        ep.close()
    assert acks == {ACK}
