"""The verifier trusts nothing that runs inside the job's process. Probes that must fail V2 (forged records, forged
or corrupted checkpoints, no training, lr=0, ...), and valid alternative repairs that must pass -- by the fast path
(state equal to the reference at every committed checkpoint) or, when not state-equal, by the independent
invariants (checkpoint format and sampler position, each transition reproduced by the pristine update rule,
logical sample accounting, restored-state consistency, bounded lost work, rank agreement)."""
import re
import textwrap

import pytest

from jobs import trusted
from jobs.build import Edit
from tests.conftest import SCEN
from traindoctor_env import reexec
from traindoctor_env.oracle import reference_solution
from traindoctor_env.reexec import Replay
from traindoctor_env.snapshot import load_truth

pytestmark = pytest.mark.executed


def run(snap, extra=(), files=None):
    """Oracle repair + extra edits (to the files as they stand) + whole-file writes, then final verification."""
    _, _, edits = reference_solution(snap, load_truth(snap))
    rp = Replay(snap)
    try:
        for e in [*edits, *extra]:
            rp.edit(e)
        for path, text in (files or {}).items():
            assert rp.write_file(path, text)
        return rp.outcome()
    finally:
        rp.close()


def run_unrepaired(snap, files=None, extra=()):
    rp = Replay(snap)
    try:
        for e in extra:
            rp.edit(e)
        for path, text in (files or {}).items():
            assert rp.write_file(path, text)
        return rp.outcome()
    finally:
        rp.close()


def line(snap, path, pattern):
    return re.search(rf"^.*{pattern}.*\n", snap.files[path], re.M).group(0)


def swap(snap, path, pattern, new):
    old = line(snap, path, pattern)
    return Edit(path, old, new if new.endswith("\n") or not new else new + "\n")


def test_valid_repairs_in_the_correctness_class_pass(snapshot):
    snap = snapshot("A2")
    # numerically different (loss scaled by 3 and unscaled before the sync) but equivalent: not bit-identical
    back, flat = line(snap, "job/train.py", r"loss.backward\(\)"), line(snap, "job/train.py", r"flat = torch.cat\(\[flat, torch.zeros")
    scaled = [Edit("job/train.py", back, back.replace("loss.backward()", "(loss * 3).backward()")),
              Edit("job/train.py", flat, flat + flat.split("flat")[0] + "flat = flat / 3\n")]
    out = run(snap, scaled)
    assert out.ok and not out.bit_identical and out.params_match_reference and out.samples_match_reference


def test_a_valid_repair_that_is_not_state_equal_passes_by_the_invariants(snapshot):
    """Rule: full-state equality is a sufficient fast path, not a necessary condition. A repair that also changes the
    checkpoint layout (an extra entry) is not state-equal to the reference -- yet it restores correctly, trains on the
    right samples and loses nothing, so the independent invariants must accept it."""
    snap = snapshot("A2")
    save = line(snap, "job/ckpt.py", r"state.update\(sampler=")
    extra = Edit("job/ckpt.py", save, save.replace("step=torch.tensor([step]))", "step=torch.tensor([step]), meta=torch.tensor([1.0, float(step)]))"))
    assert extra.new != extra.old
    out = run(snap, [extra])
    assert not out.states_match_reference and not out.bit_identical        # the fast path does not apply
    assert out.ok and out.path == "invariants"
    assert out.transitions_consistent and out.restores_consistent and out.samples_match_reference and out.lost_steps <= 4


def test_unused_prefetch_does_not_fail_training_and_the_served_log_keeps_the_prefetches(snapshot):
    """A peek at the next batch that is never used neither changes training nor hides what was served: the run
    passes (state-equal), the agent-readable sample record still names the batch each step consumed, and the data
    server's log (the ledger) lists the extra requests."""
    from jobs import trusted
    snap = snapshot("A2")
    _, _, edits = reference_solution(snap, load_truth(snap))
    batch = line(snap, "job/train.py", r"x, y = sampler.next_batch\(\)")
    peek = Edit("job/train.py", batch, batch + batch.split("x")[0] + "__import__('td_client').fetch(sampler.epoch, sampler.index)\n")
    rp = Replay(snap)
    try:
        for e in [*edits, peek]:
            rp.edit(e)
        rp.relaunch(None)
        rec = trusted.ledger(rp.jd)[-1]
        steps = snap.reference["steps"] - rec["start_step"]
        assert all(len(v) == 2 * steps - 1 or len(v) == 2 * steps for v in rec["served"].values())   # every step peeked
        recorded = [__import__("json").loads(x) for x in (trusted.trusted_dir(rp.jd) / "records" / rec["run_id"] / "samples" / "rank0.jsonl").read_text().splitlines()]
        assert {r["step"]: r["ids"] for r in recorded} == {int(s): ids for s, ids in snap.reference["samples"]["0"].items()
                                                           if int(s) >= rec["start_step"]}
        out = rp.outcome()
    finally:
        rp.close()
    assert out.ok and out.states_match_reference


def test_bit_identical_final_state_never_bypasses_the_other_invariants(snapshot):
    """Same final state as the reference, but a non-editable file was changed: samples, restores and parameters
    all check out, yet V2 fails on integrity."""
    snap = snapshot("A2")
    out = run(snap, files={"job/config.py": snap.files["job/config.py"] + "# changed\n"})
    assert out.bit_identical and out.samples_match_reference and out.restores_consistent and out.lost_steps <= 4
    assert out.tampered == ("job/config.py",) and not out.ok


@pytest.mark.parametrize("probe", ["no_training", "zero_lr", "skip_model_restore"])
def test_tamper_probes_fail_v2_task_a(snapshot, probe):
    snap = snapshot("A2")
    if probe == "no_training":
        edit = swap(snap, "job/config.yaml", r"^steps: ", "steps: 0")
    elif probe == "zero_lr":
        edit = swap(snap, "job/config.yaml", r"^lr: ", "lr: 0.0")
    else:  # restore() claims success without loading the model: the restored state is not the saved one
        edit = swap(snap, "job/ckpt.py", r"model.load_state_dict", "    pass")
    out = run(snap, [edit])
    assert not out.ok and out.path == ""
    if probe == "zero_lr":
        assert out.samples_match_reference and not out.params_match_reference and not out.transitions_consistent
    if probe == "skip_model_restore":
        assert not out.restores_consistent


def inject(snap, code):
    train = snap.files["job/train.py"]
    return {"job/train.py": train.replace('    log("training complete")\n', '    log("training complete")\n' + textwrap.indent(textwrap.dedent(code), "    "))}


def test_job_written_sample_files_are_not_evidence_for_the_verdict(snapshot):
    """B_DUP left unrepaired, plus code that overwrites the readable sample files with the reference history:
    the verdict comes from the launcher's records and still shows the replayed samples."""
    snap = snapshot("B_DUP")
    fake = f"""
        import glob
        import json as _json
        REF = {snap.reference["samples"]!r}
        for f in glob.glob("runs/*/samples/rank*.jsonl"):
            with open(f, "w") as out:
                for s, ids in REF[f.rsplit("rank", 1)[1].split(".")[0]].items():
                    out.write(_json.dumps(dict(step=int(s), ids=ids)) + "\\n")
        """
    out = run_unrepaired(snap, inject(snap, fake))
    assert not out.ok and not out.samples_match_reference and out.sample_gaps > 0 and not out.tampered


def forge_payload(snap):
    """Everything code inside the job's process can do to records kept in that process: patch the instrumentation
    module's digest functions, rewrite its in-memory event lists (found through the garbage collector), overwrite its
    files. The attacker is even handed the reference values to forge with (the reference is private; a sandboxed job
    cannot read it)."""
    ref = snap.reference
    return textwrap.dedent(f"""
        import gc, glob, os, sys
        REF = dict(samples={ref["samples"]!r}, final_params={ref["final_params"]!r}, final_digest={ref.get("final_digest", "forged")!r})
        _main = sys.modules["__main__"]
        if hasattr(_main, "state_digest"):
            _main.state_digest = lambda opt: REF["final_digest"]
        if hasattr(_main, "params_of"):
            _main.params_of = lambda opt: REF["final_params"]

        def forge():
            rank = os.environ["RANK"]
            for o in gc.get_objects():
                if isinstance(o, list) and o and isinstance(o[0], dict) and o[0].get("ev") in ("restored", "step", "end"):
                    for e in o:
                        if e.get("ev") == "step":
                            e["ids"] = REF["samples"][rank].get(str(e["step"]), e["ids"])
                            e["digest"] = REF["final_digest"]
                        elif e.get("ev") == "restored":
                            e["digest"] = REF["final_digest"]
            for f in glob.glob(".harness/*/*.jsonl") + glob.glob("runs/*/samples/*.jsonl"):
                open(f, "a").close()
        """)


def test_forged_in_process_records_cannot_turn_a_failing_job_into_a_pass(snapshot):
    """Item 1: B_DUP left unrepaired -- replayed samples, a checkpoint without sampler state -- with code that forges
    every record kept inside the job's process. Nothing the verdict rests on lives there: the checkpoints were copied
    by the launcher when committed, the batches were served by the launcher's data server, and the process history
    is the launcher's own."""
    snap = snapshot("B_DUP")
    files = inject(snap, "forge()\n")
    files["job/train.py"] = forge_payload(snap) + files["job/train.py"]
    out = run_unrepaired(snap, files)
    assert not out.ok and not out.states_match_reference and not out.tampered


def corrupt_final(snap, mutation):
    save = line(snap, "job/ckpt.py", r"torch.save\(state")
    steps = snap.reference["steps"]
    body = f"        if step == {steps}:\n            {mutation}\n"
    return Edit("job/ckpt.py", save, body + save)


def test_corrupting_only_the_final_checkpoint_with_a_matching_checksum_fails(snapshot):
    """Item 2: the manifest's checksum is computed after the corrupt state is written, so the file and its checksum
    agree; the launcher's captured copy of the final checkpoint still differs from the reference."""
    snap = snapshot("A2")
    out = run(snap, [corrupt_final(snap, 'state["model.fc1.bias"] = state["model.fc1.bias"] + 0.01')])
    assert not out.ok and not out.states_match_reference and not out.params_match_reference and out.ckpt_valid


def test_zeroing_the_final_momentum_fails_even_though_parameters_are_right(snapshot):
    """Item 2: the live parameters and the saved model entries are right; only the optimizer state is wrong."""
    snap = snapshot("A2")
    out = run(snap, [corrupt_final(snap, 'state.update({k: torch.zeros_like(v) for k, v in state.items() if k.startswith("opt.")})')])
    assert not out.ok and not out.states_match_reference and out.params_match_reference


def test_fabricated_checkpoint_state_fails_v2_task_b(snapshot):
    snap = snapshot("B_OK")
    save = line(snap, "job/ckpt.py", r"state.update\(sampler=")
    zeros = save + '        state.update({k: torch.zeros_like(v) for k, v in state.items() if k.startswith(("model.", "opt."))})\n'
    out = run_unrepaired(snap, {"job/ckpt.py": snap.files["job/ckpt.py"].replace(save, zeros)})
    assert not out.ok and not out.transitions_consistent


def test_tolerance_constants_are_the_documented_ones():
    assert reexec.FAST_TOL == reexec.TOLERANCE == 1e-5 and reexec.B_VALIDATION_SCRATCH_RUNS == 1


def test_checkpoints_replayed_from_an_earlier_run_fail_verification(snapshot):
    """Rule: verification is satisfied only by training on the data THIS verification served. A job that embeds the
    correct final checkpoint produced by an earlier (exploratory) run, writes it, forges identical per-rank digests and
    exits without training must fail V2 -- the scenario's data used to be stable across runs, which made such a
    checkpoint valid for every later run."""
    import base64
    import json
    snap = snapshot("A2")
    _, _, edits = reference_solution(snap, load_truth(snap))
    ex = Replay(snap)
    try:
        for e in edits:
            ex.edit(e)
        ex.relaunch(None)
        entries = json.loads((ex.jd / "ckpt" / "MANIFEST.json").read_text())
        last = entries[-1]
        assert last["step"] == snap.reference["steps"]
        blob = (ex.jd / "ckpt" / last["shards"][0]["file"]).read_bytes()
    finally:
        ex.close()
    forged = textwrap.dedent(f'''
        import base64 as _b64, json as _json, os as _os, sys as _sys
        import td_client as _td
        if int(_os.environ["RANK"]) == 0:
            _os.makedirs("ckpt/{last["ckpt_id"]}", exist_ok=True)
            open("ckpt/{last["shards"][0]["file"]}", "wb").write(_b64.b64decode({base64.b64encode(blob).decode()!r}))
            _m = [e for e in _json.load(open("ckpt/MANIFEST.json")) if e["step"] < {last["step"]}] + [{last!r}]
            _json.dump(_m, open("ckpt/MANIFEST.json", "w"))
        for _n in range(1, {last["step"]} + 1):
            _td.notify(dict(op="step", n=_n, digest="forged"))
        _sys.exit(0)
        ''')
    out = run_unrepaired(snap, files={"job/train.py": forged + snap.files["job/train.py"]})
    assert not out.ok and out.path == "" and not out.states_match_reference and not out.tampered
