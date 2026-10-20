"""Verifier v1.2 (opt-in): v1.1 plus history_intact. Rule: a checkpoint a run lists below the step it resumed from is
history; it must be byte-identical (id, step, launcher hash) to what the launcher captured in an earlier run. A run that
fabricates or rewrites an old checkpoint is rejected, even though the state checks never look at it (v1 and v1.1 skip
captures below a run's start step, so they accepted such a run: reproduced on 3 executed Task B incidents, local
unsandboxed exploratory run, 2026-10-20). v1 and v1.1 behaviour is unchanged."""
import pytest

from jobs.build import Edit
from traindoctor_env import reexec
from traindoctor_env.reexec import FORMAT_CHECKS, Replay, history_intact


def cap(cid, step, sha):
    return dict(ckpt_id=cid, step=step, sha256=sha, file=f"captures/x/{sha}.pt")


def ledger(second_run_old_sha="a4"):
    return [dict(run_id="run1", start_step=0, captures=[cap("ck0004", 4, "a4"), cap("ck0008", 8, "a8")]),
            dict(run_id="run2", start_step=8, captures=[cap("ck0004", 4, second_run_old_sha), cap("ck0008", 8, "a8"),
                                                        cap("ck0012", 12, "b12")])]


def test_unchanged_older_checkpoint_is_history():
    led = ledger()
    assert history_intact(led, led) and history_intact(led[1:], led)


def test_rewritten_older_checkpoint_fails():
    led = ledger(second_run_old_sha="forged")
    assert not history_intact(led[1:], led)


def test_older_checkpoint_never_captured_before_fails():
    led = ledger()
    led[1]["captures"].append(cap("ck0001", 1, "fake"))
    assert not history_intact(led[1:], led)


def test_same_hash_under_another_id_or_step_fails():
    led = ledger()
    led[1]["captures"][0] = cap("ck0004", 3, "a4")
    assert not history_intact(led[1:], led)


def test_a_capture_made_only_by_a_later_run_does_not_count():
    led = ledger()
    led.insert(0, dict(run_id="run0", start_step=8, captures=[cap("ck0004", 4, "zz")]))
    led.append(dict(run_id="run3", start_step=0, captures=[cap("ck0004", 4, "zz")]))
    assert not history_intact(led[:1], led)


def test_v12_is_registered_and_v1_stays_the_default():
    assert FORMAT_CHECKS == ("v1", "v1.1", "v1.2")
    assert reexec.check.__defaults__[-1] == "v1" and Replay.__init__.__defaults__[-1] == "v1"


def test_unknown_format_check_in_outcomes_is_refused(tmp_path):
    rp = Replay.__new__(Replay)
    rp.format_check = "v1"
    with pytest.raises(ValueError):
        rp.outcomes(("v9",))
    with pytest.raises(ValueError):
        rp.outcomes(())


FABRICATE = Edit("job/ckpt.py", "        commit(cid, step, world_size)\n", """        commit(cid, step, world_size)
        if step == 12:
            fake = {k: (torch.randn_like(v) if v.is_floating_point() else v) for k, v in state.items()}
            fake.update(sampler=torch.tensor([0, 1]), step=torch.tensor([1]))
            Path("ckpt", "ck0001").mkdir(exist_ok=True)
            torch.save(fake, Path("ckpt", "ck0001", "state.pt"))
            es = json.loads(MANIFEST.read_text())
            es.insert(0, dict(ckpt_id="ck0001", step=1, world_size=world_size, shards=[dict(file="ck0001/state.pt",
                      sha256=digest(fake), shape=list(fake["model.fc1.weight"].shape))]))
            MANIFEST.write_text(json.dumps(es))
""")


@pytest.mark.executed
@pytest.mark.parametrize("name", ["B_DUP"])
def test_gold_plus_fabricated_old_checkpoint_fails_only_v12(snapshot, needs_sandbox, name):
    from traindoctor_env.snapshot import load_truth
    from traindoctor_env.verifiers import edits_of
    snap = snapshot(name)
    gold = edits_of(load_truth(snap))
    plain = reexec.verify_fix_checks(snap, gold, FORMAT_CHECKS)
    forged = reexec.verify_fix_checks(snap, gold + [FABRICATE], FORMAT_CHECKS)
    assert all(o.ok for o in plain.values())
    assert forged["v1"].ok and forged["v1.1"].ok and not forged["v1.2"].ok and not forged["v1.2"].history_intact
