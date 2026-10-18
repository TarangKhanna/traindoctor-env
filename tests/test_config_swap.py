"""Counterfactual config swaps and label-encoding checks. Pure Python: no job is executed.

Rules (from the job's own semantics, not from current output):
R1  A cause that is visible in the config follows the CONFIG when configs are swapped between scenarios whose causes
    differ; it does not follow the scenario label.
R2  The set of config keys is identical for every kind (no field exists only to signal one cause).
R3  No config key or string value contains a label/kind name or a generic fault word.
R4  Every non-numeric config value the bug introduces is a value the job code really implements (it is a mechanism,
    not a marker); see experiments/verifier_audit/config_visibility_note.md for what remains 1:1 with a cause.
"""
import re
from pathlib import Path

from jobs.build import ALL_LABELS, KINDS
from jobs.config import dumps, flatten, loads
from rl.splits import TRAIN_SEEDS
from traindoctor_env.scenario import job_spec, make_scenario

SEEDS = list(TRAIN_SEEDS)[:150]
FAULT_WORDS = re.compile(r"bug|fault|stale|broken|inject|wrong|bad|diverg|stall|skip|dup|mismatch", re.I)


def shown(seed):
    sc = make_scenario(seed); sp = job_spec(sc)
    text = dumps(sp.config)
    for e in sp.bug:
        if e.path == "job/config.yaml":
            assert text.count(e.old) == 1
            text = text.replace(e.old, e.new)
    return sc, sp, loads(text)


def config_cause(cfg):
    """What a reader can conclude from the config alone, by the job's semantics."""
    if any(m != "all_reduce" for m in cfg["sync"]["per_rank"].values()):
        return "sync_mode_differs"
    if any(d != f"part-{r}" for r, d in cfg["data"]["shard_dirs"].items()):
        return "shard_dir_not_own"
    return None


def expected_cause(sc):
    return {"A1": "sync_mode_differs" if sc.variant == 1 else None, "A2": "shard_dir_not_own"}.get(sc.kind)


def test_config_cause_is_a_function_of_the_config_only():
    seen = set()
    for s in SEEDS:
        sc, _, cfg = shown(s)
        assert config_cause(cfg) == expected_cause(sc)
        seen.add(config_cause(cfg))
    assert seen == {None, "sync_mode_differs", "shard_dir_not_own"}


def test_swapping_configs_moves_the_visible_cause_not_the_label():
    by_cause = {}
    for s in SEEDS:
        sc, sp, cfg = shown(s)
        by_cause.setdefault(config_cause(cfg), []).append((sc, cfg))
    n = n_diff_label = 0
    for cause in ("sync_mode_differs", "shard_dir_not_own"):
        for (donor, dcfg) in by_cause[cause][:20]:
            for (recv, _) in by_cause[None][:20]:
                # recipient keeps its own label/kind; it now carries the donor's config
                swapped = dict(dcfg)
                assert config_cause(swapped) == cause != expected_cause(recv)
                n_diff_label += recv.label != donor.label
                n += 1
    # and the reverse: a clean (config-silent) config on a donor whose scenario was config-visible shows nothing
    for (sc, _) in by_cause["sync_mode_differs"][:10]:
        clean = job_spec(sc).config
        assert config_cause(loads(dumps(clean))) is None
    assert n > 100 and n_diff_label > 50   # many swaps cross labels, not just variants


def test_swapped_config_is_repaired_by_the_donor_reverse_edit_regardless_of_label():
    """The config part of the repair is a function of the config text: the reversed edit applies to the donor config
    wherever it is placed and restores a config whose visible cause is gone."""
    n = 0
    for s in SEEDS:
        sc, sp, cfg = shown(s)
        cedits = [e for e in sp.fix if e.path == "job/config.yaml"]
        if not cedits:
            continue
        n += 1
        assert config_cause(cfg) is not None      # the config-visible cause is really there before the repair
        text = dumps(cfg)
        for e in cedits:
            assert text.count(e.old) == 1
            text = text.replace(e.old, e.new)
        assert config_cause(loads(text)) is None
    assert n >= 30   # the loop must not pass vacuously (a repair with no config edits would otherwise be skipped)


def test_config_key_sets_are_identical_for_every_kind():
    keysets = {}
    for s in SEEDS:
        sc, _, cfg = shown(s)
        keysets.setdefault((sc.kind, sc.variant), set()).update(
            re.sub(r"\.\d+$", ".<rank>", k) for k in flatten(cfg))
    assert len(keysets) == 6 and len({frozenset(v) for v in keysets.values()}) == 1


def test_no_label_or_fault_name_in_any_config_key_or_string_value():
    names = {x.lower() for x in ALL_LABELS} | {k.lower() for k in KINDS}
    for s in SEEDS:
        _, _, cfg = shown(s)
        for k, v in flatten(cfg).items():
            for tok in [k] + ([v] if isinstance(v, str) else []):
                low = tok.lower()
                assert not any(n in low for n in names), (k, v)
                assert not FAULT_WORDS.search(tok), (k, v)


def test_bug_values_are_implemented_job_semantics():
    src = (Path(__file__).resolve().parents[1] / "jobs/template/train.py").read_text()
    data = (Path(__file__).resolve().parents[1] / "jobs/template/data.py").read_text()
    for s in SEEDS:
        sc, sp, cfg = shown(s)
        for m in cfg["sync"]["per_rank"].values():
            assert f'"{m}"' in src or m == "all_reduce"
        assert 'cfg["data"]["shard_dirs"]' in data
