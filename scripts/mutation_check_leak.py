"""Mutation check of tests/test_config_swap.py and tests/test_menu_leak.py.

Each mutant is applied in memory (monkeypatch / re-exec of one function's edited source) and undone afterwards; library
code on disk is never changed. A test is adequate for a mutant only if it PASSES on the real code and FAILS on the
mutant. Output: docs/audits/mutation_check_leak.json. Run: TD_KEY=... python scripts/mutation_check_leak.py
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import inspect
import json
import pathlib
import re
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jobs.build as jb  # noqa: E402
import audit.menu_leak as ml  # noqa: E402
import traindoctor_env.bundle as bundle  # noqa: E402
import traindoctor_env.scenario as scen  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CS, MLK = load("test_config_swap"), load("test_menu_leak")
CS_TESTS = [n for n in dir(CS) if n.startswith("test_")]
ML_TESTS = [n for n in dir(MLK) if n.startswith("test_")]


def run(mod, names):
    if hasattr(mod, "REP"):
        mod.REP = None
    out = {}
    for n in names:
        try:
            getattr(mod, n)()
            out[n] = "pass"
        except BaseException as e:  # noqa: BLE001  (a test failing for any reason counts as detecting the mutant)
            out[n] = f"FAIL({type(e).__name__})"
    return out


# ---------- mutation helpers (all reversible) ----------
@contextlib.contextmanager
def patched_spec(mut):
    """Wrap job_spec as seen by the config-swap tests: mut(scenario, spec) -> spec."""
    orig = CS.job_spec
    CS.job_spec = lambda sc: mut(sc, orig(sc))
    try:
        yield
    finally:
        CS.job_spec = orig


@contextlib.contextmanager
def patched_attr(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def mutated_bundle_fn(fname, old, new=None):
    """Re-exec bundle.<fname> with `old` replaced by `new` in its source; rebind it everywhere it is referenced."""
    orig = getattr(bundle, fname)
    src = inspect.getsource(orig)
    for o, n in (old if new is None else [(old, new)]):
        assert o in src, f"mutant is vacuous: {o!r} not in {fname}"
        src = src.replace(o, n)
    ns = bundle.__dict__
    exec(compile(src, f"<mutant {fname}>", "exec"), ns)
    mutant = ns[fname]
    ns[fname] = mutant
    saved = []
    for fn in (ml.episode, ml.report):
        saved.append((fn, fn.__defaults__))
        fn.__defaults__ = tuple(mutant if d is orig else d for d in (fn.__defaults__ or ()))
    if getattr(ml, fname, None) is orig:
        setattr(ml, fname, mutant)
    try:
        yield
    finally:
        ns[fname] = orig
        if getattr(ml, fname, None) is mutant:
            setattr(ml, fname, orig)
        for fn, d in saved:
            fn.__defaults__ = d


def with_cfg(spec, f):
    cfg = json.loads(json.dumps(spec.config))
    f(cfg)
    return dataclasses.replace(spec, config=cfg)


def map_bug(spec, f):
    return dataclasses.replace(spec, bug=[f(e) for e in spec.bug])


# ---------- config-swap mutants: (id, description, context manager, tests that must fail) ----------
def cs_mutants():
    C = "test_config_swap"
    yield (C, "CS1 label-encoding key: extra config key only for A2", patched_spec(
        lambda sc, sp: with_cfg(sp, lambda c: c["data"].update(prefetch=3)) if sc.kind == "A2" else sp),
        ["test_config_key_sets_are_identical_for_every_kind"])
    yield (C, "CS2 fault-word config key (stale_flag) for every kind", patched_spec(
        lambda sc, sp: with_cfg(sp, lambda c: c.update(stale_flag=0))),
        ["test_no_label_or_fault_name_in_any_config_key_or_string_value"])
    yield (C, "CS3 label name leaked into a config string value (data.root = label)", patched_spec(
        lambda sc, sp: with_cfg(sp, lambda c: c["data"].update(root=sc.label.lower()))),
        ["test_no_label_or_fault_name_in_any_config_key_or_string_value"])
    yield (C, "CS4 stop swapping: config-visible bug edits dropped (cause no longer in the config)", patched_spec(
        lambda sc, sp: dataclasses.replace(sp, bug=[e for e in sp.bug if e.path != "job/config.yaml"])),
        ["test_config_cause_is_a_function_of_the_config_only", "test_swapping_configs_moves_the_visible_cause_not_the_label"])
    yield (C, "CS5 cause follows the label: B_DUP scenarios also carry a foreign-shard config edit", patched_spec(
        lambda sc, sp: dataclasses.replace(sp, bug=sp.bug + [jb.Edit("job/config.yaml", "    0: part-0\n", "    0: part-9\n")])
        if sc.kind == "B_DUP" else sp),
        ["test_config_cause_is_a_function_of_the_config_only"])
    yield (C, "CS6 repair does not remove the config cause (config fix edits become no-ops)", patched_attr(
        jb.JobSpec, "fix", property(lambda self: [jb.Edit(e.path, e.old, e.old) if e.path == "job/config.yaml" else e
                                                  for e in [x.reversed() for x in reversed(self.bug)]])),
        ["test_swapped_config_is_repaired_by_the_donor_reverse_edit_regardless_of_label"])
    yield (C, "CS7 config repair silently absent (fix returns no config edits)", patched_attr(
        jb.JobSpec, "fix", property(lambda self: [x.reversed() for x in reversed(self.bug) if x.path != "job/config.yaml"])),
        ["test_swapped_config_is_repaired_by_the_donor_reverse_edit_regardless_of_label"])
    yield (C, "CS8 bug value not implemented by the job (reduce_bcast -> xor_mode)", patched_spec(
        lambda sc, sp: map_bug(sp, lambda e: jb.Edit(e.path, e.old, e.new.replace("reduce_bcast", "xor_mode")))),
        ["test_bug_values_are_implemented_job_semantics"])
    real = pathlib.Path.read_text
    yield (C, "CS9 shard-dir semantics not read by data.py (reference removed from template text)", patched_attr(
        pathlib.Path, "read_text", lambda self, *a, **k: real(self, *a, **k).replace('cfg["data"]["shard_dirs"]', 'cfg["data"]["x"]')
        if self.name == "data.py" else real(self, *a, **k)),
        ["test_bug_values_are_implemented_job_semantics"])


def ml_mutants():
    M = "test_menu_leak"
    yield (M, "ML1 buggy-text context for no-change item (796ba23 bug)", mutated_bundle_fn(
        "make_menu", [('clean_cfg if loc == "relaunch"', '() if loc == "relaunch"')]),
        ["test_noop_and_exclude_items_carry_no_config_bug_text"])
    yield (M, "ML2 buggy-text context for the exclude item (pre-97a49d8: exclude drawn from the edit against the buggy files)", mutated_bundle_fn(
        "make_menu", [('shown["exclude"] = _shown_parts("exclude", slots["exclude"][0], clean)', 'pass')]),
        ["test_noop_and_exclude_items_carry_no_config_bug_text"])
    yield (M, "ML3 unkeyed candidate ids: old public sha256(seed:menu:role)", mutated_bundle_fn(
        "candidate_id", 'salt_for(seed) + b"|menu-id|" + role.encode()', 'f"{seed}:menu:{role}".encode()'),
        ["test_candidate_ids_are_not_a_public_function_of_seed_and_role"])
    yield (M, "ML4 key-independent ids (seed-only salt, new public formula)", mutated_bundle_fn(
        "candidate_id", 'salt_for(seed) + b"|menu-id|" + role.encode()', 'str(seed).encode() + b"|menu-id|" + role.encode()'),
        ["test_candidate_ids_are_not_a_public_function_of_seed_and_role"])
    yield (M, "ML5 position leak: correct item moved to the front after the shuffle", mutated_bundle_fn(
        "make_menu", 'random.Random(f"menu:{snap.seed}").shuffle(cands)', 'random.Random(f"menu:{snap.seed}").shuffle(cands); cands.sort(key=lambda c: c.role != "correct")'),
        ["test_menu_structure_does_not_identify_the_repair_or_the_label"])
    yield (M, "ML6 decoys drawn from edit direction: no neutral display (shown parts dropped)", mutated_bundle_fn(
        "make_menu", 'shown[loc] = _shown_parts(loc, slots[loc][0], clean) if slots[loc][0] else ()', 'shown[loc] = ()'),
        ["test_menu_structure_does_not_identify_the_repair_or_the_label", "test_boosted_stumps_on_menu_only_features_do_not_beat_majority"])
    yield (M, "ML7 decoys only exist where the location is clean (empty decoy elsewhere shows structure)", mutated_bundle_fn(
        "make_menu", 'slots[loc] = (right, False, "correct") if loc == where else (decoy[loc], False, "decoy_" + loc)',
        'slots[loc] = (right, False, "correct") if loc == where else ([], False, "decoy_" + loc)'),
        ["test_menu_structure_does_not_identify_the_repair_or_the_label"])
    yield (M, "ML8 decoys rendered from their raw edits, only the true repair gets the neutral display", mutated_bundle_fn(
        "make_menu", 'if slots[loc][0] else ()', 'if slots[loc][0] and loc == where else ()'),
        ["test_menu_structure_does_not_identify_the_repair_or_the_label"])


def main():
    base = {**run(CS, CS_TESTS), **run(MLK, ML_TESTS)}
    res = dict(baseline_real_code=base, mutants=[])
    bad = [k for k, v in base.items() if v != "pass"]
    assert not bad, f"tests must pass on real code: {bad}"
    for suite, desc, cm, must in [*cs_mutants(), *ml_mutants()]:
        mod, names = (CS, CS_TESTS) if suite == "test_config_swap" else (MLK, ML_TESTS)
        with cm:
            r = run(mod, names)
        failing = sorted(n for n, v in r.items() if v != "pass")
        caught = [n for n in must if n in failing]
        res["mutants"].append(dict(id=desc.split()[0], description=desc, suite=suite, expected_to_fail=must,
                                   failing_tests=failing, detected_by_expected=len(caught) == len(must),
                                   detected_by_any=bool(failing), results=r))
        print(("DETECTED " if failing and len(caught) == len(must) else "MISSED   "), desc, "->", failing)
    res["undetected"] = [m["id"] for m in res["mutants"] if not m["detected_by_any"]]
    res["expected_not_detected"] = [m["id"] for m in res["mutants"] if not m["detected_by_expected"]]
    out = ROOT / "docs/audits/mutation_check_leak.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(out, "undetected:", res["undetected"], "expected_not_detected:", res["expected_not_detected"])
    return 1 if res["undetected"] or res["expected_not_detected"] else 0


if __name__ == "__main__":
    sys.exit(main())
