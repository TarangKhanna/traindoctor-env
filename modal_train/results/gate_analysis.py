"""Gate analysis: do distractor menus make V2 differ from V1? EXPLORATORY (dev_container tables, not verified recovery).

Offline, no Modal. Inputs: a menu-v2 DEV cache (modal_train/cache_menu2/dev_eps.pkl, written by modal_train.prepare)
and saved eval jsons (modal_train.app::run_eval -> `completions`, G rollouts per episode in order).
  extract(eps, comps, G)   -> one record per rollout (verdict via the unchanged scorer on the transient dev view; chosen repair's
                              outcome and role from the episode's own menu/table)
  aggregate(records)       -> the gate numbers (pure; unit-tested in tests/test_gate_analysis.py)
Run: PYTHONPATH=. python -m modal_train.results.gate_analysis <cache_dir> <out_json> name=eval.json [name=eval.json ...]"""
from __future__ import annotations

import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

GATE_V1_NE_V2 = 0.10     # pass if V1 != V2 on at least this share of rollouts 
GATE_SFT_PRESERVING = 0.80   # NO-GO (distractors too easy) if the SFT policy picks the invariant-preserving repair above this


def completes(o) -> bool:
    """The re-executed job finished: a relaunch completed and the final checkpoint was committed (full step count)."""
    return bool(o.steps_completed > 0 and o.ckpt_valid)


def failing(o, ckpt_every: int = 4) -> list[str]:
    """Every failed V2 recovery requirement of a FixOutcome, in the verifier's own terms (a copy of the red-team's
    all_failing, kept here so this analysis does not depend on that script)."""
    f = []
    if o.tampered:
        f.append("tampered")
    if o.fault_not_injected:
        f.append("fault_not_injected")
    if o.detail.startswith("I6"):
        f.append("world_size")
    if o.steps_completed == 0 and not o.ckpt_valid:
        f.append("no_completed_relaunch")
    elif not o.ckpt_valid:
        f.append("final_ckpt_not_committed")
    if o.lost_steps > ckpt_every:
        f.append("lost_work")
    if not o.ranks_consistent:
        f.append("ranks_inconsistent")
    if not o.samples_match_reference:
        f.append("sample_accounting")
    if not o.transitions_consistent:
        f.append("transitions")
    if not o.params_match_reference:
        f.append("params")
    if not o.restores_consistent:
        f.append("restore_inconsistent")
    return f


def extract(eps: list, comps: list, G: int) -> list[dict]:
    from modal_train.core import dev_episode, score
    from rl.first_update import strip_special
    from traindoctor_env.tiera import parse_answer
    assert len(comps) == G * len(eps), (len(comps), len(eps), G)
    rows = []
    for i, c in enumerate(comps):
        e = eps[i // G]
        assert e[0].kind == c["kind"]
        de = dev_episode(e)
        r2 = score(c["text"], de, "V2")
        a = parse_answer(strip_special(c["text"]), {x.id for x in e[2]})
        rec = dict(episode=i // G, seed=e[0].seed, kind=e[0].kind, valid=r2.valid, v0=bool(r2.v0), v1=bool(r2.v1), v2=bool(r2.v2),
                   chosen_role=None, chosen_ok=False, chosen_completes=False, destructive=bool(r2.unsafe))
        if a is not None:
            ch = next(x for x in e[2] if x.id == a.repair)
            o = e[3][a.repair]
            rec.update(chosen_role=ch.role, chosen_ok=bool(o.ok and not ch.destructive), chosen_completes=completes(o))
        rows.append(rec)
    return rows


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _boot(records: list[dict], stat, n: int = 2000, seed: int = 0):
    """Percentile 95% interval of stat(records) resampling EPISODES (rollouts of one episode are not independent)."""
    by = defaultdict(list)
    for r in records:
        by[r["episode"]].append(r)
    keys = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = stat([r for k in (rng.choice(keys) for _ in keys) for r in by[k]])
        if s is not None:
            vals.append(s)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


def _v1_ne_v2(rs):
    return _mean(r["v1"] and not r["v2"] for r in rs)


def _preserving_given_label(rs):
    return _mean(r["chosen_ok"] for r in rs if r["valid"] and r["v0"])


def aggregate(records: list[dict]) -> dict:
    """The gate numbers for one policy's rollouts. v1_ne_v2: V1 holds and V2 does not (V2 implies V1, so this is the
    V1/V2 disagreement); preserving_given_label_correct: among valid label-correct rollouts, the share whose chosen repair's
    re-executed outcome is a verified recovery."""
    n = len(records)
    per_kind = {}
    for k in sorted({r["kind"] for r in records}):
        rs = [r for r in records if r["kind"] == k]
        per_kind[k] = dict(n=len(rs), v0=_mean(r["v0"] for r in rs), v1=_mean(r["v1"] for r in rs), v2=_mean(r["v2"] for r in rs),
                           v1_ne_v2=_v1_ne_v2(rs), preserving_given_label_correct=_preserving_given_label(rs))
    lc = [r for r in records if r["valid"] and r["v0"]]
    roles = defaultdict(int)
    for r in lc:
        roles[r["chosen_role"]] += 1
    return dict(n=n, episodes=len({r["episode"] for r in records}), valid=_mean(r["valid"] for r in records),
                v0=_mean(r["v0"] for r in records), v1=_mean(r["v1"] for r in records), v2=_mean(r["v2"] for r in records),
                v1_ne_v2=_v1_ne_v2(records), v1_ne_v2_ci95=_boot(records, _v1_ne_v2),
                v1_ne_v2_given_v1=_mean(not r["v2"] for r in records if r["v1"]),
                n_label_correct=len(lc), preserving_given_label_correct=_preserving_given_label(records),
                preserving_ci95=_boot(records, _preserving_given_label),
                label_correct_chose_completing_invalid=_mean(r["chosen_completes"] and not r["chosen_ok"] for r in lc),
                label_correct_chose_role=dict(sorted(roles.items())), per_kind=per_kind)


def verdict(per_policy: dict[str, dict], sft_name: str = "esft_full") -> dict:
    """GO/NO-GO per the Opus criteria: (a) pooled over the trained policies (every policy except base), V1 != V2 on >= 10% of
    rollouts; (b) the SFT policy picks the invariant-preserving repair on <= 80% of its label-correct rollouts. Per-policy
    values are reported next to the pooled one; the decision uses the pooled rate and SFT's rate."""
    trained = [k for k in per_policy if k != "base"]
    tot = sum(per_policy[k]["n"] for k in trained)
    pooled = sum(per_policy[k]["v1_ne_v2"] * per_policy[k]["n"] for k in trained) / tot if tot else None
    sft = per_policy.get(sft_name, {}).get("preserving_given_label_correct")
    a = pooled is not None and pooled >= GATE_V1_NE_V2
    b = sft is not None and sft <= GATE_SFT_PRESERVING
    return dict(a_pooled_v1_ne_v2=pooled, a_pass=a, b_sft_preserving=sft, b_pass=b, go=bool(a and b),
                rule=f"GO iff pooled V1!=V2 >= {GATE_V1_NE_V2} over trained policies AND SFT preserving-given-label-correct <= {GATE_SFT_PRESERVING}")


def main(argv):
    cache, out, evals = Path(argv[0]), Path(argv[1]), [a.split("=", 1) for a in argv[2:]]
    eps = pickle.loads((cache / "dev_eps.pkl").read_bytes())    # written by modal_train.prepare
    per = {}
    for name, path in evals:
        d = json.load(open(path))
        per[name] = aggregate(extract(eps, d["completions"], d["g"]))
        per[name]["prompt_version"] = d.get("prompt_version")
    out.write_text(json.dumps(dict(label="EXPLORATORY dev_container, not verified recovery", per_policy=per, gate=verdict(per)), indent=1))
    print(json.dumps(verdict(per), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
