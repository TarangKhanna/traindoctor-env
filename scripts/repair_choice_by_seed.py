#!/usr/bin/env python3
"""Per-policy share of faulted rollouts that chose the repair that actually recovers.

Rule: a rollout "chose the recovering repair" iff its output parses to the required JSON schema and the (episode, repair)
pair it names has an ok live sandbox replay (evidence/results/test/test_replay_r*.json). Denominator: 320 faulted rollouts
(A1, A2, B_SKIP, B_DUP; 80 episodes x 4 samples) per policy. Same parser as scripts/audit_test_recount.py.
Writes evidence/results/test/AUDIT_RECOUNT.json and asserts the arm means equal the 71.41 / 71.46 printed in AUDIT_RECOUNT.txt.
"""
import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "evidence/results/test"
SEEDS = range(6)
F4 = {"A1", "A2", "B_SKIP", "B_DUP"}
G = 4

rep = {}
for i in range(5):
    for x in json.loads((D / f"test_replay_r{i}.json").read_text()):
        rep[(x["ep"], x["repair"])] = bool(x["outcome"]["ok"])


def parse(t, ep):
    try:
        j = json.loads(re.sub(r"^\s*<think>\s*</think>", "", t).strip())
        if (isinstance(j, dict) and set(j) == {"label", "evidence", "repair", "claims_fixed"} and isinstance(j["label"], str)
                and isinstance(j["repair"], str) and isinstance(j["evidence"], list)
                and all(isinstance(x, str) for x in j["evidence"]) and isinstance(j["claims_fixed"], bool)
                and (ep, j["repair"]) in rep):
            return j
    except Exception:
        pass
    return None


def chose_ok(policy):
    hit = n = 0
    for i, c in enumerate((json.loads((D / f"test_{policy}_eval.json").read_text()))["completions"]):
        if c["kind"] not in F4:
            continue
        n += 1
        j = parse(c["text"], i // G)
        hit += bool(j and rep[(i // G, j["repair"])])
    assert n == 320, (policy, n)
    return 100 * hit / n


out = {"definition": __doc__.strip().splitlines()[0], "denominator_per_policy": 320, "chose_recovering_repair_pct": {}}
for arm in ("v0", "v2"):
    for s in SEEDS:
        out["chose_recovering_repair_pct"][f"{arm}_s{s}"] = chose_ok(f"{arm}_s{s}")
vals = out["chose_recovering_repair_pct"]
out["arm_mean_pct"] = {a: sum(vals[f"{a}_s{s}"] for s in SEEDS) / 6 for a in ("v0", "v2")}

txt = (D / "AUDIT_RECOUNT.txt").read_text()
m = re.search(r"chose recovering repair \(faulted\), arm means: \{'v0': ([\d.]+), 'v2': ([\d.]+)\}", txt)
assert m, "arm-means line not found in AUDIT_RECOUNT.txt"
assert round(out["arm_mean_pct"]["v0"], 2) == float(m[1]) == 71.41, (out["arm_mean_pct"], m.groups())
assert round(out["arm_mean_pct"]["v2"], 2) == float(m[2]) == 71.46, (out["arm_mean_pct"], m.groups())
(D / "AUDIT_RECOUNT.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(json.dumps(out["arm_mean_pct"]), "ok")
