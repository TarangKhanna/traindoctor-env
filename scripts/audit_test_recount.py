"""Independent recount of the sealed TEST result from raw eval + replay JSONs.
Does not import scripts/analyze_test.py. V0/V1 truth is not available offline
(ground truth lives only in the Modal volume pickle), so V0 uses the label implied
by kind (derived from the scored file, then checked for uniqueness) and V1 is read
from test_scored.json with consistency checks. V2 is recomputed from raw."""
import json, sys, collections, hashlib
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "docs/results/test"
SEEDS = range(6)
POL = ["base", "sft", "csft"] + [f"v0_s{s}" for s in SEEDS] + [f"v2_s{s}" for s in SEEDS]
F4 = ["A1", "A2", "B_SKIP", "B_DUP"]
G = 4

scored = json.load(open(f"{D}/test_scored.json"))
sc = collections.defaultdict(list)
for r in scored["rollouts"]:
    sc[r["policy"]].append(r)

# replay table (dedup by (ep, repair))
rep, dup_conflict, modes = {}, 0, collections.Counter()
for i in range(5):
    for x in json.load(open(f"{D}/test_replay_r{i}.json")):
        k = (x["ep"], x["repair"])
        o = x["outcome"]
        modes[(o.get("exec_mode"), o.get("sandboxed"))] += 1
        rec = dict(ok=bool(o["ok"]), fni=bool(o.get("fault_not_injected")), role=x.get("role"), err=x.get("error"))
        if k in rep and rep[k] != rec:
            dup_conflict += 1
        rep[k] = rec
print("replay pairs", len(rep), "modes", dict(modes), "conflicting duplicates", dup_conflict)
print("fault_not_injected pairs", [(k, v["role"], v["ok"]) for k, v in rep.items() if v["fni"]])
print("replay ok with role!=correct:", sum(1 for v in rep.values() if v["ok"] and v["role"] != "correct"),
      "| correct-role not ok:", sum(1 for v in rep.values() if not v["ok"] and v["role"] == "correct"))
okpe = collections.Counter(k[0] for k, v in rep.items() if v["ok"])
print("episodes with >1 recovering repair:", [e for e, c in okpe.items() if c > 1])

# derive label by kind from scored v0
lab = collections.defaultdict(collections.Counter)

import re
def parse(t, ep=None):
    try:
        j = json.loads(re.sub(r"^\s*<think>\s*</think>", "", t).strip())
        if isinstance(j, dict) and set(j) == {"label", "evidence", "repair", "claims_fixed"} and isinstance(j.get("label"), str) and isinstance(j.get("repair"), str) \
           and isinstance(j.get("evidence"), list) and all(isinstance(x, str) for x in j["evidence"]) and isinstance(j.get("claims_fixed"), bool) \
           and (ep is None or (ep, j["repair"]) in rep):
            return j
    except Exception:
        pass
    return None

R = {}
for p in POL:
    d = json.load(open(f"{D}/test_{p}_eval.json"))["completions"]
    assert len(d) == 400
    R[p] = d
for p in POL:
    for i, (c, s) in enumerate(zip(R[p], sc[p])):
        j = parse(c["text"])
        if j and s["v0"]:
            lab[c["kind"]][j["label"]] += 1
print("labels among scored-V0 rollouts per kind:", {k: dict(v) for k, v in lab.items()})
KL = {k: v.most_common(1)[0][0] for k, v in lab.items()}

rows = {}
mism = collections.Counter()
for p in POL:
    out = []
    for i, c in enumerate(R[p]):
        s = sc[p][i]
        ep = i // G
        assert s["ep"] == ep and s["kind"] == c["kind"], (p, i)
        j = parse(c["text"], ep)
        valid = j is not None
        v0 = valid and j["label"] == KL.get(c["kind"], "?")
        v1 = bool(s["v1"]) and valid  # truth-dependent; from scored
        key = (ep, j["repair"]) if valid else None
        rp = rep.get(key) if key else None
        v2 = bool(v1 and rp and rp["ok"])
        # consistency checks
        if s["v1"] and not v0: mism["v1_without_v0"] += 1
        if s["v2"] and not s["v1"]: mism["v2_without_v1"] += 1
        if bool(s["valid"]) != valid: mism["valid"] += 1; print('VALID DIFF', p, i, repr(c['text'][:200]), s['valid'])
        if bool(s["v0"]) != v0: mism["v0"] += 1
        if bool(s["v2"]) != v2: mism["v2"] += 1
        if s["v2"] and not (rp and rp["ok"]): mism["v2_without_ok_replay"] += 1
        if rp and rp["ok"] and valid and not s["v2"]: mism["ok_replay_not_v2(v1 false)"] += 1
        if valid and key not in rep: mism["valid_no_replay"] += 1
        claimed = valid and j["claims_fixed"]
        out.append(dict(ep=ep, kind=c["kind"], valid=valid, v0=v0, v1=v1, v2=v2, claimed=claimed))
    rows[p] = out
print("mismatches vs scored:", dict(mism))

def rate(rs, k, kinds=None):
    rs = [r for r in rs if kinds is None or r["kind"] in kinds]
    return 100 * sum(r[k] for r in rs) / len(rs)

print("\npolicy | V0 all | V1 all | V2 all | V0 f4 | V1 f4 | V2 f4 | claimed | claimed-not-V2")
for p in POL:
    rs = rows[p]; cl = [r for r in rs if r["claimed"]]
    print(f"{p:6s} {rate(rs,'v0'):5.1f} {rate(rs,'v1'):5.1f} {rate(rs,'v2'):5.1f} | {rate(rs,'v0',F4):5.1f} {rate(rs,'v1',F4):5.1f} {rate(rs,'v2',F4):5.1f} | {len(cl)} {sum(not r['v2'] for r in cl)}")

def eps_mean(p, k, kinds):
    a = collections.defaultdict(list)
    for r in rows[p]:
        if r["kind"] in kinds: a[r["ep"]].append(float(r[k]))
    es = sorted(a)
    return es, np.array([100 * np.mean(a[e]) for e in es])

kinds_of = {r["ep"]: r["kind"] for r in rows["base"]}
def contrast(k, kinds=F4, arms=("v2", "v0")):
    diffs = []
    for s in SEEDS:
        es, a = eps_mean(f"{arms[0]}_s{s}", k, kinds); _, b = eps_mean(f"{arms[1]}_s{s}", k, kinds)
        diffs.append(float((a - b).mean()))
    d = np.array(diffs); n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    # two-sided p for t with df=5 via numeric integration
    from math import gamma, pi, sqrt
    df = n - 1
    c = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    xs = np.linspace(abs(t), 200, 2000001)
    p = 2 * float(np.trapezoid(c * (1 + xs ** 2 / df) ** (-(df + 1) / 2), xs))
    # two-level bootstrap
    rng = np.random.default_rng(20261018)
    mats = {}
    for arm in arms:
        mats[arm] = np.stack([eps_mean(f"{arm}_s{s}", k, kinds)[1] for s in SEEDS])
    es = eps_mean("v0_s0", k, kinds)[0]
    strata = [[i for i, e in enumerate(es) if kinds_of[e] == kk] for kk in kinds]
    M = mats[arms[0]] - mats[arms[1]]
    bs = []
    for _ in range(10000):
        si = rng.integers(0, len(SEEDS), len(SEEDS))
        tot = 0; cnt = 0
        parts = []
        for st in strata:
            ei = rng.choice(st, len(st))
            parts.append(M[np.ix_(si, ei)].mean(axis=1))
        bs.append(float(np.mean(parts, axis=0).mean()))  # equal strata sizes
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d, t, p, lo, hi

for name, k, kinds, arms in [("PRIMARY V2 live, V2arm-V0arm, 4 faulted", "v2", F4, ("v2", "v0")),
                             ("V2 live, all 5 kinds", "v2", F4 + ["B_OK"], ("v2", "v0")),
                             ("V0 label, faulted", "v0", F4, ("v2", "v0")),
                             ("V1 evidence, faulted", "v1", F4, ("v2", "v0"))]:
    d, t, p, lo, hi = contrast(k, kinds, arms)
    print(f"\n{name}: mean {d.mean():+.2f}  per-seed {[round(float(x),1) for x in d]}  t={t:.2f} p={p:.4f}  boot95 [{lo:+.2f},{hi:+.2f}]")

# swap sensitivity & V1-vs-V2 coincidence
print("\nV1 true but V2 false (rollouts, per arm, faulted):",
      {a: sum(1 for s in SEEDS for r in rows[f'{a}_s{s}'] if r['kind'] in F4 and r['v1'] and not r['v2']) for a in ("v0", "v2")})
print("V2 true but V1 false:", sum(1 for p in POL for r in rows[p] if r["v2"] and not r["v1"]))
print("V0-only (V0 true, V1 false) per arm, faulted:",
      {a: sum(1 for s in SEEDS for r in rows[f'{a}_s{s}'] if r['kind'] in F4 and r['v0'] and not r['v1']) for a in ("v0", "v2")})
print("V1-only contrast (V1 true, V0 any) = V1 contrast above. V0-only share of V0-true rollouts, faulted:",
      {a: round(100 * sum(1 for s in SEEDS for r in rows[f'{a}_s{s}'] if r['kind'] in F4 and r['v0'] and not r['v1']) /
                max(1, sum(1 for s in SEEDS for r in rows[f'{a}_s{s}'] if r['kind'] in F4 and r['v0'])), 1) for a in ("v0", "v2")})
# per-kind V2 by arm
print("per-kind V2 by arm (mean over seeds):")
for kk in F4 + ["B_OK"]:
    print(" ", kk, {a: round(float(np.mean([rate(rows[f'{a}_s{s}'], 'v2', [kk]) for s in SEEDS])), 1) for a in ("v0", "v2")},
          "V1", {a: round(float(np.mean([rate(rows[f'{a}_s{s}'], 'v1', [kk]) for s in SEEDS])), 1) for a in ("v0", "v2")},
          "V0", {a: round(float(np.mean([rate(rows[f'{a}_s{s}'], 'v0', [kk]) for s in SEEDS])), 1) for a in ("v0", "v2")})

# repair-choice decomposition: rollout picked the recovering repair (live replay ok), regardless of label/evidence
def chose_ok(p, kinds):
    rs = []
    for i, c in enumerate(R[p]):
        if c["kind"] not in kinds: continue
        j = parse(c["text"], i // G)
        rs.append(bool(j and rep[(i // G, j["repair"])]["ok"]))
    return 100 * sum(rs) / len(rs)
print("\nchose recovering repair (faulted), arm means:",
      {a: round(float(np.mean([chose_ok(f"{a}_s{s}", F4) for s in SEEDS])), 2) for a in ("v0", "v2")},
      "| label&repair but evidence wrong (V0 & ok & !V1):",
      {a: sum(1 for s in SEEDS for i, c in enumerate(R[f"{a}_s{s}"]) if c["kind"] in F4 and rows[f"{a}_s{s}"][i]["v0"] and not rows[f"{a}_s{s}"][i]["v1"]
              and (lambda j: bool(j and rep[(i // G, j['repair'])]["ok"]))(parse(c["text"], i // G))) for a in ("v0", "v2")})
