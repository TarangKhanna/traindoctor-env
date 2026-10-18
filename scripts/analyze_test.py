"""Sealed-TEST analysis (frozen before any TEST rollout; protocol: evidence/prereg/test_protocol.md).

Input: a scored-rollouts JSON written by `modal_train/cpu_app.py::run_score`:
  {"rollouts": [{policy, ep, kind, valid, v0, v1, v2, claimed, replay, unsafe}, ...], ...}
  v0 = label correct, v1 = label + causal evidence, v2 = live Modal-Sandbox verified recovery (strict gate, evidence, safe).
  replay: "ok" | "error" (replay crash/timeout: counted as unverified-failure) | "none" (invalid output, nothing to replay).
Output: tables + primary analysis (V2-arm minus V0-arm, 6 paired seeds), JSON and Markdown. Pure Python + numpy; no Modal.
Run: PYTHONPATH=. .venv/bin/python scripts/analyze_test.py <scored.json> <out_prefix>
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np

KINDS = ["A1", "A2", "B_SKIP", "B_DUP", "B_OK"]
FAULT_KINDS = ["A1", "A2", "B_SKIP", "B_DUP"]
SEEDS = list(range(6))
MMD = 7.2          # minimum meaningful difference, percentage points (evidence/prereg/power_analysis_amendment.md)
BOOT_N = 10_000
BOOT_SEED = 20261018


# ---- statistics -----------------------------------------------------------------------------------------------
def t_cdf(t: float, df: int) -> float:
    """Student t CDF by Simpson integration of the pdf (no scipy)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    pdf = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)  # noqa: E731
    n = 4000
    h = abs(t) / n
    s = pdf(0) + pdf(abs(t)) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n))
    half = s * h / 3
    return 0.5 + math.copysign(half, t)


def t_crit(df: int, level: float = 0.95) -> float:
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < 0.5 + level / 2 else (lo, mid)
    return (lo + hi) / 2


def paired_t(d: list[float]) -> dict:
    """Paired t over per-seed differences (conditional on the TEST episodes; ignores episode sampling variance)."""
    n, m = len(d), float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("inf") * (1 if m > 0 else -1 if m < 0 else 0)
    p = 2 * (1 - t_cdf(abs(t), n - 1)) if math.isfinite(t) else 0.0
    tc = t_crit(n - 1)
    return dict(mean=m, sd=sd, t=t, df=n - 1, p=p, ci95=[m - tc * se, m + tc * se], n=n)


def classify(lo: float, hi: float, mmd: float = MMD) -> str:
    """Outcome class from the 95% CI [lo, hi] of the V2-arm minus V0-arm difference (points). Fixed in advance."""
    if lo >= mmd:
        return "effect >= minimum meaningful difference (CI lower bound >= MMD)"
    if lo > 0 and hi < mmd:
        return "practically insignificant (effect > 0 but CI entirely below MMD)"
    if hi < mmd:
        return "falsified (CI excludes any positive effect >= MMD)" + ("; V2 arm worse (hi < 0)" if hi < 0 else "")
    if lo > 0:
        return "positive effect, size unresolved (CI excludes 0, upper bound >= MMD)"
    return "inconclusive (underpowered): CI spans 0 and MMD"


def boot_diff(D: np.ndarray, strata: list[list[int]], seeds_axis: bool, rng: np.random.Generator, n: int = BOOT_N) -> list[float]:
    """D: [S, E] (or [1, E]) per-seed per-episode differences in points. Two-level percentile bootstrap: resample the S training
    seeds (if seeds_axis; paired by construction) and, within each kind stratum, the episodes (cluster = episode)."""
    S, _ = D.shape
    out = np.empty(n)
    for b in range(n):
        si = rng.integers(0, S, S) if seeds_axis else np.zeros(1, dtype=int)
        ei = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])
        out[b] = D[np.ix_(si, ei)].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


# ---- aggregation ----------------------------------------------------------------------------------------------
def policies() -> list[str]:
    return ["base", "sft", "csft"] + [f"v0_s{s}" for s in SEEDS] + [f"v2_s{s}" for s in SEEDS]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def policy_table(rolls: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rolls:
        by[r["policy"]].append(r)
    out = {}
    for p, rs in by.items():
        row = dict(n=len(rs), valid=mean(r["valid"] for r in rs))
        for k in ("v0", "v1", "v2"):
            row[k] = mean(r[k] for r in rs)
            row[k + "_fault4"] = mean(r[k] for r in rs if r["kind"] in FAULT_KINDS)
        claimed = [r for r in rs if r["claimed"]]
        row.update(claimed=len(claimed), claimed_and_verified=sum(r["v2"] for r in claimed),
                   claimed_not_verified=sum(not r["v2"] for r in claimed),
                   replay_errors=sum(r["replay"] == "error" for r in rs), unsafe=sum(r["unsafe"] for r in rs))
        row["per_kind"] = {k: {m: mean(r[m] for r in rs if r["kind"] == k) for m in ("v0", "v1", "v2")} for k in KINDS if any(r["kind"] == k for r in rs)}
        out[p] = row
    return out


def ep_matrix(rolls: list[dict], policy: str, key: str, n_eps: int) -> np.ndarray:
    """[n_eps] mean over the rollouts of one policy per episode, in points."""
    acc = defaultdict(list)
    for r in rolls:
        if r["policy"] == policy:
            acc[r["ep"]].append(float(r[key]))
    assert set(acc) == set(range(n_eps)), f"{policy}: episodes {sorted(set(range(n_eps)) - set(acc))[:5]} missing"
    return np.array([100.0 * mean(acc[e]) for e in range(n_eps)])


def analyse(data: dict) -> dict:
    rolls = data["rollouts"]
    kinds = {}
    for r in rolls:
        kinds[r["ep"]] = r["kind"]
    n_eps = max(kinds) + 1
    eps = list(range(n_eps))
    fault = [e for e in eps if kinds[e] in FAULT_KINDS]
    strata_fault = [[e for e in fault if kinds[e] == k] for k in FAULT_KINDS]
    strata_all = [[e for e in eps if kinds[e] == k] for k in KINDS]
    rng = np.random.default_rng(BOOT_SEED)
    res = dict(n_episodes=n_eps, episodes_per_kind={k: sum(1 for e in eps if kinds[e] == k) for k in KINDS},
               policy_table=policy_table(rolls), mmd=MMD, boot_n=BOOT_N, boot_seed=BOOT_SEED)

    def arm(prefix, key):   # [S, E]
        return np.stack([ep_matrix(rolls, f"{prefix}_s{s}", key, n_eps) for s in SEEDS])

    def block(D, strata, seeds_axis=True):
        pooled = float(D.mean())
        lo, hi = boot_diff(D, strata, seeds_axis, rng)
        return dict(diff=pooled, boot_ci95=[lo, hi], n_seed_rows=int(D.shape[0]))

    # PRIMARY: V2 arm minus V0 arm, recovery-verified rate (V2), the 4 faulted kinds, 6 paired seeds
    A2, A0 = arm("v2", "v2"), arm("v0", "v2")
    D = (A2 - A0)[:, fault]
    prim = block(D, [[fault.index(e) for e in st] for st in strata_fault])
    prim["paired_t_over_seeds"] = paired_t([float(x) for x in D.mean(axis=1)])
    prim["per_seed_diff"] = [float(x) for x in D.mean(axis=1)]
    prim["outcome_class"] = classify(*prim["boot_ci95"])
    prim["metric"] = "V2 (live-verified recovery), kinds A1,A2,B_SKIP,B_DUP"
    res["primary"] = prim
    # Sensitivity / secondary descriptive contrasts (not used for the outcome class)
    Dall = A2 - A0
    res["primary_all5_pooled"] = block(Dall, [[e for e in st] for st in strata_all])
    res["primary_all5_pooled"]["paired_t_over_seeds"] = paired_t([float(x) for x in Dall.mean(axis=1)])
    Dv0 = arm("v2", "v0") - arm("v0", "v0")
    res["label_accuracy_diff_v2arm_minus_v0arm_fault4"] = block(Dv0[:, fault], [[fault.index(e) for e in st] for st in strata_fault])
    Dv1 = arm("v2", "v1") - arm("v0", "v1")
    res["evidence_grounded_diff_v2arm_minus_v0arm_fault4"] = block(Dv1[:, fault], [[fault.index(e) for e in st] for st in strata_fault])
    # Each arm vs continued SFT (csft is one run; arm seeds and episodes resampled, csft treated as one fixed policy)
    cs, sft, base = ep_matrix(rolls, "csft", "v2", n_eps), ep_matrix(rolls, "sft", "v2", n_eps), ep_matrix(rolls, "base", "v2", n_eps)
    sec = {}
    for name, A in (("v2_arm", A2), ("v0_arm", A0)):
        sec[f"{name}_minus_csft"] = block((A - cs)[:, fault], [[fault.index(e) for e in st] for st in strata_fault])
    sec["sft_minus_base"] = block((sft - base)[None, fault], [[fault.index(e) for e in st] for st in strata_fault], seeds_axis=False)
    sec["csft_minus_sft"] = block((cs - sft)[None, fault], [[fault.index(e) for e in st] for st in strata_fault], seeds_axis=False)
    res["secondary"] = sec
    # family breakdown (descriptive): Task A = A1+A2, Task B faulted = B_SKIP+B_DUP, B_OK
    fam = {"A": ["A1", "A2"], "B_faulted": ["B_SKIP", "B_DUP"], "B_OK": ["B_OK"]}
    res["by_family_v2arm_minus_v0arm"] = {f: float((A2 - A0)[:, [e for e in eps if kinds[e] in ks]].mean()) for f, ks in fam.items()}
    res["replay_errors_total"] = sum(r["replay"] == "error" for r in rolls)
    return res


def markdown(res: dict) -> str:
    L = ["# Sealed TEST results (EXPLORATORY: two task families; no claim about unseen mechanisms)", "",
         f"Episodes: {res['n_episodes']} ({res['episodes_per_kind']}); G=4 rollouts per episode per policy. V2 = live Modal Sandbox replay (exec_mode=modal_sandbox).", "",
         "| policy | n | valid | V0 label | V1 evid | V2 verified | V2 (4 fault kinds) | claimed | claimed, not verified | replay errors | " + " | ".join(f"V2 {k}" for k in KINDS) + " |",
         "|" + "---|" * (10 + len(KINDS))]
    for p in policies():
        r = res["policy_table"].get(p)
        if not r:
            continue
        f = lambda x: f"{100 * x:.1f}"  # noqa: E731
        L.append(f"| {p} | {r['n']} | {f(r['valid'])} | {f(r['v0'])} | {f(r['v1'])} | {f(r['v2'])} | {f(r['v2_fault4'])} | {r['claimed']} | {r['claimed_not_verified']} | {r['replay_errors']} | "
                 + " | ".join(f(r['per_kind'].get(k, {}).get('v2', float('nan'))) for k in KINDS) + " |")
    pr = res["primary"]
    t = pr["paired_t_over_seeds"]
    L += ["", "## Primary: V2-arm minus V0-arm, live-verified recovery (V2), faulted kinds, 6 paired seeds",
          f"- difference: **{pr['diff']:+.2f} points**; two-level bootstrap (seeds, episodes within kind) 95% CI [{pr['boot_ci95'][0]:+.2f}, {pr['boot_ci95'][1]:+.2f}]",
          f"- paired t over seeds (conditional on these episodes): mean {t['mean']:+.2f}, sd {t['sd']:.2f}, t={t['t']:.2f} (df {t['df']}), p={t['p']:.3f}, 95% CI [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}]",
          f"- per-seed differences: {', '.join(f'{x:+.1f}' for x in pr['per_seed_diff'])}",
          f"- MMD {res['mmd']} points; **outcome class: {pr['outcome_class']}**", "", "## Other contrasts (descriptive)"]
    for k in ("primary_all5_pooled", "label_accuracy_diff_v2arm_minus_v0arm_fault4", "evidence_grounded_diff_v2arm_minus_v0arm_fault4"):
        b = res[k]
        L.append(f"- {k}: {b['diff']:+.2f} [{b['boot_ci95'][0]:+.2f}, {b['boot_ci95'][1]:+.2f}]")
    for k, b in res["secondary"].items():
        L.append(f"- {k} (V2, fault kinds): {b['diff']:+.2f} [{b['boot_ci95'][0]:+.2f}, {b['boot_ci95'][1]:+.2f}]")
    L.append(f"- V2-arm minus V0-arm by family (points, no CI): {json.dumps(res['by_family_v2arm_minus_v0arm'])}")
    L.append(f"- replay crash/timeout rollouts (counted as unverified-failure): {res['replay_errors_total']}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    out = analyse(data)
    open(sys.argv[2] + ".json", "w").write(json.dumps(out, indent=1, sort_keys=True))
    open(sys.argv[2] + ".md", "w").write(markdown(out))
    print(markdown(out))
