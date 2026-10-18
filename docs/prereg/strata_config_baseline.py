"""Config-only baseline and stratum counts. Pure Python (no job executed, no model). TRAIN seeds only for the baseline;
DEV is used ONLY to count scenarios per stratum from the generator (no outcome of any model is read)."""
import json, sys
from collections import Counter
from jobs.build import apply_edits  # noqa: F401  (import check only)
from jobs.config import dumps, loads
from rl.splits import DEV_SEEDS, TRAIN_SEEDS
from traindoctor_env.scenario import job_spec, make_scenario


def variant_name(sc):
    return "A1_cfg" if (sc.kind == "A1" and sc.variant == 1) else "A1_src" if sc.kind == "A1" else sc.kind


def stratum(spec):
    """RULE: config-obvious iff the injected bug edits job/config.yaml (the repair location is itself the shown config)."""
    return "config_obvious" if any(e.path == "job/config.yaml" for e in spec.bug) else "evidence_ambiguous"


def shown_config(spec):
    text = dumps(spec.config)
    for e in spec.bug:
        if e.path == "job/config.yaml":
            assert text.count(e.old) == 1
            text = text.replace(e.old, e.new)
    return loads(text)


def config_rule(cfg):
    """Config-only classifier written from the config semantics, not fitted: deviation in sync mode -> A1; shard dir not
    the rank's own -> A2; otherwise the training-set majority label among config-silent scenarios."""
    if any(v != "all_reduce" for v in cfg["sync"]["per_rank"].values()):
        return "COLLECTIVE_DIVERGENCE"
    if any(d != f"part-{r}" for r, d in cfg["data"]["shard_dirs"].items()):
        return "RANK_STALLED_BEFORE_COLLECTIVE"
    return None


def main():
    out = {"train": {}, "dev_counts": {}}
    rows = []
    for s in TRAIN_SEEDS:
        sc = make_scenario(s); sp = job_spec(sc)
        rows.append((variant_name(sc), stratum(sp), sc.label, config_rule(shown_config(sp))))
    silent = Counter(l for _, _, l, p in rows if p is None)
    fallback = silent.most_common(1)[0][0]   # majority among config-silent TRAIN scenarios
    for st in ("config_obvious", "evidence_ambiguous"):
        sub = [r for r in rows if r[1] == st]
        ok = sum((r[3] or fallback) == r[2] for r in sub)
        out["train"][st] = dict(n=len(sub), label_acc=ok / len(sub), by_variant=dict(Counter(r[0] for r in sub)))
    ok = sum((r[3] or fallback) == r[2] for r in rows)
    out["train"]["overall"] = dict(n=len(rows), label_acc=ok / len(rows), fallback_label=fallback,
                                   majority_label_acc=max(Counter(r[2] for r in rows).values()) / len(rows))
    dev = Counter(); dv = Counter()
    for s in DEV_SEEDS:
        sc = make_scenario(s); dev[stratum(job_spec(sc))] += 1; dv[variant_name(sc)] += 1
    out["dev_counts"] = dict(strata=dict(dev), variants=dict(dv), n=sum(dev.values()))
    json.dump(out, sys.stdout, indent=1, sort_keys=True); print()


main()
