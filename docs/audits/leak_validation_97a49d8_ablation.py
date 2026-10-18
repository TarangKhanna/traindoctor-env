"""Ablation for the 97a49d8 validation: which menu text carries the residual menu-only signal? Run with PYTHONPATH=.:docs/audits."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import leak_validation_97a49d8 as L

eps = [L.natural(s) for s in L.SEEDS]
out = {}
# (a) where does `reduce_bcast` appear OUTSIDE the "~" forms (i.e. in context lines), per slot, per kind?
where = Counter()
for e in eps:
    for it in e["items"]:
        ctx = "\n".join(x for x in it["text"].splitlines() if not x.startswith("~"))
        if "reduce_bcast" in ctx:
            where[(e["kind"] + ("_cfg" if e["variant"] and e["kind"] == "A1" else ""), it["slot"])] += 1
out["reduce_bcast_in_context_lines"] = {f"{k[0]}|{k[1]}": v for k, v in where.items()}
# (b) rerun the strongest predictors with every token touching reduce_bcast removed
def strip(e):
    """drop the two items (no-change relaunch, exclude-rank) whose CONTEXT is drawn from the buggy job text"""
    e2 = dict(e)
    e2["items"] = [it for it in e["items"] if it["slot"] not in ("relaunch", "exclude")]
    return e2
ms = L.models_with_majority()
for name, tr in (("original", lambda e: e), ("without_relaunch_and_exclude_items", strip)):
    ee = [tr(e) for e in eps]
    sub = {}
    for what, k, tgt in (("repair_slot", 6, lambda e: L.SLOTS.index(e["target"])), ("label", 4, lambda e: L.ALL_LABELS.index(e["label"]))):
        hit, preds, y = L.cv_eval(ee, tgt, k, {m: ms[m] for m in ("majority", "nb", "stumps")})
        sub[what] = {m: L.acc_ci(hit[m], len(ee)) for m in hit}
    out[name] = sub
# (c) A1_cfg detection from the menu alone
flag = [("reduce_bcast" in "\n".join(it["text"] for it in e["items"] if it["slot"] in ("relaunch", "exclude"))) for e in eps]
truth = [e["kind"] == "A1" and e["variant"] == 1 for e in eps]
out["A1cfg_flag_from_noop_exclude_context"] = dict(tp=sum(f and t for f, t in zip(flag, truth)), fp=sum(f and not t for f, t in zip(flag, truth)),
                                                   n_A1cfg=sum(truth), n=len(eps))
Path("docs/audits/leak_validation_97a49d8_ablation.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(json.dumps(out, indent=1, sort_keys=True))
