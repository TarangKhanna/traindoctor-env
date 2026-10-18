"""indep2: is the overlap signal just the shown config (a section every bundle contains anyway)? config-only vs overlap-only models."""
import sys, json, pickle, re, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from indep2_feats import build, slot_of, SLOTS
from indep2_models import *
nat = pickle.load(open(sys.argv[1], "rb")); cf = pickle.load(open(sys.argv[2], "rb")); res = {}
def cfgvec(rows):
    # every 'path: value' of the shown config, as numeric/ordinal features (value hash for non-numbers) + per-rank mode patterns
    allk = []
    D = []
    for b in rows:
        d = {}; stack = []
        for l in b["config_text"].splitlines():
            ind = len(l) - len(l.lstrip()); k, _, v = l.strip().partition(":"); v = v.strip()
            stack = stack[:ind // 2]; 
            if v == "": stack.append(k); continue
            key = ".".join(stack + [k if not k.isdigit() else "rank"]) 
            try: d[key + "|" + (k if k.isdigit() else "")] = float(v)
            except ValueError: d[key + "|" + (k if k.isdigit() else "") + "=" + v] = 1.0
            if k.isdigit() and v.startswith("part-"): d["shard_mismatch"] = d.get("shard_mismatch", 0) + (v != "part-" + k)
            if v == "reduce_bcast": d["has_reduce_bcast"] = 1.0
        D.append(d); allk += list(d)
    names = sorted(set(allk)); X = np.array([[d.get(n, 0.0) for n in names] for d in D]); keep = X.std(0) > 0
    return X[:, keep]
def ysl(b): return SLOTS.index(slot_of(next(i["text"] for i in b["items"] if i["role"] == "correct")))
for nm, rows in (("nat", nat), ("cf", cf)):
    X = cfgvec(rows); y = np.array([ysl(b) for b in rows]); g = np.array([b["group"] for b in rows]); base = oof_majority(y, g, 6)
    for mn, fn in (("gb_stumps", lambda: StumpGB(60)), ("rf", lambda: RF(80, 6))):
        p = oof(fn, X, y, g, 6); k = int((p == y).sum()); res[f"{nm}.config_only.slot.{mn}"] = dict(acc=k/len(y), ci=wilson(k, len(y)), majority=float((base == y).mean()), nfeat=X.shape[1])
    Xs, _ = build(rows); names = Xs["MO"][1]; Xm = Xs["MO"][0]
    blk = [i for i, n in enumerate(names) if any(s in n for s in ("in_cfg", "new_tok_vs_cfg"))]
    ok = [i for i in range(len(names)) if i not in set(blk)]
    for tag, cols in (("menu_no_config_comparison", ok), ("config_comparison_only", blk)):
        p = oof(lambda: StumpGB(60), Xm[:, cols], y, g, 6); k = int((p == y).sum()); res[f"{nm}.{tag}.slot.gb_stumps"] = dict(acc=k/len(y), ci=wilson(k, len(y)), majority=float((base == y).mean()))
    print(nm, {k: round(v["acc"], 3) for k, v in res.items() if k.startswith(nm)}, flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
