"""indep2: which overlap feature carries the signal, and for which kinds. usage: indep2_dissect.py nat.pkl out.json"""
import sys, json, pickle, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from indep2_feats import build, slot_of, SLOTS
from indep2_models import *
nat = pickle.load(open(sys.argv[1], "rb")); X, _ = build(nat); Xm, names = X["MO"]
y = np.array([SLOTS.index(slot_of(next(i["text"] for i in b["items"] if i["role"] == "correct"))) for b in nat]); g = np.array([b["group"] for b in nat])
kinds = np.array([b["kind"] for b in nat]); out = {}
def acc(cols):
    p = oof(lambda: StumpGB(60), Xm[:, cols], y, g, 6); return p, float((p == y).mean())
allc = np.arange(len(names))
groups = {"A_in_cfg/B_in_cfg/bugform/cleanform": [i for i, n in enumerate(names) if any(s in n for s in ("A_in_cfg", "B_in_cfg", "bugform_in_cfg", "cleanform_in_cfg", "first_form_in_cfg", "second_form_in_cfg"))],
          "ctx_in_cfg/tok_in_cfg/new_tok": [i for i, n in enumerate(names) if any(s in n for s in ("ctx_in_cfg", "tok_in_cfg", "new_tok_vs_cfg"))],
          "jac_item_item": [i for i, n in enumerate(names) if "jac_" in n],
          "A_bug/B_bug": [i for i, n in enumerate(names) if n.endswith(".A_bug") or n.endswith(".B_bug")]}
for gname, cols in groups.items():
    p, a = acc(cols); out["only:" + gname] = a
    rest = [i for i in allc if i not in set(cols)]; p2, a2 = acc(rest); out["drop:" + gname] = a2
    print(gname, "only", round(a, 3), "drop", round(a2, 3), flush=True)
p, a = acc(allc); out["all"] = a
out["per_kind_acc_all"] = {k: float((p[kinds == k] == y[kinds == k]).mean()) for k in sorted(set(kinds))}
out["per_kind_n"] = {k: int((kinds == k).sum()) for k in sorted(set(kinds))}
print(out["per_kind_acc_all"])
# simple rule: single-feature stump importance via first-round split of the full model
m = StumpGB(1).fit(Xm, y, 6); out["top_first_round_features"] = [names[j] for j, *_ in m.trees[0]]
print(out["top_first_round_features"])
json.dump(out, open(sys.argv[2], "w"), indent=1)
