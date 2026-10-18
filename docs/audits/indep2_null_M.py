"""indep2: permutation null (200) for pure-menu features (M: item text/ids/positions, NO config comparison). usage: nat.pkl out.json"""
import sys, json, pickle, numpy as np
from multiprocessing import Pool
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from indep2_feats import build, slot_of, SLOTS
from indep2_models import *
nat = pickle.load(open(sys.argv[1], "rb")); X, names = build(nat)[0]["M"]; g = np.array([b["group"] for b in nat])
LAB = sorted({b["label"] for b in nat})
ys = {"slot": (np.array([SLOTS.index(slot_of(next(i["text"] for i in b["items"] if i["role"] == "correct"))) for b in nat]), 6),
      "label": (np.array([LAB.index(b["label"]) for b in nat]), len(LAB))}
def f(a):
    s, tn = a; y, K = ys[tn]; yp = y[np.random.default_rng(s).permutation(len(y))]
    return float((oof(lambda: StumpGB(40), X, yp, g, K) == yp).mean())
if __name__ == "__main__":
    out = {}
    with Pool(4) as p:
        for tn, (y, K) in ys.items():
            obs = float((oof(lambda: StumpGB(40), X, y, g, K) == y).mean()); base = float((oof_majority(y, g, K) == y).mean())
            nl = np.array(p.map(f, [(s, tn) for s in range(200)]))
            k = int(round(obs * len(y)))
            out[tn] = dict(observed=obs, ci=wilson(k, len(y)), majority=base, null_mean=float(nl.mean()), null_p95=float(np.quantile(nl, .95)), null_max=float(nl.max()),
                           p_value=float((1 + (nl >= obs).sum()) / 201), nperm=200); print(tn, out[tn], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
