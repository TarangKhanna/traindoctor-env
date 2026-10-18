"""indep2: out-of-fold leak predictors on menu-only features. usage: indep2_run.py nat.pkl cf.pkl out.json [nperm]"""
import sys, json, pickle, time, hashlib, numpy as np
from multiprocessing import Pool
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from indep2_feats import build, slot_of, SLOTS, TOK
from indep2_models import *

nat = pickle.load(open(sys.argv[1], "rb")); cf = pickle.load(open(sys.argv[2], "rb")); OUT = sys.argv[3]; NPERM = int(sys.argv[4]) if len(sys.argv) > 4 else 200
LABELS = sorted({b["label"] for b in nat}); KINDS_N = sorted({b["kind"] for b in nat})
KC = ["A1v0", "A1v1", "A2", "B_SKIP", "B_DUP", "B_OK"]
def kc(b): return b["kind"] + (f"v{b['variant']}" if b["kind"] == "A1" else "")
def y_slot(b): return SLOTS.index(slot_of(next(i["text"] for i in b["items"] if i["role"] == "correct")))
def y_pos(b): return [i["role"] for i in b["items"]].index("correct")

def shingle_mat(rows, n=4, dim=1 << 14):
    M = np.zeros((len(rows), dim), np.float32)
    for r, b in enumerate(rows):
        t = "\n".join(i["text"] for i in sorted(b["items"], key=lambda x: slot_of(x["text"])))
        for k in range(len(t) - n + 1): M[r, int(hashlib.md5(t[k:k+n].encode()).hexdigest()[:6], 16) % dim] = 1
    return M
def jaccard(M): inter = M @ M.T; s = M.sum(1); return inter / (s[:, None] + s[None, :] - inter + 1e-9)

def report(name, pred, y, base):
    n = len(y); k = int((pred == y).sum()); kb = int((base == y).sum())
    return dict(task=name, n=n, acc=k/n, acc_ci=wilson(k, n), majority_acc=kb/n, majority_ci=wilson(kb, n), margin=k/n - kb/n,
                diff_ci_excludes_zero=bool(wilson(k, n)[0] > kb/n))

def _null(args):
    seed, X, y, g, K = args
    rng = np.random.default_rng(1000 + seed); yp = y[rng.permutation(len(y))]
    return float((oof(lambda: StumpGB(40), X, yp, g, K) == yp).mean())

if __name__ == "__main__":
    t0 = time.time(); res = {}
    Xn, _ = build(nat); gn = np.array([b["group"] for b in nat])
    tasks = {"slot": (np.array([y_slot(b) for b in nat]), 6), "label": (np.array([LABELS.index(b["label"]) for b in nat]), len(LABELS)),
             "kind": (np.array([KINDS_N.index(b["kind"]) for b in nat]), len(KINDS_N)), "correct_pos": (np.array([y_pos(b) for b in nat]), 6)}
    S = jaccard(shingle_mat(nat)); np.fill_diagonal(S, -1)
    for tn, (y, K) in tasks.items():
        base = oof_majority(y, gn, K)
        for fs in ("M", "MO", "MOC"):
            X = Xn[fs][0]
            for mn, fn in (("gb_stumps", lambda: StumpGB(60)), ("rf", lambda: RF(80, 6))):
                res[f"nat.{tn}.{fs}.{mn}"] = report(f"{tn}/{fs}/{mn}", oof(fn, X, y, gn, K), y, base)
                print(res[f"nat.{tn}.{fs}.{mn}"]["task"], round(res[f"nat.{tn}.{fs}.{mn}"]["acc"], 3), round(res[f"nat.{tn}.{fs}.{mn}"]["majority_acc"], 3), int(time.time()-t0), flush=True)
        res[f"nat.{tn}.text.knn"] = report(f"{tn}/text/knn", knn_oof(S, y, gn, K), y, base)
        print(tn, "knn", res[f"nat.{tn}.text.knn"]["acc"], flush=True)
    # counterfactual groups: same seed, every kind
    Xc, _ = build(cf); gc = np.array([b["group"] for b in cf])
    ctasks = {"kind6": (np.array([KC.index(kc(b)) for b in cf]), 6), "label": (np.array([LABELS.index(b["label"]) for b in cf]), len(LABELS)),
              "slot": (np.array([y_slot(b) for b in cf]), 6)}
    Sc = jaccard(shingle_mat(cf)); np.fill_diagonal(Sc, -1)
    for tn, (y, K) in ctasks.items():
        base = oof_majority(y, gc, K)
        for fs in ("M", "MO", "MOC"):
            X = Xc[fs][0]
            for mn, fn in (("gb_stumps", lambda: StumpGB(60)), ("rf", lambda: RF(80, 6))):
                res[f"cf_heldout_groups.{tn}.{fs}.{mn}"] = report(f"CF {tn}/{fs}/{mn}", oof(fn, X, y, gc, K), y, base)
                print(res[f"cf_heldout_groups.{tn}.{fs}.{mn}"]["task"], round(res[f"cf_heldout_groups.{tn}.{fs}.{mn}"]["acc"], 3), round(res[f"cf_heldout_groups.{tn}.{fs}.{mn}"]["majority_acc"], 3), int(time.time()-t0), flush=True)
        res[f"cf_heldout_groups.{tn}.text.knn"] = report(f"CF {tn}/text/knn", knn_oof(Sc, y, gc, K), y, base)
    # train on natural seeds, test on disjoint CF seeds (feature columns are the same names, rebuild jointly)
    Xj, _ = build(nat + cf); nN = len(nat)
    for tn, (yf, K) in (("label", (lambda b: LABELS.index(b["label"]), len(LABELS))), ("slot", (y_slot, 6))):
        yn = np.array([yf(b) for b in nat]); yc = np.array([yf(b) for b in cf]); base = np.full(len(yc), np.bincount(yn).argmax())
        for fs in ("MO",):
            X = Xj[fs][0]
            for mn, m in (("gb_stumps", StumpGB(60)), ("rf", RF(80, 6))):
                p = m.fit(X[:nN], yn, K).predict(X[nN:]); res[f"nat_to_cf.{tn}.{fs}.{mn}"] = report(f"nat->cf {tn}/{fs}/{mn}", p, yc, base)
    # permutation null (label-permutation, same pipeline) for the stump learner on MO features
    X_slot = Xn["MO"][0]; null = {}
    with Pool(4) as pool:
        for tn in ("slot", "label"):
            y, K = tasks[tn]
            obs = float((oof(lambda: StumpGB(40), X_slot, y, gn, K) == y).mean())
            nl = np.array(pool.map(_null, [(s, X_slot, y, gn, K) for s in range(NPERM)]))
            null[tn] = dict(observed=obs, null_mean=float(nl.mean()), null_p95=float(np.quantile(nl, .95)), null_max=float(nl.max()), nperm=NPERM,
                            p_value=float((1 + (nl >= obs).sum()) / (1 + NPERM)))
            print("null", tn, null[tn], int(time.time()-t0), flush=True)
    res["permutation_null_stumps_MO"] = null
    res["meta"] = dict(n_nat=len(nat), n_cf=len(cf), nat_seeds="5000-5799", cf_seeds="7000-7199 x6 kinds", labels=LABELS, seconds=time.time()-t0)
    json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
