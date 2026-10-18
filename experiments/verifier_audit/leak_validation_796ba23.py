"""Independent validation of commit 796ba23 (repair-menu leak closed?). CPU only, pure Python + numpy, no job executed.
Usage: PYTHONPATH=. python experiments/verifier_audit/leak_validation_796ba23.py  -> experiments/verifier_audit/leak_validation_796ba23.json

Part 1  alternative predictors on menu-only features (repair slot and label), 5-fold CV ACROSS seeds, DEV 10000..10399,
        label-permutation null, held-out counterfactual pairs, menu+config (no records) predictors.
Part 3  edit fidelity: displayed forms (Candidate.shown) vs executed edits (Candidate.edits).
Part 4  id->role hash probe.
Part 2 (bundle-evidence oracle) needs executed records; here only a PARTIAL chooser on documented-format certificate facts.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit.menu_leak import wilson
from jobs.build import ALL_LABELS, LABELS
from jobs.build import build as build_job, materialize
from jobs.config import job_config
from traindoctor_env.bundle import _location, make_menu, padded_config
from traindoctor_env.scenario import Scenario, Truth, job_spec
from traindoctor_env.snapshot import job_files

SEEDS = list(range(10000, 10400))
SLOTS = ["sync", "perrank", "shard", "ckpt", "relaunch", "exclude"]
OUT = Path(__file__).with_suffix(".json")


def slot_of_edits(c) -> str:
    if not c.edits:
        return "relaunch"
    if c.destructive:
        return "exclude"
    return _location(list(c.edits))


def slot_from_text(diff: str) -> str:
    """slot read from the DISPLAYED text only (what a model sees)."""
    if "no change" in diff:
        return "relaunch"
    if "--- job/train.py" in diff:
        return "sync"
    if "--- job/ckpt.py" in diff:
        return "ckpt"
    forms = re.findall(r"^~ (.*)$", diff, re.M)
    if any("world_size" in f for f in forms):
        return "exclude"
    if any("part-" in f for f in forms):
        return "shard"
    return "perrank"


def make_episode(sc: Scenario) -> dict:
    spec = job_spec(sc)
    with tempfile.TemporaryDirectory() as t:
        materialize(Path(t) / "jd", spec.config, spec.bug)
        snap = SimpleNamespace(files=job_files(Path(t) / "jd"), config=job_config(Path(t) / "jd"), seed=sc.seed)
    truth = Truth(sc.seed, sc.kind, sc.variant, LABELS[sc.kind], spec.task, [asdict(e) for e in spec.fix], spec.details)
    menu = make_menu(snap, truth)
    items = [dict(id=c.id, role=c.role, slot=slot_of_edits(c), text=c.diff(snap.files)) for c in menu]
    for it in items:
        assert slot_from_text(it["text"]) == it["slot"], (sc, it["slot"], it["text"][:80])
    correct = [i for i, it in enumerate(items) if it["role"] == "correct"]
    assert len(correct) == 1
    return dict(seed=sc.seed, kind=sc.kind, variant=sc.variant, label=truth.label, items=items,
                target=items[correct[0]]["slot"], cfg=padded_config(snap.files["job/config.yaml"]),
                cands=menu, files=snap.files, details=spec.details, world=sc.world_size)


def natural(seed):
    from traindoctor_env.scenario import make_scenario
    return make_episode(make_scenario(seed))


# ---------------------------------------------------------------- features
def forms_of(text: str) -> list:
    blocks = re.split(r"^~ ---$", "\n".join(x for x in text.splitlines() if x.startswith("~")), flags=re.M)
    return [tuple(x[2:].strip() if x.startswith("~ ") else "" for x in b.splitlines() if x.strip()) for b in blocks]


def item_tokens(it: dict, pos: int, with_numbers=True) -> list:
    t = it["text"]
    num = (lambda s: s) if with_numbers else (lambda s: re.sub(r"\d+", "N", s))
    toks = [f"slot={it['slot']}", f"pos={pos}", f"idc1={it['id'][1]}", f"idlast={it['id'][-1]}",
            f"idd={sum(ch.isdigit() for ch in it['id'])}", f"len={len(t.rstrip())}"]
    fs = forms_of(t)
    for k, f in enumerate(fs):
        for ln in f:
            toks += [f"form{k}:{num(ln)}", f"any:{num(ln)}"]
    toks += [f"set:{num(' | '.join(sorted(' ; '.join(f) for f in fs)))}", f"first:{num(' ; '.join(fs[0]) if fs else '')}"]
    w = re.findall(r"[A-Za-z_]+|\d+", t)
    toks += [f"u:{x}" for x in w] + [f"b:{a}_{b}" for a, b in zip(w, w[1:])]
    c4 = re.sub(r"\d+", "N", t) if not with_numbers else t
    toks += [f"c4:{c4[i:i+4]}" for i in range(0, len(c4) - 3, 2)]
    return toks


def episode_tokens(e: dict, with_cfg=False) -> list:
    toks = []
    for p, it in enumerate(e["items"]):
        for x in item_tokens(it, p):
            if not x.startswith(("pos=", "c4:", "b:", "u:")):
                toks.append(it["slot"] + "/" + x)
            else:
                toks.append(x)
        toks.append(f"pos_of_{it['slot']}={p}")
    if with_cfg:
        cfg = e["cfg"]
        toks += ["cfg:reduce_bcast" if "reduce_bcast" in cfg else "cfg:no_rb",
                 f"cfg:part_world={'part-%d' % e['world'] in cfg}"]
        # does an item's CONTEXT disagree with the bundle config (clean text vs buggy text)?
        cfg_lines = set(x.strip() for x in cfg.splitlines())
        for it in e["items"]:
            ctx = [x[2:].strip() for x in it["text"].splitlines() if x.startswith("  ")]
            bad = [x for x in ctx if re.match(r"\d+: ", x) and x not in cfg_lines]
            toks.append(f"ctxdiff:{it['slot']}:{len(bad) > 0}")
    return toks


class Vocab:
    def __init__(self, docs, min_count=5):
        c = Counter(t for d in docs for t in set(d))
        self.idx = {t: i for i, t in enumerate(sorted(t for t, n in c.items() if n >= min_count))}

    def mat(self, docs):
        X = np.zeros((len(docs), len(self.idx)), dtype=np.float32)
        for r, d in enumerate(docs):
            for t in d:
                j = self.idx.get(t)
                if j is not None:
                    X[r, j] = 1
        return X


# ---------------------------------------------------------------- classifiers (multiclass, numpy)
def fit_nb(X, y, k, alpha=1.0):
    pri = np.log((np.bincount(y, minlength=k) + 1) / (len(y) + k))
    th = np.stack([(X[y == c].sum(0) + alpha) / ((y == c).sum() + 2 * alpha) for c in range(k)])
    return lambda Z: pri + Z @ np.log(th).T + (1 - Z) @ np.log(1 - th).T


def fit_lr(X, y, k, l2=1e-2, it=150, lr=0.5):
    n, d = X.shape
    W, b = np.zeros((d, k)), np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(it):
        Z = X @ W + b
        P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (X.T @ G * 5 + l2 * W); b -= lr * G.sum(0) * 5
    return lambda Z: Z @ W + b


def fit_stumps(X, y, k, rounds=30, eta=0.5):
    """One-vs-rest gradient-boosted decision stumps on binary features (logistic loss)."""
    n, d = X.shape
    F = np.zeros((n, k))
    stumps = []
    Y = np.eye(k)[y]
    for _ in range(rounds):
        P = np.exp(F - F.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        R = Y - P                                             # residual per class
        for c in range(k):
            s1 = X.T @ R[:, c]; s0 = R[:, c].sum() - s1
            n1 = X.sum(0) + 1; n0 = n - X.sum(0) + 1
            gain = s1 ** 2 / n1 + s0 ** 2 / n0
            j = int(np.argmax(gain))
            v1, v0 = eta * s1[j] / n1[j] * 2, eta * s0[j] / n0[j] * 2
            F[:, c] += np.where(X[:, j] > 0, v1, v0)
            stumps.append((c, j, v1, v0))
    def pred(Z):
        out = np.zeros((len(Z), k))
        for c, j, v1, v0 in stumps:
            out[:, c] += np.where(Z[:, j] > 0, v1, v0)
        return out
    return pred


def fit_lookup(docs_train, y, k, keyfn):
    tab = defaultdict(lambda: np.zeros(k))
    for d, t in zip(docs_train, y):
        tab[keyfn(d)][t] += 1
    glob = np.bincount(y, minlength=k)
    return lambda ds: np.stack([tab[keyfn(d)] + 1e-3 * glob if keyfn(d) in tab else glob.astype(float) for d in ds])


def acc_ci(hit, n):
    lo, hi = wilson(int(hit), n)
    return dict(accuracy=hit / n, ci95=[round(lo, 4), round(hi, 4)], n=n)


def cv_eval(eps, target, k, models, folds=5, with_cfg=False, rng_perm=None):
    """5-fold CV across seeds: out-of-fold accuracy of each model on `target(e)` (class index); permutation: labels
    shuffled in TRAIN folds only, scored against the TRUE held-out labels."""
    y_all = np.array([target(e) for e in eps])
    docs = [episode_tokens(e, with_cfg) for e in eps]
    res = {m: 0 for m in models}
    preds = {m: np.zeros(len(eps), int) for m in models}
    for f in range(folds):
        tr = [i for i in range(len(eps)) if i % folds != f]
        te = [i for i in range(len(eps)) if i % folds == f]
        ytr = y_all[tr].copy()
        if rng_perm is not None:
            ytr = rng_perm.permutation(ytr)
        V = Vocab([docs[i] for i in tr])
        Xtr, Xte = V.mat([docs[i] for i in tr]), V.mat([docs[i] for i in te])
        for m, mk in models.items():
            if m.startswith("lookup"):
                key = {"lookup_sigma": lambda d: tuple(sorted(t for t in d if t.startswith(("sync/set:", "perrank/set:", "shard/set:", "ckpt/set:")))),
                       "lookup_nums": lambda d: tuple(sorted(t for t in d if re.search(r"(sync|perrank|shard)/form\d:.*\d", t) and "any" not in t)),
                       "lookup_slotorder": lambda d: tuple(sorted(t for t in d if t.startswith("pos_of_")))}[m]
                p = fit_lookup([docs[i] for i in tr], ytr, k, key)([docs[i] for i in te])
            else:
                p = mk(Xtr, ytr, k)(Xte)
            preds[m][te] = p.argmax(1)
    for m in models:
        res[m] = int((preds[m] == y_all).sum())
    return res, preds, y_all


MODELS = {"majority": None, "nb": fit_nb, "lr": fit_lr, "stumps": fit_stumps,
          "lookup_sigma": None, "lookup_nums": None, "lookup_slotorder": None}


def models_with_majority():
    m = dict(MODELS)
    m["majority"] = lambda X, y, k: (lambda Z: np.tile(np.bincount(y, minlength=k).astype(float), (len(Z), 1)))
    return m


def remap_eval(eps, k, target, preds_oof, y_all, folds=5):
    """Learned confusion remap: predicted class -> most frequent true class, estimated on the TRAIN folds' OOF
    predictions only, applied to held-out. Best-of-both guard against 'below chance is reversible'."""
    out = {}
    for m, p in preds_oof.items():
        hit = 0
        for f in range(folds):
            tr = [i for i in range(len(eps)) if i % folds != f]
            te = [i for i in range(len(eps)) if i % folds == f]
            conf = np.zeros((k, k))
            for i in tr:
                conf[p[i], y_all[i]] += 1
            mp = conf.argmax(1)
            hit += int(sum(mp[p[i]] == y_all[i] for i in te))
        out[m] = hit
    return out


def run_predictors(eps, tag, with_cfg=False, n_perm=100):
    ms = models_with_majority()
    result = {}
    for what, k, tgt in (("repair_slot", len(SLOTS), lambda e: SLOTS.index(e["target"])),
                         ("label", len(ALL_LABELS), lambda e: ALL_LABELS.index(e["label"])),
                         ("kind5", 5, lambda e: ["A1", "A2", "B_SKIP", "B_DUP", "B_OK"].index(e["kind"]))):
        hit, preds, y = cv_eval(eps, tgt, k, ms, with_cfg=with_cfg)
        rm = remap_eval(eps, k, tgt, preds, y)
        n = len(eps)
        maj_cls = Counter(y.tolist()).most_common(1)[0][0]
        res = {m: dict(acc_ci(hit[m], n), remapped=acc_ci(rm[m], n)) for m in ms}
        res["_chance"] = dict(majority_class_frequency=float((y == maj_cls).mean()), uniform=1 / k)
        # permutation null (NB + LR, fast): train on shuffled labels, score vs true held-out labels
        null = {}
        rng = np.random.default_rng(0)
        for m in ("nb", "lr"):
            accs = []
            sub = {m: ms[m]}
            for _ in range(n_perm if m == "nb" else max(10, n_perm // 5)):
                h, _, _ = cv_eval(eps, tgt, k, sub, with_cfg=with_cfg, rng_perm=rng)
                accs.append(h[m] / n)
            null[m] = dict(mean=float(np.mean(accs)), p5=float(np.percentile(accs, 5)), p95=float(np.percentile(accs, 95)),
                           max=float(np.max(accs)), n_perm=len(accs),
                           observed=res[m]["accuracy"], frac_null_ge_observed=float(np.mean(np.array(accs) >= res[m]["accuracy"])))
        res["_perm_null"] = null
        result[what] = res
    return result


# ---------------------------------------------------------------- item-level (single item scored) predictor
def item_level(eps, folds=5):
    """Binary item scorer on item tokens only (NB/LR/stumps), pick argmax per menu; remap = none needed because slot is
    readable from the text (slot-level classifiers above already cover the remap case)."""
    docs = [[item_tokens(it, p) for p, it in enumerate(e["items"])] for e in eps]
    out = {}
    for m, mk in (("nb", fit_nb), ("lr", fit_lr), ("stumps", fit_stumps)):
        hit = 0
        for f in range(folds):
            tr = [i for i in range(len(eps)) if i % folds != f]
            te = [i for i in range(len(eps)) if i % folds == f]
            rows = [(docs[i][p], int(it["role"] == "correct")) for i in tr for p, it in enumerate(eps[i]["items"])]
            V = Vocab([r[0] for r in rows])
            X, y = V.mat([r[0] for r in rows]), np.array([r[1] for r in rows])
            mod = mk(X, y, 2)
            for i in te:
                s = mod(V.mat(docs[i])); s = s[:, 1] - s[:, 0]
                hit += eps[i]["items"][int(np.argmax(s))]["role"] == "correct"
        out[m] = acc_ci(hit, len(eps))
    return out


# ---------------------------------------------------------------- fidelity
def norm(s):
    return re.sub(r"\s+", " ", s.strip())


def fidelity(eps):
    rows = []
    for e in eps:
        for c, it in zip(e["cands"], e["items"]):
            if not c.edits or not c.shown:
                continue
            exec_old = [norm(x) for ed in c.edits for x in ed.old.splitlines()]
            exec_new = [norm(x) for ed in c.edits for x in ed.new.splitlines()]
            shown_clean = [norm(x) for _, cl, bg, _ in c.shown for x in cl]
            shown_bug = [norm(x) for _, cl, bg, _ in c.shown for x in bg]
            if c.role == "correct":     # edit turns the faulty form into the clean form
                direction = (exec_old, exec_new)
            else:                        # decoy edit turns the clean form into a wrong form
                direction = (exec_new, exec_old)
            faulty, clean = direction     # faulty form executed, clean form executed
            issues = []
            if not all(x in clean or x == "" for x in exec_new if False): pass
            ex_clean_in_shown = all(x in shown_clean + shown_bug for x in clean)
            ex_faulty_in_shown = all(x in shown_clean + shown_bug for x in faulty)
            if not (ex_clean_in_shown and ex_faulty_in_shown):
                issues.append("executed text not literally displayed")
            n_sites_shown, n_sites_exec = len(c.shown), len(c.edits)
            if n_sites_shown != n_sites_exec:
                issues.append(f"shows {n_sites_shown} sites, executes {n_sites_exec}")
            rows.append(dict(seed=e["seed"], kind=e["kind"], variant=e["variant"], role=c.role, slot=it["slot"],
                             issues=issues, exec_edit_old=exec_old, exec_edit_new=exec_new))
    return rows


def hash_probe(eps):
    """Is a candidate id a public function of (seed, role)? A program that guesses the seed range recovers the role."""
    ok = 0
    for e in eps[:100]:
        ids = {it["id"]: it["role"] for it in e["items"]}
        hits = {}
        for s in range(10000, 10400):
            for r in ("correct", "noop", "exclude_rank"):
                if "R" + hashlib.sha256(f"{s}:menu:{r}".encode()).hexdigest()[:5] in ids:
                    hits.setdefault(s, []).append(r)
        good = [s for s, rs in hits.items() if len(rs) >= 2]
        ok += (good == [e["seed"]])
    return dict(menus_tested=100, seed_recovered_from_ids_alone=ok,
                note="seed range 10000..10399 only (unknown-range search is a 20-bit id check per candidate role)")


def main():
    eps = [natural(s) for s in SEEDS]
    print("episodes", len(eps), Counter(e["kind"] for e in eps), flush=True)
    res = dict(seeds=[SEEDS[0], SEEDS[-1] + 1], n=len(eps), kind_counts=dict(Counter(e["kind"] for e in eps)))
    res["target_slot_counts"] = dict(Counter(e["target"] for e in eps))
    res["menu_only"] = run_predictors(eps, "menu_only")
    res["menu_item_level"] = item_level(eps)
    res["menu_plus_config_no_records"] = run_predictors(eps, "cfg", with_cfg=True, n_perm=40)
    # counterfactual pairs: same (seed, world, job_seed), all six variants
    pair_eps, mismatch = [], 0
    for s in SEEDS[:200]:
        from traindoctor_env.scenario import make_scenario
        base = make_scenario(s)
        grp = [make_episode(Scenario(s, k, v, base.world_size, base.job_seed)) for k, v in
               (("A1", 0), ("A1", 1), ("A2", 0), ("B_SKIP", 0), ("B_DUP", 0), ("B_OK", 0))]
        pair_eps.append(grp)
        flat = {tuple(sorted(it["text"] for it in g["items"])) for g in grp}
        mismatch += len(flat) < len(grp)
    res["counterfactual"] = dict(n_groups=len(pair_eps),
                                 groups_with_identical_menu_texts_across_variants=mismatch)
    # train on natural seeds, evaluate on held-out pair groups (balanced over the 6 variants): menu-only
    train_eps = eps[200:]
    held = [g for grp in pair_eps for g in grp]   # seeds 10000..10199, disjoint from train_eps
    ms = models_with_majority()
    cf = {}
    for what, k, tgt in (("repair_slot", len(SLOTS), lambda e: SLOTS.index(e["target"])),
                         ("label", len(ALL_LABELS), lambda e: ALL_LABELS.index(e["label"]))):
        ytr = np.array([tgt(e) for e in train_eps]); yte = np.array([tgt(e) for e in held])
        dtr, dte = [episode_tokens(e) for e in train_eps], [episode_tokens(e) for e in held]
        V = Vocab(dtr); Xtr, Xte = V.mat(dtr), V.mat(dte)
        cf[what] = dict(majority_on_balanced_heldout=float(np.bincount(yte).max() / len(yte)))
        for m in ("nb", "lr", "stumps"):
            cf[what][m] = acc_ci(int((ms[m](Xtr, ytr, k)(Xte).argmax(1) == yte).sum()), len(yte))
    res["counterfactual"]["train_natural_seeds_10200_10399__test_pair_groups_seeds_10000_10199"] = cf
    # same symptoms (shared seed) different cause: menus of A1_src vs A2 vs ...: which SLOT texts are byte-identical?
    same = Counter()
    for grp in pair_eps:
        for g in grp[1:]:
            for it in g["items"]:
                same[(it["slot"])] += any(it["text"] == jt["text"] for jt in grp[0]["items"] if jt["slot"] == it["slot"])
    res["counterfactual"]["slot_text_identical_to_A1src_variant_of_same_seed"] = {k: v / (5 * len(pair_eps)) for k, v in same.items()}
    fid = fidelity(eps)
    summ = defaultdict(lambda: Counter())
    for r in fid:
        summ[(r["kind"], r["variant"], r["role"], r["slot"])][tuple(r["issues"])] += 1
    res["fidelity"] = [dict(kind=k[0], variant=k[1], role=k[2], slot=k[3], issues=[list(i) for i in v], counts=list(v.values()))
                       for k, v in sorted(summ.items(), key=str)]
    res["id_hash_probe"] = hash_probe(eps)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
