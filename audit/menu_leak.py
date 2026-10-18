"""Menu-only leakage: how well do features of the repair menu alone (no records, no logs) predict the correct repair
and the label? Pure Python: menus are built from the buggy job's files and config without executing anything.

RULE (the project rules: statistical, not a string scan): over >= 200 held-out seeds, a lookup learner trained on the other
seeds, using only menu-derived features, must not beat chance (repair: mean fraction of correct candidates per menu;
label: the majority class) by more than z=3 standard errors. Every feature group is asserted, content included: a
menu whose edit shapes identify the cause hands the answer over without reading the incident."""
from __future__ import annotations

import math
import random
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from jobs.build import materialize
from jobs.config import job_config
from traindoctor_env.bundle import _location, make_menu
from traindoctor_env.scenario import Truth, job_spec, make_scenario
from traindoctor_env.snapshot import job_files

DEV_SEEDS = range(10000, 10400)
_KEY = re.compile(r"^[+~] *([A-Za-z_]+):", re.M)
_NUM = re.compile(r"\d+")


def _lines(diff: str, sign: str) -> tuple:
    """The added (+) or removed (-) lines of a menu diff with numbers masked (ranks, steps, shard numbers)."""
    return tuple(_NUM.sub("N", x) for x in diff.splitlines() if x[:1] == sign and x[:3] not in ("+++", "---"))


def _forms(diff: str) -> list:
    """The "~" blocks of a neutral diff (forms separated by '~ ---'), numbers masked, in display order."""
    out = []
    for blk in re.split(r"^~ ---$", "\n".join(x for x in diff.splitlines() if x[:1] == "~" or x == "~ ---"), flags=re.M):
        out.append(tuple(_NUM.sub("N", x) for x in blk.splitlines() if x.strip()))
    return out


def _first_form(diff: str) -> tuple:
    f = _forms(diff)
    return f[0] if f else ()


def _variants(diff: str) -> tuple:
    """The unordered pair of forms: all the information in the display once side and sign are removed."""
    return tuple(sorted(_forms(diff)))


def chance_margin(p: float, n: int, z: float = 3.0) -> float:
    return z * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def episode(seed: int, make_menu_fn=make_menu) -> dict:
    """seed, label, kind, menu [(id, diff)] and the correct ids: no execution, no records."""
    sc = make_scenario(seed)
    spec = job_spec(sc)
    with tempfile.TemporaryDirectory() as t:
        materialize(Path(t) / "jd", spec.config, spec.bug)
        snap = SimpleNamespace(files=job_files(Path(t) / "jd"), config=job_config(Path(t) / "jd"), seed=sc.seed)
    truth = Truth(sc.seed, sc.kind, sc.variant, sc.label, spec.task, [asdict(e) for e in spec.fix], spec.details)
    menu = make_menu_fn(snap, truth)
    slot = "relaunch" if sc.kind == "B_OK" else _location([e for e in spec.fix])
    return dict(seed=seed, kind=sc.kind, label=sc.label, menu=[(c.id, c.diff(snap.files)) for c in menu], slot=slot,
                correct={c.id for c in menu if c.role == "correct"})


def _touched(diff: str) -> tuple:
    head = tuple(sorted(set(re.findall(r"^--- (\S+)", diff, re.M))))
    return head + tuple(sorted(set(_KEY.findall(diff))))


def _cand(menu, i) -> dict:
    cid, diff = menu[i]
    return dict(
        opaque={"position": i, "id_c1": cid[1], "id_c12": cid[1:3], "id_last": cid[-1],
                "id_digits": sum(ch.isdigit() for ch in cid), "diff_len": len(diff)},
        content={"no_change": "no change" in diff, "touched": _touched(diff), "n_plus": diff.count("\n+"),
                 "n_minus": diff.count("\n-"), "rstrip_len": len(diff.rstrip()),
                 "shape": (diff.count("\n+"), diff.count("\n-"), len(_touched(diff))),
                 "added_text": _lines(diff, "+"), "removed_text": _lines(diff, "-"),
                 "first_form_text": _first_form(diff), "variant_set_text": _variants(diff)})


def repair_leakage(eps: list[dict]) -> dict:
    """Per-feature lookup learner (even seeds train, odd seeds test): pick the menu item whose feature value was
    most often correct in training (ties at random)."""
    train, test = [e for e in eps if e["seed"] % 2 == 0], [e for e in eps if e["seed"] % 2 == 1]
    res = {}
    for g in ("opaque", "content"):
        for f in _cand(eps[0]["menu"], 0)[g]:
            hit, tot = defaultdict(int), defaultdict(int)
            for e in train:
                for i, (cid, _) in enumerate(e["menu"]):
                    v = _cand(e["menu"], i)[g][f]
                    tot[v] += 1
                    hit[v] += cid in e["correct"]
            acc = 0.0
            for e in test:
                rng = random.Random(f"leak:{e['seed']}")
                sc = [(hit[v] / tot[v] if tot[v] else 0.0, rng.random(), cid)
                      for i, (cid, _) in enumerate(e["menu"]) for v in [_cand(e["menu"], i)[g][f]]]
                acc += max(sc)[2] in e["correct"]
            n = len(test)
            # baseline = the best constant-location guess ("always the checkpoint repair"), the repair-side analogue
            # of the majority label: the class prior is not menu leakage, so it is the bar a feature must clear
            chance = max(sum(len(e["correct"]) / len(e["menu"]) for e in test) / n,
                         max(Counter(e["slot"] for e in test).values()) / n)
            res[f"{g}.{f}"] = dict(group=g, accuracy=acc / n, chance=chance, n=n, margin=chance_margin(chance, n),
                                   above_chance=acc / n - chance)
    return res


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _ngrams(diff: str, n: int = 4) -> Counter:
    """Character n-grams of the menu text only (ids removed, digits masked, so ids/ranks are not features)."""
    t = _NUM.sub("N", diff)
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def _train_lr(rows, epochs: int = 40, lr: float = 0.5, l2: float = 1e-3) -> dict:
    """Logistic regression on sparse n-gram counts (log1p), plain SGD; rows = [(Counter, 0/1)]."""
    w: dict = defaultdict(float)
    b = 0.0
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(rows)
        for x, y in rows:
            z = b + sum(w[g] * math.log1p(c) for g, c in x.items())
            pr = 1 / (1 + math.exp(-max(min(z, 30), -30)))
            err = pr - y
            b -= lr * err * 0.1
            for g, c in x.items():
                w[g] -= lr * (err * math.log1p(c) + l2 * w[g]) * 0.1
    return dict(w=w, b=b)


def ngram_lr_leakage(eps: list[dict], folds: int = 5) -> dict:
    """Second, independent predictor: a character n-gram logistic regression on menu text only, cross-validated
    ACROSS SEEDS (a seed's menu is never in its own training folds). In each held-out menu it picks the item with the
    highest score; baseline = max(fraction correct per menu, best constant location), as for the lookup learner."""
    hit = n = 0
    base = 0.0
    slots = Counter(e["slot"] for e in eps)
    for k in range(folds):
        tr = [e for i, e in enumerate(eps) if i % folds != k]
        te = [e for i, e in enumerate(eps) if i % folds == k]
        rows = [(_ngrams(d), int(cid in e["correct"])) for e in tr for cid, d in e["menu"]]
        m = _train_lr(rows)
        for e in te:
            sc = []
            for cid, d in e["menu"]:
                x = _ngrams(d)
                sc.append((m["b"] + sum(m["w"].get(g, 0.0) * math.log1p(c) for g, c in x.items()),
                           random.Random(f"lr:{e['seed']}:{cid}").random(), cid))
            hit += max(sc)[2] in e["correct"]
            n += 1
            base += len(e["correct"]) / len(e["menu"])
    chance = max(base / n, max(slots.values()) / len(eps))
    lo, hi = wilson(hit, n)
    return dict(accuracy=hit / n, ci95=[lo, hi], chance=chance, n=n, margin=chance_margin(chance, n),
                above_chance=hit / n - chance)


def label_leakage(eps: list[dict]) -> dict:
    def feats(e):
        m = e["menu"]
        return dict(
            opaque={"first_chars": "".join(sorted(c[1] for c, _ in m)), "n_items": len(m),
                    "padded_lengths": tuple(sorted(len(d) for _, d in m))},
            content={"touched_keys": tuple(sorted({t for _, d in m for t in _touched(d)})),
                     "stripped_lengths": tuple(sorted(len(d.rstrip()) for _, d in m)),
                     "shapes": tuple(sorted(_cand(m, i)["content"]["shape"] for i in range(len(m))))})
    train, test = [e for e in eps if e["seed"] % 2 == 0], [e for e in eps if e["seed"] % 2 == 1]
    maj = Counter(e["label"] for e in train).most_common(1)[0][0]
    base = sum(e["label"] == maj for e in test) / len(test)
    res = {}
    for g in ("opaque", "content"):
        for f in feats(eps[0])[g]:
            tab: dict = defaultdict(Counter)
            for e in train:
                tab[feats(e)[g][f]][e["label"]] += 1
            ok = sum((tab[feats(e)[g][f]].most_common(1)[0][0] if tab[feats(e)[g][f]] else maj) == e["label"] for e in test)
            res[f"{g}.{f}"] = dict(group=g, accuracy=ok / len(test), majority=base, n=len(test),
                                   margin=chance_margin(base, len(test)), above_majority=ok / len(test) - base)
    return res


SLOTS = ("sync", "perrank", "shard", "ckpt", "relaunch", "exclude")


def _slot_of_text(diff: str) -> str:
    """The repair slot, read from the DISPLAYED text only."""
    if "no change" in diff:
        return "relaunch"
    if "--- job/train.py" in diff:
        return "sync"
    if "--- job/ckpt.py" in diff:
        return "ckpt"
    forms = re.findall(r"^~ (.*)$", diff, re.M)
    if any("world_size" in f for f in forms):
        return "exclude"
    return "shard" if any("part-" in f for f in forms) else "perrank"


def _tokens(e: dict) -> list:
    """Menu-only tokens, numbers kept: per slot, the words and word pairs of the item's whole text, plus slot order."""
    toks = []
    for pos, (_, diff) in enumerate(e["menu"]):
        slot = _slot_of_text(diff)
        w = re.findall(r"[A-Za-z_]+|\d+", diff)
        toks += [f"{slot}/u:{x}" for x in w] + [f"{slot}/b:{a}_{b}" for a, b in zip(w, w[1:])] + [f"pos_of_{slot}={pos}"]
    return toks


def _stumps(X, y, k, rounds=30, eta=0.5):
    """One-vs-rest gradient-boosted decision stumps on binary features (the predictor that found the buggy-context leak)."""
    import numpy as np
    n = len(X)
    F, Y, stumps = np.zeros((n, k)), np.eye(k)[y], []
    for _ in range(rounds):
        P = np.exp(F - F.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        R = Y - P
        for c in range(k):
            s1 = X.T @ R[:, c]
            s0 = R[:, c].sum() - s1
            n1, n0 = X.sum(0) + 1, n - X.sum(0) + 1
            j = int(np.argmax(s1 ** 2 / n1 + s0 ** 2 / n0))
            v1, v0 = eta * s1[j] / n1[j] * 2, eta * s0[j] / n0[j] * 2
            F[:, c] += np.where(X[:, j] > 0, v1, v0)
            stumps.append((c, j, v1, v0))

    def pred(Z):
        out = np.zeros((len(Z), k))
        for c, j, v1, v0 in stumps:
            out[:, c] += np.where(Z[:, j] > 0, v1, v0)
        return out
    return pred


def stumps_leakage(eps: list[dict], folds: int = 5) -> dict:
    """RULE: boosted stumps on menu-only tokens, cross-validated ACROSS seeds, must not beat the majority class by more
    than z=3 standard errors, for the repair slot and the label. (The config legitimately reveals A1-cfg and A2: that is
    evidence in the bundle, not a menu feature, so it is not an input here.)"""
    import numpy as np
    docs = [_tokens(e) for e in eps]
    out = {}
    for what, classes, tgt in (("repair_slot", SLOTS, [e["slot"] for e in eps]),
                               ("label", sorted({e["label"] for e in eps}), [e["label"] for e in eps])):
        y = np.array([classes.index(t) for t in tgt])
        hit = 0
        for f in range(folds):
            tr = [i for i in range(len(eps)) if i % folds != f]
            te = [i for i in range(len(eps)) if i % folds == f]
            cnt = Counter(t for i in tr for t in set(docs[i]))
            vocab = {t: j for j, t in enumerate(sorted(t for t, c in cnt.items() if c >= 5))}

            def mat(idx):
                X = np.zeros((len(idx), len(vocab)), dtype=np.float32)
                for r, i in enumerate(idx):
                    for t in docs[i]:
                        if t in vocab:
                            X[r, vocab[t]] = 1
                return X
            pred = _stumps(mat(tr), y[tr], len(classes))(mat(te))
            hit += int((pred.argmax(1) == y[te]).sum())
        base = Counter(y.tolist()).most_common(1)[0][1] / len(y)
        acc = hit / len(y)
        out[what] = dict(accuracy=acc, majority=base, n=len(y), ci95=wilson(hit, len(y)),
                         margin=chance_margin(base, len(y)), above_majority=acc - base)
    return out


def id_recovery(eps: list[dict], probes: int = 50) -> dict:
    """RULE: a candidate id must not be a function of public data. Tries the OLD public formula
    sha256(f"{seed}:menu:{role}") over the seed range for the correct id of each menu."""
    import hashlib
    roles = ("correct", "noop", "exclude_rank")
    ok = 0
    for e in eps[:probes]:
        ids = {c for c, _ in e["menu"]}
        seeds = [s for s in DEV_SEEDS
                 if sum("R" + hashlib.sha256(f"{s}:menu:{r}".encode()).hexdigest()[:5] in ids for r in roles) >= 2]
        ok += seeds == [e["seed"]]
    return dict(menus=min(probes, len(eps)), recovered=ok)


def report(seeds=DEV_SEEDS, make_menu_fn=make_menu) -> dict:
    eps = [episode(s, make_menu_fn) for s in seeds]
    return dict(n_episodes=len(eps), repair_leakage=repair_leakage(eps), label_leakage=label_leakage(eps),
                ngram_lr=ngram_lr_leakage(eps), stumps=stumps_leakage(eps), id_recovery=id_recovery(eps))


def violations(rep: dict) -> list[str]:
    """Every feature (opaque or content) whose holdout accuracy exceeds its baseline by more than its margin."""
    out = [f"repair:{k}" for k, v in rep["repair_leakage"].items() if v["above_chance"] > v["margin"]]
    lr = rep["ngram_lr"]
    out += ["repair:ngram_lr"] if lr["above_chance"] > lr["margin"] else []
    out += [f"stumps:{k}" for k, v in rep["stumps"].items() if v["above_majority"] > v["margin"]]
    out += ["id_recovery"] if rep["id_recovery"]["recovered"] else []
    return out + [f"label:{k}" for k, v in rep["label_leakage"].items() if v["above_majority"] > v["margin"]]


if __name__ == "__main__":
    import json
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/verifier_audit/menu_leak_dev400.json")
    rep = report()
    rep["violations"] = violations(rep)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(out, "violations:", rep["violations"])
