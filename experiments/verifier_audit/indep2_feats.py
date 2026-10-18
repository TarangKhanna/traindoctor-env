"""indep2: hand-built menu-only features (written independently of the fix author's tests)."""
import re, numpy as np, hashlib

SLOTS = ["sync", "perrank", "shard", "ckpt", "relaunch", "exclude"]

def slot_of(text):
    if text.startswith("no change"):
        return "relaunch"
    head = text.splitlines()[0]
    if "train.py" in head: return "sync"
    if "ckpt.py" in head: return "ckpt"
    til = "\n".join(l for l in text.splitlines() if l.startswith("~"))
    if "world_size" in til: return "exclude"
    if "reduce" in til: return "perrank"
    return "shard"

def forms(text):
    til = [l[2:] for l in text.splitlines() if l.startswith("~")]
    if "---" in til:
        i = til.index("---"); return til[:i], til[i+1:]
    return til, []

def bug_form(slot, f):
    j = "\n".join(f)
    if slot == "sync": return "rank ==" in j
    if slot == "perrank": return "reduce_bcast" in j
    if slot == "ckpt": return ("_progress(" in j) or ("sampler=" not in j and "state.update" in j)
    if slot == "shard":
        m = re.search(r"(\d+): part-(\d+)", j); return bool(m) and m.group(1) != m.group(2)
    return None

def ngr(s, n=3, dim=96):
    v = np.zeros(dim)
    for i in range(len(s) - n + 1):
        v[int(hashlib.md5(s[i:i+n].encode()).hexdigest()[:6], 16) % dim] += 1
    return v

TOK = re.compile(r"[A-Za-z_]+|\d+")

def item_feats(b, i, vocab):
    it = b["items"][i]; t = it["text"]; slot = slot_of(t)
    lines = t.splitlines(); cfg = {l.strip() for l in b["config_text"].splitlines()}
    A, B = forms(t); body = [l for l in lines if not l.startswith("~") and not l.startswith("---") and set(l.strip()) != {"."}]
    firstt = next((k for k, l in enumerate(lines) if l.startswith("~")), 0)
    M = {}   # pure item features
    M["pos"] = i
    for s in SLOTS: M["slot_" + s] = float(slot == s)
    M["nlines"] = len(lines); M["ntilde"] = sum(l.startswith("~") for l in lines)
    M["pre_ctx"] = firstt - (1 if lines[0].startswith("---") else 0)
    M["post_ctx"] = len(body) - M["pre_ctx"] - (0 if not lines[0].startswith("---") else 0)
    M["strip_len"] = len(t.rstrip(".\n "))
    M["dots"] = len(t) - len(t.rstrip("."))
    M["lenA"], M["lenB"] = len(A), len(B)
    M["A_bug"] = float(bool(bug_form(slot, A))); M["B_bug"] = float(bool(bug_form(slot, B)))
    nums = [int(x) for x in re.findall(r"\d+", " ".join(A + B))]
    M["maxnum"] = max(nums) if nums else -1; M["nnum"] = len(nums)
    h = it["id"]; v = int(h[1:], 16)
    M["id_val"] = v / 2**20; M["id_par"] = v % 2; M["id_m3"] = v % 3; M["id_m6"] = v % 6; M["id_nib"] = int(h[1], 16)
    M["id_digits"] = sum(c.isdigit() for c in h[1:])
    M["id_rank"] = sorted(x["id"] for x in b["items"]).index(h)
    toks = set(TOK.findall(t))
    V = {("tok_" + w): float(w in toks) for w in vocab}
    NG = {f"ng{k}": x for k, x in enumerate(ngr(t))}
    # overlap features (with other bundle sections)
    O = {}
    inA = [l.strip() in cfg for l in A]; inB = [l.strip() in cfg for l in B]
    O["A_in_cfg"] = float(np.mean(inA)) if inA else 0; O["B_in_cfg"] = float(np.mean(inB)) if inB else 0
    O["bugform_in_cfg"] = float((M["A_bug"] and O["A_in_cfg"] == 1) or (M["B_bug"] and O["B_in_cfg"] == 1))
    O["cleanform_in_cfg"] = float((not M["A_bug"] and slot not in ("relaunch",) and O["A_in_cfg"] == 1 and bool(A)) or (not M["B_bug"] and O["B_in_cfg"] == 1 and bool(B)))
    O["first_form_in_cfg"] = O["A_in_cfg"]; O["second_form_in_cfg"] = O["B_in_cfg"]
    ctx = [l.strip() for l in body if l.strip() and not l.startswith("no change")]
    O["ctx_in_cfg"] = float(np.mean([c in cfg for c in ctx])) if ctx else 0
    mine = set(l.strip() for l in lines if l.strip() and set(l.strip()) != {"."})
    js = []
    for j, o in enumerate(b["items"]):
        if j == i: continue
        other = set(l.strip() for l in o["text"].splitlines() if l.strip() and set(l.strip()) != {"."})
        js.append(len(mine & other) / max(1, len(mine | other)))
    O["jac_mean"] = float(np.mean(js)); O["jac_max"] = max(js); O["jac_min"] = min(js)
    ctoks = set(TOK.findall(b["config_text"]))
    O["tok_in_cfg"] = float(np.mean([w in ctoks for w in toks])) if toks else 0
    O["new_tok_vs_cfg"] = len(toks - ctoks)
    return slot, M, V, NG, O

def config_feats(b):
    c = b["config_text"]
    return dict(world=b["world"], n_reduce=c.count("reduce_bcast"), n_perrank=len(re.findall(r"^\s+\d+: (all_reduce|reduce_bcast)", c, re.M)),
                part_mismatch=sum(a != b_ for a, b_ in re.findall(r"(\d+): part-(\d+)", c)),
                nlines=len(c.splitlines()))

def build(rows, vocab=None):
    """returns dict of feature-set name -> (X, names), y arrays, groups."""
    if vocab is None:
        from collections import Counter
        df = Counter()
        for b in rows:
            for it in b["items"]: df.update(set(TOK.findall(it["text"])))
        vocab = [w for w, c in df.most_common(120)]
    sets = {"M": [], "MO": [], "MOC": []}; names = None
    for b in rows:
        per = {}
        for i in range(len(b["items"])):
            slot, M, V, NG, O = item_feats(b, i, vocab); per[slot] = (M, V, NG, O)
        vM, vO, vC = {}, {}, {}
        for s in SLOTS:
            M, V, NG, O = per[s]
            for k, x in {**M, **V, **NG}.items(): vM[f"{s}.{k}"] = x
            for k, x in O.items(): vO[f"{s}.{k}"] = x
        for k, x in config_feats(b).items(): vC[k] = x
        sets["M"].append(vM); sets["MO"].append({**vM, **vO}); sets["MOC"].append({**vM, **vO, **vC})
    out = {}
    for k, lst in sets.items():
        nm = list(lst[0]); X = np.array([[d[n] for n in nm] for d in lst], float)
        keep = X.std(0) > 0; out[k] = (X[:, keep], [n for n, kk in zip(nm, keep) if kk])
    return out, vocab
