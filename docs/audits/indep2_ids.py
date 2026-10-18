"""indep2: can seed / role / correct candidate id be recovered from ids without the key? usage: indep2_ids.py nat.pkl out.json
Brute force: for hash formats x keyless guesses x seeds 0..N, count bundles where >=2 of the 6 ids match H(seed,role) (chance ~ 0).
Positive control: with the real key (salt_for) the same harness must recover the roles (so it CAN fail)."""
import sys, json, pickle, hashlib, os, itertools
sys.path.insert(0, os.getcwd())
rows = pickle.load(open(sys.argv[1], "rb"))[:60]
ROLES = ["correct", "noop", "exclude_rank", "decoy_sync", "decoy_perrank", "decoy_shard", "decoy_ckpt"]
obs = [{it["id"] for it in b["items"]} for b in rows]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
def sal(key, seed): return hashlib.sha256(key + b"|salt|" + str(seed).encode()).digest()
FORMATS = {
 "sha256(salt|menu-id|role) [true form, guessed key]": lambda key, s, r: "R" + hashlib.sha256(sal(key, s) + b"|menu-id|" + r.encode()).hexdigest()[:5],
 "sha256(seed|menu-id|role)": lambda key, s, r: "R" + hashlib.sha256(f"{s}|menu-id|{r}".encode()).hexdigest()[:5],
 "sha256(seed:role)": lambda key, s, r: "R" + hashlib.sha256(f"{s}:{r}".encode()).hexdigest()[:5],
 "sha256(role+seed)": lambda key, s, r: "R" + hashlib.sha256(f"{r}{s}".encode()).hexdigest()[:5],
 "md5(seed|role)": lambda key, s, r: "R" + hashlib.md5(f"{s}|{r}".encode()).hexdigest()[:5],
 "sha1(seed|role)": lambda key, s, r: "R" + hashlib.sha1(f"{s}|{r}".encode()).hexdigest()[:5],
 "sha256(sha256(seed)|role)": lambda key, s, r: "R" + hashlib.sha256(hashlib.sha256(str(s).encode()).digest() + b"|menu-id|" + r.encode()).hexdigest()[:5],
}
KEYS = [b"", b"test-suite-key", b"dev", b"key", b"traindoctor", b"td", b"TD_KEY", b"secret", b"password", b"0"*64]
res = {"n_bundles_tested": len(rows), "seed_range": N, "brute_force": {}}
# per-seed lookup index: id -> set of bundles containing it
idx = {}
for bi, ids in enumerate(obs):
    for x in ids: idx.setdefault(x, set()).add(bi)
def scan(fmt, key, seeds):
    hits = []
    for s in seeds:
        c = {}
        for r in ROLES:
            h = FORMATS[fmt](key, s, r)
            for bi in idx.get(h, ()): c[bi] = c.get(bi, 0) + 1
        for bi, n in c.items():
            if n >= 3: hits.append((s, bi, n))     # >=3 matching roles in one bundle: chance ~ (7*6/2^20)^3 per seed*bundle, negligible
    return hits
for fmt in FORMATS:
    keys = KEYS if fmt.startswith("sha256(salt") else [b""]
    tot = 0
    for key in keys: tot += len(scan(fmt, key, range(N if fmt.startswith("sha256(salt") is False else N // 10)))
    res["brute_force"][fmt] = dict(keys_tried=len(keys), hits_ge3_roles=tot)
    print(fmt, tot, flush=True)
# positive control: the real key (TD_KEY from env) with the true seed range around the dev seeds must hit
real = os.environ["TD_KEY"].encode()
pc = scan("sha256(salt|menu-id|role) [true form, guessed key]", real, range(5000, 5060))
res["positive_control_with_real_key"] = dict(bundles_recovered=len({bi for _, bi, _ in pc}), of=len(rows))
print("positive control", res["positive_control_with_real_key"])
# statistical: is the correct id distinguishable by id-only properties? chi-square on first nibble / parity / rank-in-bundle across all bundles
import collections, math
allrows = pickle.load(open(sys.argv[1], "rb"))
def chi(counter, k):
    n = sum(counter.values()); e = n / k; return sum((counter.get(i, 0) - e) ** 2 / e for i in range(k)), k - 1
prop = {"first_hex_nibble": collections.Counter(), "parity": collections.Counter(), "rank_of_id_in_bundle": collections.Counter(), "position_in_menu": collections.Counter()}
for b in allrows:
    for pos, it in enumerate(b["items"]):
        if it["role"] != "correct": continue
        prop["first_hex_nibble"][int(it["id"][1], 16)] += 1; prop["parity"][int(it["id"][1:], 16) % 2] += 1
        prop["rank_of_id_in_bundle"][sorted(x["id"] for x in b["items"]).index(it["id"])] += 1; prop["position_in_menu"][pos] += 1
res["id_property_chi2_of_correct_item"] = {}
for k, c in prop.items():
    K = {"first_hex_nibble": 16, "parity": 2, "rank_of_id_in_bundle": 6, "position_in_menu": 6}[k]; x, df = chi(c, K)
    res["id_property_chi2_of_correct_item"][k] = dict(chi2=x, df=df, note="df+3*sqrt(2df) rule of thumb: leak if chi2 >> " + str(round(df + 3 * math.sqrt(2 * df), 1)))
# seed leakage: do ids of one role repeat/correlate with seed mod k? (same id for same role across seeds would be a constant)
rid = collections.defaultdict(set)
for b in allrows:
    for it in b["items"]: rid[it["role"]].add(it["id"])
res["distinct_ids_per_role"] = {r: len(v) for r, v in rid.items()}; res["n_bundles"] = len(allrows)
print(json.dumps(res["id_property_chi2_of_correct_item"], indent=1))
json.dump(res, open(sys.argv[2], "w"), indent=1)
