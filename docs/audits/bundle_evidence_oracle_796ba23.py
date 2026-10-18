"""PARTIAL bundle-evidence-only chooser (item 2 of the 796ba23 validation). No Gloo: real records cannot be produced here, so the
certificate facts are SYNTHESIZED in the documented record format (snapshot.collect) from the scenario details. The chooser then sees
ONLY: those record texts, the padded config text and the rendered menu text (ids + displayed diffs); never roles, slots or Candidate
objects. What this tests: is the displayed menu sufficient, given the diagnostic facts, to pick the repair and never a decoy, for every
variant? What it does NOT test: that the real records in the 2048-token bundle carry those facts (needs executed snapshots)."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from leak_validation_796ba23 import SEEDS, forms_of, make_episode, natural  # noqa: E402
from audit.menu_leak import wilson  # noqa: E402


def synth_records(e):
    d, w = e["details"], e["world"]
    k, r, s = e["kind"], d["rank"], d.get("step")
    recs = []
    if k == "A1":
        op = d["bad_op"]
        recs += [f"[run1] rank {r} step={s} seq=7 {op} bucket=0 numel=12 pending"] + \
                [f"[run1] rank {q} step={s} seq=7 all_reduce bucket=0 numel=12 pending" for q in range(w) if q != r]
    elif k == "A2":
        recs += [f"[run1] rank {r} INFO waiting for shard 1"] + \
                [f"[run1] rank {q} step={s} seq=7 all_reduce bucket=0 numel=12 pending" for q in range(w) if q != r]
    else:
        recs += [f"[run1] rank {r} step=3 samples=[1, 2, 3, 4]", f"[run2] rank {r} step=3 samples={'[1, 2, 3, 4]' if k == 'B_OK' else '[5, 6, 7, 8]'}"]
    return recs


def choose(recs, cfg, menu):
    """menu = [(id, displayed_text)]; returns chosen id. Uses only displayed text + record text."""
    txt = "\n".join(recs)
    items = [(i, t, forms_of(t)) for i, t in menu]
    def has(t, rx): return re.search(rx, t) is not None
    m = re.search(r"rank (\d+) step=(\d+) seq=\d+ (broadcast|reduce) ", txt)
    if m:
        r, s, op = m.group(1), m.group(2), m.group(3)
        if op == "broadcast":      # sync-code item whose displayed rank-keyed form names this rank and step
            c = [i for i, t, f in items if "--- job/train.py" in t and has(t, rf"rank == {r} and step >= {s} ")]
        else:                      # config item with a `R: all_reduce` / `R: reduce_bcast` pair for this rank
            c = [i for i, t, f in items if "--- job/config.yaml" in t and has(t, rf"~ +{r}: reduce_bcast") and has(t, rf"~ +{r}: all_reduce")]
        return c[0] if len(c) == 1 else ("AMBIG", c)
    m = re.search(r"rank (\d+) INFO waiting for shard", txt)
    if m:
        r = m.group(1)
        c = [i for i, t, f in items if "--- job/config.yaml" in t and has(t, rf"~ +{r}: part-{r}\b") and has(t, rf"~ +{r}: part-\d+")
             and not has(t, rf"~ +{r}: reduce_bcast")]
        return c[0] if len(c) == 1 else ("AMBIG", c)
    sm = re.findall(r"\[run\d\] rank \d+ step=\d+ samples=(.*)", txt)
    if len(sm) == 2:
        if sm[0] == sm[1]:
            return [i for i, t, f in items if "no change" in t][0]
        c = [i for i, t, f in items if "--- job/ckpt.py" in t]
        return c[0] if len(c) == 1 else ("AMBIG", c)
    return ("NONE", [])


def main():
    per = defaultdict(lambda: [0, 0])
    detail = Counter()
    for s in SEEDS:
        e = natural(s)
        menu = [(it["id"], it["text"]) for it in e["items"]]
        got = choose(synth_records(e), e["cfg"], menu)
        want = next(it["id"] for it in e["items"] if it["role"] == "correct")
        key = f"{e['kind']}{'_cfg' if e['kind'] == 'A1' and e['variant'] else ''}"
        per[key][1] += 1
        per[key][0] += got == want
        if got != want:
            detail[(key, str(got)[:40])] += 1
    out = {k: dict(correct=v[0], n=v[1], accuracy=v[0] / v[1], ci95=[round(x, 4) for x in wilson(v[0], v[1])]) for k, v in sorted(per.items())}
    res = dict(partial=True, caveat="synthesized certificate records; real-record sufficiency in the 2048-token bundle NOT RUN",
               per_variant=out, failures={f"{k[0]}|{k[1]}": v for k, v in detail.items()})
    Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
