"""Stdlib recount of results/status_audit/<run-id>/raw.jsonl -> per-kind accuracy and 3x3 confusion matrices.
Post-hoc diagnostic audit, not part of the original experiment. Usage: python audit/status_only_recount.py <run-dir>"""
import json
import sys
from collections import Counter

CLASSES = ("A1_src", "A1_cfg", "A2")


def tables(rows):
    out = {}
    for name in ("pred_process", "pred_job"):
        conf = Counter((r["truth"], r[name]) for r in rows)
        per = {c: (conf[(c, c)], sum(conf[(c, p)] for p in CLASSES)) for c in CLASSES}
        out[name] = dict(conf=conf, per=per, overall=(sum(v[0] for v in per.values()), sum(v[1] for v in per.values())))
    return out


def render(rows) -> str:
    lines = []
    for name, t in tables(rows).items():
        lines += [f"### {name}", "rows = truth, cols = predicted", "| truth | " + " | ".join(CLASSES) + " | acc |", "|---|---|---|---|---|"]
        for c in CLASSES:
            k, n = t["per"][c]
            lines.append(f"| {c} | " + " | ".join(str(t["conf"][(c, p)]) for p in CLASSES) + f" | {k}/{n} = {k / n:.3f} |")
        k, n = t["overall"]
        lines += [f"overall: {k}/{n} = {k / n:.3f}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    rows = [json.loads(x) for x in open(sys.argv[1].rstrip("/") + "/raw.jsonl")]
    print(f"n={len(rows)}\n" + render(rows))
