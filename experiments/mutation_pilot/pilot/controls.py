"""Positive/negative controls through the full agent scoring path: gold patch (must verify) and decoy patch (must not)."""
import difflib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot import agent, core
from generator.mutate import arm_edits

for arm in ("gold", "decoy"):
    for tid in sorted(core.TASKS):
        snap, m = core.build_snapshot(tid), core.mutant(tid)
        files = dict(snap.files); new = dict(files)
        for e in arm_edits(m, arm):
            new[e.path] = new[e.path].replace(e.old, e.new)
        patch = "".join("".join(difflib.unified_diff(files[p].splitlines(True), new[p].splitlines(True), f"a/{p}", f"b/{p}")) for p in files if files[p] != new[p])
        d = agent.ANS / f"ctl-{arm}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}_k1.json").write_text("```json\n" + json.dumps(dict(label="control", evidence_ids=[], fix_claim=arm, patch=patch)) + "\n```")
print("controls written")
