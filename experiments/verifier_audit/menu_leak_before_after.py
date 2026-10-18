"""Before/after menu-leak numbers (pure Python). Before = traindoctor_env/bundle.py at commit e4b3d15 (signed +/- diff),
after = working tree (neutral display). Usage: python experiments/verifier_audit/menu_leak_before_after.py [out.json]
Held-out predictors, 400 DEV seeds: lookup learner on diff direction/variant text, and the n-gram logistic regression
(5-fold across seeds). Accuracies are fractions of held-out menus where the top-scored item is the correct repair."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from audit.menu_leak import DEV_SEEDS, report, violations, wilson

BEFORE = "e4b3d15"


def load_old():
    src = subprocess.run(["/usr/bin/git", "show", f"{BEFORE}:traindoctor_env/bundle.py"], capture_output=True, text=True, check=True).stdout
    path = Path(__file__).parent / "_bundle_before.py"
    path.write_text(src)
    try:
        spec = importlib.util.spec_from_file_location("bundle_before", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["bundle_before"] = mod
        spec.loader.exec_module(mod)
    finally:
        path.unlink()
    return mod


def summarize(rep):
    keys = ["content.added_text", "content.removed_text", "content.first_form_text", "content.variant_set_text"]
    out = {}
    for k in keys:
        v = rep["repair_leakage"].get(k)
        if v:
            lo, hi = wilson(round(v["accuracy"] * v["n"]), v["n"])
            out[k] = dict(accuracy=v["accuracy"], ci95=[lo, hi], chance=v["chance"], n=v["n"])
    out["ngram_lr"] = rep["ngram_lr"]
    out["violations"] = violations(rep)
    return out


if __name__ == "__main__":
    old = load_old()
    res = dict(seeds=[DEV_SEEDS.start, DEV_SEEDS.stop], before=summarize(report(DEV_SEEDS, old.make_menu)),
               after=summarize(report(DEV_SEEDS)))
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/verifier_audit/menu_leak_before_after.json")
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1, sort_keys=True))
