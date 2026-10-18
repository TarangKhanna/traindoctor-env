#!/usr/bin/env python3
"""Release check: relative links resolve, mermaid fences balance, SVGs are well-formed, and the key README numbers
match their source files. Exits non-zero on any failure.   Usage: python scripts/check_readme.py (repo root)."""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# 1. links and images
for doc in ("README.md", "ARCHITECTURE.md", "docs/README.md"):
    p = ROOT / doc
    txt = p.read_text()
    bad = []
    for m in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)\)", txt):
        tgt = m[1].split("#")[0]
        if not tgt or re.match(r"[a-z]+:", tgt):
            continue
        if not (p.parent / tgt).exists():
            bad.append(tgt)
    check(not bad, f"{doc}: relative links resolve" + (f" (missing {bad})" if bad else ""))
    fences = txt.count("```")
    check(fences % 2 == 0, f"{doc}: code fences balanced ({fences})")

# 2. svgs
for f in sorted((ROOT / "docs/figures").glob("*.svg")):
    try:
        ET.parse(f)
        ok = True
    except ET.ParseError:
        ok = False
    check(ok, f"{f.name} well-formed")

# 3. numbers
R = (ROOT / "README.md").read_text()
A = json.loads((ROOT / "evidence/results/test/TEST_REPORT_analysis.json").read_text())
RC = json.loads((ROOT / "evidence/results/test/AUDIT_RECOUNT.json").read_text())
RT = (ROOT / "evidence/results/REDTEAM_TABLE.md").read_text()
TR = (ROOT / "evidence/results/TEST_REPORT.md").read_text()
pt = A["policy_table"]
mean = lambda a: sum(a) / len(a)
pct = lambda k: 100 * pt[k]["v2_fault4"]
p = A["primary"]
check(f"{p['diff']:+.2f}" == "+10.36" and "**10.36 percentage points**" in R, "primary diff +10.36 in analysis JSON and README")
check([f"{x:+.2f}" for x in p["boot_ci95"]] == ["+5.83", "+15.26"] and "**+5.83 to +15.26**" in R, "primary CI [+5.83, +15.26]")
check(round(mean([pct(f"v0_s{i}") for i in range(6)]), 1) == 23.0 and "23.0%" in R, "label-only arm mean 23.0")
check(round(mean([pct(f"v2_s{i}") for i in range(6)]), 1) == 33.4 and "33.4%" in R, "verified arm mean 33.4")
check(f"{RC['arm_mean_pct']['v0']:.1f}/{RC['arm_mean_pct']['v2']:.1f}" == "71.4/71.5" and "71.4%" in R and "71.5%" in R, "repair-choice 71.4/71.5")
check(round(pct("base"), 1) == 5.0 and round(pct("sft"), 1) == 17.2 and "| Base model | 5.0% |" in R and "| 17.2% |" in R, "SFT 5.0 -> 17.2")
s = A["secondary"]["v2_arm_minus_csft"]
check(f"{s['diff']:+.2f}" == "+3.70" and [f"{x:+.2f}" for x in s["boot_ci95"]] == ["-1.67", "+9.11"] and "+3.7 points; interval \u22121.7 to +9.1" in R,
      "continued-SFT diff +3.70 CI [-1.67, +9.11]")
check(A["n_episodes"] == 100 and "100 held-out incidents" in R, "100 episodes")
check("369" in R and "369" in TR, "369 replays in README and TEST_REPORT")
check("0 accepted of 86" in RT and "48 invariant, 38 execution_failure" in RT and "| Rejected by a specific correctness invariant | **48** |" in R and "| Failed during execution, before usable recovery artifacts | **38** |" in R and "| Accepted fake repairs | **0** |" in R and "0 of 86 attempted fake repairs" in R,
      "red team 86/48/38/0")
check("gold 5/5" in RT and "alt-gold 9/10" in RT and "| Accepted known-correct reference repairs | **5/5** |" in R and "| Accepted alternative legitimate repairs | **9/10** |" in R, "gold 5/5, alt 9/10")
check("Null (no diff): 0 accepted of 5" in RT and "no-op 0/5" in (ROOT / "docs/VERIFICATION.md").read_text(), "null 0/5")
check("72 missing and 8 duplicated samples" in R and "gaps=72 dups=8" in RT, "F11 A1_src 72 gaps / 8 dups")
sys.exit(1 if fails else 0)
