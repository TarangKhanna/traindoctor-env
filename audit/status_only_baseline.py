"""Status-only baselines (rule: audit/status_only_rule.md). Post-hoc diagnostic audit, not part of the original
experiment. Classifiers take ONLY the bundle text (a str) and return one of CLASSES."""
from __future__ import annotations

import re

CLASSES = ("A1_src", "A1_cfg", "A2")
_PROC = re.compile(r"\[run\d+\] rank \d+ (failed|killed) exit=(-?\d+)")


def _record_lines(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("classifiers accept bundle text (str) only")
    body = text.split("== records ==", 1)[-1].split("== repairs ==", 1)[0]
    return [ln for ln in body.splitlines() if ln.strip()]


def classify_process(text: str) -> str:
    procs = [m for ln in _record_lines(text) if (m := _PROC.search(ln))]
    if any(m.group(1) == "killed" or int(m.group(2)) < 0 for m in procs):
        return "A2"
    return "A1_src"


def classify_job(text: str) -> str:
    lines = [ln for ln in _record_lines(text)
             if re.search(r"\[run\d+\] resume_from=.* -> ", ln) or re.match(r"\S+ (submitted|failed): run\d+", ln)]
    if any(re.search(r"killed|stalled|watchdog|exit=-", ln) for ln in lines):
        return "A2"
    return "A1_src"
