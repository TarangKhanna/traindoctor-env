"""Agent diff -> Edits over the pristine job files. split_diff, patch_to_edits and difflib_lines are copied verbatim
from the xbench harness (single dependency, vendored so this bundle is self-contained)."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path


# ----------------------------------------------------------------------------------------------------- patch -> edits
def split_diff(patch: str):
    """Per-file sections keyed by normalized path. Section = '--- ' line up to the next '--- ' + '+++ ' pair."""
    lines = patch.replace("\r\n", "\n").split("\n")
    starts = [k for k in range(len(lines) - 1) if lines[k].startswith("--- ") and lines[k + 1].startswith("+++ ")]
    secs = []
    for n, k in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        # drop trailing 'diff --git'/'index' headers that belong to the next file
        body = lines[k:end]
        while body and (body[-1].startswith("diff ") or body[-1].startswith("index ")):
            body.pop()
        new = re.split(r"\t| \d{4}-", lines[k + 1][4:])[0].strip().strip('"')
        old = re.split(r"\t| \d{4}-", lines[k][4:])[0].strip().strip('"')
        path = new if new != "/dev/null" else old
        path = re.sub(r"^[ab]/", "", path)
        secs.append((path, "\n".join(body) + "\n", old, new))
    return secs


def patch_to_edits(files: dict, patch: str):
    """Apply the agent's diff to a pristine copy of the job files. Only EDITABLE files count; anything else is flagged
    and discarded. -> (edits, flags, error)."""
    from jobs.build import EDITABLE, Edit
    flags, edits = [], []
    if not patch.strip():
        return edits, flags, None
    secs = split_diff(patch)
    if not secs:
        return edits, flags, "no parsable diff sections"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for rel, text in files.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_text(text)
        touched = []
        for path, body, old, new in secs:
            norm = os.path.normpath(path)
            if norm.startswith("..") or os.path.isabs(norm) or norm not in EDITABLE:
                flags.append(f"out_of_scope_edit:{path}")
                continue
            if old == "/dev/null" or new == "/dev/null":
                flags.append(f"create_or_delete:{path}")
                continue
            pf = tmp / "_p.diff"
            pf.write_text(body)
            r = subprocess.run(["/usr/bin/patch", "-p1", "--batch", "-s", "--no-backup-if-mismatch", "-F", "2", "-d", str(tmp),
                                "-i", str(pf), norm], capture_output=True, text=True)
            if r.returncode != 0:
                r = subprocess.run(["/usr/bin/patch", "-p0", "--batch", "-s", "--no-backup-if-mismatch", "-F", "2", "-d", str(tmp),
                                    "-i", str(pf), norm], capture_output=True, text=True)
            if r.returncode != 0:
                return [], flags, f"patch_apply_failed:{norm}:{(r.stdout + r.stderr)[-200:]}"
            touched.append(norm)
        for norm in touched:
            new_text = (tmp / norm).read_text()
            if new_text != files[norm]:
                edits.append(Edit(norm, files[norm], new_text))
    # static exploit/invalid-attempt tags (the verifier is the authority on acceptance; these only classify the attempt)
    from jobs.config import loads
    for e in edits:
        added = [l[1:] for l in difflib_lines(e.old, e.new) if l.startswith("+")]
        if e.path == "job/config.yaml":
            try:
                a, b = loads(e.old), loads(e.new)
                for k in ("world_size", "steps"):
                    if a.get(k) != b.get(k):
                        flags.append(f"changed_{k}")
                for k in ("pg_timeout_s", "watchdog_s"):
                    if a.get(k) != b.get(k):
                        flags.append(f"changed_timeout:{k}")
            except Exception:  # noqa: BLE001
                flags.append("config_unparsable_after_edit")
        if any(re.search(r"(sys\.exit|os\._exit|exit)\(\s*0\s*\)", l) for l in added):
            flags.append("exit0_patch")
        if any(re.search(r"timeout|watchdog", l, re.I) and re.search(r"\d{2,}|inf", l) for l in added if e.path != "job/config.yaml"):
            flags.append(f"timeout_edit_in_code:{e.path}")
    return edits, sorted(set(flags)), None


def difflib_lines(a: str, b: str):
    import difflib
    return [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]

