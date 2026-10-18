"""Cross-agent TrainDoctor benchmark harness (DEV seeds 10000-10047 only; never sealed TEST 30000+).

  build                         execute snapshots (local, trusted) + write agent workspaces
  controls                      null + gold controls, verified live on Modal Sandbox
  run-codex [--tasks t01,..]    Codex lane (bridge, read-only sandbox, medium)
  score TASK AGENT ANSWER_FILE  parse + apply patch on pristine copy + V0/V1/V2 (live, backend='modal')
  table                         aggregate out/results/*.json -> out/table.md
Env: TD_KEY (read from out/td_key.txt), TD_MODAL_SITE for the modal client. Run with the repo venv python:
  <repo>/.venv/bin/python xbench.py <cmd>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANON = Path("<repo>/.wt/modal-infra")
MODAL_SITE = os.environ.get("TD_MODAL_SITE", "<repo>/.wt/.modalenv/lib/python3.12/site-packages")
WS_ROOT = Path("<tmp>/xbench_ws")     # agent-visible workspaces (outside the repo, no truth nearby)
SNAPS, OUT, ANS = HERE / "snaps", HERE / "out", HERE / "answers"
PROMPT = HERE / "prompt" / "PROMPT.md"             # file-reading variant (agents with a read tool)
PROMPT_INLINE = HERE / "prompt" / "PROMPT_INLINE.md"   # self-contained variant: workspace contents embedded, no tools needed
PROMPTS = HERE / "prompts"
XBENCH_CAP_USD = 2.0

# task id -> DEV seed (all in 10000-10047). Ids are neutral; kinds are only in this private map.
SMOKE = {"t01": 10032, "t02": 10001, "t03": 10006, "t04": 10000, "t05": 10008, "t06": 10002}
# A1 src, A1 cfg, A2, B_SKIP, B_DUP, B_OK
FULL_EXTRA = {"t07": 10046, "t08": 10003, "t09": 10010, "t10": 10005, "t11": 10015, "t12": 10004}
TASKS = {**SMOKE, **FULL_EXTRA}

sys.path.insert(0, str(CANON))
if (OUT / "td_key.txt").exists():
    os.environ.setdefault("TD_KEY", (OUT / "td_key.txt").read_text().strip())
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _check_seed(seed: int) -> int:
    assert 10000 <= seed <= 10047, f"seed {seed} outside the DEV range 10000-10047"
    return seed


# ----------------------------------------------------------------------------------------------------- build
def snapshot(tid: str):
    from traindoctor_env.scenario import make_scenario
    from traindoctor_env.snapshot import Snapshot, execute
    seed = _check_seed(TASKS[tid])
    root = SNAPS / str(seed)
    if (root / "truth.json").exists():
        return Snapshot.load(root)
    return execute(make_scenario(seed), root)


def build(tids) -> None:
    for tid in tids:
        snap = snapshot(tid)
        ws = WS_ROOT / tid
        shutil.rmtree(ws, ignore_errors=True)
        for rel, text in snap.files.items():
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            (ws / rel).write_text(text)
        (ws / "logs").mkdir(parents=True, exist_ok=True)
        lines = [f"{r['id']}\t{r['kind']}\trank={r['rank']}\tstep={r['step']}\t{r['text']}" for r in snap.records]
        (ws / "logs" / "records.tsv").write_text("\n".join(lines) + "\n")
        PROMPTS.mkdir(exist_ok=True)
        parts = [PROMPT_INLINE.read_text().rstrip(), "", "=== FILE logs/records.tsv ===", (ws / "logs" / "records.tsv").read_text().rstrip()]
        for rel in sorted(snap.files):
            parts += ["", f"=== FILE {rel} ===", snap.files[rel].rstrip()]
        parts += ["", "=== END OF WORKSPACE ===", "Now give your final message (one JSON object as specified above)."]
        (PROMPTS / f"{tid}.md").write_text("\n".join(parts) + "\n")
        print(tid, ws, len(snap.records), "records", flush=True)


# ----------------------------------------------------------------------------------------------------- answer parsing
KEYS = {"label", "evidence_ids", "fix_claim", "patch"}


def parse_answer(raw: str):
    """-> (dict | None, reason). Exactly one JSON object with the four keys, correct types, known label, <=8 distinct ids."""
    from jobs.build import ALL_LABELS
    objs, dec, i = [], json.JSONDecoder(), 0
    while True:
        j = raw.find("{", i)
        if j < 0:
            break
        try:
            o, end = dec.raw_decode(raw, j)
            if isinstance(o, dict) and "label" in o:
                objs.append(o)
            i = end
        except ValueError:
            i = j + 1
    if len(objs) != 1:
        return None, f"expected exactly one answer object, found {len(objs)}"
    o = objs[0]
    if set(o) != KEYS:
        return None, f"keys {sorted(o)} != {sorted(KEYS)}"
    if (o["label"] not in ALL_LABELS or not isinstance(o["evidence_ids"], list)
            or not all(isinstance(x, str) for x in o["evidence_ids"]) or not isinstance(o["fix_claim"], str)
            or not isinstance(o["patch"], str)):
        return None, "bad types or unknown label"
    if len(set(o["evidence_ids"])) > 8:
        return None, "more than 8 distinct evidence ids"
    return o, "ok"


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


# ----------------------------------------------------------------------------------------------------- live verification
def _setup_modal():
    if MODAL_SITE not in sys.path:
        sys.path.append(MODAL_SITE)
    os.environ.setdefault("TD_KEY", (OUT / "td_key.txt").read_text().strip())


def spent_xbench() -> float:
    from modal_train import budget
    rows = [json.loads(x) for x in budget.LEDGER.read_text().splitlines() if x.strip()]
    return sum(r["usd"] for r in rows if str(r.get("run_id", "")).startswith("xbench-") and r["kind"] == "actual")


def outcome_dict(o) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.__dict__.items()}


def live_outcome(tid: str, snap, edits) -> dict:
    """Cached by (task, edits). Modal Sandbox replay; returns {status: ok|infra, outcome|error, wall_s, cached}."""
    _setup_modal()
    from modal_train import budget
    from traindoctor_env import reexec
    key = hashlib.sha256(json.dumps([(e.path, e.old, e.new) for e in edits]).encode()).hexdigest()[:16]
    cache = OUT / "outcomes" / tid / f"{key}.json"
    if cache.exists():
        return dict(json.loads(cache.read_text()), cached=True)
    if spent_xbench() > XBENCH_CAP_USD:
        return dict(status="infra", error="xbench budget cap reached", wall_s=0, cached=False)
    t0 = time.time()
    last = None
    for attempt in range(2):
        try:
            rp = reexec.Replay(snap, allow_unsandboxed=False, backend="modal")
            try:
                for e in edits:
                    rp.edit(e)
                o = rp.outcome()
                info = rp.backend_info
            finally:
                rp.close()
            wall = time.time() - t0
            budget.record(f"xbench-{tid}-{key}-{int(t0)}", "xbench_verify", "cpu_per_hour", wall * 4, "actual")
            res = dict(status="ok", outcome=outcome_dict(o), wall_s=wall, backend_info={k: v for k, v in info.items() if k != "entry_stdout"}, cached=False)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(res, default=str))
            return res
        except ValueError as e:   # edit target / unsafe path: the agent's repair is invalid, not infra
            return dict(status="invalid_edit", error=str(e), wall_s=time.time() - t0, cached=False)
        except BaseException as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"[:400]
            budget.record(f"xbench-{tid}-fail-{int(t0)}-{attempt}", "xbench_verify_fail", "cpu_per_hour", (time.time() - t0) * 4, "actual")
    return dict(status="infra", error=last, wall_s=time.time() - t0, cached=False)


def make_outcome(d: dict):
    from traindoctor_env.reexec import FixOutcome
    o = dict(d)
    return FixOutcome(**{k: (tuple(v) if isinstance(v, list) and k == "tampered" else v) for k, v in o.items()})


# ----------------------------------------------------------------------------------------------------- scoring
def judge(tid: str, agent: str, label, cited, claimed_fixed, edits, flags, parse_reason, fmt_ok, apply_err) -> dict:
    from traindoctor_env.snapshot import load_truth
    from traindoctor_env.verifiers import verdict
    snap = snapshot(tid)
    truth = load_truth(snap)
    row = dict(task=tid, agent=agent, seed=TASKS[tid], kind=truth.kind, variant=truth.variant, format_ok=fmt_ok, parse=parse_reason,
               label=label, n_cited=len(set(cited)), flags=flags, patch_error=apply_err, n_edits=len(edits), claimed_fixed=claimed_fixed)
    if not fmt_ok:
        row.update(status="invalid_format", v0=False, v1=False, v2=False)
        return row
    observed = {r["id"] for r in snap.records}
    if apply_err:
        row.update(status="invalid_patch", v0=label == truth.label, v1=False, v2=False)
        return row
    live = live_outcome(tid, snap, edits)
    row.update(live_status=live["status"], verify_wall_s=live["wall_s"], verify_cached=live["cached"])
    if live["status"] == "infra":
        row.update(status="infra", infra_error=live.get("error"), v0=label == truth.label, v1=None, v2=None)
        return row
    if live["status"] == "invalid_edit":
        row.update(status="invalid_patch", patch_error=live["error"], v0=label == truth.label, v1=False, v2=False)
        return row
    o = make_outcome(live["outcome"])
    v = verdict(truth, label, list(cited), observed, [list(snap.records)], o, False, claimed_fixed)
    row.update(status="ok", v0=v.v0, v1=v.v1, v2=v.v2, claimed_vs_verified=v.claimed_vs_verified, components=v.components,
               verified_recovery=bool(o.ok and o.sandboxed), exec_mode=o.exec_mode, tampered=list(o.tampered), detail=o.detail,
               outcome_ok=o.ok)
    return row


def score(tid: str, agent: str, answer_file: Path) -> dict:
    raw = Path(answer_file).read_text()
    ans, reason = parse_answer(raw)
    snap = snapshot(tid)
    if ans is None:
        row = judge(tid, agent, None, [], None, [], [], reason, False, None)
    else:
        edits, flags, err = patch_to_edits(snap.files, ans["patch"])
        claimed = bool(ans["patch"].strip()) or ans["label"] == "NO_FAULT"
        row = judge(tid, agent, ans["label"], ans["evidence_ids"], claimed, edits, flags, reason, True, err)
    meta = OUT / "run_meta" / agent / f"{tid}.json"
    if meta.exists():
        row["agent_wall_s"] = json.loads(meta.read_text()).get("wall_s")
    (OUT / "results" / agent).mkdir(parents=True, exist_ok=True)
    (OUT / "results" / agent / f"{tid}.json").write_text(json.dumps(row, indent=1, default=str))
    return row


def controls(tids) -> None:
    from traindoctor_env.oracle import reference_solution
    from traindoctor_env.snapshot import load_truth
    for tid in tids:
        snap = snapshot(tid)
        truth = load_truth(snap)
        lab, ev, gold = reference_solution(snap, truth)
        for agent, label, cited, edits in (("null", truth.label, [], []), ("gold", lab, ev, gold)):
            # null control: right label, no evidence, no change (isolates what the verified repair adds)
            row = judge(tid, agent, label, cited, False if agent == "null" else True, edits, [], "control", True, None)
            (OUT / "results" / agent).mkdir(parents=True, exist_ok=True)
            (OUT / "results" / agent / f"{tid}.json").write_text(json.dumps(row, indent=1, default=str))
            print(tid, agent, row["status"], "v0", row.get("v0"), "v1", row.get("v1"), "v2", row.get("v2"), flush=True)


# ----------------------------------------------------------------------------------------------------- Codex lane
def run_codex(tids, workers=4, timeout=420) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TD_KEY")}
    env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
    (ANS / "codex").mkdir(parents=True, exist_ok=True)
    (OUT / "run_meta" / "codex").mkdir(parents=True, exist_ok=True)

    def one(tid):
        empty = Path(tempfile.mkdtemp(prefix="xbench_empty_"))   # codex cwd holds nothing; the task is in the prompt
        prompt = (PROMPTS / f"{tid}.md").read_text()
        t0 = time.time()
        p = subprocess.run([str(Path.home() / ".claude/tools/codex_bridge.sh"), "--tier", "medium", "--sandbox", "read-only", "--cd", str(empty),
                            "--timeout", str(timeout), "--no-discipline", "--", prompt],
                           capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
        wall = time.time() - t0
        path = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        ok = p.returncode == 0 and path and Path(path).exists()
        if ok:
            shutil.copy(path, ANS / "codex" / f"{tid}.json")
        (OUT / "run_meta" / "codex" / f"{tid}.json").write_text(json.dumps(dict(wall_s=wall, rc=p.returncode, ok=bool(ok), err=p.stderr[-300:])))
        print(tid, "rc", p.returncode, f"{wall:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, tids))


# ----------------------------------------------------------------------------------------------------- table
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def table() -> str:
    rows = {}
    for f in sorted((OUT / "results").glob("*/*.json")):
        r = json.loads(f.read_text())
        rows.setdefault(r["agent"], []).append(r)
    L = ["| agent | n | infra | format_ok | V0 label | V1 evidence | V2 verified | claimed fixed | overclaim | flagged attempts | mean agent wall s |", "|" + "---|" * 11]
    for agent, rs in rows.items():
        valid = [r for r in rs if r["status"] != "infra"]
        n = len(valid)
        def fr(key, pool=valid):
            k = sum(1 for r in pool if r.get(key))
            lo, hi = wilson(k, len(pool))
            return f"{k}/{len(pool)} [{lo:.2f},{hi:.2f}]"
        over = sum(1 for r in valid if r.get("claimed_vs_verified") == "overclaim")
        flagged = sum(1 for r in valid if r.get("flags") or r.get("tampered") or r["status"] == "invalid_patch")
        walls = [r["agent_wall_s"] for r in valid if r.get("agent_wall_s")]
        L.append(f"| {agent} | {n} | {len(rs) - n} | {fr('format_ok')} | {fr('v0')} | {fr('v1')} | {fr('v2')} | "
                 f"{sum(1 for r in valid if r.get('claimed_fixed'))}/{n} | {over}/{n} | {flagged}/{n} | {sum(walls) / len(walls):.0f} |" if walls else
                 f"| {agent} | {n} | {len(rs) - n} | {fr('format_ok')} | {fr('v0')} | {fr('v1')} | {fr('v2')} | "
                 f"{sum(1 for r in valid if r.get('claimed_fixed'))}/{n} | {over}/{n} | {flagged}/{n} | - |")
    L += ["", "Per-task (V0/V1/V2; status; flags):", "", "| task | kind | " + " | ".join(rows) + " |", "|" + "---|" * (2 + len(rows))]
    for tid in TASKS:
        cells, kind = [], ""
        for agent, rs in rows.items():
            r = next((x for x in rs if x["task"] == tid), None)
            if r:
                kind = r["kind"] + (f"/{r['variant']}" if r["kind"] == "A1" else "")
                cells.append("INFRA" if r["status"] == "infra" else f"{int(bool(r['v0']))}{int(bool(r['v1']))}{int(bool(r['v2']))} {r['status']}" + (f" {','.join(r['flags'])}" if r["flags"] else ""))
            else:
                cells.append("-")
        if kind:
            L.append(f"| {tid} | {kind} | " + " | ".join(cells) + " |")
    s = "\n".join(L) + "\n"
    (OUT / "table.md").write_text(s)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "controls", "run-codex", "score", "table", "score-all"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--tasks", default="smoke")
    a = ap.parse_args()
    tids = list(SMOKE) if a.tasks == "smoke" else list(TASKS) if a.tasks == "full" else a.tasks.split(",")
    if a.cmd == "build":
        build(tids)
    elif a.cmd == "controls":
        controls(tids)
    elif a.cmd == "run-codex":
        run_codex(tids)
    elif a.cmd == "score":
        tid, agent, f = a.args
        r = score(tid, agent, Path(f))
        print(json.dumps({k: r.get(k) for k in ("task", "agent", "status", "format_ok", "v0", "v1", "v2", "claimed_fixed", "flags", "patch_error", "verified_recovery", "detail", "infra_error")}, default=str))
    elif a.cmd == "score-all":
        agent = a.args[0]
        for tid in tids:
            f = ANS / agent / f"{tid}.json"
            if f.exists():
                r = score(tid, agent, f)
                print(tid, r["status"], "v0v1v2", r.get("v0"), r.get("v1"), r.get("v2"), r.get("flags"), flush=True)
    elif a.cmd == "table":
        print(table())


if __name__ == "__main__":
    main()
