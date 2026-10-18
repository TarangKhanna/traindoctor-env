"""Episode: the agent-facing tool surface over a frozen snapshot (plain Python, no framework).

Reads return snapshot records (each line prefixed with its evidence id), capped at MAX_CHARS
(~400 tokens). Fix tools are shape-validated only and always answer ACK: no correctness feedback.
The agent's workspace is job/ (source and config), data/, runs/ and ckpt/; edits are plain file writes under
job/. Record reads replay the frozen snapshot until the first relaunch: the relaunch goes through the real
resubmit path on a per-episode working copy of the private fixture and a fresh public snapshot (new evidence ids)
is captured; later reads come from that snapshot only. The fixture itself is never modified.
"""
from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import asdict

from jobs import safefs
from jobs.build import ALL_LABELS
from jobs.config import flatten, job_config
from traindoctor_env.reexec import FixOutcome, Replay
from traindoctor_env.snapshot import Snapshot, public_records
from traindoctor_env.verifiers import EVIDENCE_CAP, claimed_vs_verified

MAX_CHARS = 1600
MAX_FILE_CHARS = 8000
ACK = "recorded"
ID_RE = re.compile(r"\bE[0-9a-f]{6}\b")
# What the agent may read in its workspace (job directory); everything else is the harness's.
VISIBLE = re.compile(r"^(job/[^/]+|data/.+|runs/run\d+/(logs|trace|samples)/rank\d+\.jsonl|ckpt/MANIFEST\.json|"
                     r"progress/rank\d+\.json)$")

# tool -> {arg: (accepted types, required)}
READ_TOOLS = {
    "job_status": {},
    "job_config": {},
    "rank_timeline": {"ranks": (list, False), "window": (list, False)},
    "collective_trace": {"step": (int, True)},
    "sample_ids": {"step": (int, True)},
    "grep_logs": {"pattern": (str, True), "ranks": (list, False), "window": (list, False)},
    "ckpt_manifest": {},
    "process_state": {"rank": (int, True)},
    "scheduler_events": {},
    "list_files": {"path": (str, False)},
    "read_file": {"path": (str, True), "start": (int, False), "end": (int, False)},
}
FIX_TOOLS = {
    "write_file": {"path": (str, True), "content": (str, True)},
    "relaunch": {"resume_from": ((str, type(None)), False), "chaos": (bool, False)},
    "kill_ranks": {"ranks": (list, True)},
    "escalate": {"reason": (str, True)},
}
END_TOOLS = {
    "submit_diagnosis": {"label": (str, True), "evidence_ids": (list, True), "claimed_fixed": (bool, True),
                         "fix_summary": (str, True)},
    "finish": {},
}
INTRO = ("A distributed training job failed or misbehaved. Investigate with the read tools, edit the job's files "
         "under job/ and relaunch if appropriate, then call submit_diagnosis and finish.\n"
         f"Cite at most {EVIDENCE_CAP} distinct record ids as evidence.\n"
         "Diagnosis labels: " + ", ".join(ALL_LABELS) + "\nTools: " + ", ".join([*READ_TOOLS, *FIX_TOOLS, *END_TOOLS]))


def truncate_lines(lines: list[str], limit: int = MAX_CHARS) -> str:
    """Join lines up to `limit` characters; deterministic, says so when it cuts."""
    full = "\n".join(lines)
    if len(full) <= limit:
        return full
    out, size = [], 0
    for i, line in enumerate(lines):
        marker = f"... truncated ({len(lines) - i} more lines)"
        if size + len(line) + 1 + len(marker) > limit:
            out.append(marker)
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)[:limit]


def _validate(args: dict, spec: dict) -> str | None:
    for k in args:
        if k not in spec:
            return f"unexpected argument '{k}'"
    for k, (typ, required) in spec.items():
        if k not in args:
            if required:
                return f"missing argument '{k}'"
        elif typ is not object and (not isinstance(args[k], typ) or (typ is int and isinstance(args[k], bool))):
            return f"argument '{k}' has the wrong type"
    ints = lambda v: all(isinstance(x, int) and not isinstance(x, bool) for x in v)  # noqa: E731
    if args.get("window") is not None and not (len(args["window"]) == 2 and ints(args["window"])):
        return "argument 'window' must be [first_step, last_step]"
    if args.get("ranks") is not None and not ints(args["ranks"]):
        return "argument 'ranks' must be a list of integers"
    return None


class Episode:
    def __init__(self):
        self.snap: Snapshot | None = None
        self.replay: Replay | None = None

    def reset(self, snapshot: Snapshot) -> str:
        self.close()
        self.snap = snapshot
        self.phase = 0
        self.records = list(snapshot.records)
        self.phase_records = [self.records]
        self.calls: list[dict] = []
        self.actions: list[dict] = []
        self.observed_ids: set[str] = set()
        self.diagnosis: dict | None = None
        self.result: dict | None = None
        self.done = False
        return INTRO

    def close(self) -> None:
        if self.replay:
            self.replay.close()
            self.replay = None

    @property
    def root(self):
        return self.replay.jd if self.replay else self.snap.fixture_dir

    def outcome(self) -> FixOutcome:
        """Verdict on the workspace as it is now (same code as reexec.verify_fix)."""
        if self.replay is None:
            self.replay = Replay(self.snap)
        return self.replay.outcome()

    def _apply_fix(self, name: str, args: dict) -> str:
        if self.replay is None:
            self.replay = Replay(self.snap)
        if name == "write_file":
            path = args["path"].strip("/")
            if not (path.startswith("job/") and "/" not in path[4:] and self.replay.write_file(path, args["content"])):
                return "error: invalid arguments: path must be a file directly under job/"
        elif name == "relaunch":
            self.replay.relaunch(args.get("resume_from"), args.get("chaos", False))
            self.phase += 1
            self.records = public_records(self.replay.jd, self.snap.seed, self.phase)
            self.phase_records.append(self.records)
        return ACK

    def state(self) -> dict:
        return dict(calls=list(self.calls), actions=list(self.actions), observed_ids=sorted(self.observed_ids),
                    diagnosis=self.diagnosis, done=self.done, result=self.result)

    def step(self, action: dict) -> tuple[str, bool]:
        if self.done:
            return "episode finished", True
        name, args = action.get("tool"), action.get("args", {})
        spec = {**READ_TOOLS, **FIX_TOOLS, **END_TOOLS}.get(name)
        if spec is None:
            return f"error: unknown tool '{name}'", False
        if not isinstance(args, dict) or (err := _validate(args, spec)):
            return f"error: invalid arguments: {err or 'args must be an object'}", False
        self.calls.append(dict(tool=name, args=args))
        if name in FIX_TOOLS:
            if name in ("write_file", "relaunch") and (msg := self._apply_fix(name, args)) != ACK:
                self.calls.pop()
                return msg, False
            self.actions.append(dict(tool=name, args=args))
            return ACK, False
        if name == "submit_diagnosis":
            if not args["label"] or not all(isinstance(x, str) for x in args["evidence_ids"]):
                self.calls.pop()
                return "error: invalid arguments: label must be non-empty and evidence_ids a list of strings", False
            self.diagnosis = dict(args)
            return ACK, False
        if name == "finish":
            out = self.outcome()  # the final verification: re-execute the agent's final job/
            claimed = (self.diagnosis or {}).get("claimed_fixed")
            self.result = dict(claimed_fixed=claimed, verified_fixed=out.ok, sandboxed=out.sandboxed, outcome=asdict(out),
                               claimed_vs_verified=claimed_vs_verified(claimed, out.ok))
            self.done = True
            return "episode finished", True
        obs = getattr(self, f"_{name}")(**args)
        self.observed_ids |= set(ID_RE.findall(obs)) & {r["id"] for r in self.records}
        return obs, False

    # --- read tools -------------------------------------------------------------------------
    def _records(self, kind: str, rank=None, ranks=None, window=None, pattern=None) -> str:
        out = []
        for r in self.records:
            if r["kind"] != kind:
                continue
            if rank is not None and r["rank"] != rank:
                continue
            if ranks is not None and r["rank"] not in ranks:
                continue
            if window is not None and not (window[0] <= r["step"] <= window[1]):
                continue
            if pattern is not None and not pattern.search(r["text"]):
                continue
            out.append(f"{r['id']} {r['text']}")
        return truncate_lines(out) if out else "(no matching records)"

    def _job_status(self):
        return self._records("job")

    def _job_config(self):
        try:
            cfg = job_config(self.root)
        except (OSError, ValueError):
            return "error: job/config.yaml is not a readable regular file"
        return truncate_lines([f"{k} = {json.dumps(v)}" for k, v in sorted(flatten(cfg).items())])

    def _rank_timeline(self, ranks=None, window=None):
        return self._records("log", ranks=ranks, window=window)

    def _collective_trace(self, step):
        return self._records("trace", window=[step, step])

    def _sample_ids(self, step):
        return self._records("samples", window=[step, step])

    def _grep_logs(self, pattern, ranks=None, window=None):
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"error: invalid pattern: {e}"
        return self._records("log", ranks=ranks, window=window, pattern=rx)

    def _ckpt_manifest(self):
        return self._records("ckpt")

    def _process_state(self, rank):
        return self._records("process", rank=rank)

    def _scheduler_events(self):
        return self._records("scheduler")

    def _visible_files(self) -> list[str]:
        """Plain regular files only, found without following a single symlink (the workspace is job-writable)."""
        out = []
        for dirpath, dirs, files in os.walk(self.root, followlinks=False):
            rel_dir = os.path.relpath(dirpath, self.root)
            for n in files:
                rel = n if rel_dir == "." else f"{rel_dir}/{n}"
                if VISIBLE.match(rel) and stat.S_ISREG(os.lstat(os.path.join(dirpath, n)).st_mode):
                    out.append(rel)
        return sorted(out)

    def _list_files(self, path="."):
        prefix = "" if path in (".", "", "/") else path.strip("/") + "/"
        names = [f for f in self._visible_files() if f.startswith(prefix)]
        return truncate_lines(names, MAX_FILE_CHARS) if names else "(no such directory)"

    def _read_file(self, path, start=1, end=None):
        path = path.strip("/")
        if path not in self._visible_files():
            return f"error: no such file: {path}"
        try:
            lines = safefs.read_bytes(self.root, path, 1 << 20).decode(errors="replace").splitlines()
        except (OSError, ValueError):
            return f"error: no such file: {path}"
        return truncate_lines([f"{i}: {x}" for i, x in enumerate(lines[max(start, 1) - 1:end], max(start, 1))], MAX_FILE_CHARS)
