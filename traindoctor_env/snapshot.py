"""execute(scenario, out_dir): run the buggy job once (including the failure and its restart) and freeze the result.

out_dir/
  public/   records.jsonl config.json manifest.json files.json   (all an agent can ever see)
  fixture/  the frozen job directory (private; used only by reexec and the oracle)
  fixture/_private/  reference.json meta.json
  truth.json                                                      (private)

Determinism: records contain only logical time (step, collective seq), canonical error classes and normalized
text, ordered by (kind, rank, run, step, seq). Ids depend only on (seed, phase, record index).
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from jobs import manifest, safefs, trusted
from jobs.dataserver import salt_for
from jobs.build import JobSpec, materialize
from jobs.config import dumps, job_config, loads
from jobs.resubmit import resubmit
from traindoctor_env.scenario import Scenario, Truth, job_spec

SOURCE = "executed"
_REFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Snapshot:
    root: Path
    seed: int
    records: tuple
    config: dict
    manifest: list
    files: dict   # job/ source and config text, as of the incident

    @property
    def public_dir(self) -> Path:
        return self.root / "public"

    @property
    def fixture_dir(self) -> Path:
        return self.root / "fixture"

    @property
    def truth_path(self) -> Path:
        return self.root / "truth.json"

    @property
    def salt(self) -> bytes:
        return salt_for(self.seed)

    @property
    def meta(self) -> dict:
        return json.loads((self.fixture_dir / "_private" / "meta.json").read_text())

    @property
    def reference(self) -> dict:
        return json.loads((self.fixture_dir / "_private" / "reference.json").read_text())

    @classmethod
    def load(cls, root: Path) -> "Snapshot":
        pub = Path(root) / "public"
        recs = tuple(json.loads(line) for line in (pub / "records.jsonl").read_text().splitlines())
        return cls(Path(root), Truth.load(Path(root) / "truth.json").seed, recs, json.loads((pub / "config.json").read_text()),
                   json.loads((pub / "manifest.json").read_text()), json.loads((pub / "files.json").read_text()))


def load_truth(snap: Snapshot) -> Truth:
    return Truth.load(snap.truth_path)


_NORMALIZE = [(re.compile(r"pid=\d+"), "pid=<pid>"), (re.compile(r"0x[0-9a-fA-F]+"), "<addr>"),
              (re.compile(r"127\.0\.0\.1:\d+"), "<addr>"), (re.compile(r"port=\d+"), "port=<port>"),
              (re.compile(r"(?<![\w.])(?:/[\w.\-]+){2,}"), "<path>")]


def normalize(text: str) -> str:
    for rx, sub in _NORMALIZE:
        text = rx.sub(sub, text)
    return text


def evidence_ids(seed: int, n: int, phase: int = 0) -> list[str]:
    """Opaque ids from (seed, phase, index) only (phase = number of state-changing actions so far, so
    pre- and post-action ids never collide); on the rare collision the next salt is used."""
    ids, used = [], set()
    for i in range(n):
        salt = 0
        while True:
            h = "E" + hashlib.sha256(f"{seed}:{phase}:{i}:{salt}".encode()).hexdigest()[:6]
            if h not in used:
                break
            salt += 1
        used.add(h)
        ids.append(h)
    return ids


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _jsonl_job(job_dir: Path, rel: str) -> list[dict]:
    """A job-written jsonl, read through a no-symlink walk; missing -> empty, a planted symlink -> refused."""
    try:
        blob = trusted.read_nofollow(job_dir, rel)
    except FileNotFoundError:
        return []
    return [json.loads(x) for x in blob.decode().splitlines()]


def run_names(job_dir: Path) -> list[str]:
    """Runs in the order the launcher started them (from its ledger, not from the job-writable runs/ directory)."""
    return trusted.run_ids(job_dir)


def collect(job_dir: Path) -> list[tuple]:
    """Raw records (kind, rank, run, step, seq, text) from a job_dir."""
    out = []
    for ri, (run, rec) in enumerate((r["run_id"], r) for r in trusted.ledger(job_dir)):
        job = json.loads(trusted.read_run_record(job_dir, run, "job.json"))   # how the run was launched: the ledger, not a job-writable start.json
        out.append(("job", -1, ri, job["last_step"], 0,
                    f"[{run}] resume_from={rec['resume_from'] or 'none'} started_at_step={rec['start_step']} "
                    f"-> {job['status']} last_step={job['last_step']}" + (f" error={job['error_class']}" if job["error_class"] else "")))
        for r in job["ranks"]:
            rk = r["rank"]
            state = {"ok": "exited", "error": "failed", "killed": "killed"}[r["status"]]
            err = f" {r['exc_class']}" + (f": {r['exc_msg']}" if r["exc_class"] == "PYTHON_ERROR" else "") if r["exc_class"] else ""
            out.append(("process", rk, ri, r["last_step"], 0,
                        f"[{run}] rank {rk} {state} exit={r['exitcode']} last_step={r['last_step']}{err}"))
            for i, d in enumerate(_jsonl_job(job_dir, f"runs/{run}/logs/rank{rk}.jsonl")):
                out.append(("log", rk, ri, d["step"], i, f"[{run}] rank {rk} {d['level']} {d['msg']}"))
            for i, d in enumerate(_jsonl(trusted.trusted_dir(job_dir) / "records" / run / "samples" / f"rank{rk}.jsonl")):
                out.append(("samples", rk, ri, d["step"], i, f"[{run}] rank {rk} step={d['step']} samples={d['ids']}"))
            calls: dict[int, dict] = {}
            for d in _jsonl_job(job_dir, f"runs/{run}/trace/rank{rk}.jsonl"):
                if d["ev"] == "enter":
                    calls[d["seq"]] = dict(d, done=False)
                else:
                    calls[d["seq"]]["done"] = True
            for seq, c in sorted(calls.items()):
                out.append(("trace", rk, ri, c["step"], seq,
                            f"[{run}] rank {rk} step={c['step']} seq={seq} {c['op']} bucket={c['bucket']} "
                            f"numel={c['numel']} {'done' if c['done'] else 'pending'}"))
    for i, e in enumerate(trusted.events(job_dir)):
        out.append(("scheduler", -1, 0, i, 0, f"{e['event']}: {e['detail']}"))
    for i, e in enumerate(manifest.load(job_dir)):
        out.append(("ckpt", -1, 0, e["step"], i, f"{e['ckpt_id']} step={e['step']} world_size={e['world_size']}"))
    return out


def public_records(job_dir: Path, seed: int, phase: int = 0) -> list[dict]:
    raw = sorted(collect(job_dir), key=lambda r: r[:5])
    return [dict(id=i, kind=k, rank=rk, step=st, text=normalize(tx), source=SOURCE)
            for i, (k, rk, _, st, _, tx) in zip(evidence_ids(seed, len(raw), phase), raw)]


def job_files(job_dir: Path) -> dict:
    """job/* as plain regular files, read without following a symlink (anything else is left out)."""
    out = {}
    for n in safefs.listdir(job_dir, "job"):
        try:
            out[f"job/{n}"] = safefs.read_bytes(job_dir, f"job/{n}", 1 << 20).decode()
        except (OSError, ValueError):
            continue
    return out


def committed_end(rec: dict, steps: int) -> int:
    """Steps of this run that persist: a completed run commits all of them; a run that ended otherwise only what it
    had checkpointed (the rest is lost to the failure and recomputed by the restart)."""
    if rec["completed"]:
        return steps
    return max([c["step"] for c in rec["captures"]], default=rec["start_step"])


def step_batches(served: list) -> dict[int, tuple]:
    """window -> (epoch, index, ids): the batch a step consumed is the FIRST one the data server served while that
    many steps of the run were acknowledged. Later requests in the same window are prefetches (served is not
    consumed). An entry without a window (older ledgers) falls back to its position in the log."""
    out: dict[int, tuple] = {}
    for k, e in enumerate(served):
        out.setdefault(e[3] if len(e) > 3 else k, tuple(e[:3]))
    return out


def lineage(job_dir: Path, steps: int) -> dict[int, dict[int, list[int]]]:
    """rank -> step -> sample ids of the history that produced the final state, from the data server's log and
    the committed-step boundaries: the first batch served in window k of a run is the one for step start+k, and each run
    supersedes everything from its start step on (a restart rolls back to its checkpoint)."""
    lin: dict[int, dict[int, list[int]]] = {}
    for rec in trusted.ledger(job_dir):
        end = committed_end(rec, steps)
        for r, served in rec["served"].items():
            r = int(r)
            d = {s: v for s, v in lin.get(r, {}).items() if s < rec["start_step"]}
            d.update({rec["start_step"] + w: b[2] for w, b in step_batches(served).items() if rec["start_step"] + w < end})
            lin[r] = d
    return lin


def reference_run(spec: JobSpec, salt) -> dict:
    """Uninterrupted run of the pristine job, checkpointed at EVERY step by the launcher: the recovery target.
    states[step] is the full committed state (model, optimizer momentum, sampler position) after `step` steps.
    It depends only on the pristine config and the salt, so identical (config, salt) share one run."""
    with _REFERENCE_LOCK:  # parallel executions of one config wait for the first
        return _reference(dumps(spec.config), salt)


def reference_for_config(config: dict, salt) -> dict:
    """The recovery target for the pristine `config` under `salt` (bytes, or a per-verification SaltSchedule)."""
    with _REFERENCE_LOCK:
        return _reference(dumps(config), salt)


@functools.lru_cache(maxsize=64)
def _reference(config_text: str, salt) -> dict:
    import torch
    config = loads(config_text)
    with tempfile.TemporaryDirectory() as tmp:
        jd = Path(tmp) / "ref"
        materialize(jd, {**config, "ckpt_every": 1})
        r = resubmit(jd, salt=salt)
        if not r.ok:
            raise RuntimeError(f"reference run failed: {r.status}")
        rec = trusted.ledger(jd)[0]
        states = {}
        for c in rec["captures"]:
            blob = torch.load(trusted.trusted_dir(jd) / c["file"], weights_only=True)
            states[str(c["step"])] = {k: v.tolist() for k, v in blob.items()}
        assert sorted(map(int, states)) == list(range(1, config["steps"] + 1)), sorted(states)
        final = states[str(config["steps"])]
        return dict(steps=config["steps"], states=states,
                    config=dict(lr=config["lr"], hidden=config["hidden"], seed=config["seed"], world_size=config["world_size"],
                                batch_size=config["batch_size"], ckpt_every=config["ckpt_every"],
                                num_samples=config["data"]["num_samples"]),
                    final_params=[x for k in sorted(final) if k.startswith("model.") for x in _flat(final[k])],
                    samples={str(k): {str(s): v for s, v in d.items()} for k, d in lineage(jd, config["steps"]).items()})


def _flat(v):
    return [x for y in v for x in _flat(y)] if isinstance(v, list) else [v]


def execute(scenario: Scenario, out_dir: Path) -> Snapshot:
    out = Path(out_dir)
    fx = out / "fixture"
    spec = job_spec(scenario)
    salt = salt_for(scenario.seed)
    materialize(fx, spec.config, spec.bug)
    if spec.task == "A":
        res = resubmit(fx, salt=salt)
        expect = res.status == "failed" and res.error_class == "COLLECTIVE_TIMEOUT"
    else:
        res = resubmit(fx, kill=spec.kill, restart_on_failure=True, salt=salt)
        runs = [json.loads(trusted.read_run_record(fx, r, "job.json")) for r in run_names(fx)]
        expect = res.ok and [r["status"] for r in runs] == ["failed", "completed"] and trusted.ledger(fx)[0]["kill_done"]
    frontier = res.job.last_step + 1 if res.job else 0
    if not expect:
        raise RuntimeError(f"{scenario} did not play out as designed: {res.status} {res.error_class}")
    priv = fx / "_private"
    priv.mkdir()
    (priv / "reference.json").write_text(json.dumps(dict(reference_run(spec, salt), pristine_config=spec.config)))
    (priv / "meta.json").write_text(json.dumps(dict(frontier=frontier, kill=spec.kill, task=spec.task)))
    pub = out / "public"
    pub.mkdir()
    (pub / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in public_records(fx, scenario.seed)))
    (pub / "config.json").write_text(json.dumps(job_config(fx), indent=2, sort_keys=True) + "\n")
    (pub / "files.json").write_text(json.dumps(job_files(fx), indent=1, sort_keys=True) + "\n")
    shutil.copy(manifest.path(fx), pub / "manifest.json")
    Truth(scenario.seed, scenario.kind, scenario.variant, scenario.label, spec.task,
          [asdict(e) for e in spec.fix], spec.details).save(out / "truth.json")
    return Snapshot.load(out)
