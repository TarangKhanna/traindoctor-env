"""One state-only recovery demonstration, separate from historical Task B.

Trusted, fixed recovery actions operate on artifact copies; arbitrary code is not
accepted here. The selected continuation itself is graded. Correctness and raw
recomputation cost are separate; scratch is never rejected merely for its cost.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from jobs import manifest, sandbox, trusted
from jobs.build import TEMPLATE, base_config, materialize
from jobs.dataserver import SaltSchedule, salt_for
from jobs.resubmit import SCRATCH, resubmit
from jobs.template.ckpt import digest
from traindoctor_env.reexec import check, lost_work
from traindoctor_env.snapshot import reference_for_config

VERSION = "state_recovery_v1"
CAUSES = ("sampler", "optimizer")  # fixture construction only; never in observations
ACTIONS = ("resume", "restore_sampler", "restore_optimizer", "restore_full", "older", "scratch")
ACTION_TEXT = {
    "resume": "Resume the delivered step-8 checkpoint unchanged.",
    "restore_sampler": "Restore only its sampler component from the committed step-8 archive; resume at step 8.",
    "restore_optimizer": "Restore only its optimizer components from that archive; resume at step 8.",
    "restore_full": "Restore the complete committed step-8 archive; resume at step 8.",
    "older": "Resume the coherent committed step-4 checkpoint.",
    "scratch": "Restart the correct training program from step 0.",
}


def source_hashes(jd: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((jd / "job").glob("*.py"))}


def read_state(jd: Path, rel: str) -> dict:
    return torch.load(io.BytesIO(trusted.read_nofollow(jd, rel)), weights_only=True)


def capture_at(jd: Path, step: int) -> dict:
    # Only the original preemption run is authoritative for archived components.
    return next(c for c in trusted.ledger(jd)[0]["captures"] if c["step"] == step)


def archived_state(jd: Path, step: int) -> dict:
    cap = capture_at(jd, step)
    return torch.load(trusted.trusted_dir(jd) / cap["file"], weights_only=True)


def _write_checkpoint(jd: Path, state: dict) -> None:
    torch.save(state, jd / "ckpt/ck0008/state.pt")
    _update_manifest(jd, state)


def _update_manifest(jd: Path, state: dict) -> None:
    entries = manifest.load(jd)
    for entry in entries:
        if entry["step"] == 8:
            entry["shards"][0]["sha256"] = digest(state)
    manifest.path(jd).write_text(json.dumps(entries, sort_keys=True) + "\n")


def _summary(state: dict) -> dict:
    # Tensor measurements, not validity flags or hidden construction labels.
    def stats(prefix):
        return {k: {"shape": list(v.shape), "sum": float(v.double().sum()),
                    "l2": float(v.double().square().sum().sqrt())}
                for k, v in state.items() if k.startswith(prefix)}
    return {"step": int(state["step"].item()), "sampler_position": state["sampler"].tolist(),
            "model": stats("model."), "optimizer": stats("opt.")}


@dataclass
class RecoveryFixture:
    root: Path
    config: dict
    schedule: SaltSchedule
    reference: dict
    source: dict
    frontier: int

    @classmethod
    def create(cls, root: Path, seed: int = 777) -> "RecoveryFixture":
        if "TD_KEY" not in os.environ:
            raise ValueError("set an explicit demonstration TD_KEY; never read or create an owner key")
        if not sandbox.available():
            raise RuntimeError("local macOS job sandbox unavailable; refusing an unlabelled fallback")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        cfg = base_config(2, seed, 5, 5, 0)
        base = salt_for(seed)
        # One challenge for the paired comparison, fixed BEFORE any action. Values
        # below step 8 remain compatible with genuine pre-failure checkpoints.
        schedule = SaltSchedule(base, hashlib.sha256(base + b"|state-recovery-v1").digest(), 8)
        jd = root / "prefix"
        materialize(jd, cfg)
        result = resubmit(jd, kill=(0, 9), sandbox=True, salt=base)
        rec = trusted.ledger(jd)[0]
        if result.ok or not rec["kill_done"] or {c["step"] for c in rec["captures"]} != {4, 8}:
            raise RuntimeError("the specified preemption incident did not execute")
        hashes = source_hashes(jd)
        assert all(hashlib.sha256((Path(TEMPLATE) / name).read_bytes()).hexdigest() == sha
                   for name, sha in hashes.items())
        ref = reference_for_config(cfg, schedule)
        frontier = max(rec["acked"].values())
        return cls(root, cfg, schedule, ref, hashes, frontier)

    def incident(self, root: Path, cause: str) -> "RecoveryIncident":
        if cause not in CAUSES:
            raise ValueError("unknown fixture construction")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        jd = root / "jobdir"
        trusted.copy_job(self.root / "prefix", jd)
        state, earlier = archived_state(jd, 8), archived_state(jd, 4)
        keys = ["sampler"] if cause == "sampler" else [k for k in state if k.startswith("opt.")]
        for key in keys:
            state[key] = earlier[key].clone()
        _write_checkpoint(jd, state)
        # This event does not manufacture a committed capture of the mixed state.
        trusted.append_event(jd, "artifact_delivery", "checkpoint bundle assembled by external recovery storage")
        manifest.validate(jd, manifest.load(jd))
        assert source_hashes(jd) == self.source
        return RecoveryIncident(self, root)


@dataclass
class RecoveryIncident:
    fixture: RecoveryFixture
    root: Path

    @property
    def jd(self):
        return self.root / "jobdir"

    def status(self, metadata: str = "incident") -> dict:
        # Both constructions have precisely this public initial observation.
        return {"version": VERSION, "source": "executed", "display_name": metadata,
                "last_run": "worker failure", "committed_steps": [4, 8],
                "delivered_checkpoint": "ck0008", "manifest_integrity": "passes",
                "actions": dict(ACTION_TEXT)}

    def inspect_checkpoint(self) -> dict:
        return {"source": "executed", "checkpoint": "ck0008",
                "delivered": _summary(read_state(self.jd, "ckpt/ck0008/state.pt"))}

    def committed_history(self) -> dict:
        rec = trusted.ledger(self.jd)[0]
        # Genuine archived component measurements and launcher-owned sample IDs.
        return {"source": "executed", "records": [
            {"committed_step": step, "state": _summary(archived_state(self.jd, step))}
            for step in (4, 8)], "served_before_failure": rec["served"],
            "acknowledged_updates_by_rank": rec["acked"]}

    def recover(self, action: str, root: Path) -> dict:
        if action not in ACTIONS:
            raise ValueError("unknown recovery action")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        jd = root / "jobdir"
        trusted.copy_job(self.jd, jd)
        before = frozenset(trusted.run_ids(jd))
        if action.startswith("restore_"):
            original = archived_state(jd, 8)
            if action == "restore_full":
                shutil.copyfile(trusted.trusted_dir(jd) / capture_at(jd, 8)["file"], jd / "ckpt/ck0008/state.pt")
                _update_manifest(jd, original)
            else:
                state = read_state(jd, "ckpt/ck0008/state.pt")
                keys = ["sampler"] if action == "restore_sampler" else [k for k in state if k.startswith("opt.")]
                for key in keys:
                    state[key] = original[key].clone()
                _write_checkpoint(jd, state)
        start = 0 if action == "scratch" else 4 if action == "older" else 8
        # Explicit rollback: discarded future bundles are not commitments of this
        # continuation. Archived launcher captures remain intact in trusted storage.
        manifest.path(jd).write_text(json.dumps([e for e in manifest.load(jd) if e["step"] <= start]) + "\n")
        manifest.validate(jd, manifest.load(jd))
        result = resubmit(jd, SCRATCH if start == 0 else f"ck{start:04d}", sandbox=True,
                          salt=self.fixture.schedule)
        rec = trusted.ledger(jd)[-1]
        if rec["run_id"] != result.run_id or rec["run_id"] in before:
            raise RuntimeError(f"recovery was not launched: {result.status} {result.error_class}")
        redo = lost_work(jd, self.fixture.frontier, before, scratch_allowance=0)
        # This NEW protocol projects the existing semantic/provenance checks with
        # cost excluded. It does not claim to be historical Task B verification.
        semantic = check(jd, None, lost=0, before=before, ref=self.fixture.reference,
                         salt=self.fixture.schedule, format_check="v1.2", relaunches=1)
        provenance = source_hashes(jd) == self.fixture.source
        valid = bool(result.ok and semantic.ok and provenance)
        receipt = {"version": VERSION, "source": "executed", "action": action,
                   "platform": platform.platform(), "machine": platform.machine(),
                   "isolation_backend": "macOS sandbox-exec", "exec_mode": "local_os_sandbox",
                   "completed": result.ok, "run_id": result.run_id, "resume_from": rec["resume_from"],
                   "start_step": rec["start_step"], "semantic_valid": valid,
                   "checkpoint_continuity": semantic.restores_consistent if start else None,
                   "completion_status": result.status, "execution_error": result.error_class,
                   "source_preserved": provenance, "source_sha256": source_hashes(jd),
                   "input_manifest_validated": True,
                   "redone_optimizer_steps": redo, "scratch_exemption": 0,
                   "cost_score": max(0.0, 1 - redo / self.fixture.config["steps"]) if valid else 0.0,
                   "checks_with_cost_excluded": asdict(semantic),
                   "infrastructure_error": None}
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt


def state_rule(checkpoint: dict, history: dict) -> str:
    """Transparent selective-repair baseline; consumes public measurements only."""
    current = checkpoint["delivered"]
    expected = next(r["state"] for r in history["records"] if r["committed_step"] == current["step"])
    if current["sampler_position"] != expected["sampler_position"]:
        return "restore_sampler"
    if current["optimizer"] != expected["optimizer"]:
        return "restore_optimizer"
    return "resume"
