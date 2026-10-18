"""Reference solver. Allowed to read Truth; used to prove each scenario is solvable and as the
positive example for the verifier."""
from __future__ import annotations

from jobs.build import Edit
from traindoctor_env.scenario import Truth
from traindoctor_env.snapshot import Snapshot
from traindoctor_env.verifiers import edits_of, evidence_requirements


def reference_solution(snap: Snapshot, truth: Truth) -> tuple[str, list[str], list[Edit]]:
    """(label, evidence ids, repair edits). The evidence is the first alternative of each required role."""
    evidence = sorted({i for group in evidence_requirements(list(snap.records), truth) for i in sorted(group, key=sorted)[0]})
    return truth.label, evidence, edits_of(truth)
