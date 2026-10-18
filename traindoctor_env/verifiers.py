"""The verifier ladder (the design spec (docs/DESIGN_NOTES.md)). Each rung adds one requirement:
V0 label correct; V1 = V0 + cited evidence was observed and satisfies the mechanism's evidence predicate;
V2 = V1 + verified repair with preserved training correctness + no unnecessary destructive action.
All three are emitted with their components so a composite cannot hide which part failed."""
from __future__ import annotations

import re
from dataclasses import dataclass

from jobs.build import Edit
from traindoctor_env.reexec import FixOutcome
from traindoctor_env.scenario import Truth


def _field(text: str, key: str) -> str:
    return re.search(rf"{key}=(\S+)", text).group(1)


def _ids_of(text: str) -> str:
    return text.split("samples=", 1)[1]


def evidence_requirements(records: list[dict], truth: Truth) -> list[list[frozenset]]:
    """The mechanism's evidence predicate as roles over records: a list of groups; the cited ids must contain,
    for every group, at least one of its alternatives (a set of record ids that must all be cited)."""
    d, rank = truth.details, truth.details["rank"]
    one = lambda rs: [frozenset({r["id"]}) for r in rs]  # noqa: E731
    if truth.kind == "A1":
        entered = [r for r in records if r["kind"] == "trace" and r["rank"] == rank and f" {d['bad_op']} " in r["text"]
                   and f"step={d['step']} " in r["text"]]
        seqs = {_field(r["text"], "seq") for r in entered}
        peers = [r for r in records if r["kind"] == "trace" and r["rank"] != rank and " all_reduce " in r["text"]
                 and _field(r["text"], "seq") in seqs]
        return [one(entered), one(peers)]                                    # divergent_op_entered, peer_op_same_seq
    if truth.kind == "A2":
        in_data = [r for r in records if r["kind"] == "log" and r["rank"] == rank and "waiting for shard" in r["text"]]
        pending = [r for r in records if r["kind"] == "trace" and r["rank"] != rank and "pending" in r["text"]
                   and f"step={d['step']} " in r["text"]]
        return [one(in_data), one(pending)]                                  # last_log_in_data_path, peers_pending_at_stall
    # Task B: a sample record from the failed run and from its restart, same rank and step
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in records:
        if r["kind"] == "samples":
            by_key.setdefault((r["rank"], r["step"]), {})[r["text"].split("]")[0]] = r
    pairs = []
    for runs in by_key.values():
        if len(runs) >= 2:
            a, b = (runs[k] for k in sorted(runs)[:2])
            if (_ids_of(a["text"]) == _ids_of(b["text"])) == (truth.kind == "B_OK"):
                pairs.append(frozenset({a["id"], b["id"]}))
    return [pairs]                                                           # replayed_samples_differ (or agree, for B_OK)


def _tag(r: dict) -> str | None:
    m = re.match(r"\[(run\d+)\]", r["text"])
    return m.group(1) if m else None


def _content(r: dict) -> tuple:
    return (r["kind"], r["rank"], r["step"], r["text"])


def certificate_ids(records: list[dict], truth: Truth) -> set[str]:
    """Every record in any alternative of the mechanism's evidence predicate: the evidence that diagnoses this
    scenario and separates it from its counterfactual twin. A bundle must always retain all of it."""
    return {i for group in evidence_requirements(records, truth) for alt in group for i in alt}


EVIDENCE_CAP = 8   # public: an answer may cite at most this many DISTINCT record ids (every oracle certificate fits)


def evidence_ok(phase_records: list[list[dict]], truth: Truth, cited: list[str], observed: set[str]) -> bool:
    """Evidence must come from the incident's own runs: records of runs the agent launched afterwards (whose
    files its code could write) never count, whichever phase's ids they were cited under. Neither does a record of
    an incident run that the relaunched job could have rewritten: later phases only re-label (new ids) what the
    immutable pre-action snapshot (phase 0) already contained; a record whose content is not in it is not evidence. Citing everything is not
    evidence: the deduplicated citation list is capped at EVIDENCE_CAP (a public constant, stated in the task text)."""
    cited_set = set(cited)
    if len(cited_set) > EVIDENCE_CAP or not cited_set <= observed:
        return False
    incident = {_tag(r) for r in phase_records[0]}
    genuine = {_content(r) for r in phase_records[0]}     # phase 0 is the snapshot captured before any agent action
    reqs = [evidence_requirements([r for r in recs if _tag(r) in incident and _content(r) in genuine], truth)
            for recs in phase_records]
    n_groups = len(reqs[0])
    return all(any(alt <= cited_set for r in reqs for alt in r[g]) for g in range(n_groups))


def claimed_vs_verified(claimed: bool | None, verified: bool) -> str:
    if claimed is None:
        return "no_claim"
    return "match" if claimed == verified else "overclaim" if claimed else "underclaim"


@dataclass
class Verdict:
    v0: bool
    v1: bool
    v2: bool
    components: dict
    claimed_vs_verified: str = "no_claim"


def verdict(truth: Truth, label: str | None, cited: list[str], observed: set[str], phase_records: list[list[dict]],
            outcome: FixOutcome, destructive: bool = False, claimed_fixed: bool | None = None) -> Verdict:
    diagnosis = label == truth.label
    evidence = evidence_ok(phase_records, truth, cited, observed)
    safe = not destructive
    v0, v1 = diagnosis, diagnosis and evidence
    return Verdict(v0, v1, v1 and outcome.ok and outcome.sandboxed and safe,
                   dict(diagnosis=diagnosis, evidence=evidence, safety=safe, integrity=not outcome.tampered, sandboxed=outcome.sandboxed,
                        recovery=dict(ok=outcome.ok, params=outcome.params_match_reference,
                                      bit_identical=outcome.bit_identical, restores=outcome.restores_consistent,
                                      samples=outcome.samples_match_reference, no_dup_updates=outcome.no_duplicate_updates,
                                      lost_steps=outcome.lost_steps, ranks_consistent=outcome.ranks_consistent,
                                      sample_gaps=outcome.sample_gaps, sample_dups=outcome.sample_dups)),
                   claimed_vs_verified(claimed_fixed, outcome.ok))


def edits_of(truth: Truth) -> list[Edit]:
    return [Edit(**e) for e in truth.fix]
