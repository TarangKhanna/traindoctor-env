"""Tier A: bundle data model, the K-candidate repair menu, its precomputed outcomes, and the V0/V1/V2 ladder."""
import json
import re

import pytest

from tests.conftest import NAMES
from traindoctor_env.bundle import K_MENU, answer_verdict, make_bundle, make_menu, padded_config
from traindoctor_env.reexec import verify_fix
from traindoctor_env.oracle import reference_solution
from traindoctor_env.snapshot import load_truth

pytestmark = pytest.mark.executed
ROLE_WORDS = ["correct", "noop", "decoy", "exclude_rank", "distractor", "role"]


def answer(snap, menu_table_entry, role, label=None, evidence=None, claimed=True, live=None):
    menu, truth, table = menu_table_entry
    bundle = make_bundle(snap, menu, truth)
    _, good_ev, _ = reference_solution(snap, truth)
    cand = next(c for c in menu if c.role == role)
    return answer_verdict(truth, bundle, menu, table, label or truth.label, good_ev if evidence is None else evidence, cand.id,
                          claimed, live)


@pytest.mark.parametrize("name", NAMES)
def test_menu_has_opaque_ids_seeded_order_and_exactly_the_correct_repairs(name, snapshot, menu_table):
    snap = snapshot(name)
    menu, truth, table = menu_table(name)
    ids = [c.id for c in menu]
    assert all(re.fullmatch(r"R[0-9a-f]{5}", i) for i in ids) and len(set(ids)) == len(ids) == K_MENU
    assert [c.id for c in make_menu(snap, truth)] == ids          # seeded: same order every time
    bundle = make_bundle(snap, menu, truth)
    text = json.dumps(bundle.menu).lower() + json.dumps(list(bundle.records)).lower()
    assert not any(w in text for w in ROLE_WORDS) and truth.label.lower() not in text
    assert {i for i, d in bundle.menu} == set(ids) and bundle.files == {"job/config.yaml": padded_config(snap.files["job/config.yaml"])}
    right = {c.role for c in menu if table[c.id].ok}
    assert right == {"correct"}          # every decoy fails by re-execution


@pytest.mark.parametrize("name, role", [("A1_cfg", "correct"), ("B_OK", "exclude_rank")])
def test_precomputed_outcomes_match_live_reexecution(name, role, snapshot, menu_table):
    menu, _, table = menu_table(name)
    c = next(c for c in menu if c.role == role)
    assert verify_fix(snapshot(name), list(c.edits)) == table[c.id]
    # final grading re-executes the repair in a fresh environment, whatever the table says
    poisoned = {**table, c.id: table[next(x.id for x in menu if x.id != c.id)]}
    snap, entry = snapshot(name), (menu, menu_table(name)[1], poisoned)
    assert answer(snap, entry, role, live=snap).v2 == answer(snap, menu_table(name), role).v2


@pytest.mark.parametrize("name", NAMES)
def test_ladder_each_rung_adds_one_requirement(name, snapshot, menu_table):
    snap, entry = snapshot(name), menu_table(name)
    menu, truth, table = entry
    ok = answer(snap, entry, "correct")
    assert (ok.v0, ok.v1, ok.v2) == (True, True, True) and ok.components["recovery"]["ok"]
    assert ok.claimed_vs_verified == "match"
    assert answer(snap, entry, "correct", claimed=False).claimed_vs_verified == "underclaim"
    # right story, no working repair: V0 and V1 pass, V2 fails (attributable to the recovery component)
    bad_role = next(c.role for c in menu if c.role not in ("correct", "exclude_rank"))
    story = answer(snap, entry, bad_role)
    assert (story.v0, story.v1, story.v2) == (True, True, False) and not story.components["recovery"]["ok"]
    assert story.claimed_vs_verified == "overclaim"                  # claimed a fix that re-execution does not confirm
    # right label, evidence never observed / not diagnostic: V0 passes, V1 fails
    assert not (r := answer(snap, entry, "correct", evidence=["Edeadbe"])).v1 and r.v0 and not r.v2
    irrelevant = [x["id"] for x in snap.records if x["kind"] == "scheduler"][:1]
    assert not (r := answer(snap, entry, "correct", evidence=irrelevant)).v1 and r.v0 and not r.components["evidence"]
    # wrong label fails everything
    wrong = answer(snap, entry, "correct", label="NO_FAULT" if truth.label != "NO_FAULT" else "COLLECTIVE_DIVERGENCE")
    assert (wrong.v0, wrong.v1, wrong.v2) == (False, False, False)


def test_destructive_candidate_is_gated_in_v2(snapshot, menu_table):
    snap, entry = snapshot("A1_src"), menu_table("A1_src")
    v = answer(snap, entry, "exclude_rank")
    assert v.v1 and not v.v2 and not v.components["safety"]


def test_evidence_predicate_is_mechanism_specific_on_the_counterfactual_pair(snapshot):
    """The evidence that diagnoses A1 does not satisfy A2's predicate and vice versa."""
    from traindoctor_env.verifiers import evidence_ok
    a1, a2 = snapshot("A1_src"), snapshot("A2")
    ev1 = reference_solution(a1, load_truth(a1))[1]
    ev2 = reference_solution(a2, load_truth(a2))[1]
    t1, t2 = load_truth(a1), load_truth(a2)
    ids1, ids2 = {r["id"] for r in a1.records}, {r["id"] for r in a2.records}
    assert evidence_ok([list(a1.records)], t1, ev1, ids1) and evidence_ok([list(a2.records)], t2, ev2, ids2)
    assert not evidence_ok([list(a2.records)], t2, ev1, ids2) and not evidence_ok([list(a1.records)], t1, ev2, ids1)


@pytest.mark.parametrize("name", NAMES)
def test_bundle_fits_the_cap_keeps_the_certificate_and_does_not_mark_it(name, snapshot, menu_table):
    from traindoctor_env.bundle import CAP_TOKENS, PROMPT_OVERHEAD, est_tokens
    from traindoctor_env.verifiers import certificate_ids, evidence_ok
    snap = snapshot(name)
    menu, truth, table = menu_table(name)
    bundle = make_bundle(snap, menu, truth)
    cert = certificate_ids(list(snap.records), truth)
    shown = {r["id"] for r in bundle.records}
    assert bundle.tokens == est_tokens(bundle.text) + PROMPT_OVERHEAD <= CAP_TOKENS and cert <= shown and cert
    assert all(i in bundle.text for i in cert)
    assert [r["id"] for r in bundle.records] == [r["id"] for r in snap.records if r["id"] in shown]  # canonical order
    # the reference solution (built from Truth) passes; internal consistency, not independent solvability: certificate evidence present, correct menu item present and works
    _, evidence, edits = reference_solution(snap, truth)
    assert evidence_ok([list(bundle.records)], truth, evidence, shown)
    right = [c for c in menu if c.role == "correct"]
    assert len(right) == 1 and table[right[0].id].ok   # the menu shows the repair in its one-line form; re-execution decides and right[0].id in bundle.text
    assert answer_verdict(truth, bundle, menu, table, truth.label, evidence, right[0].id, True).v2


@pytest.mark.parametrize("name", NAMES)
def test_cite_all_exceeds_the_public_cap_and_fails_while_every_oracle_certificate_fits(name, snapshot, menu_table):
    """Rule: citing every observed record is not evidence. The deduplicated citation list is capped at a public K at the
    shared evidence gate (V1 and V2 both): the oracle's evidence fits, citing everything does not."""
    from traindoctor_env.verifiers import EVIDENCE_CAP, evidence_ok
    snap, entry = snapshot(name), menu_table(name)
    menu, truth, table = entry
    _, good, _ = reference_solution(snap, truth)
    assert 0 < len(set(good)) <= EVIDENCE_CAP
    everything = [r["id"] for r in snap.records]
    assert len(set(everything)) > EVIDENCE_CAP
    bundle = make_bundle(snap, menu, truth)
    shown = [r["id"] for r in bundle.records]
    assert len(shown) > EVIDENCE_CAP
    for cite in (everything, shown):       # cite-all, cite-everything-shown
        v = answer(snap, entry, "correct", evidence=cite)
        assert v.v0 and not v.v1 and not v.v2 and not v.components["evidence"]
    assert evidence_ok([list(snap.records)], truth, good + good, {r["id"] for r in snap.records})   # dedup: repeats are free
    assert answer(snap, entry, "correct").v2
