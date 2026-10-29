"""State-dependent evidence and the absence of an initial identity cue."""
import copy
import json

import torch

from jobs import manifest
from jobs.template.ckpt import digest
from traindoctor_env.state_recovery_v1 import RecoveryIncident, _update_manifest, state_rule


def evidence():
    state = {"step": 8, "sampler_position": [0, 8], "model": {"sum": 3}, "optimizer": {"sum": 7}}
    return {"delivered": copy.deepcopy(state)}, {"records": [{"committed_step": 8, "state": state}]}


def test_decision_changes_with_relevant_component_evidence():
    ckpt, history = evidence()
    assert state_rule(ckpt, history) == "resume"
    ckpt["delivered"]["sampler_position"] = [0, 4]
    assert state_rule(ckpt, history) == "restore_sampler"
    ckpt, history = evidence()
    ckpt["delivered"]["optimizer"] = {"sum": 5}
    assert state_rule(ckpt, history) == "restore_optimizer"


def test_initial_observation_and_actions_do_not_read_private_state():
    # No fixture paths, hidden cause or correctness fields are inspected by status.
    incident = RecoveryIncident(None, None)
    a, b = incident.status("node-A"), incident.status("node-B")
    assert a.pop("display_name") == "node-A"
    assert b.pop("display_name") == "node-B"
    assert a == b
    assert not ({"cause", "label", "oracle", "valid", "correct_action"} & a.keys())


def test_full_restore_refreshes_manifest_integrity(tmp_path):
    path = tmp_path / "ckpt/ck0008"
    path.mkdir(parents=True)
    restored = {"model.fc1.weight": torch.ones((2, 2)), "step": torch.tensor([8])}
    torch.save(restored, path / "state.pt")
    manifest.path(tmp_path).write_text(json.dumps([{"ckpt_id": "ck0008", "step": 8,
        "shards": [{"file": "ck0008/state.pt", "sha256": "digest-of-previous-delivery"}]}]))
    _update_manifest(tmp_path, restored)
    manifest.validate(tmp_path, manifest.load(tmp_path))
    assert manifest.load(tmp_path)[0]["shards"][0]["sha256"] == digest(restored)
