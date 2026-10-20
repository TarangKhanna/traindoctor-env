"""Agent config is refused before any shard allocation or worker launch, without numeric coercion."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jobs import launcher, trusted
from jobs.build import base_config
from jobs.config import BadConfig, validate_config
from jobs.dataserver import DataSpec
from jobs.resubmit import resubmit

LIMITS = [
    ("world_size", 1, 16), ("data.num_samples", 1, 4096), ("batch_size", 1, 256),
    ("steps", 1, 1000), ("hidden", 1, 1024), ("ckpt_every", 1, 1000),
    ("seed", 0, 2**31 - 1), ("data.shard_steps", 1, 1000), ("sync.warmup_steps", 0, 1000),
    ("lr", 1e-8, 10), ("pg_timeout_s", 0.001, 120), ("watchdog_s", 0.001, 120),
    ("step_time_s", 0, 1), ("data.producer.shard_mb", 0.001, 1024),
    ("data.producer.cached_shards", 0, 1000), ("data.producer.throttle_mb_s.0", 0.001, 1e6),
]


def config():
    cfg = base_config(2, 7, 4, 4, 0)
    cfg["data"]["producer"] = dict(shard_mb=4, cached_shards=1, throttle_mb_s={"0": 400, "1": 400})
    return cfg


def set_value(cfg, path, value):
    parts = path.split(".")
    for key in parts[:-1]:
        cfg = cfg[key]
    cfg[parts[-1]] = value


@pytest.mark.parametrize("path,lower,upper", LIMITS)
@pytest.mark.parametrize("bad", [10**100, -1, float("nan"), float("inf"), -float("inf"), "4", None, True, [], {}])
def test_pathological_values_are_rejected_directly(path, lower, upper, bad):
    cfg = config()
    set_value(cfg, path, bad)
    with pytest.raises(BadConfig):
        validate_config(cfg)


@pytest.mark.parametrize("path,lower,upper", LIMITS)
def test_declared_bounds_are_enforced(path, lower, upper):
    for bad in (lower - 1, upper + 1):
        cfg = config()
        set_value(cfg, path, bad)
        with pytest.raises(BadConfig):
            validate_config(cfg)
    if lower > 0:
        cfg = config()
        set_value(cfg, path, 0)
        with pytest.raises(BadConfig):
            validate_config(cfg)


@pytest.mark.parametrize("path", ["world_size", "data.num_samples", "batch_size", "steps", "hidden", "ckpt_every",
                                  "seed", "data.shard_steps", "sync.warmup_steps", "data.producer.cached_shards"])
def test_integer_fields_reject_fractional_values(path):
    cfg = config()
    set_value(cfg, path, 1.5)
    with pytest.raises(BadConfig):
        validate_config(cfg)


def test_shipped_and_boundary_configs_validate_without_allocating():
    validate_config(base_config(4, 31337, 5, 6, 1))
    cfg = config()
    for path, lower, upper in LIMITS:
        set_value(cfg, path, upper)
    cfg["data"]["producer"]["throttle_mb_s"] = {str(r): 1e6 for r in range(16)}
    validate_config(cfg)
    cfg["step_time_s"] = cfg["seed"] = cfg["sync"]["warmup_steps"] = 0
    cfg["data"]["producer"]["cached_shards"] = 0
    validate_config(cfg)


@pytest.mark.parametrize("changes", [dict(world=10**100), dict(num_samples=10**100), dict(batch_size=10**100),
                                     dict(seed=10**100), dict(world=0), dict(batch_size=0),
                                     dict(num_samples=1), dict(num_samples=True)])
def test_data_spec_cannot_bypass_validation(changes):
    params = dict(world=2, num_samples=96, batch_size=4, seed=7)
    with pytest.raises(BadConfig):
        DataSpec(**{**params, **changes})


@pytest.mark.parametrize("launch", [launcher.run_job, resubmit])
def test_invalid_config_is_a_recorded_rejection_before_launch(tmp_path, monkeypatch, launch):
    cfg = config()
    cfg["data"]["num_samples"] = 10**100
    monkeypatch.setattr(launcher, "job_config", lambda _: deepcopy(cfg))
    forbidden = Mock(side_effect=AssertionError("resource construction before validation"))
    monkeypatch.setattr(launcher, "DataServer", forbidden)
    monkeypatch.setattr(launcher, "free_port", forbidden)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    result = launch(tmp_path)
    assert result.status == "refused" and result.error_class == "BAD_CONFIG"
    assert "refused" in trusted.events(tmp_path)[-1]["event"]
    assert not trusted.ledger(tmp_path) and not (tmp_path / ".lock").exists()
    forbidden.assert_not_called()


def test_launch_params_does_not_coerce_wrong_types(tmp_path, monkeypatch):
    cfg = config()
    cfg["world_size"] = "2"
    monkeypatch.setattr(launcher, "job_config", lambda _: cfg)
    with pytest.raises(BadConfig):
        launcher.launch_params(tmp_path)
