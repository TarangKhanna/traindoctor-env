"""Tier A metadata must not identify the variant. Rule: within a task family, nothing about a bundle other than its
CONTENT (menu size, menu item sizes, bundle length, record counts per kind, file sizes) predicts the answer better
than always guessing the majority class."""
import random
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.build import materialize
from jobs.config import job_config
from traindoctor_env.bundle import ITEM_CHARS, K_MENU, make_bundle, make_menu
from traindoctor_env.scenario import Scenario, job_spec
from traindoctor_env.snapshot import execute, job_files, load_truth
from traindoctor_env.scenario import Truth

VARIANTS = [("A1", 0), ("A1", 1), ("A2", 0), ("B_SKIP", 0), ("B_DUP", 0), ("B_OK", 0)]
KINDS_SEEN = ("job", "process", "scheduler", "ckpt", "log", "trace", "samples")


def fake_snapshot(sc: Scenario, tmp: Path):
    """Enough of a Snapshot for make_menu: the buggy job's files and config, no execution."""
    spec = job_spec(sc)
    materialize(tmp, spec.config, spec.bug)
    truth = Truth(sc.seed, sc.kind, sc.variant, sc.label, spec.task, [asdict(e) for e in spec.fix], spec.details)
    return SimpleNamespace(files=job_files(tmp), config=job_config(tmp), seed=sc.seed), truth


@pytest.mark.parametrize("kind, variant", VARIANTS)
def test_every_variant_shows_the_same_number_of_equally_long_menu_items(kind, variant):
    for seed in range(20):
        with tempfile.TemporaryDirectory() as t:
            snap, truth = fake_snapshot(Scenario(seed, kind, variant, 2 + seed % 3, 1000 + seed), Path(t) / "jd")
            menu = make_menu(snap, truth)
            assert len(menu) == K_MENU and len({c.id for c in menu}) == K_MENU
            assert {len(c.diff(snap.files)) for c in menu} == {ITEM_CHARS}, (kind, seed)


def features(bundle, menu_texts):
    counts = {k: sum(1 for r in bundle.records if r["kind"] == k) for k in KINDS_SEEN}
    lens = sorted(len(t) for t in menu_texts)
    return {"menu_size": len(menu_texts), "tokens": bundle.tokens, "chars": len(bundle.text), "records": len(bundle.records),
            **{f"n_{k}": v for k, v in counts.items()}, "item_min": lens[0], "item_max": lens[-1],
            "item_mean": statistics.mean(lens), "config_chars": len(bundle.files["job/config.yaml"]),
            "files_chars": sum(len(v) for v in bundle.files.values())}


def loo_stump_accuracy(rows, labels, feature):
    """Leave-one-out accuracy of the best single-threshold rule on one feature (a weak but fair metadata predictor)."""
    hits = 0
    for i in range(len(rows)):
        tr = [(r[feature], l) for j, (r, l) in enumerate(zip(rows, labels)) if j != i]
        best, rule = -1, None
        for thr in sorted({v for v, _ in tr}):
            for lo, hi in ((a, b) for a in set(labels) for b in set(labels) if a != b):
                acc = sum((lo if v <= thr else hi) == l for v, l in tr)
                if acc > best:
                    best, rule = acc, (thr, lo, hi)
        thr, lo, hi = rule
        hits += (lo if rows[i][feature] <= thr else hi) == labels[i]
    return hits / len(rows)


def loo_nn_accuracy(rows, labels):
    """Leave-one-out 1-nearest-neighbour on all features, each standardized."""
    keys = sorted(rows[0])
    sd = {k: statistics.pstdev([r[k] for r in rows]) or 1.0 for k in keys}
    dist = lambda a, b: sum(((a[k] - b[k]) / sd[k]) ** 2 for k in keys)  # noqa: E731
    return sum(labels[i] == labels[min((j for j in range(len(rows)) if j != i), key=lambda j: dist(rows[i], rows[j]))]
               for i in range(len(rows))) / len(rows)


@pytest.mark.sweep
@pytest.mark.executed
def test_metadata_alone_does_not_beat_the_majority_class_within_a_task_family(tmp_path):
    """60 executed scenarios (world 2): per family, the best single-feature threshold rule and a nearest-neighbour rule
    over all metadata features, leave-one-out, must not beat always answering the majority label by more than a small
    margin (n is small: chance alone moves accuracy a few points). Features: menu size, item sizes, bundle size,
    record counts per kind, file sizes."""
    plan = [(i, kind, var) for i, (kind, var) in enumerate(VARIANTS * 10)]
    scen = [Scenario(4000 + i, kind, var, 2, 9000 + i) for i, kind, var in plan]
    with ThreadPoolExecutor(2) as ex:
        snaps = list(ex.map(lambda sc: execute(sc, tmp_path / f"s{sc.seed}"), scen))
    families: dict[str, tuple[list, list]] = {"A": ([], []), "B": ([], [])}
    for sc, snap in zip(scen, snaps):
        truth = load_truth(snap)
        menu = make_menu(snap, truth)
        bundle = make_bundle(snap, menu, truth)
        rows, labels = families[sc.kind[0]]
        rows.append(features(bundle, [d for _, d in bundle.menu]))
        labels.append(truth.label)
    for fam, (rows, labels) in families.items():
        base = max(labels.count(l) for l in set(labels)) / len(labels)
        margin = 0.15
        worst = {f: loo_stump_accuracy(rows, labels, f) for f in rows[0]}
        print(f"family {fam}: n={len(rows)} majority={base:.2f} nn={loo_nn_accuracy(rows, labels):.2f} stumps={ {k: round(v, 2) for k, v in worst.items()} }")
        assert all(a <= base + margin for a in worst.values()), (fam, base, worst)
        assert loo_nn_accuracy(rows, labels) <= base + margin
