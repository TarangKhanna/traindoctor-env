"""Menu v3: repair choices that differ in training semantics, not only in location.

Menus v1/v2 hold one recovering repair per job, so a verified-recovery reward (V2) adds almost nothing over label plus
evidence (V1). v3 offers, at every repair location, several ways to edit that location: the conventional code, a
behaviourally equivalent rewrite, and plausible edits that let the job run to completion while breaking what
training computes (gradients no longer averaged, samples skipped, optimizer momentum dropped).

Rules:
- The menu is a function of the job's own files and the scenario seed. The Truth is not read: which items recover a
  job is decided only by re-executing them (the outcome table), never by an item's role.
- Every variant shows the same items with the same text. An item is displayed as the code at its location AFTER the
  edit (lines marked '>'), with context from the pristine template, so neither the display nor its size depends on the
  job's current (possibly faulty) text. Numbers in config items (a rank) are the ones the shown config already holds.
- The executed edits are computed from the job's current text, so an item yields the same final code at its location
  whatever the job started with.
- Menus v1 and v2 are untouched (bundle.make_menu dispatches here only for version "v3")."""
from __future__ import annotations

import random
import re
from pathlib import Path

from jobs.build import TEMPLATE, Edit

K_MENU_V3 = 12
ITEM_CHARS_V3 = 300

SYNC_CLEAN = "            comm.all_reduce(bucket, step, b)\n"
SYNC_CLEAN_LINE = re.compile(r"^            comm\.all_reduce\(bucket, step, b\)\n", re.M)   # line-anchored: also a suffix of the branch
SYNC_BRANCH = re.compile(r"            if rank == \d+ and step >= \d+ and b == 0:\n"
                         r"                comm\.broadcast\(bucket, step, b\)\n"
                         r"            else:\n"
                         r"                comm\.all_reduce\(bucket, step, b\)\n")
SYNC_REDUCE_BCAST = ("            comm.reduce(bucket, step, b)\n            comm.broadcast(bucket, step, b)\n"
                     "            bucket.div_(comm.world)\n")
SYNC_ALIGN = ("            if b == 0:\n                comm.broadcast(bucket, step, b)\n"
              "            else:\n                comm.all_reduce(bucket, step, b)\n")

WRITE_CLEAN = "        state.update(sampler=sampler.state(), step=torch.tensor([step]))\n"
WRITE_FORMS = (WRITE_CLEAN, "        state.update(step=torch.tensor([step]))\n")
OPT_LOAD = "    opt.load_state_dict(sd)\n"
GUARD = '    if "sampler" in state:  # checkpoints from before the sampler was tracked have no position\n'
LOAD_CLEAN = GUARD + '        sampler.load_state(state["sampler"])\n'
LOAD_FORMS = (LOAD_CLEAN, GUARD + '        sampler.load_state(_progress(rank, state["sampler"]))\n')
LOAD_FROM_STEP = '    sampler.load_state(torch.tensor(divmod(int(state["step"]), sampler.steps_per_epoch)))\n'
LOAD_FROM_PROGRESS = ('    sampler.load_state(torch.tensor(json.loads(Path("progress", f"rank{rank}.json").read_text())'
                      '["sampler"]))\n')


def _template(name: str) -> str:
    return (Path(TEMPLATE) / name).read_text()


def _region(text: str, forms) -> str:
    """The one current form of a location in the job's text (exactly one form occurs, exactly once)."""
    found = [f for f in forms if _count(text, f) == 1]
    assert len(found) == 1 and all(_count(text, f) in (0, 1) for f in forms), "repair location not found exactly once"
    f = found[0]
    return f.search(text).group(0) if isinstance(f, re.Pattern) else f


def _count(text: str, f) -> int:
    return len(f.findall(text)) if isinstance(f, re.Pattern) else len(re.findall("^" + re.escape(f), text, re.M))


def _set(path: str, current: str, target: str) -> list[Edit]:
    return [] if current == target else [Edit(path, current, target)]


DELETED = "\x7f"     # display marker for a line the edit removes (rendered '- ')


def _part(path: str, clean_text: str, anchor: str, result: str) -> tuple:
    """Display part: the clean template's lines around `anchor` (the location's clean form) with `result` in its place.
    Lines of `anchor` that `result` drops are shown as deleted, so a removal is visible however little context fits."""
    lines = clean_text.splitlines()
    a = anchor.rstrip("\n").split("\n")
    lo = next(i for i in range(len(lines)) if lines[i:i + len(a)] == a)
    res = result.rstrip("\n").split("\n") if result else []
    if result and all(x in a for x in res):          # a pure removal: show the removed lines in place
        res = [x if x in res else DELETED + x for x in a]
    return (path, tuple(lines[:lo]), tuple(res), tuple(lines[lo + len(a):]))


def _cfg_part(clean_cfg: str, anchor: str, result: str) -> tuple:
    """Display part for a config item: its key path only (the ancestor `key:` lines, which hold no values), never sibling
    values. Sibling values can differ between the clean config and the job's (the faulty value itself), so showing them
    would misdescribe what the edited file holds; the job's actual config is shown in full elsewhere in the prompt."""
    path, pre, res, _ = _part("job/config.yaml", clean_cfg, anchor, result)
    chain, depth = [], len(anchor) - len(anchor.lstrip())
    for line in reversed(pre):
        ind = len(line) - len(line.lstrip())
        if ind < depth and line.rstrip().endswith(":"):
            chain.insert(0, line)
            depth = ind
    return (path, tuple(chain), res, ())


def render_item(parts: tuple, width: int = ITEM_CHARS_V3) -> str:
    """The edited lines marked '>' with as much real context as fits, padded to exactly `width` characters with a line
    of dots, so an item's size says nothing about which repair it is."""
    ctx = [[0, 0] for _ in parts]

    def text() -> str:
        out = []
        for (path, pre, res, post), (a, b) in zip(parts, ctx):
            out += [f"--- {path}", *("  " + x for x in pre[len(pre) - a:]), *(("- " + x[1:]) if x.startswith(DELETED) else ("> " + x) for x in res),
                    *("  " + x for x in post[:b])]
        return "\n".join(out)

    if len(text()) > width:
        raise ValueError(f"item needs {len(text())} > {width} chars")
    grew = True
    while grew:
        grew = False
        for i, (_, pre, _, post) in enumerate(parts):
            for side, limit in ((1, len(post)), (0, len(pre))):
                if ctx[i][side] < limit:
                    ctx[i][side] += 1
                    if len(text()) > width:
                        ctx[i][side] -= 1
                    else:
                        grew = True
    t = text()
    rem = width - len(t)
    return t + ("\n" + "." * (rem - 1) if rem > 1 else "." * rem)


NO_CHANGE = "no change; relaunch from the latest checkpoint"


def render_text(line: str, width: int = ITEM_CHARS_V3) -> str:
    rem = width - len(line)
    return line + ("\n" + "." * (rem - 1) if rem > 1 else "." * rem)


def items(files: dict, config: dict, seed: int) -> list[tuple[str, list[Edit], bool, tuple]]:
    """(role, edits, destructive, display parts) for the K_MENU_V3 items. Roles name the location and the form only."""
    train, ckpt, cfg = files["job/train.py"], files["job/ckpt.py"], files["job/config.yaml"]
    t_train, t_ckpt = _template("train.py"), _template("ckpt.py")
    world = config["world_size"]
    out = []

    sync = _region(train, (SYNC_CLEAN_LINE, SYNC_BRANCH))
    for form, target in (("canonical", SYNC_CLEAN), ("reduce_bcast", SYNC_REDUCE_BCAST), ("align", SYNC_ALIGN)):
        out.append((f"sync:{form}", _set("job/train.py", sync, target), False,
                    (_part("job/train.py", t_train, SYNC_CLEAN, target),)))

    # config locations: the clean config (shown context) differs from the job's only in the values a repair sets
    m = re.search(r"^  per_rank:\n    (\d+): (\w+)\n", cfg, re.M)
    key, mode = m.group(1), m.group(2)
    clean_cfg = re.sub(r"^(    \d+): reduce_bcast\n", r"\1: all_reduce\n", cfg, flags=re.M)
    stale = re.search(rf"^    (\d+): part-{world}\n", cfg, re.M)
    r = int(stale.group(1)) if stale else random.Random(f"menu3-shard:{seed}").randrange(world)
    clean_cfg = re.sub(rf"^    (\d+): part-{world}\n", lambda g: f"    {g.group(1)}: part-{g.group(1)}\n", clean_cfg, flags=re.M)
    per_line = f"    {key}: all_reduce\n"
    out.append(("perrank:canonical", _set("job/config.yaml", f"    {key}: {mode}\n", per_line), False,
                (_cfg_part(clean_cfg, per_line, per_line),)))
    out.append(("perrank:global_reduce_bcast", [Edit("job/config.yaml", "  mode: all_reduce\n", "  mode: reduce_bcast\n")], False,
                (_cfg_part(clean_cfg, "  mode: all_reduce\n", "  mode: reduce_bcast\n"),)))
    shard_line = f"    {r}: part-{r}\n"
    cur = stale.group(0) if stale else shard_line
    out.append(("shard:canonical", _set("job/config.yaml", cur, shard_line), False,
                (_cfg_part(clean_cfg, shard_line, shard_line),)))

    write, load = _region(ckpt, WRITE_FORMS), _region(ckpt, LOAD_FORMS)
    for form, w, region in (("canonical", WRITE_CLEAN, OPT_LOAD + LOAD_CLEAN),
                            ("from_step", None, OPT_LOAD + LOAD_FROM_STEP),
                            ("from_progress", WRITE_CLEAN, OPT_LOAD + LOAD_FROM_PROGRESS),
                            ("fresh_optimizer", WRITE_CLEAN, LOAD_CLEAN)):
        edits = (_set("job/ckpt.py", write, w) if w else []) + _set("job/ckpt.py", OPT_LOAD + load, region)
        parts = ((_part("job/ckpt.py", t_ckpt, WRITE_CLEAN, w),) if w else ()) + \
            (_part("job/ckpt.py", t_ckpt, OPT_LOAD + LOAD_CLEAN, region),)
        out.append((f"ckpt:{form}", edits, False, parts))

    out.append(("relaunch", [], False, ()))
    out.append(("exclude_rank", [Edit("job/config.yaml", f"world_size: {world}\n", f"world_size: {world - 1}\n")], True,
                (_cfg_part(clean_cfg, f"world_size: {world}\n", f"world_size: {world - 1}\n"),)))
    assert len(out) == K_MENU_V3
    return out
