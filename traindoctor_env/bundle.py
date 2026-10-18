"""Tier A data model: the snapshot's records and job files plus a menu of K candidate repairs (concrete edits,
opaque ids, seeded order) and each candidate's outcome, precomputed by real re-execution (the fix-outcome
table). Rendering this as text for a model is M3; here is only the data and the answer check."""
from __future__ import annotations

import hashlib
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from jobs.build import Edit
from traindoctor_env.reexec import FixOutcome, verify_fix
from traindoctor_env.scenario import Truth
from traindoctor_env.snapshot import Snapshot
from traindoctor_env.verifiers import EVIDENCE_CAP, Verdict, certificate_ids, edits_of, verdict


K_MENU = 6            # every variant shows exactly this many candidate repairs (menu v1)
K_MENU_V2 = 9         # menu v2: the v1 six plus three plausible-but-invalid repairs shown in every variant (see _v2_extras)
MENU_VERSIONS = ("v1", "v2")
ITEM_CHARS = 360      # every menu item is rendered with exactly this many characters (real context lines, then blanks)
CONFIG_WIDTH = 28     # job/config.yaml is shown with every line padded to this width
QUOTA = dict(log=5, trace=6, samples=3)   # records per kind in a bundle (at least the certificate's), whatever the label


class ItemTooLong(Exception):
    pass


@dataclass(frozen=True)
class Candidate:
    id: str
    role: str                 # private: correct | noop | exclude_rank | decoy_<location>
    edits: tuple
    destructive: bool = False
    shown: tuple = ()         # neutral display parts (path, clean_lines, bug_lines, clean_file_lines); see _shown_parts
    clean_cfg: tuple = ()     # the CLEAN job/config.yaml lines: the no-change item's context (never the buggy text)

    def diff(self, files: dict | None = None) -> str:
        """The candidate as a unified-diff-like text of exactly ITEM_CHARS characters: the changed lines, then real
        unchanged lines around them while they fit, then blank padding, so the size of an item says nothing about
        which repair it is."""
        files = files or {}
        if not self.edits:
            lines = ["no change; relaunch from the latest checkpoint"]
            ctx = ["  " + x for x in (self.clean_cfg or files.get("job/config.yaml", "").splitlines())]
            return _fit(lines, [ctx])
        if self.shown:
            parts = [dict(salt=self.id, head=f"--- {path}", old=list(clean), new=list(bug), lines=list(lines),
                          lo=(lo := _find(lines, clean)), hi=lo + max(1, len(clean)))
                     for path, clean, bug, lines in self.shown]
            return _fit_parts(parts)
        parts = []
        for e in self.edits:
            text = files.get(e.path, "")
            at = text.find(e.old)
            first = text.count("\n", 0, at) if at >= 0 else 0
            lines = text.splitlines()
            old, new = e.old.splitlines(), e.new.splitlines()
            parts.append(dict(salt=self.id, head=f"--- {e.path}", old=old, new=new, lines=lines, lo=first, hi=first + max(1, len(old))))
        return _fit_parts(parts)


def _find(lines, block) -> int:
    n = len(block)
    return next((i for i in range(len(lines) - n + 1) if list(lines[i:i + n]) == list(block)), 0)


def _fit(lines: list[str], contexts: list[list[str]]) -> str:
    out = list(lines)
    for ctx in contexts:
        for x in ctx:
            if len("\n".join(out + [x])) > ITEM_CHARS:
                break
            out.append(x)
    return _blank_pad("\n".join(out))


def _blank_pad(text: str) -> str:
    """Pad to exactly ITEM_CHARS with a filler line of dots (not blanks), so the stripped length of an item is the
    same for every candidate and says nothing about how much real context happened to fit."""
    text = text.rstrip()
    if len(text) > ITEM_CHARS:
        raise ItemTooLong(f"{len(text)} chars > {ITEM_CHARS}")
    rem = ITEM_CHARS - len(text)
    return text + ("\n" + "." * (rem - 1) if rem > 1 else "." * rem)


def _fit_parts(parts: list[dict]) -> str:
    pre = [[] for _ in parts]
    post = [[] for _ in parts]

    def render() -> str:
        out = []
        for p, a, b in zip(parts, pre, post):
            # neutral display: the two forms of the changed lines, both marked "~", in an order fixed by a hash of the
            # candidate and file (never by which form is the job's current one), so side and sign carry no signal
            forms = [p["old"], p["new"]]
            if hashlib.sha256((p["salt"] + p["head"]).encode()).digest()[0] & 1:
                forms.reverse()
            out += [p["head"], *("  " + x for x in a), *("~ " + x for x in forms[0]), "~ ---", *("~ " + x for x in forms[1]),
                    *("  " + x for x in b)]
        return "\n".join(out)

    _blank_pad(render())          # the changed lines alone must fit
    grew = True
    while grew:
        grew = False
        for i, p in enumerate(parts):
            for side in ("post", "pre"):
                if side == "post" and p["hi"] + len(post[i]) < len(p["lines"]):
                    post[i].append(p["lines"][p["hi"] + len(post[i])])
                    if len(render()) > ITEM_CHARS:
                        post[i].pop()
                        continue
                elif side == "pre" and p["lo"] - len(pre[i]) > 0:
                    pre[i].insert(0, p["lines"][p["lo"] - len(pre[i]) - 1])
                    if len(render()) > ITEM_CHARS:
                        pre[i].pop(0)
                        continue
                else:
                    continue
                grew = True
    return _blank_pad(render())


CAP_TOKENS = 3072      # the whole Tier A prompt, in Qwen3 tokens (was 2048: the Task B certificate does not fit)
PROMPT_OVERHEAD = 160  # chat-template tokens + the answer-format instruction around the bundle text (tiera.FORMAT); a test checks it
ALWAYS = ("job", "process", "scheduler", "ckpt")   # the incident summary: kept for every scenario, label-independent
_PIECE = re.compile(r"[A-Za-z]+|[0-9]|\s+|(.)\1*")
SAFETY = 1.10


def est_tokens(text: str) -> int:
    """A deterministic UPPER BOUND on the Qwen3 token count of `text`, with no tokenizer (so bundles are byte-identical
    on every machine). Letter runs cost 1 token per 4 letters, every digit 1 (Qwen splits digits), a run of one
    repeated symbol 1 per 16, any other symbol 1, a lone space 0 (it joins the next word), a newline or an indent 1;
    then x1.10. tests/test_token_cap.py checks bound >= the real tokenizer on prompts built from this repo's job
    text, and every executed bundle against the real count."""
    n = 0
    for m in _PIECE.finditer(text):
        t = m.group(0)
        if t.isalpha():
            n += -(-len(t) // 4)
        elif t.isdigit():
            n += 1
        elif t.isspace():
            n += ("\n" in t) + (len(t.replace("\n", "")) > 1)
        else:
            n += -(-len(t) // 16) if len(t) > 1 else 1
    return -(-int(n * SAFETY * 1000) // 1000)


class BundleTooLarge(Exception):
    pass


@dataclass(frozen=True)
class Bundle:
    seed: int
    records: tuple            # the retained records, in canonical order
    files: dict               # job/config.yaml only: the full source does not fit the cap; the menu diffs carry the code
    menu: tuple               # (id, diff) pairs, in presentation order
    text: str                 # the rendering the token cap applies to
    tokens: int


def _set(files: dict, key: str, value) -> Edit:
    """An edit setting one `key: value` line of job/config.yaml."""
    m = re.search(rf"^( *){key}: .*\n", files["job/config.yaml"], re.M)
    return Edit("job/config.yaml", m.group(0), f"{m.group(1)}{key}: {value}\n")


SYNC_LINE = "            comm.all_reduce(bucket, step, b)\n"
SAMPLER_WRITE = "state.update(sampler=sampler.state(), step=torch.tensor([step]))"


def _decoy_edits(files: dict, config: dict, seed: int) -> dict[str, list[Edit]]:
    """One WRONG edit per repair location (the places a real repair touches), built from the job's own text with a
    stream that depends on the seed alone. Each re-introduces the fault that location can hold, so it never repairs
    the job it is shown for; only the location that really holds this job's cause gets the true repair instead."""
    rng = random.Random(f"decoy:{seed}")
    world = config["world_size"]
    rank, step = rng.randrange(world), rng.choice([5, 6, 7])
    cfg = files["job/config.yaml"]
    mode = re.search(r"^    (\d+): all_reduce\n", cfg, re.M)
    shard = re.findall(r"^    (\d+): part-\1\n", cfg, re.M)
    out: dict[str, list[Edit]] = {"sync": [], "perrank": [], "shard": [], "ckpt": []}
    if SYNC_LINE in files["job/train.py"]:
        out["sync"] = [Edit("job/train.py", SYNC_LINE, f"            (comm.broadcast if rank == {rank} and step >= {step} and b == 0 "
                                                       f"else comm.all_reduce)(bucket, step, b)\n")]
    if mode:
        out["perrank"] = [Edit("job/config.yaml", mode.group(0), f"    {mode.group(1)}: reduce_bcast\n")]
    if shard:
        r = shard[rank % len(shard)]
        out["shard"] = [Edit("job/config.yaml", f"    {r}: part-{r}\n", f"    {r}: part-{world}\n")]
    if SAMPLER_WRITE in files["job/ckpt.py"]:
        out["ckpt"] = [Edit("job/ckpt.py", SAMPLER_WRITE, "state.update(step=torch.tensor([step]))")]
    return out


def _sync_bug(rank: int, step: int) -> str:
    return f"            (comm.broadcast if rank == {rank} and step >= {step} and b == 0 else comm.all_reduce)(bucket, step, b)"


CKPT_WRITE_CLEAN = SAMPLER_WRITE
CKPT_WRITE_BUG = "state.update(step=torch.tensor([step]))"
CKPT_LOAD_CLEAN = '        sampler.load_state(state["sampler"])'
CKPT_LOAD_BUG = '        sampler.load_state(_progress(rank, state["sampler"]))'


def _shown_parts(loc: str, edits, clean: dict) -> tuple:
    """Neutral display of a repair-location candidate: ALWAYS the two forms (the correct code and the faulty code) of
    the location, drawn from the CLEAN job text (the buggy job with the true repair applied), with context from the
    clean file. The true repair and a decoy therefore render identically in shape and context; which location is
    the faulty one has to be read from the incident records. Display only: Candidate.edits is what is executed."""
    txt = "".join(e.old + e.new for e in edits)
    if loc == "sync":
        m = re.search(r"rank == (\d+) and step >= (\d+)", txt)
        L = clean["job/train.py"].splitlines()
        return (("job/train.py", (SYNC_LINE.rstrip("\n"),), (_sync_bug(int(m.group(1)), int(m.group(2))),), tuple(L)),)
    if loc == "ckpt":
        L = tuple(clean["job/ckpt.py"].splitlines())
        return (("job/ckpt.py", (CKPT_WRITE_CLEAN,), (CKPT_WRITE_BUG,), L),
                ("job/ckpt.py", (CKPT_LOAD_CLEAN,), (CKPT_LOAD_BUG,), L))
    e = edits[0]
    a, b = e.old.rstrip("\n"), e.new.rstrip("\n")
    c, g = (a, b) if a + "\n" in clean["job/config.yaml"] else (b, a)
    return (("job/config.yaml", (c,), (g,), tuple(clean["job/config.yaml"].splitlines())),)


def _location(edits) -> str | None:
    """Which repair location a true repair touches (None for the no-change relaunch)."""
    if not edits:
        return None
    e = edits[0]
    if e.path == "job/train.py":
        return "sync"
    if e.path == "job/ckpt.py":
        return "ckpt"
    return "perrank" if "reduce" in e.old + e.new else "shard"


def _menu_repair(truth: Truth) -> list[Edit]:
    """The true repair as the menu shows it. B_SKIP's full revert also deletes a now-unused helper (a large diff no
    other repair has, which would mark it); the one-line revert of the restore call is the same repair for the
    verifier (re-execution decides). Likewise A1's source repair is shown as disabling the rank-keyed branch rather
    than deleting its four lines, so every repair and every decoy is a one-line change of the same shape."""
    fix = edits_of(truth)
    if truth.kind == "B_SKIP":
        return fix[:1]
    if truth.kind == "A1" and fix[0].path == "job/train.py":   # disable the rank-keyed branch: a one-line edit
        return [Edit("job/train.py", fix[0].old.splitlines(True)[0], "            if False:\n")]
    return list(fix)


def candidate_id(seed: int, role: str) -> str:
    """Opaque candidate id, keyed by the per-scenario secret (jobs.dataserver.salt_for, derived from TD_KEY), so the seed
    and the role cannot be recovered from the id without the key. Deterministic for a given key and seed."""
    from jobs.dataserver import salt_for
    return "R" + hashlib.sha256(salt_for(seed) + b"|menu-id|" + role.encode()).hexdigest()[:5]


RESTORE_DEF = "def restore(cid: str, model, opt, sampler, rank: int) -> int:\n"
ALIGN_BLOCK = ("            if b == 0:\n                comm.broadcast(bucket, step, b)\n"
               "            else:\n                comm.all_reduce(bucket, step, b)\n")


def _v2_extras(f: dict, truth: Truth, clean: dict) -> list[tuple]:
    """Menu v2's three extra candidates, present in EVERY variant with the same text, so their presence says nothing about
    the label. Each is a plausible repair that does not preserve recovery-correctness; whether it completes the job and which
    invariant it breaks is decided by real re-execution (the outcome table), never assigned here.
      timeout  raise the collective timeouts (the reflex for a hang; Task A keeps hanging)
      scratch  restore() ignores the checkpoint and trains from step 0 (the wrong restart: lost work / sample accounting)
      align    make every rank run the same collective, but broadcast bucket 0 instead of averaging it (consistent, so a
               divergent-branch job completes, but the gradients are no longer the data-parallel average)
    Returns (role, edits, shown) tuples; `shown` is the neutral display (clean text vs the candidate's text)."""
    cfg_lines = tuple(clean["job/config.yaml"].splitlines())
    cfg_edits, shown_t = [], []
    for key, val in (("pg_timeout_s", 20), ("watchdog_s", 40)):
        e = _set(f, key, val)
        cfg_edits.append(e)
        shown_t.append(("job/config.yaml", (e.old.rstrip("\n"),), (e.new.rstrip("\n"),), cfg_lines))
    ck = tuple(clean["job/ckpt.py"].splitlines())
    scratch = [Edit("job/ckpt.py", RESTORE_DEF, RESTORE_DEF + "    return 0\n")]
    scratch_shown = (("job/ckpt.py", (RESTORE_DEF.rstrip("\n"),), (RESTORE_DEF.rstrip("\n"), "    return 0"), ck),)
    fix = edits_of(truth)
    if truth.kind == "A1" and fix[0].path == "job/train.py":      # the job holds the rank-keyed branch: rewrite its condition
        align = [Edit("job/train.py", fix[0].old.splitlines(True)[0], "            if b == 0:\n")]
    else:
        align = [Edit("job/train.py", SYNC_LINE, ALIGN_BLOCK)]
    align_shown = (("job/train.py", (SYNC_LINE.rstrip("\n"),), tuple(ALIGN_BLOCK.rstrip("\n").split("\n")),
                    tuple(clean["job/train.py"].splitlines())),)
    return [("distractor_timeout", cfg_edits, tuple(shown_t)), ("distractor_scratch", scratch, scratch_shown),
            ("distractor_align", align, align_shown)]


def make_menu(snap: Snapshot, truth: Truth, version: str = "v1") -> list[Candidate]:
    """K_MENU candidates, the same SIX repair slots for every variant, so which files or keys a menu touches says
    nothing about the cause: one slot per repair location (training-step sync code, per-rank sync mode, data shard
    directory, checkpoint code), the no-change relaunch, and the destructive rank exclusion. The slot that holds this
    job's cause carries the true repair; the others carry a wrong edit at their own location (a decoy built from the
    job's own text by a seed-only stream). For the no-fault variant the correct answer is the relaunch slot.
    version="v2" appends the three _v2_extras (K_MENU_V2 items); "v1" (default) is unchanged."""
    assert version in MENU_VERSIONS, version
    f = snap.files
    world = snap.config["world_size"]
    decoy = _decoy_edits(f, snap.config, snap.seed)
    where = _location(edits_of(truth)) if truth.kind != "B_OK" else None
    right = _menu_repair(truth)
    slots: dict[str, tuple[list[Edit], bool, str]] = {}
    clean = dict(f)
    for e in (edits_of(truth) if truth.kind != "B_OK" else []):
        clean[e.path] = clean[e.path].replace(e.old, e.new)
    shown: dict[str, tuple] = {}
    for loc in ("sync", "perrank", "shard", "ckpt"):
        slots[loc] = (right, False, "correct") if loc == where else (decoy[loc], False, "decoy_" + loc)
        shown[loc] = _shown_parts(loc, slots[loc][0], clean) if slots[loc][0] else ()
    slots["relaunch"] = ([], False, "correct" if where is None else "noop")
    slots["exclude"] = ([_set(f, "world_size", world - 1)], True, "exclude_rank")
    # every item's context comes from the CLEAN text, shown the same way (no item's context may identify the fault)
    shown["exclude"] = _shown_parts("exclude", slots["exclude"][0], clean)
    assert len(slots) == K_MENU and sum(r == "correct" for _, _, r in slots.values()) == 1, (truth.kind, sorted(slots))
    clean_cfg = tuple(clean["job/config.yaml"].splitlines())
    cands = [Candidate(candidate_id(snap.seed, role), role, tuple(e), d, shown.get(loc, ()), clean_cfg if loc == "relaunch" else ())
             for loc, (e, d, role) in slots.items()]
    if version == "v2":
        cands += [Candidate(candidate_id(snap.seed, role), role, tuple(e), False, sh) for role, e, sh in _v2_extras(f, truth, clean)]
        assert len(cands) == K_MENU_V2 and len({c.id for c in cands}) == K_MENU_V2
    random.Random(f"menu:{snap.seed}" if version == "v1" else f"menu2:{snap.seed}").shuffle(cands)
    return cands


def render(records, config_text: str, menu_pairs, labels) -> str:
    out = ["A distributed training job failed or misbehaved. Answer with: label, the ids of the records that "
           f"support it (at most {EVIDENCE_CAP}), the id of the repair to apply, and whether you claim it fixes the job.",
           "Labels: " + ", ".join(labels), "", "== job/config.yaml ==", config_text.rstrip(), "", "== records =="]
    out += [f"{r['id']} {r['text']}" for r in records]
    out += ["", "== repairs =="]
    for rid, diff in menu_pairs:
        out += [f"{rid}:", diff]
    return "\n".join(out)


def padded_config(text: str) -> str:
    """job/config.yaml with every line padded to CONFIG_WIDTH, so its size does not depend on the edited values."""
    return "\n".join(line.ljust(CONFIG_WIDTH) for line in text.splitlines())


def make_bundle(snap: Snapshot, menu: list[Candidate], truth: Truth) -> Bundle:
    """The snapshot as one compact text with evidence ids and a repair menu, <= CAP_TOKENS. Always kept: the incident
    summary and every certificate record. Then, per record kind, other records are drawn (from a stream seeded by
    the scenario seed and the kind alone, independent of the label) until the kind has QUOTA[kind] records, so the
    number of records of each kind is the same for every variant; retained records are shown in canonical order,
    so neither order nor placement marks the certificate. Menu items and the config are rendered at fixed sizes
    (see Candidate.diff, padded_config) so sizes carry no label either."""
    from jobs.build import ALL_LABELS
    certificate = certificate_ids(list(snap.records), truth)
    pairs = tuple((c.id, c.diff(snap.files)) for c in menu)
    cfg = padded_config(snap.files["job/config.yaml"])
    keep = {r["id"] for r in snap.records if r["kind"] in ALWAYS} | certificate
    for kind, quota in QUOTA.items():
        have = sum(1 for r in snap.records if r["kind"] == kind and r["id"] in keep)
        pool = [r["id"] for r in snap.records if r["kind"] == kind and r["id"] not in keep]
        random.Random(f"bundle:{snap.seed}:{kind}").shuffle(pool)
        keep |= set(pool[:max(0, quota - have)])
    # Compaction: if the prompt is still over the cap, drop non-certificate records, noise kinds before the incident
    # summary, each group in the order of a stream seeded by the scenario seed alone. The certificate is never dropped.
    droppable = [r["id"] for r in snap.records if r["id"] in keep and r["id"] not in certificate]
    random.Random(f"compact:{snap.seed}").shuffle(droppable)
    kind_of = {r["id"]: r["kind"] for r in snap.records}
    droppable.sort(key=lambda i: kind_of[i] in ALWAYS)          # stable: noise first, summary last
    while True:
        kept = tuple(r for r in snap.records if r["id"] in keep)
        text = render(kept, cfg, pairs, ALL_LABELS)
        if est_tokens(text) + PROMPT_OVERHEAD <= CAP_TOKENS:
            break
        if not droppable:
            raise BundleTooLarge(f"seed {snap.seed}: the certificate alone needs {est_tokens(text) + PROMPT_OVERHEAD} > {CAP_TOKENS} tokens")
        keep.discard(droppable.pop(0))
    return Bundle(snap.seed, kept, {"job/config.yaml": cfg}, pairs, text, est_tokens(text) + PROMPT_OVERHEAD)


def outcome_table(snap: Snapshot, menu: list[Candidate], workers: int = 2) -> dict[str, FixOutcome]:
    """Real re-execution of every candidate (two at a time)."""
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip((c.id for c in menu), ex.map(lambda c: verify_fix(snap, list(c.edits)), menu)))


def answer_verdict(truth: Truth, bundle: Bundle, menu: list[Candidate], table: dict[str, FixOutcome],
                   label: str, evidence_ids: list[str], repair_id: str, claimed_fixed: bool,
                   live: Snapshot | None = None) -> Verdict:
    """V0/V1/V2 for a Tier A answer (label, evidence ids, repair id, claimed_fixed). Every record in the bundle
    counts as observed. The table is the RL fast path; pass the snapshot as `live` to re-execute the repair
    in a fresh trusted environment instead (final grading, audits)."""
    chosen = next(c for c in menu if c.id == repair_id)
    if live is not None:
        table = {**table, repair_id: verify_fix(live, list(chosen.edits))}
    return verdict(truth, label, evidence_ids, {r["id"] for r in bundle.records}, [list(bundle.records)],
                   table[repair_id], chosen.destructive, claimed_fixed)
