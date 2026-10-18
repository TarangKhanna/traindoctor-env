#!/usr/bin/env python3
"""Regenerate the README figures in docs/figures/ (pure-Python SVG, standard library only).

Every number drawn comes from a committed file:
  docs/REDTEAM_TABLE.md                      -> fig_redteam_funnel.svg, fig_fake_repairs.svg
  docs/results/test/TEST_REPORT_analysis.json -> fig_test_results.svg, fig_two_panel.svg
  docs/results/test/AUDIT_RECOUNT.json       -> fig_two_panel.svg (right panel)
fig_failure.svg is a schematic of the quickstart scenario (seed 123, Task A1); it draws no measured numbers.

Usage:  python scripts/make_figures.py          (from the repo root; writes docs/figures/*.svg)
Colors are the Okabe-Ito colorblind-safe palette.
"""
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"

BLUE, ORANGE, GREEN, SKY, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00", "#999999"
INK, MUTED, GRID, BG = "#1f2328", "#59636e", "#d0d7de", "#ffffff"
FONT = "font-family=\"Helvetica, Arial, sans-serif\""


def t(x, y, s, size=13, anchor="start", weight="normal", fill=INK, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}" {FONT} {extra}>{escape(s)}</text>')


def rect(x, y, w, h, fill, stroke="none", rx=4, sw=1):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(x1, y1, x2, y2, stroke=INK, sw=1.5, dash="", marker=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arr)"' if marker else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'


def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
            f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
            f'{rect(0, 0, w, h, BG, GRID, 8)}{body}</svg>\n')


# ---------------------------------------------------------------- red-team table parsing
def parse_redteam():
    rows = []
    for ln in (ROOT / "docs" / "REDTEAM_TABLE.md").read_text().splitlines():
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) == 9 and c[0] not in ("ID", "---") and not c[0].startswith("-"):
            rows.append(dict(id=c[0], task=c[1], accepted=c[2] == "yes", cls=c[3], first=c[4], allfail=c[5],
                             launch=c[6], mode=c[8]))
    return rows


def is_cheat(r):
    return r["id"][0] in "FX"


# ---------------------------------------------------------------- shared helpers for the narrative figures
BODY, HEAD, TITLE = 16, 18, 22          # px; readable when the 800 px canvas is scaled to a 390 px phone
W0 = 800
WHITE = "#ffffff"


def tl(x, y, lines, size=BODY, anchor="start", weight="normal", fill=INK, lh=None):
    lh = lh or size * 1.3
    return "".join(t(x, y + i * lh, s, size, anchor, weight, fill) for i, s in enumerate(lines))


def badge(cx, cy, n, fill=INK):
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="15" fill="{fill}"/>'
            + t(cx, cy + 6, str(n), 18, "middle", "bold", WHITE))


def arrow(x1, y1, x2, y2, stroke=INK, sw=2.5, both=False):
    """Line with explicit polygon heads (no <marker>), so it renders identically in every SVG viewer."""
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    hl, hw = 11, 6

    def head(x, y, a):
        bx, by = x - hl * math.cos(a), y - hl * math.sin(a)
        px, py = -math.sin(a) * hw, math.cos(a) * hw
        return f'<polygon points="{x:.1f},{y:.1f} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" fill="{stroke}"/>'
    ax, ay = x1 + (hl * math.cos(ang) if both else 0), y1 + (hl * math.sin(ang) if both else 0)
    bx, by = x2 - hl * math.cos(ang), y2 - hl * math.sin(ang)
    out = f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{stroke}" stroke-width="{sw}"/>' + head(x2, y2, ang)
    return out + (head(x1, y1, ang + math.pi) if both else "")


def title_block(text, lines=1):
    return t(W0 / 2, 40, text, TITLE, "middle", "bold") if lines == 1 else tl(W0 / 2, 40, text, TITLE, "middle", "bold", 28)


# ---------------------------------------------------------------- (a) the failure
def fig_failure():
    W, H = W0, 476
    b = [title_block("Rank 2 called a different collective, so all four ranks hang")]
    px = [20, 280, 540]
    pw = 240
    for i, x in enumerate(px):
        b.append(rect(x, 70, pw, 308, "#f6f8fa", GRID, 10))
        b.append(badge(x + 24, 96, i + 1))
    for x in (px[0] + pw, px[1] + pw):
        b.append(arrow(x + 1, 215, x + 19, 215))
    # panel 1: who calls what
    x = px[0]
    b.append(t(x + 48, 102, "Step 6, call #12", HEAD, weight="bold"))
    for k, (name, op, col) in enumerate([("rank 0", "all_reduce", BLUE), ("rank 1", "all_reduce", BLUE),
                                         ("rank 2", "broadcast", ORANGE), ("rank 3", "all_reduce", BLUE)]):
        y = 126 + k * 58
        b.append(rect(x + 14, y, pw - 28, 44, col, rx=8))
        b.append(t(x + 28, y + 28, name, BODY, weight="bold", fill=WHITE if col == BLUE else INK))
        b.append(t(x + pw - 28, y + 28, op, BODY, "end", "bold", WHITE if col == BLUE else INK))
    b.append(t(x + pw / 2, 366, "Only rank 2 differs", BODY, "middle", "bold", "#8a5a00"))
    # panel 2: deadlock
    x = px[1]
    b.append(t(x + 48, 102, "Deadlock", HEAD, weight="bold"))
    ox, ow = x + 150, 76
    b.append(rect(ox, 168, ow, 90, ORANGE, rx=8))
    b.append(t(ox + ow / 2, 218, "rank 2", BODY, "middle", "bold"))
    for k, name in enumerate(["rank 0", "rank 1", "rank 3"]):
        y = 130 + k * 64
        b.append(rect(x + 14, y, 76, 44, BLUE, rx=8))
        b.append(t(x + 52, y + 28, name, BODY, "middle", "bold", WHITE))
        b.append(arrow(x + 93, y + 22, ox - 2, 213 + (k - 1) * 18, MUTED, 2, both=True))
    b.append(t(x + pw / 2, 326, "each waits for the others", BODY, "middle", fill=MUTED))
    b.append(t(x + pw / 2, 354, "nobody can finish", HEAD, "middle", "bold", VERM))
    # panel 3: timeout
    x = px[2]
    b.append(t(x + 48, 102, "Timeout", HEAD, weight="bold"))
    for k, name in enumerate(["rank 0", "rank 1", "rank 2", "rank 3"]):
        cx = x + 14 + (k % 2) * 108
        cy = 124 + (k // 2) * 62
        b.append(rect(cx, cy, 100, 50, VERM, rx=8))
        b.append(t(cx + 50, cy + 21, name, 16, "middle", "bold", WHITE))
        b.append(t(cx + 50, cy + 41, "exit 1", 16, "middle", fill=WHITE))
    b.append(t(x + pw / 2, 288, "same error on every rank:", BODY, "middle", fill=MUTED))
    b.append(t(x + pw / 2, 316, "COLLECTIVE_TIMEOUT", BODY, "middle", "bold", VERM))
    b.append(t(x + pw / 2, 342, "(job fails)", BODY, "middle", fill=MUTED))
    # callout
    b.append(rect(20, 392, 760, 62, "#fff4e5", ORANGE, 10, 2))
    b.append(t(W / 2, 430, "The error never names rank 2. Only the per-rank trace shows who diverged.", BODY + 1, "middle", "bold"))
    return svg(W, H, "".join(b), "Rank 2 called a different collective, so all four ranks hang",
               "Three steps. 1: at step 6 ranks 0, 1 and 3 call all_reduce while rank 2 calls broadcast. "
               "2: the ranks wait on each other and none can finish. 3: all four ranks exit 1 with the same COLLECTIVE_TIMEOUT error; "
               "only the per-rank trace shows who diverged.")


# ---------------------------------------------------------------- (c) red-team funnel
def fig_funnel(rows):
    cheats = [r for r in rows if is_cheat(r)]
    n = len(cheats)
    acc = sum(r["accepted"] for r in cheats)
    inv = sum(r["cls"] == "invariant" for r in cheats)
    exe = sum(r["cls"] == "execution_failure" for r in cheats)
    assert n == inv + exe + acc, "unexpected class in red-team table"
    ctl = {}
    for key, pred in (("null", lambda r: r["id"] == "null"), ("gold", lambda r: r["id"] == "gold"),
                      ("alt", lambda r: r["id"].startswith("alt"))):
        sel = [r for r in rows if pred(r)]
        ctl[key] = (sum(r["accepted"] for r in sel), len(sel))
    W, H = W0, 500
    b = [title_block(f"No fake repair got through: {acc} of {n} accepted")]
    b.append(rect(20, 130, 170, 120, ORANGE, rx=10))
    b.append(t(105, 182, str(n), 36, "middle", "bold"))
    b.append(t(105, 212, "fake repairs", BODY, "middle", "bold"))
    b.append(t(105, 232, "replayed in sandbox", 14, "middle"))
    # two branches
    b.append(rect(300, 80, 250, 100, BLUE, rx=10))
    b.append(t(425, 122, str(inv), 30, "middle", "bold", WHITE))
    b.append(t(425, 148, "rejected by", BODY, "middle", "bold", WHITE))
    b.append(t(425, 168, "correctness invariants", BODY, "middle", "bold", WHITE))
    b.append(rect(300, 200, 250, 100, VERM, rx=10))
    b.append(t(425, 242, str(exe), 30, "middle", "bold", WHITE))
    b.append(t(425, 268, "job failed before", BODY, "middle", "bold", WHITE))
    b.append(t(425, 288, "producing artifacts", BODY, "middle", "bold", WHITE))
    b.append(arrow(192, 170, 298, 135))
    b.append(arrow(192, 210, 298, 250))
    b.append(arrow(552, 130, 628, 170))
    b.append(arrow(552, 250, 628, 210))
    b.append(rect(630, 130, 150, 120, WHITE, GREEN, 10, 4))
    b.append(t(705, 182, str(acc), 36, "middle", "bold", GREEN))
    b.append(t(705, 212, "accepted", BODY, "middle", "bold", GREEN))
    b.append(t(W / 2, 325, f"The {exe} show those cheats do not run, not invariant coverage.", BODY, "middle", fill=MUTED))
    # controls row
    b.append(f'<line x1="20" y1="348" x2="780" y2="348" stroke="{GRID}" stroke-width="1.5"/>')
    b.append(t(20, 380, "Controls on the same suite (accepted / tried)", HEAD, weight="bold"))
    for x, label, key, good in ((20, "no-op patch", "null", False), (280, "gold repair", "gold", True),
                                (540, "other valid repairs", "alt", True)):
        a, m = ctl[key]
        b.append(rect(x, 398, 240, 80, WHITE, GREEN if good else VERM, 10, 3))
        b.append(t(x + 120, 430, f"{a} / {m}", 28, "middle", "bold", GREEN if good else VERM))
        b.append(t(x + 120, 460, label, BODY, "middle", "bold"))
    return svg(W, H, "".join(b), f"No fake repair got through: {acc} of {n} accepted",
               f"{n} fake repairs: {inv} rejected by correctness invariants, {exe} failed before producing artifacts, {acc} accepted. "
               f"Controls: no-op patch accepted {ctl['null'][0]}/{ctl['null'][1]}, gold repair {ctl['gold'][0]}/{ctl['gold'][1]}, "
               f"other valid repairs {ctl['alt'][0]}/{ctl['alt'][1]}.")


# (e) why exit code 0 is not recovery
def fig_fake(rows):
    desc = {"F01": "Swallow the exception around the sync", "F02": "Exit 0 at the fault",
            "F11": "Set world_size: 1 (one process)", "F06": "Raise the collective timeout"}
    W = W0
    hdr = 100
    rh = 126
    b = [title_block("Fake repairs can finish the job, yet none was accepted")]
    cols = [24, 290, 450]
    for x, s in zip(cols, ("fake repair", "job result", "what the verifier saw")):
        b.append(t(x, hdr - 14, s, HEAD, weight="bold", fill=MUTED))
    b.append(line(20, hdr, W - 20, hdr, INK, 2))
    y = hdr
    for cid, text in desc.items():
        rs = [r for r in rows if r["id"] == cid]
        fin = sum(r["launch"].startswith("completed") for r in rs)
        acc = sum(r["accepted"] for r in rs)
        assert acc == 0
        ex = next((r for r in rs if r["cls"] == "invariant"), None)
        if ex:
            parts = [p.strip() for p in ex["allfail"].split(";")]
            gaps = next((p for p in parts if p.startswith("samples gaps") and p != "samples gaps=0 dups=0"), None)
            seen = [f"{ex['first']}" + (f"; {gaps}" if gaps and gaps not in ex["first"] else "") + f" (on {ex['task']})"]
        else:
            seen = ["job never completed", "a relaunch (I1)"]
        b += [tl(cols[0], y + 36, wrap(text, 22), BODY, weight="bold", lh=22)]
        b.append(rect(cols[1], y + 18, 140, 44, "#e8f1f8" if fin else "#fbe9e0", rx=8))
        b.append(t(cols[1] + 70, y + 46, f"finished {fin}/{len(rs)}" if fin else "failed", BODY, "middle", "bold", BLUE if fin else VERM))
        b += [tl(cols[2], y + 30, [f"rejected {len(rs)}/{len(rs)}" if True else ""], BODY, weight="bold", fill=GREEN)]
        b += [tl(cols[2], y + 54, [x for s_ in seen for x in wrap(s_, 30)], 15, fill=INK, lh=20)]
        y += rh
        b.append(line(20, y, W - 20, y, GRID, 1.5))
    H = y + 70
    b.append(t(W / 2, y + 38, "Finished = the relaunch ended cleanly. Rejected = verifier invariant failed.", BODY, "middle", fill=MUTED))
    return svg(W, H, "".join(b), "Fake repairs can finish the job, yet none was accepted",
               "Four fake repairs (swallow the exception, exit 0, world_size 1, raise the timeout), whether the job finished, "
               "and the check that rejected each; 0 accepted.")


def wrap(s, n):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n and cur:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + [cur]


# ---------------------------------------------------------------- (f) two-panel headline result
def fig_two_panel():
    d = json.loads((ROOT / "docs/results/test/TEST_REPORT_analysis.json").read_text())
    rc = json.loads((ROOT / "docs/results/test/AUDIT_RECOUNT.json").read_text())
    pt = d["policy_table"]
    mean = lambda a: sum(a) / len(a)
    v0 = [100 * pt[f"v0_s{i}"]["v2_fault4"] for i in range(6)]
    v2 = [100 * pt[f"v2_s{i}"]["v2_fault4"] for i in range(6)]
    csft = 100 * pt["csft"]["v2_fault4"]
    diff, (lo, hi) = d["primary"]["diff"], d["primary"]["boot_ci95"]
    assert abs((mean(v2) - mean(v0)) - diff) < 1e-9, "arm means do not reproduce primary.diff"
    sd = d["primary"]["per_seed_diff"]
    assert all(abs((b_ - a_) - s_) < 1e-6 for a_, b_, s_ in zip(v0, v2, sd)), "per-seed diffs differ from primary.per_seed_diff"
    r0 = [rc["chose_recovering_repair_pct"][f"v0_s{i}"] for i in range(6)]
    r2 = [rc["chose_recovering_repair_pct"][f"v2_s{i}"] for i in range(6)]
    assert abs(mean(r0) - rc["arm_mean_pct"]["v0"]) < 1e-9 and abs(mean(r2) - rc["arm_mean_pct"]["v2"]) < 1e-9
    W, H = W0, 940

    def panel(y0, title, sub, a, bb, ymax, step, ann, extra_ref=None):
        o = [t(W / 2, y0, title, TITLE - 2, "middle", "bold"), t(W / 2, y0 + 26, sub, BODY, "middle", fill=MUTED)]
        top, bot = y0 + 70, y0 + 300
        L, R = 110, 740
        Y = lambda v: bot - (bot - top) * v / ymax
        for v in range(0, int(ymax) + 1, step):
            o.append(line(L, Y(v), R, Y(v), GRID, 1))
            o.append(t(L - 10, Y(v) + 6, f"{v}%", 15, "end", fill=MUTED))
        xa, xb = 270, 580
        for xx, vals, col, name in ((xa, a, ORANGE, "label-only reward"), (xb, bb, GREEN, "verified reward")):
            m = mean(vals)
            o.append(f'<line x1="{xx - 70}" y1="{Y(m):.1f}" x2="{xx + 70}" y2="{Y(m):.1f}" stroke="{col}" stroke-width="6"/>')
            o.append(t(xx, bot + 28, name, BODY, "middle", "bold"))
            o.append(t(xx, bot + 48, "GRPO, mean of 6 seeds", 14, "middle", fill=MUTED))
        for va, vb in zip(a, bb):
            o.append(line(xa, Y(va), xb, Y(vb), "#b0b7bf", 1.5))
        for xx, vals in ((xa, a), (xb, bb)):
            for v in vals:
                o.append(f'<circle cx="{xx}" cy="{Y(v):.1f}" r="6" fill="{INK}" stroke="#fff" stroke-width="1.5"/>')
        o.append(t(xa - 78, Y(mean(a)) + 6, f"{mean(a):.1f}%", BODY, "end", "bold"))
        o.append(t(xb + 78, Y(mean(bb)) + 6, f"{mean(bb):.1f}%", BODY, "start", "bold"))
        if extra_ref is not None:
            o.append(line(L, Y(extra_ref), R, Y(extra_ref), BLUE, 2, "7 5"))
            o.append(t(L + 6, Y(extra_ref) - 8, f"continued SFT, 1 run: {extra_ref:.1f}%", 15, "start", fill=BLUE))
        o.append(t(W / 2, bot + 84, ann, BODY, "middle", "bold"))
        return o

    b = []
    b += panel(44, f"Strict verified success: +{diff:.2f} pp for the verified-reward arm",
               "Share of faulted rollouts that fully recover; each dot is one seed, grey lines join paired seeds",
               v0, v2, 50, 10, f"+{diff:.2f} pp, 95% CI [+{lo:.2f}, +{hi:.2f}], positive in all 6 pairs", csft)
    b.append(line(20, 462, W - 20, 462, GRID, 2))
    diff2 = mean(r2) - mean(r0)
    b += panel(492, f"Chose the recovering repair: {mean(r0):.1f}% vs {mean(r2):.1f}%, no difference",
               "Share of faulted rollouts whose chosen repair recovers in live replay; same seeds",
               r0, r2, 100, 20, f"difference {diff2:+.2f} pp")
    b.append(t(W / 2, H - 20, "So the gain is evidence grounding, not better repairs. Sealed TEST, 100 episodes, exploratory.", BODY, "middle", "bold"))
    return svg(W, H, "".join(b), f"Verified reward: +{diff:.2f} pp strict success, same repair choice",
               f"Top: strict verified success, label-only arm {mean(v0):.1f}% vs verified-reward arm {mean(v2):.1f}% (+{diff:.2f} pp, "
               f"95% CI +{lo:.2f} to +{hi:.2f}), six paired seeds, continued SFT {csft:.1f}%. "
               f"Bottom: share choosing the recovering repair, {mean(r0):.1f}% vs {mean(r2):.1f}%.")


# ---------------------------------------------------------------- (d) sealed TEST results
def fig_test():
    d = json.loads((ROOT / "docs/results/test/TEST_REPORT_analysis.json").read_text())
    pt = d["policy_table"]
    pct = lambda k: 100 * pt[k]["v2_fault4"]
    v0 = [pct(f"v0_s{i}") for i in range(6)]
    v2 = [pct(f"v2_s{i}") for i in range(6)]
    mean = lambda a: sum(a) / len(a)
    diff, (lo, hi) = d["primary"]["diff"], d["primary"]["boot_ci95"]
    assert abs((mean(v2) - mean(v0)) - diff) < 1e-6, "arm means do not reproduce the reported primary difference"
    bars = [("Base model", pct("base"), None, GREY, "single run"),
            ("SFT", pct("sft"), None, SKY, "single run"),
            ("Continued SFT", pct("csft"), None, SKY, "single run, matched compute"),
            ("GRPO\nlabel-only reward", mean(v0), v0, ORANGE, "mean of 6 seeds"),
            ("GRPO\nverified reward", mean(v2), v2, GREEN, "mean of 6 seeds")]
    W, H = 760, 450
    px0, px1, py0, py1 = 70, 730, 360, 90  # plot box; y from 0..45
    ymax = 45.0
    Y = lambda v: py0 - (py0 - py1) * v / ymax
    b = [t(40, 28, "Sealed TEST: live-verified recovery of the repaired job", 16, weight="bold"),
         t(40, 47, f"Strict success on the 4 faulted task kinds ({d['n_episodes'] * 4 // 5} episodes x 4 samples per model); each repair replayed in the sandbox. Exploratory.", 11.5, fill=MUTED)]
    for v in range(0, 46, 10):
        b.append(line(px0, Y(v), px1, Y(v), GRID, 1))
        b.append(t(px0 - 8, Y(v) + 4, f"{v}%", 11, "end", fill=MUTED))
    slot = (px1 - px0) / len(bars)
    xc = []
    for i, (name, val, dots, col, note) in enumerate(bars):
        cx = px0 + slot * (i + 0.5)
        xc.append(cx)
        bw = 78
        b.append(rect(cx - bw / 2, Y(val), bw, py0 - Y(val), col, rx=3))
        b.append(t(cx, Y(val) - (28 if dots else 8), f"{val:.1f}%", 13, "middle", "bold"))
        if dots:
            for j, dv in enumerate(sorted(dots)):
                b.append(f'<circle cx="{cx - 25 + j * 10:.1f}" cy="{Y(dv):.1f}" r="3.6" fill="{INK}" fill-opacity="0.85" stroke="#fff" stroke-width="1"/>')
        for k, part in enumerate(name.split("\n")):
            b.append(t(cx, py0 + 18 + 15 * k, part, 12, "middle", "bold"))
        b.append(t(cx, py0 + 50, note, 10.5, "middle", fill=MUTED))
    # difference bracket between the two GRPO arms
    xa, xb = xc[3], xc[4]
    yb = Y(max(max(v0), max(v2))) - 40
    b.append(f'<path d="M{xa:.1f},{yb + 8:.1f} V{yb:.1f} H{xb:.1f} V{yb + 8:.1f}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    b.append(t((xa + xb) / 2, yb - 20, f"+{diff:.2f} pp (95% CI +{lo:.2f} to +{hi:.2f})", 12.5, "middle", "bold"))
    b.append(t((xa + xb) / 2, yb - 6, "verified minus label-only, paired over 6 seeds", 10.5, "middle", fill=MUTED))
    b.append(t(40, 438, "Dots are individual training seeds. Continued SFT is a single run and is not distinguishable from the verified-reward arm. Source: docs/results/test/", 10.5, fill=MUTED))
    return svg(W, H, "".join(b), "Sealed TEST results",
               "Strict live-verified success: base, SFT, continued SFT, GRPO label-only and GRPO verified-reward arms.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = parse_redteam()
    assert len(rows) == 107, f"expected 107 red-team rows, got {len(rows)}"
    figs = {"fig_failure.svg": fig_failure(), "fig_redteam_funnel.svg": fig_funnel(rows),
            "fig_test_results.svg": fig_test(), "fig_fake_repairs.svg": fig_fake(rows),
            "fig_two_panel.svg": fig_two_panel()}
    for name, s in figs.items():
        (OUT / name).write_text(s)
        print("wrote", (OUT / name).relative_to(ROOT), len(s), "bytes")


if __name__ == "__main__":
    sys.exit(main())
