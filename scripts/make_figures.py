#!/usr/bin/env python3
"""Regenerate the README figures in docs/figures/ (pure-Python SVG, standard library only).

Every number drawn comes from a committed file:
  docs/REDTEAM_TABLE.md                      -> fig_redteam_funnel.svg, fig_fake_repairs.svg
  docs/results/test/TEST_REPORT_analysis.json -> fig_test_results.svg
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


# ---------------------------------------------------------------- (a) the failure
def fig_failure():
    W, H = 760, 330
    b = [t(W / 2, 28, "The failure: one rank takes a different branch, so everyone times out", 16, "middle", "bold")]
    b.append(t(W / 2, 48, "4-rank job, step 6, collective #12 (quickstart scenario, seed 123, Task A1). Schematic of the recorded trace.", 11.5, "middle", fill=MUTED))
    xs = [95, 285, 475, 665]
    ranks = [("rank 0", "all_reduce", BLUE), ("rank 1", "all_reduce", BLUE),
             ("rank 2", "broadcast", VERM), ("rank 3", "all_reduce", BLUE)]
    for x, (name, op, col) in zip(xs, ranks):
        b.append(rect(x - 70, 70, 140, 34, "#f6f8fa", GRID))
        b.append(t(x, 92, name, 14, "middle", "bold"))
        b.append(line(x, 104, x, 126, GRID, 2))
        b.append(line(x, 162, x, 170, GRID, 2))
        b.append(rect(x - 62, 126, 124, 36, col, rx=6))
        b.append(t(x, 149, op, 14, "middle", "bold", "#ffffff"))
    b.append(t(xs[2], 182, "enters a different collective", 11.5, "middle", fill=VERM, weight="bold"))
    for x in (xs[0], xs[1], xs[3]):
        b.append(t(x, 182, "waits for its peers", 11.5, "middle", fill=MUTED))
    b.append(rect(60, 212, 640, 52, "#fff4e5", ORANGE, 6))
    b.append(t(W / 2, 234, "all_reduce never completes; the watchdog fires after the timeout", 13.5, "middle", "bold"))
    b.append(t(W / 2, 253, "Every rank reports the same COLLECTIVE_TIMEOUT. Only the per-rank collective trace shows who diverged.", 12, "middle", fill=MUTED))
    b.append(line(W / 2, 266, W / 2, 292, INK, 1.5, marker=True))
    b.append(t(W / 2, 314, "Job fails: exit 1 on all ranks", 13, "middle", "bold", VERM))
    return svg(W, H, "".join(b), "The failure: a diverging collective",
               "Ranks 0, 1 and 3 call all_reduce while rank 2 calls broadcast; all ranks time out with the same error.")


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
    W, H = 760, 330
    x0, bw = 40, 680
    b = [t(40, 30, "Red team: no fake repair was accepted", 16, weight="bold"),
         t(40, 49, f"{n} attempted fake repairs: {len({r['id'] for r in cheats})} cheat classes x {len({r['task'] for r in cheats})} task variants, each replayed in the network-less sandbox", 11.5, fill=MUTED)]
    b.append(rect(x0, 66, bw, 38, GREY))
    b.append(t(x0 + bw / 2, 90, f"{n} fake repairs attempted", 14, "middle", "bold", "#ffffff"))
    wi = bw * inv / n
    we = bw * exe / n
    b.append(line(x0 + bw / 2, 104, x0 + wi / 2, 128, MUTED, 1.2, marker=True))
    b.append(line(x0 + bw / 2, 104, x0 + wi + we / 2, 128, MUTED, 1.2, marker=True))
    b.append(rect(x0, 130, wi, 54, BLUE))
    b.append(t(x0 + wi / 2, 153, f"{inv} rejected by", 13, "middle", "bold", "#ffffff"))
    b.append(t(x0 + wi / 2, 172, "correctness invariants", 13, "middle", "bold", "#ffffff"))
    b.append(rect(x0 + wi, 130, we, 54, ORANGE))
    b.append(t(x0 + wi + we / 2, 153, f"{exe} failed before", 13, "middle", "bold", INK))
    b.append(t(x0 + wi + we / 2, 172, "producing artifacts", 13, "middle", "bold", INK))
    b.append(t(x0, 204, "Invariants: sample accounting, restored state, rank consistency, world size, protected files.", 11.5, fill=MUTED))
    b.append(t(x0, 221, f"The {exe} execution failures show those cheats do not run; invariant coverage is claimed for the {inv} only.", 11.5, fill=MUTED))
    b.append(rect(x0, 236, 150, 34, GREEN))
    b.append(t(x0 + 75, 258, f"{acc} accepted", 15, "middle", "bold", "#ffffff"))
    b.append(t(x0 + 165, 258, f"(rule-of-three 95% upper bound on the acceptance rate: 3/{n} = {300 / n:.1f}%)", 11.5, fill=MUTED))
    b.append(t(x0, 298, "Controls on the same suite:", 12.5, weight="bold"))
    b.append(t(x0 + 170, 298,
               f"no-op patch accepted {ctl['null'][0]}/{ctl['null'][1]}  |  gold repair {ctl['gold'][0]}/{ctl['gold'][1]}  |  "
               f"alternative valid repairs {ctl['alt'][0]}/{ctl['alt'][1]}", 12.5))
    b.append(t(x0, 316, "The one rejected valid repair (B_DUP alt2) fails only a checkpoint-format check. Source: docs/REDTEAM_TABLE.md", 11, fill=MUTED))
    return svg(W, H, "".join(b), "Red-team funnel",
               f"{n} fake repairs: {inv} rejected by invariants, {exe} failed before artifacts, {acc} accepted.")


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


# ---------------------------------------------------------------- (e) why exit code 0 is not recovery
def fig_fake(rows):
    desc = {"F01": "Swallow the exception around the sync", "F02": "Exit 0 at the fault",
            "F11": "Set world_size: 1 (one process)", "F06": "Raise the collective timeout"}
    W = 760
    hdr = 92
    rh = 58
    b = [t(30, 28, "Why \"the job exits 0\" is not recovery", 16, weight="bold"),
         t(30, 47, "Four fake repairs from the red-team suite, as recorded in docs/REDTEAM_TABLE.md (all runs in the sandbox)", 11.5, fill=MUTED)]
    cols = [30, 290, 400, 485]
    for x, s in zip(cols, ("Fake repair", "Job finished", "Accepted", "What the verifier saw (example run)")):
        b.append(t(x, hdr - 8, s, 12, weight="bold"))
    b.append(line(30, hdr, W - 30, hdr, INK, 1.2))
    y = hdr
    for cid, text in desc.items():
        rs = [r for r in rows if r["id"] == cid]
        fin = sum(r["launch"].startswith("completed") for r in rs)
        acc = sum(r["accepted"] for r in rs)
        ex = next((r for r in rs if r["cls"] == "invariant"), None)
        if ex:
            parts = [p.strip() for p in ex["allfail"].split(";")]
            gaps = next((p for p in parts if p.startswith("samples gaps") and p != "samples gaps=0 dups=0"), None)
            seen = ex["first"] + (f"; {gaps}" if gaps and gaps not in ex["first"] else "")
            seen += f"  ({ex['task']})"
        else:
            seen = "job never completed a relaunch (I1 no_completed_relaunch)"
        b.append(t(cols[0], y + 24, cid, 13, weight="bold"))
        b.append(t(cols[0] + 36, y + 24, text, 13))
        b.append(t(cols[1], y + 24, f"cleanly in {fin} of {len(rs)}", 12.5))
        b.append(rect(cols[2], y + 8, 62, 24, "#e6f4ef" if acc == 0 else "#fdecea", rx=12))
        b.append(t(cols[2] + 31, y + 25, f"{acc} of {len(rs)}", 12.5, "middle", "bold", GREEN if acc == 0 else VERM))
        for k, part in enumerate(wrap(seen, 44)):
            b.append(t(cols[3], y + 22 + 14 * k, part, 11.5, fill=INK))
        y += rh
        b.append(line(30, y, W - 30, y, GRID, 1))
    H = y + 46
    b.append(t(30, y + 22, 'Job finished = the relaunch ended cleanly (exit 0). None of those runs was accepted; each was rejected on an invariant.', 10.5, fill=MUTED))
    b.append(t(30, y + 36, "I1 relaunch completes, I3 params match the reference, I6 world size, samples gaps/dups = scheduled samples skipped/consumed twice.", 10.5, fill=MUTED))
    return svg(W, H + 10, "".join(b), "Why exit code 0 is not recovery",
               "Fake repairs that make the job stop failing, and the invariant that rejected them.")


def wrap(s, n):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n and cur:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + [cur]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = parse_redteam()
    assert len(rows) == 107, f"expected 107 red-team rows, got {len(rows)}"
    figs = {"fig_failure.svg": fig_failure(), "fig_redteam_funnel.svg": fig_funnel(rows),
            "fig_test_results.svg": fig_test(), "fig_fake_repairs.svg": fig_fake(rows)}
    for name, s in figs.items():
        (OUT / name).write_text(s)
        print("wrote", (OUT / name).relative_to(ROOT), len(s), "bytes")


if __name__ == "__main__":
    sys.exit(main())
