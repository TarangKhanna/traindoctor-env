#!/usr/bin/env python3
"""Regenerate the README figures in docs/figures/ (pure-Python SVG, standard library only).

Every number drawn comes from a committed file:
  evidence/results/REDTEAM_TABLE.md -> fig_redteam_funnel.svg
fig_failure.svg is a schematic of the quickstart scenario (seed 123, Task A1); it draws no measured numbers.

Usage:  python scripts/make_figures.py          (from the repo root; writes docs/figures/*.svg)
Colors are the Okabe-Ito colorblind-safe palette.
"""
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
    for ln in (ROOT / "evidence/results/REDTEAM_TABLE.md").read_text().splitlines():
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = parse_redteam()
    assert len(rows) == 107, f"expected 107 red-team rows, got {len(rows)}"
    figs = {"fig_failure.svg": fig_failure(), "fig_redteam_funnel.svg": fig_funnel(rows)}
    for name, s in figs.items():
        (OUT / name).write_text(s)
        print("wrote", (OUT / name).relative_to(ROOT), len(s), "bytes")


if __name__ == "__main__":
    sys.exit(main())
