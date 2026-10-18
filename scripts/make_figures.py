#!/usr/bin/env python3
"""Regenerate docs/figures/fig_failure.svg (standard library only; draws no measured numbers).

The figure is a schematic of the quickstart scenario (Task A1): at one step three ranks call all_reduce while one
calls broadcast, and every rank times out. Light and dark mode are handled by CSS classes inside the SVG.
fig_trust_boundary.svg is hand-written and not generated.

Usage:  python scripts/make_figures.py          (from the repo root)
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "fig_failure.svg"


def fig_failure():
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 480" width="360" height="480" role="img" aria-labelledby="title description">
<title id="title">Four ranks, one mismatched collective</title>
<desc id="description">At step 6, call #12, ranks 0,1,3 call all_reduce while rank 2 calls broadcast. All four ranks time out with the same COLLECTIVE_TIMEOUT.</desc>
<style>
text{font-family:Arial,Helvetica,sans-serif;font-size:14px}
.bg{fill:#f6f8fa}.box{fill:#fff;stroke:#a1aab4}.ink{fill:#1f2328}.sub{fill:#59636e}.sbox{fill:#eef4fd;stroke:#0969da}.untrusted{fill:#fff1ee;stroke:#cf222e}.owned{fill:#ebf3ff;stroke:#0969da}.outside{fill:#e8f7ec;stroke:#1a7f37}.arrow{stroke:#57606a;fill:none}
.orange{fill:#fff1dc;stroke:#b86e00}.blue{fill:#ddebfb;stroke:#0969da}.exit{fill:#ffe3e0;stroke:#cf222e}
.heading{font-size:20px;font-weight:700}
@media(prefers-color-scheme:dark){
.bg{fill:#0d1117}.box{fill:#161b22;stroke:#697580}.ink{fill:#e6edf3}.sub{fill:#a7b0bc}.sbox{fill:#12253a;stroke:#58a6ff}.untrusted{fill:#301d23;stroke:#f85149}.owned{fill:#152a43;stroke:#58a6ff}.outside{fill:#132c20;stroke:#3fb950}.arrow{stroke:#a7b0bc}
.orange{fill:#3a2a10;stroke:#f0ad4e}.blue{fill:#152a43;stroke:#58a6ff}.exit{fill:#301d23;stroke:#f85149}
}
</style>
<rect class="bg" width="360" height="480" rx="12"/>
<g text-anchor="middle">
<text class="ink heading" x="180" y="28">Four ranks,</text>
<text class="ink heading" x="180" y="51">one mismatched collective</text>
<text class="sub" x="180" y="77">Step 6, call #12</text>
</g>
<rect class="blue" x="20" y="88" width="320" height="30" rx="6"/>
<rect class="blue" x="20" y="124" width="320" height="30" rx="6"/>
<rect class="orange" x="20" y="160" width="320" height="30" rx="6"/>
<rect class="blue" x="20" y="196" width="320" height="30" rx="6"/>
<g class="ink">
<text x="34" y="108">rank 0</text><text x="154" y="108">all_reduce</text>
<text x="34" y="144">rank 1</text><text x="154" y="144">all_reduce</text>
<text x="34" y="180" font-weight="700">rank 2</text><text x="154" y="180" font-weight="700">broadcast (differs)</text>
<text x="34" y="216">rank 3</text><text x="154" y="216">all_reduce</text>
</g>
<path class="arrow" d="M180 232v16m-4-5 4 5 4-5" stroke-width="2"/>
<g class="sub" text-anchor="middle">
<text x="180" y="266">All four ranks wait:</text>
<text x="180" y="283">no collective can complete</text>
</g>
<rect class="exit" x="20" y="298" width="154" height="32" rx="6"/>
<rect class="exit" x="186" y="298" width="154" height="32" rx="6"/>
<rect class="exit" x="20" y="338" width="154" height="32" rx="6"/>
<rect class="exit" x="186" y="338" width="154" height="32" rx="6"/>
<g class="ink" text-anchor="middle">
<text x="97" y="319">rank 0 · exit 1</text>
<text x="263" y="319">rank 1 · exit 1</text>
<text x="97" y="359">rank 2 · exit 1</text>
<text x="263" y="359">rank 3 · exit 1</text>
<text x="180" y="392">COLLECTIVE_TIMEOUT on every rank</text>
</g>
<rect class="box" x="20" y="405" width="320" height="67" rx="8"/>
<g class="ink">
<text x="32" y="425">The error never names rank 2.</text>
<text x="32" y="444">Only the per-rank trace shows</text>
<text x="32" y="463">who diverged.</text>
</g>
</svg>
"""


if __name__ == "__main__":
    OUT.write_text(fig_failure())
    print(f"wrote {OUT.relative_to(OUT.parent.parent.parent)}")
