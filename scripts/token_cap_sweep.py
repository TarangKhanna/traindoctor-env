"""Offline token sweep: python scripts/token_cap_sweep.py CAP PESSIMISTIC(0|1) N. Synthetic records, no Gloo; true Qwen3 tokens."""
import sys, tempfile, statistics as st, collections
from pathlib import Path
from traindoctor_env.bundle import make_menu, make_bundle, BundleTooLarge
import traindoctor_env.bundle as B
from traindoctor_env.scenario import make_scenario
from traindoctor_env.tiera import training_prompt
from tests.synth_snapshot import synth_snapshot
import transformers
tok = transformers.AutoTokenizer.from_pretrained("mlx-community/Qwen3-0.6B-4bit", local_files_only=True)
cap = int(sys.argv[1]); B.CAP_TOKENS = cap
pess = sys.argv[2] == "1"
by = collections.defaultdict(list); fails = 0; n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
for s in range(10000, 10000 + n):
    sc = make_scenario(s)
    with tempfile.TemporaryDirectory() as t:
        snap, truth = synth_snapshot(sc, Path(t) / "jd", pess)
        menu = make_menu(snap, truth)
        try: b = make_bundle(snap, menu, truth)
        except BundleTooLarge as e: fails += 1; by[(sc.kind, sc.world_size)].append(-1); continue
        by[(sc.kind, sc.world_size)].append(len(tok.encode(training_prompt(b))))
allv = [x for v in by.values() for x in v if x > 0]
for k, v in sorted(by.items()): print(k, len(v), "max", max(v), "median", st.median(v))
q = sorted(allv)
print("cap", cap, "pess", pess, "TooLarge", fails, "n", len(allv), "max", q[-1], "median", st.median(q), "p95", q[int(.95*len(q))-1])
