# Sanitization of the public copies

The files in `docs/results/` were copied from private run directories. Before publication they were edited as follows; the unsanitized originals (with their original SHA-256 sums, `ORIGINAL_SHA256SUMS`) are retained privately and are not part of this repository.

- **Home-directory paths.** The absolute path prefix of the machine that produced the runs (the user's home directory, which contained the username) was replaced by `<HOME>` in the six raw result JSONs: `redteam_modal_v11.json`, `redteam_isolate_long.json`, `redteam_a2v2_v2a.json`, `redteam_isolate.json`, `redteam_a2v2_v2b_f06.json`, `modal_sandbox_proof.json`. The occurrences were Python traceback paths from rank stderr. No result values were changed; the JSON still parses.
- **Sealed TEST seed values.** `TEST_REPORT.md` no longer states the numeric range of the sealed TEST seeds. The TEST result files already identify episodes by index only. The seed list itself is private; only its SHA-256 is published in `docs/prereg/test_protocol.md`.
- **New file.** `quickstart_verified_replay.txt` (output of a real Modal Sandbox replay, 2026-10-19) was added to the sums.

`SHA256SUMS` was regenerated for the sanitized copies. Verify with `cd docs && shasum -a 256 -c results/SHA256SUMS`.

Files whose hash changed (original -> sanitized, first 12 hex digits):

| file | original | sanitized |
|---|---|---|
| `docs/results/modal_sandbox_proof.json` | `5ce520f9402f` | `f156bfe71284` |
| `docs/results/redteam_a2v2_v2a.json` | `60ba89223567` | `733e30de777d` |
| `docs/results/redteam_a2v2_v2b_f06.json` | `d9f41e45fdbd` | `4bce81064c5e` |
| `docs/results/redteam_isolate_long.json` | `3bb74fa99bb6` | `271a5609ad27` |
| `docs/results/redteam_isolate.json` | `26ba4488cff3` | `809400900ac7` |
| `docs/results/redteam_modal_v11.json` | `3abc30d283bc` | `622a3b8d51b3` |
| `docs/results/TEST_REPORT.md` | `8c00cea13b35` | `3385aa523fbc` |

- **Derived file added after sanitization:** `docs/results/test/AUDIT_RECOUNT.json` (per-seed repair-choice rates, produced by `scripts/repair_choice_by_seed.py` from the committed scored/replay files; no other hashed file changed).
