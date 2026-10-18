# Evidence

| path | contents |
|---|---|
| [`results/`](results/) | raw results and reports; verify with `cd evidence/results && shasum -a 256 -c SHA256SUMS` |
| [`prereg/`](prereg/) | frozen pre-registrations and protocols |
| [`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md) | each headline number mapped to its file, command and commit |
| [`quickstart_local_output.txt`](quickstart_local_output.txt) | recorded output of the local quickstart |
| [`archive/`](archive/) | superseded specs, development notes and loose development JSON; kept for the record |
| [`MOVES.md`](MOVES.md) | old path to new path for every moved file |

Files hash-pinned by `results/SHA256SUMS` and the frozen protocols are byte-identical to their original versions. They may mention pre-reorganization paths (for example `docs/results/`); see `MOVES.md` for the mapping.
