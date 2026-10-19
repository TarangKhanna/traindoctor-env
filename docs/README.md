# Documentation

Guides for the TrainDoctor environment and verifier. Data, scripts and write-ups of the training and evaluation results are in [traindoctor-research](https://github.com/TarangKhanna/traindoctor-research).

## Reading paths

- **5 minutes:** the top-level [`README`](../README.md), then [`ARCHITECTURE.md`](../ARCHITECTURE.md).
- **See one example end to end:** [`DEMO.md`](DEMO.md) (oracle demonstration: evidence, diagnosis, patch, replay, verdicts).
- **How it works:** [`VERIFICATION.md`](VERIFICATION.md), [`TASKS.md`](TASKS.md).
- **Run something:** [`QUICKSTART.md`](QUICKSTART.md) (local, no account) and [`TESTING.md`](TESTING.md).
- **Build on it:** [`EXTENDING.md`](EXTENDING.md).

## Guides

| file | contents |
|---|---|
| [`QUICKSTART.md`](QUICKSTART.md) | install, local and sandboxed quickstart, unit-test commands |
| [`DEMO.md`](DEMO.md) | one oracle demonstration: failure, patch, replay, gold and fake verdicts |
| [`TASKS.md`](TASKS.md) | the five task kinds and the two tiers |
| [`VERIFICATION.md`](VERIFICATION.md) | invariants, `exec_mode` trust table, red-team details |
| [`EXTENDING.md`](EXTENDING.md) | using the packages, adding a task |
| [`TESTING.md`](TESTING.md) | test selections, counts and known skips |
| [`figures/`](figures/) | `fig_failure` (a hang that the world_size:1 fake removes) and `fig_trust_boundary` |
