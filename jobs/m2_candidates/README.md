# M2 candidates (not part of M1 v2)

The four faults of the first M1 (`COLLECTIVE_MISMATCH`, `SHARDING_MISCONFIG`, `PREEMPTION_NO_CLEANUP`,
`NONIDEMPOTENT_RESUBMIT`) injected failures through config overrides and scheduler state in the old job. They
are kept here for M2, to be re-expressed as source/config patches on `jobs/template/` (see `jobs/build.py`) if
they add a distinct failure mode. They do not run against the current job. The last tree where they, their
tests and the old infrastructure all worked is the tag `m1-v1-4fault` (`git show m1-v1-4fault:jobs/train.py`).
