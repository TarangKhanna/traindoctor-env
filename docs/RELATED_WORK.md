# Related work and differentiation

Brief and non-exhaustive; claims are limited to what the linked pages say.
- [AIOpsLab](https://arxiv.org/abs/2501.06706) and [Cloud-OpsBench](https://arxiv.org/abs/2603.00468): agent benchmarks for cloud/microservice operations (detection, localization, mitigation). TrainDoctor targets distributed ML training jobs and grades recovery with training-correctness invariants.
- [DDBench](https://arxiv.org/abs/2608.14863): agentic repair of 60 historical bugs in open-source distributed systems, graded by a hidden test oracle. TrainDoctor uses executable multi-rank ML-training failures and re-executes the fix to check training state.
- [SWE-Gym](https://arxiv.org/abs/2412.21139), [R2E-Gym](https://arxiv.org/abs/2504.07164), [SWE-RL](https://arxiv.org/abs/2502.18449), and SWE-bench-style environments: executable software-engineering tasks (issue fixing) for evaluation or RL, graded by unit tests. TrainDoctor's tasks are runtime failures of multi-process training, graded by recovery invariants.
- A Kubernetes SRE RL environment listed as "kube-sre-gym" / SRE-GYM (OpenEnv-compatible): only third-party listings were found, not a canonical page, so no comparison is made beyond noting it exists.
- [PyTorch Flight Recorder tutorial](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html): shows how to debug a stuck collective with trace analysis (NCCL). It is a debugging aid, not an agent environment. In our own check on Gloo/CPU the recorder logged ops but did not dump on timeout (`evidence/archive/m0/fault_proof.md`), so TrainDoctor writes its own per-rank collective trace.
- [Agentic Diagnosis for LLM Training at Scale (AMD)](https://rocm.blogs.amd.com/software-tools-optimization/maxtext-slurm-agentic-diagnosis/README.html): an agent workflow for diagnosing large training failures; a tool, not a benchmark with a recovery-checking verifier.
- RunSleuth: no verifiable source was found, so it is not described here.
- Reward-hacking and verifier-audit work such as [BenchJack](https://arxiv.org/abs/2605.12673) motivates auditing the reward with mutation tests and exploit probes, as done here.

TrainDoctor's combination: executable multi-rank ML-training failures, replay-verified recovery in a sandbox, and an audited reward. No claim of being first or only.
