# GRT-360 reproducibility profiles

Each profile is a declarative snapshot of a milestone.  Paths are resolved
relative to `GRT360_MODEL_ROOT` at delivery time; no profile contains a local
absolute path, credential, sequence name, ground truth, or result lookup.

`v5_final.json` is the current final candidate.  The older profiles are kept
for regression comparison and continued research.  Use `scripts/run_profile.py`
to materialize a command and `scripts/benchmark_cuda_backends.py` to compare
the CUDA backends on the same input set.
