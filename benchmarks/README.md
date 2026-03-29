# Colab Benchmark Harness

This directory contains a comparison harness for legacy `csrc` custom ops and
the `nkernels` implementation.

## Scope

- Inventories all canonical methods retained in
  `csrc/extensions/generated_legacy_api_manifest.inc`.
- Benchmarks methods that have a workload spec in `benchmark_specs.py`.
- Reports unported, missing, and unspec'd methods explicitly.

## Outputs

The harness writes:

- `baseline_results.json`
- `candidate_results.json`
- `joined_results.json`
- `joined_results.csv`
- `summary.md`

## Colab Usage

1. Clone the baseline repo, for example `vllm`.
2. Clone your repo.
3. Build or install the legacy extension from the cloned baseline repo.
4. Build or install the `nkernels` extension from your cloned repo.
5. Determine the `torch.ops` namespaces exposed by each build.
6. Run:

```bash
python nkernels/benchmarks/compare_ops.py \
  --baseline-source-root /content/vllm \
  --candidate-source-root /content/neuralatlasai_world/nkernels \
  --baseline /path/to/legacy_extension.so \
  --candidate /path/to/nkernels_extension.so \
  --baseline-namespace-map '{"cuda":"_C","cpu":"_C","moe":"_moe_C","rocm":"_rocm_C","stable":"_C"}' \
  --candidate-namespace-map '{"cuda":"_C","cpu":"_C","moe":"_moe_C","rocm":"_rocm_C","stable":"_C"}' \
  --device cuda \
  --warmup 20 \
  --iters 100 \
  --output-dir /content/bench_results
```

Or use the wrapper:

```bash
bash nkernels/benchmarks/run_repo_compare.sh \
  /content/vllm \
  /content/neuralatlasai_world/nkernels \
  /path/to/legacy_extension.so \
  /path/to/nkernels_extension.so \
  '{"cuda":"_C","cpu":"_C","moe":"_moe_C","rocm":"_rocm_C","stable":"_C"}' \
  '{"cuda":"_C","cpu":"_C","moe":"_moe_C","rocm":"_rocm_C","stable":"_C"}' \
  /content/bench_results
```

## Notes

- Namespace maps are build-specific. Adjust them to the actual namespaces
  registered in `torch.ops`.
- A method present in the manifest but absent at runtime is reported as
  `missing_op`.
- A method present at runtime but without a workload spec is reported as
  `no_spec`.
- The harness isolates baseline and candidate in separate subprocesses so
  conflicting custom op namespaces can be benchmarked safely in one run.
- When `--baseline-source-root` is provided, the script derives the compared
  method surface directly from the cloned baseline repo bindings.
