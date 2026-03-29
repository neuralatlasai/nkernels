#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  cat <<'EOF'
Usage:
  run_repo_compare.sh <baseline_repo_root> <candidate_repo_root> <baseline_endpoint> <candidate_endpoint> <baseline_namespace_map_json> <candidate_namespace_map_json> [output_dir]

Example:
  run_repo_compare.sh \
    /content/vllm \
    /content/neuralatlasai_world/nkernels \
    /content/vllm/build/libvllm_ext.so \
    /content/neuralatlasai_world/nkernels/build/libnkernels_ext.so \
    '{"cuda":"_C","cpu":"_C","moe":"_moe_C","rocm":"_rocm_C","stable":"_C"}' \
    '{"cuda":"_C","cpu":"_C","moe":"_moe_C","rocm":"_rocm_C","stable":"_C"}' \
    /content/bench_results
EOF
  exit 1
fi

baseline_repo_root="$1"
candidate_repo_root="$2"
baseline_endpoint="$3"
candidate_endpoint="$4"
baseline_namespace_map="$5"
candidate_namespace_map="$6"
output_dir="${7:-${PWD}/bench_results}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${script_dir}/compare_ops.py" \
  --baseline-source-root "${baseline_repo_root}" \
  --candidate-source-root "${candidate_repo_root}" \
  --baseline "${baseline_endpoint}" \
  --candidate "${candidate_endpoint}" \
  --baseline-namespace-map "${baseline_namespace_map}" \
  --candidate-namespace-map "${candidate_namespace_map}" \
  --device cuda \
  --warmup 20 \
  --iters 100 \
  --output-dir "${output_dir}"
