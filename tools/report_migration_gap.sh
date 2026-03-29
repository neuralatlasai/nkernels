#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nkernels_dir="$(cd "${script_dir}/.." && pwd)"
workspace_root="${1:-${nkernels_dir}/..}"

legacy_csrc="${workspace_root}/csrc"
nkernels_csrc="${nkernels_dir}/csrc"
report_dir="${nkernels_dir}/reports"
report_path="${report_dir}/legacy_migration_status.md"

mkdir -p "${report_dir}"

legacy_file_count="$(rg --files "${legacy_csrc}" | wc -l | tr -d ' ')"
nkernels_file_count="$(rg --files "${nkernels_csrc}" | wc -l | tr -d ' ')"
overlap_file_count="$(
  comm -12 \
    <(cd "${legacy_csrc}" && rg --files | sort) \
    <(cd "${nkernels_csrc}" && rg --files | sort) | wc -l | tr -d ' '
)"
missing_file_count="$(
  comm -23 \
    <(cd "${legacy_csrc}" && rg --files | sort) \
    <(cd "${nkernels_csrc}" && rg --files | sort) | wc -l | tr -d ' '
)"

legacy_method_count="$(
  perl -0777 -ne '
    while (/(?:ops|cache_ops|cuda_utils|custom_ar|rocm_ops|m)\.def\(\s*"([^"]+)"/sg) {
      $s = $1;
      $s =~ s/\(.+$//s;
      $s =~ s/\($//;
      next if $s eq q{};
      print "$s\n";
    }
  ' \
    "${workspace_root}/csrc/torch_bindings.cpp" \
    "${workspace_root}/csrc/cpu/torch_bindings.cpp" \
    "${workspace_root}/csrc/rocm/torch_bindings.cpp" \
    "${workspace_root}/csrc/moe/torch_bindings.cpp" \
    "${workspace_root}/csrc/libtorch_stable/torch_bindings.cpp" |
    sort -u | wc -l | tr -d ' '
)"

nkernels_binding_count="$(
  (rg -n 'ops\.def\(|m\.def\(|STABLE_TORCH_LIBRARY_FRAGMENT|TORCH_LIBRARY' \
    "${nkernels_csrc}" || true) | wc -l | tr -d ' '
)"

manifest_method_count="$(
  rg -c '^NKERNELS_API_METHOD' \
    "${nkernels_csrc}/extensions/generated_legacy_api_manifest.inc" |
    tr -d ' '
)"

{
  printf '%s\n' '# Legacy Migration Status'
  printf '\n%s\n' "Source tree comparison generated on $(date -u '+%Y-%m-%d %H:%M:%S UTC')."
  printf '\n%s\n' '## Summary'
  printf -- '- Legacy `csrc` files: `%s`\n' "${legacy_file_count}"
  printf -- '- `nkernels/csrc` files: `%s`\n' "${nkernels_file_count}"
  printf -- '- Relative-path overlap: `%s`\n' "${overlap_file_count}"
  printf -- '- Legacy files still missing in `nkernels/csrc`: `%s`\n' "${missing_file_count}"
  printf -- '- Unique legacy exported methods: `%s`\n' "${legacy_method_count}"
  printf -- '- Canonical manifest entries retained in `nkernels`: `%s`\n' "${manifest_method_count}"
  printf -- '- Actual binding registrations currently present in `nkernels/csrc`: `%s`\n' "${nkernels_binding_count}"
  printf '\n%s\n' '## Interpretation'
  printf '%s\n' 'The current `nkernels` tree contains migration infrastructure and API inventorying, but it does not yet contain the legacy binding layer or the full set of legacy kernel implementation units.'
  printf '\n%s\n' '## Missing File Samples'
  comm -23 \
    <(cd "${legacy_csrc}" && rg --files | sort) \
    <(cd "${nkernels_csrc}" && rg --files | sort) | sed -n '1,80p' | sed 's/^/- `/; s/$/`/'
} > "${report_path}"

printf '%s\n' "${report_path}"
