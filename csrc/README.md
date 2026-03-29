# nkernels `csrc`

This source tree is organized by engineering responsibility rather than by
individual benchmark targets.

- `core/`: common scalar types, kernel contracts, and error handling.
- `core/`: common scalar types, kernel contracts, performance envelopes, and
  error handling.
- `hardware_architecture/`: backend normalization, runtime capability discovery,
  and architecture feature gating.
- `quantization/`: scale-granularity and accumulator policies.
- `architecture/compound/`: algorithm-level dispatch that selects the correct
  kernel family for a full request.
- `extensions/`: registration points for future kernels so new implementations
  can be inserted without editing central dispatch code.
  This layer also owns API-surface retention so refactors do not drop exported
  methods.

Placement rules:

- Put architecture and runtime gates in `hardware_architecture/`.
- Put dtype and scaling policy in `quantization/`.
- Put algorithm routing in `architecture/compound/`.
- Keep raw kernel implementations out of `core/`.
- Preserve the exported method surface through the `extensions/` API manifest.
  Use canonical `nkernels.<domain>.<method>` identifiers for all new work.
- Track source and API migration gaps with `tools/report_migration_gap.sh` and
  review `reports/legacy_migration_status.md` before claiming parity.
