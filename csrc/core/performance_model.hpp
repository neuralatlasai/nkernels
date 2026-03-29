#pragma once

#include <cstdint>

#include "core/kernel_contract.hpp"
#include "hardware_architecture/device_profile.hpp"
#include "quantization/policy.hpp"

namespace nkernels::core {

enum class SchedulerModel : uint8_t {
  kDefault = 0,
  kPersistent,
  kSplitK,
  kStreamK,
};

struct LaunchTile {
  int32_t cta_m = 0;
  int32_t cta_n = 0;
  int32_t cta_k = 0;
  int32_t worker_groups = 0;
  int32_t pipeline_stages = 0;
  int32_t cluster_m = 1;
  int32_t cluster_n = 1;
};

struct NumericGuardRails {
  bool widen_reduction = true;
  bool deterministic_split_k = false;
  bool bound_scale_reciprocal = true;
  bool require_finite_output = true;
};

struct KernelTuningHint {
  SchedulerModel scheduler = SchedulerModel::kDefault;
  LaunchTile tile;
  int64_t max_workspace_bytes = 0;
  double target_occupancy = 0.0;
  bool prefer_async_tma = false;
  bool prefer_multicast = false;
  NumericGuardRails numerics;
};

constexpr const char* to_string(SchedulerModel scheduler) {
  switch (scheduler) {
    case SchedulerModel::kPersistent:
      return "persistent";
    case SchedulerModel::kSplitK:
      return "split_k";
    case SchedulerModel::kStreamK:
      return "stream_k";
    case SchedulerModel::kDefault:
    default:
      return "default";
  }
}

inline KernelTuningHint make_default_tuning_hint(
    const hardware_architecture::DeviceProfile& profile,
    const KernelContract& contract,
    const quantization::MatmulExecutionPolicy& execution,
    const MatmulShape& shape) {
  KernelTuningHint hint{};
  hint.prefer_async_tma = profile.features.async_tma;
  hint.prefer_multicast = profile.features.shared_memory_multicast &&
                          shape.n >= 2048;
  hint.numerics.widen_reduction =
      execution.accumulator_type == ScalarType::kFloat32 ||
      execution.accumulator_type == ScalarType::kInt32;
  hint.numerics.deterministic_split_k = contract.deterministic;

  if (profile.arch >= 120) {
    hint.tile = LaunchTile{128, 256, 128, 8, 4, 2, 1};
    hint.target_occupancy = 0.72;
  } else if (profile.arch >= 100) {
    hint.tile = LaunchTile{128, 128, 128, 8, 4, 2, 1};
    hint.target_occupancy = 0.68;
  } else if (profile.arch >= 90) {
    hint.tile = LaunchTile{128, 128, 64, 8, 4, 1, 1};
    hint.target_occupancy = 0.64;
  } else if (profile.arch >= 80) {
    hint.tile = LaunchTile{128, 128, 64, 8, 3, 1, 1};
    hint.target_occupancy = 0.60;
  } else if (profile.arch >= 75) {
    hint.tile = LaunchTile{64, 128, 64, 4, 2, 1, 1};
    hint.target_occupancy = 0.54;
  } else {
    hint.tile = LaunchTile{64, 64, 32, 4, 2, 1, 1};
    hint.target_occupancy = 0.50;
  }

  if (execution.scheme == quantization::QuantizationScheme::kFp8Blockwise ||
      execution.scheme == quantization::QuantizationScheme::kNvFp4Blockwise) {
    hint.scheduler = SchedulerModel::kPersistent;
  } else if (shape.k >= 8192) {
    hint.scheduler = SchedulerModel::kSplitK;
    hint.max_workspace_bytes = shape.m * shape.n * sizeof(float);
  } else if (shape.m >= profile.multiprocessor_count * hint.tile.cta_m) {
    hint.scheduler = SchedulerModel::kStreamK;
  } else {
    hint.scheduler = SchedulerModel::kPersistent;
  }

  return hint;
}

}  // namespace nkernels::core
