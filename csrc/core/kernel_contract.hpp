#pragma once

#include <cstdint>
#include <string>

#include "core/scalar_type.hpp"

namespace nkernels::core {

enum class MemoryLayout : uint8_t {
  kUnknown = 0,
  kRowMajor,
  kColumnMajor,
  kInterleaved32,
  kPaged,
};

enum class LaunchPriority : uint8_t {
  kStability = 0,
  kBalanced,
  kThroughput,
};

struct AlignmentRequirement {
  int64_t base_pointer_alignment_bytes = 1;
  int64_t m_multiple = 1;
  int64_t n_multiple = 1;
  int64_t k_multiple = 1;
};

struct MatmulShape {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

struct KernelContract {
  std::string algorithm_name;
  ScalarType activation_type = ScalarType::kUnknown;
  ScalarType weight_type = ScalarType::kUnknown;
  ScalarType output_type = ScalarType::kUnknown;
  ScalarType accumulator_type = ScalarType::kUnknown;
  MemoryLayout activation_layout = MemoryLayout::kUnknown;
  MemoryLayout weight_layout = MemoryLayout::kUnknown;
  MemoryLayout output_layout = MemoryLayout::kUnknown;
  AlignmentRequirement alignment;
  bool deterministic = false;
  bool requires_contiguous_scale_buffers = true;
  bool allows_runtime_fallback = true;
  LaunchPriority launch_priority = LaunchPriority::kBalanced;
};

constexpr const char* to_string(MemoryLayout layout) {
  switch (layout) {
    case MemoryLayout::kRowMajor:
      return "row_major";
    case MemoryLayout::kColumnMajor:
      return "column_major";
    case MemoryLayout::kInterleaved32:
      return "interleaved32";
    case MemoryLayout::kPaged:
      return "paged";
    case MemoryLayout::kUnknown:
    default:
      return "unknown";
  }
}

constexpr const char* to_string(LaunchPriority priority) {
  switch (priority) {
    case LaunchPriority::kStability:
      return "stability";
    case LaunchPriority::kBalanced:
      return "balanced";
    case LaunchPriority::kThroughput:
      return "throughput";
    default:
      return "balanced";
  }
}

inline bool satisfies_alignment(const MatmulShape& shape,
                                const AlignmentRequirement& requirement) {
  return shape.m > 0 && shape.n > 0 && shape.k > 0 &&
         shape.m % requirement.m_multiple == 0 &&
         shape.n % requirement.n_multiple == 0 &&
         shape.k % requirement.k_multiple == 0;
}

}  // namespace nkernels::core
