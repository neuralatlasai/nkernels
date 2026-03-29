#pragma once

#include <cstdint>

namespace nkernels::core {

enum class ScalarType : uint8_t {
  kUnknown = 0,
  kFloat32,
  kFloat16,
  kBFloat16,
  kFloat8E4M3FN,
  kFloat8E5M2,
  kNvFloat4,
  kInt8,
  kUInt8,
  kInt32,
};

constexpr const char* to_string(ScalarType type) {
  switch (type) {
    case ScalarType::kFloat32:
      return "fp32";
    case ScalarType::kFloat16:
      return "fp16";
    case ScalarType::kBFloat16:
      return "bf16";
    case ScalarType::kFloat8E4M3FN:
      return "fp8_e4m3fn";
    case ScalarType::kFloat8E5M2:
      return "fp8_e5m2";
    case ScalarType::kNvFloat4:
      return "nvfp4";
    case ScalarType::kInt8:
      return "int8";
    case ScalarType::kUInt8:
      return "uint8";
    case ScalarType::kInt32:
      return "int32";
    case ScalarType::kUnknown:
    default:
      return "unknown";
  }
}

constexpr bool is_fp8(ScalarType type) {
  return type == ScalarType::kFloat8E4M3FN ||
         type == ScalarType::kFloat8E5M2;
}

constexpr bool is_fp4(ScalarType type) {
  return type == ScalarType::kNvFloat4;
}

constexpr bool is_low_precision_float(ScalarType type) {
  return type == ScalarType::kFloat16 || type == ScalarType::kBFloat16 ||
         is_fp8(type) || is_fp4(type);
}

constexpr bool is_integer_quantized(ScalarType type) {
  return type == ScalarType::kInt8 || type == ScalarType::kUInt8;
}

}  // namespace nkernels::core
