#pragma once

#include <cstdint>
#include <string>

#include "core/scalar_type.hpp"
#include "hardware_architecture/device_profile.hpp"

namespace nkernels::quantization {

enum class ScaleGranularity : uint8_t {
  kNone = 0,
  kTensor,
  kChannel,
  kToken,
  kGroup,
  kBlock,
};

enum class QuantizationScheme : uint8_t {
  kNone = 0,
  kInt8Symmetric,
  kInt8Asymmetric,
  kFp8Tensorwise,
  kFp8Blockwise,
  kNvFp4Blockwise,
};

struct MatmulQuantizationPolicy {
  core::ScalarType activation_type = core::ScalarType::kUnknown;
  core::ScalarType weight_type = core::ScalarType::kUnknown;
  core::ScalarType output_type = core::ScalarType::kUnknown;
  ScaleGranularity activation_scale_granularity = ScaleGranularity::kNone;
  ScaleGranularity weight_scale_granularity = ScaleGranularity::kNone;
  bool with_bias = false;
  bool prefer_max_throughput = false;
  bool require_strict_stability = true;
};

struct MatmulExecutionPolicy {
  QuantizationScheme scheme = QuantizationScheme::kNone;
  core::ScalarType accumulator_type = core::ScalarType::kFloat32;
  bool use_tensor_cores = false;
  bool use_two_stage_accumulation = false;
  bool fuse_dequantize_in_mainloop = false;
  std::string diagnostic;
};

constexpr const char* to_string(ScaleGranularity granularity) {
  switch (granularity) {
    case ScaleGranularity::kTensor:
      return "tensor";
    case ScaleGranularity::kChannel:
      return "channel";
    case ScaleGranularity::kToken:
      return "token";
    case ScaleGranularity::kGroup:
      return "group";
    case ScaleGranularity::kBlock:
      return "block";
    case ScaleGranularity::kNone:
    default:
      return "none";
  }
}

constexpr const char* to_string(QuantizationScheme scheme) {
  switch (scheme) {
    case QuantizationScheme::kInt8Symmetric:
      return "int8_symmetric";
    case QuantizationScheme::kInt8Asymmetric:
      return "int8_asymmetric";
    case QuantizationScheme::kFp8Tensorwise:
      return "fp8_tensorwise";
    case QuantizationScheme::kFp8Blockwise:
      return "fp8_blockwise";
    case QuantizationScheme::kNvFp4Blockwise:
      return "nvfp4_blockwise";
    case QuantizationScheme::kNone:
    default:
      return "none";
  }
}

inline QuantizationScheme infer_quantization_scheme(
    const MatmulQuantizationPolicy& policy) {
  if (core::is_fp4(policy.activation_type) || core::is_fp4(policy.weight_type)) {
    return QuantizationScheme::kNvFp4Blockwise;
  }

  if (core::is_fp8(policy.activation_type) || core::is_fp8(policy.weight_type)) {
    const bool block_scaled =
        policy.activation_scale_granularity == ScaleGranularity::kBlock ||
        policy.weight_scale_granularity == ScaleGranularity::kBlock;
    return block_scaled ? QuantizationScheme::kFp8Blockwise
                        : QuantizationScheme::kFp8Tensorwise;
  }

  if (core::is_integer_quantized(policy.activation_type) ||
      core::is_integer_quantized(policy.weight_type)) {
    return QuantizationScheme::kInt8Symmetric;
  }

  return QuantizationScheme::kNone;
}

inline MatmulExecutionPolicy make_matmul_execution_policy(
    const hardware_architecture::DeviceProfile& profile,
    const MatmulQuantizationPolicy& policy) {
  MatmulExecutionPolicy execution{};
  execution.scheme = infer_quantization_scheme(policy);
  execution.use_tensor_cores = profile.features.tensor_cores;

  switch (execution.scheme) {
    case QuantizationScheme::kNvFp4Blockwise:
      execution.accumulator_type =
          !policy.require_strict_stability &&
                  hardware_architecture::supports_bf16_math(profile)
              ? core::ScalarType::kBFloat16
              : core::ScalarType::kFloat32;
      execution.use_two_stage_accumulation = true;
      execution.fuse_dequantize_in_mainloop = true;
      execution.diagnostic =
          "Block-scaled FP4 path with widened accumulation.";
      return execution;

    case QuantizationScheme::kFp8Blockwise:
    case QuantizationScheme::kFp8Tensorwise:
      execution.accumulator_type = core::ScalarType::kFloat32;
      execution.use_two_stage_accumulation = true;
      execution.fuse_dequantize_in_mainloop = true;
      execution.diagnostic =
          "FP8 path with widened accumulation for stable reduction.";
      return execution;

    case QuantizationScheme::kInt8Symmetric:
    case QuantizationScheme::kInt8Asymmetric:
      execution.accumulator_type = core::ScalarType::kInt32;
      execution.use_two_stage_accumulation = true;
      execution.fuse_dequantize_in_mainloop = true;
      execution.diagnostic = "INT8 path with integer accumulation.";
      return execution;

    case QuantizationScheme::kNone:
    default:
      execution.accumulator_type = core::ScalarType::kFloat32;
      execution.use_two_stage_accumulation = false;
      execution.fuse_dequantize_in_mainloop = false;
      execution.diagnostic =
          "Floating-point path with widened accumulation.";
      return execution;
  }
}

}  // namespace nkernels::quantization
