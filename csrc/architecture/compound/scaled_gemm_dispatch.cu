#include "architecture/compound/scaled_gemm_dispatch.hpp"

#include <utility>

#include "extensions/registry.hpp"

namespace nkernels::architecture::compound {

namespace {

using hardware_architecture::Backend;
using hardware_architecture::DeviceProfile;

bool is_tensor_core_aligned(const ScaledGemmRequest& request) {
  return request.alignment_bytes >= 16 && request.problem.m % 8 == 0 &&
         request.problem.n % 8 == 0 && request.problem.k % 16 == 0;
}

core::AlignmentRequirement default_alignment_for(
    const quantization::MatmulExecutionPolicy& execution) {
  if (!execution.use_tensor_cores) {
    return core::AlignmentRequirement{4, 1, 1, 1};
  }
  return core::AlignmentRequirement{16, 8, 8, 16};
}

ScaledGemmDispatchResult make_unsupported(
    quantization::MatmulExecutionPolicy execution, std::string diagnostic) {
  ScaledGemmDispatchResult result{};
  result.supported = false;
  result.execution = std::move(execution);
  result.diagnostic = std::move(diagnostic);
  return result;
}

ScaledGemmDispatchResult make_supported(
    const DeviceProfile& profile, const ScaledGemmRequest& request,
    quantization::MatmulExecutionPolicy execution, CompoundKernelFamily family,
    std::string kernel_symbol, std::string diagnostic) {
  ScaledGemmDispatchResult result{};
  result.supported = true;
  result.family = family;
  result.kernel_symbol = std::move(kernel_symbol);
  result.execution = std::move(execution);
  result.diagnostic = std::move(diagnostic);

  result.contract.algorithm_name = result.kernel_symbol;
  result.contract.activation_type = request.quantization.activation_type;
  result.contract.weight_type = request.quantization.weight_type;
  result.contract.output_type = request.quantization.output_type;
  result.contract.accumulator_type = result.execution.accumulator_type;
  result.contract.activation_layout = request.activation_layout;
  result.contract.weight_layout = request.weight_layout;
  result.contract.output_layout = request.output_layout;
  result.contract.alignment = default_alignment_for(result.execution);
  result.contract.deterministic = request.quantization.require_strict_stability;
  result.contract.requires_contiguous_scale_buffers =
      result.execution.scheme != quantization::QuantizationScheme::kNone;
  result.contract.allows_runtime_fallback = true;
  result.contract.launch_priority =
      request.quantization.require_strict_stability
          ? core::LaunchPriority::kStability
          : (request.quantization.prefer_max_throughput
                 ? core::LaunchPriority::kThroughput
                 : core::LaunchPriority::kBalanced);
  result.tuning = core::make_default_tuning_hint(
      profile, result.contract, result.execution, request.problem);
  return result;
}

ScaledGemmDispatchResult try_cuda_builtin(
    const DeviceProfile& profile, const ScaledGemmRequest& request,
    quantization::MatmulExecutionPolicy execution) {
  if (execution.use_tensor_cores && !is_tensor_core_aligned(request)) {
    return make_unsupported(
        std::move(execution),
        "Tensor-core path requires 16-byte aligned operands with M/N multiples "
        "of 8 and K multiple of 16.");
  }

  switch (execution.scheme) {
    case quantization::QuantizationScheme::kNvFp4Blockwise:
      if (!hardware_architecture::supports_nvfp4(profile)) {
        return make_unsupported(
            std::move(execution),
            "NVFP4 requires CUDA runtime >= 12.8 and SM100-class or newer "
            "hardware.");
      }
      return make_supported(
          profile, request, std::move(execution),
          CompoundKernelFamily::kCudaNvFp4,
          profile.arch >= 120 ? "cutlass_scaled_fp4_mm_sm120a"
                              : "cutlass_scaled_fp4_mm_sm100a",
          "Selected block-scaled FP4 CUDA route.");

    case quantization::QuantizationScheme::kFp8Blockwise:
      if (!hardware_architecture::supports_fp8_blockwise(profile)) {
        return make_unsupported(
            std::move(execution),
            "Block-scaled FP8 requires Hopper-class or newer hardware with a "
            "compatible CUDA runtime.");
      }
      return make_supported(
          profile, request, std::move(execution),
          CompoundKernelFamily::kCudaC3x,
          profile.arch >= 120 ? "cutlass_scaled_mm_blockwise_sm120_fp8"
          : profile.arch >= 100 ? "cutlass_scaled_mm_blockwise_sm100_fp8"
                                : "cutlass_scaled_mm_blockwise_sm90_fp8",
          "Selected block-scaled FP8 CUDA route.");

    case quantization::QuantizationScheme::kFp8Tensorwise:
      if (!hardware_architecture::supports_fp8_tensorwise(profile)) {
        return make_unsupported(
            std::move(execution),
            "Tensorwise FP8 requires Ada-class or newer hardware with a "
            "compatible CUDA runtime.");
      }
      if (profile.arch >= 120) {
        return make_supported(
            profile, request, std::move(execution),
            CompoundKernelFamily::kCudaC3x,
            "cutlass_scaled_mm_sm120_fp8",
            "Selected SM120 tensorwise FP8 CUDA route.");
      }
      if (profile.arch >= 100) {
        return make_supported(
            profile, request, std::move(execution),
            CompoundKernelFamily::kCudaC3x,
            "cutlass_scaled_mm_sm100_fp8",
            "Selected SM100 tensorwise FP8 CUDA route.");
      }
      if (profile.arch >= 90) {
        return make_supported(
            profile, request, std::move(execution),
            CompoundKernelFamily::kCudaC3x,
            "cutlass_scaled_mm_sm90_fp8",
            "Selected SM90 tensorwise FP8 CUDA route.");
      }
      return make_supported(
          profile, request, std::move(execution),
          CompoundKernelFamily::kCudaC2x,
          "cutlass_scaled_mm_sm89",
          "Selected SM89 tensorwise FP8 CUDA route.");

    case quantization::QuantizationScheme::kInt8Symmetric:
    case quantization::QuantizationScheme::kInt8Asymmetric:
      if (profile.arch >= 89) {
        return make_supported(
            profile, request, std::move(execution),
            CompoundKernelFamily::kCudaC2x,
            "cutlass_scaled_mm_sm89", "Selected INT8 CUDA route for SM89.");
      }
      if (profile.arch >= 80) {
        return make_supported(
            profile, request, std::move(execution),
            CompoundKernelFamily::kCudaC2x,
            "cutlass_scaled_mm_sm80", "Selected INT8 CUDA route for SM80+.");
      }
      if (profile.arch >= 75) {
        return make_supported(
            profile, request, std::move(execution),
            CompoundKernelFamily::kCudaC2x,
            "cutlass_scaled_mm_sm75", "Selected INT8 CUDA route for SM75.");
      }
      return make_unsupported(std::move(execution),
                              "INT8 tensor-core kernels require SM75 or newer.");

    case quantization::QuantizationScheme::kNone:
    default:
      if (request.quantization.activation_type == core::ScalarType::kBFloat16 ||
          request.quantization.weight_type == core::ScalarType::kBFloat16) {
        if (!hardware_architecture::supports_bf16_math(profile)) {
          return make_unsupported(
              std::move(execution),
              "BF16 tensor-core kernels require SM80 or newer hardware.");
        }
        return make_supported(
            profile, request, std::move(execution),
            profile.arch >= 90 ? CompoundKernelFamily::kCudaC3x
                               : CompoundKernelFamily::kCudaC2x,
            profile.arch >= 90 ? "tensorop_bf16_gemm_sm90_or_later"
                               : "tensorop_bf16_gemm_sm80",
            "Selected BF16 CUDA route with widened accumulation.");
      }

      if (request.quantization.activation_type == core::ScalarType::kFloat16 ||
          request.quantization.weight_type == core::ScalarType::kFloat16) {
        if (profile.arch < 75) {
          return make_unsupported(
              std::move(execution),
              "FP16 tensor-core kernels require SM75 or newer hardware.");
        }
        return make_supported(
            profile, request, std::move(execution),
            profile.arch >= 90 ? CompoundKernelFamily::kCudaC3x
                               : CompoundKernelFamily::kCudaC2x,
            profile.arch >= 90 ? "tensorop_f16_gemm_sm90_or_later"
                               : "tensorop_f16_gemm_sm75_to_sm89",
            "Selected FP16 CUDA route with widened accumulation.");
      }

      return make_supported(profile, request, std::move(execution),
                            CompoundKernelFamily::kUnknown,
                            "simt_fp32_reference_gemm",
                            "Selected SIMT FP32 reference route.");
  }
}

ScaledGemmDispatchResult try_rocm_builtin(
    const DeviceProfile& profile, const ScaledGemmRequest& request,
    quantization::MatmulExecutionPolicy execution) {
  (void)request;
  return make_unsupported(
      std::move(execution),
      std::string("No built-in ROCm kernel family is registered for ") +
          hardware_architecture::to_string(profile.family) +
          ". Insert CK or backend-specific kernels through the extension "
          "registry.");
}

}  // namespace

ScaledGemmDispatchResult dispatch_scaled_gemm(
    const DeviceProfile& profile, const ScaledGemmRequest& request) {
  if (request.problem.m <= 0 || request.problem.n <= 0 || request.problem.k <= 0) {
    return make_unsupported({}, "Scaled GEMM shape must be strictly positive.");
  }

  const auto execution =
      quantization::make_matmul_execution_policy(profile, request.quantization);

  ScaledGemmDispatchResult result{};
  if (profile.backend == Backend::kCuda) {
    result = try_cuda_builtin(profile, request, execution);
  } else if (profile.backend == Backend::kRocm) {
    result = try_rocm_builtin(profile, request, execution);
  } else {
    result = make_unsupported(
        execution, "Only CUDA and ROCm backends are modeled in this dispatch.");
  }

  if (!result.supported && request.allow_extensions) {
    for (const auto& extension : extensions::scaled_gemm_extensions()) {
      ScaledGemmDispatchResult extension_result{};
      if (extension.support != nullptr &&
          extension.support(profile, request, &extension_result)) {
        extension_result.family = CompoundKernelFamily::kExtension;
        if (extension_result.kernel_symbol.empty()) {
          extension_result.kernel_symbol = extension.symbol;
        }
        if (extension_result.contract.algorithm_name.empty()) {
          extension_result.contract.algorithm_name =
              extension_result.kernel_symbol;
        }
        return extension_result;
      }
    }
  }

  return result;
}

ScaledGemmDispatchResult dispatch_scaled_gemm(const ScaledGemmRequest& request) {
  return dispatch_scaled_gemm(hardware_architecture::query_active_device_profile(),
                              request);
}

}  // namespace nkernels::architecture::compound
