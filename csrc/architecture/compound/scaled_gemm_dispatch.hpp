#pragma once

#include <string>

#include "core/kernel_contract.hpp"
#include "core/performance_model.hpp"
#include "hardware_architecture/device_profile.hpp"
#include "quantization/policy.hpp"

namespace nkernels::architecture::compound {

enum class CompoundKernelFamily : uint8_t {
  kUnknown = 0,
  kCudaC2x,
  kCudaC3x,
  kCudaNvFp4,
  kRocmComposableKernel,
  kExtension,
};

struct ScaledGemmRequest {
  core::MatmulShape problem;
  quantization::MatmulQuantizationPolicy quantization;
  core::MemoryLayout activation_layout = core::MemoryLayout::kRowMajor;
  core::MemoryLayout weight_layout = core::MemoryLayout::kColumnMajor;
  core::MemoryLayout output_layout = core::MemoryLayout::kRowMajor;
  int64_t alignment_bytes = 16;
  bool allow_extensions = true;
};

struct ScaledGemmDispatchResult {
  bool supported = false;
  CompoundKernelFamily family = CompoundKernelFamily::kUnknown;
  std::string kernel_symbol;
  core::KernelContract contract;
  core::KernelTuningHint tuning;
  quantization::MatmulExecutionPolicy execution;
  std::string diagnostic;
};

constexpr const char* to_string(CompoundKernelFamily family) {
  switch (family) {
    case CompoundKernelFamily::kCudaC2x:
      return "cuda_cutlass_2x";
    case CompoundKernelFamily::kCudaC3x:
      return "cuda_cutlass_3x";
    case CompoundKernelFamily::kCudaNvFp4:
      return "cuda_nvfp4";
    case CompoundKernelFamily::kRocmComposableKernel:
      return "rocm_composable_kernel";
    case CompoundKernelFamily::kExtension:
      return "extension";
    case CompoundKernelFamily::kUnknown:
    default:
      return "unknown";
  }
}

ScaledGemmDispatchResult dispatch_scaled_gemm(
    const hardware_architecture::DeviceProfile& profile,
    const ScaledGemmRequest& request);

ScaledGemmDispatchResult dispatch_scaled_gemm(
    const ScaledGemmRequest& request);

}  // namespace nkernels::architecture::compound
