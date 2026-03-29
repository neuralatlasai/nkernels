#pragma once

#include <cstdint>
#include <string>

namespace nkernels::hardware_architecture {

enum class Backend : uint8_t {
  kUnknown = 0,
  kCuda,
  kRocm,
  kOneApi,
  kCpu,
};

enum class Vendor : uint8_t {
  kUnknown = 0,
  kNVIDIA,
  kAMD,
  kIntel,
};

enum class ArchitectureFamily : uint8_t {
  kUnknown = 0,
  kTuring,
  kAmpere,
  kAda,
  kHopper,
  kBlackwell,
  kPostBlackwell,
  kCDNA2,
  kCDNA3,
  kCDNA4,
  kFutureAMD,
};

struct FeatureSet {
  bool tensor_cores = false;
  bool bf16 = false;
  bool fp8 = false;
  bool block_fp8 = false;
  bool nvfp4 = false;
  bool async_tma = false;
  bool shared_memory_multicast = false;
  bool paged_attention = false;
};

struct DeviceProfile {
  Backend backend = Backend::kUnknown;
  Vendor vendor = Vendor::kUnknown;
  ArchitectureFamily family = ArchitectureFamily::kUnknown;
  int device_index = -1;
  int runtime_version = 0;
  int driver_version = 0;
  int major = 0;
  int minor = 0;
  int arch = 0;
  int multiprocessor_count = 0;
  int max_threads_per_block = 0;
  size_t shared_memory_per_block = 0;
  size_t shared_memory_per_multiprocessor = 0;
  std::string name;
  std::string native_arch_name;
  FeatureSet features;
};

DeviceProfile query_active_device_profile();
DeviceProfile query_device_profile(int device_index);

constexpr const char* to_string(Backend backend) {
  switch (backend) {
    case Backend::kCuda:
      return "cuda";
    case Backend::kRocm:
      return "rocm";
    case Backend::kOneApi:
      return "oneapi";
    case Backend::kCpu:
      return "cpu";
    case Backend::kUnknown:
    default:
      return "unknown";
  }
}

constexpr const char* to_string(Vendor vendor) {
  switch (vendor) {
    case Vendor::kNVIDIA:
      return "nvidia";
    case Vendor::kAMD:
      return "amd";
    case Vendor::kIntel:
      return "intel";
    case Vendor::kUnknown:
    default:
      return "unknown";
  }
}

constexpr const char* to_string(ArchitectureFamily family) {
  switch (family) {
    case ArchitectureFamily::kTuring:
      return "turing";
    case ArchitectureFamily::kAmpere:
      return "ampere";
    case ArchitectureFamily::kAda:
      return "ada";
    case ArchitectureFamily::kHopper:
      return "hopper";
    case ArchitectureFamily::kBlackwell:
      return "blackwell";
    case ArchitectureFamily::kPostBlackwell:
      return "post_blackwell";
    case ArchitectureFamily::kCDNA2:
      return "cdna2";
    case ArchitectureFamily::kCDNA3:
      return "cdna3";
    case ArchitectureFamily::kCDNA4:
      return "cdna4";
    case ArchitectureFamily::kFutureAMD:
      return "future_amd";
    case ArchitectureFamily::kUnknown:
    default:
      return "unknown";
  }
}

inline bool is_cuda(const DeviceProfile& profile) {
  return profile.backend == Backend::kCuda;
}

inline bool is_rocm(const DeviceProfile& profile) {
  return profile.backend == Backend::kRocm;
}

inline bool supports_bf16_math(const DeviceProfile& profile) {
  return profile.features.bf16;
}

inline bool supports_fp8_tensorwise(const DeviceProfile& profile) {
  return profile.features.fp8;
}

inline bool supports_fp8_blockwise(const DeviceProfile& profile) {
  return profile.features.block_fp8;
}

inline bool supports_nvfp4(const DeviceProfile& profile) {
  return profile.features.nvfp4;
}

}  // namespace nkernels::hardware_architecture
