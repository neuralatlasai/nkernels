#include "hardware_architecture/device_profile.hpp"

#include <string_view>

#include "core/error.hpp"
#include "hardware_architecture/runtime_compat.hpp"

namespace nkernels::hardware_architecture {

namespace {

void runtime_check(runtime::Error error, const char* operation) {
  if (error != runtime::kSuccess) {
    core::fail("hardware_architecture",
               std::string(operation) + " failed with runtime error: " +
                   runtime::get_error_string(error));
  }
}

ArchitectureFamily classify_cuda_family(int arch) {
  if (arch >= 130) {
    return ArchitectureFamily::kPostBlackwell;
  }
  if (arch >= 100) {
    return ArchitectureFamily::kBlackwell;
  }
  if (arch >= 90) {
    return ArchitectureFamily::kHopper;
  }
  if (arch == 89) {
    return ArchitectureFamily::kAda;
  }
  if (arch >= 80) {
    return ArchitectureFamily::kAmpere;
  }
  if (arch >= 75) {
    return ArchitectureFamily::kTuring;
  }
  return ArchitectureFamily::kUnknown;
}

int parse_gfx_arch(std::string_view name) {
  if (name.rfind("gfx", 0) != 0) {
    return 0;
  }
  int value = 0;
  for (size_t idx = 3; idx < name.size(); ++idx) {
    const char c = name[idx];
    if (c < '0' || c > '9') {
      break;
    }
    value = value * 10 + static_cast<int>(c - '0');
  }
  return value;
}

ArchitectureFamily classify_rocm_family(int arch) {
  if (arch >= 960) {
    return ArchitectureFamily::kFutureAMD;
  }
  if (arch >= 950) {
    return ArchitectureFamily::kCDNA4;
  }
  if (arch >= 940) {
    return ArchitectureFamily::kCDNA3;
  }
  if (arch >= 900) {
    return ArchitectureFamily::kCDNA2;
  }
  return ArchitectureFamily::kUnknown;
}

FeatureSet infer_features(const DeviceProfile& profile) {
  FeatureSet features{};
  features.paged_attention = profile.backend == Backend::kCuda ||
                             profile.backend == Backend::kRocm;

  if (profile.backend == Backend::kCuda) {
    features.tensor_cores = profile.arch >= 75;
    features.bf16 = profile.arch >= 80;

    if (profile.arch >= 100) {
      const bool enabled = profile.runtime_version >= 12080;
      features.fp8 = enabled;
      features.block_fp8 = enabled;
      features.nvfp4 = enabled;
      features.async_tma = true;
      features.shared_memory_multicast = true;
      return features;
    }

    if (profile.arch >= 90) {
      const bool enabled = profile.runtime_version >= 12000;
      features.fp8 = enabled;
      features.block_fp8 = enabled;
      features.async_tma = true;
      features.shared_memory_multicast = true;
      return features;
    }

    if (profile.arch == 89) {
      features.fp8 = profile.runtime_version >= 12040;
    }

    return features;
  }

  if (profile.backend == Backend::kRocm) {
    const bool matrix_core_class =
        profile.family == ArchitectureFamily::kCDNA2 ||
        profile.family == ArchitectureFamily::kCDNA3 ||
        profile.family == ArchitectureFamily::kCDNA4 ||
        profile.family == ArchitectureFamily::kFutureAMD;
    features.tensor_cores = matrix_core_class;
    features.bf16 = matrix_core_class;

    // ROCm enablement remains conservative until backend-specific kernel
    // families are registered in the extension layer.
    features.fp8 = false;
    features.block_fp8 = false;
    features.nvfp4 = false;
  }

  return features;
}

}  // namespace

DeviceProfile query_active_device_profile() {
  int device_index = 0;
  runtime_check(runtime::get_device(&device_index), "get_device");
  return query_device_profile(device_index);
}

DeviceProfile query_device_profile(int device_index) {
  runtime::DeviceProperties properties{};
  runtime_check(runtime::get_device_properties(&properties, device_index),
                "get_device_properties");

  int runtime_version = 0;
  int driver_version = 0;
  runtime_check(runtime::runtime_get_version(&runtime_version),
                "runtime_get_version");
  runtime_check(runtime::driver_get_version(&driver_version),
                "driver_get_version");

  DeviceProfile profile{};
  profile.device_index = device_index;
  profile.runtime_version = runtime_version;
  profile.driver_version = driver_version;
  profile.multiprocessor_count = properties.multiProcessorCount;
  profile.max_threads_per_block = properties.maxThreadsPerBlock;
  profile.shared_memory_per_block = properties.sharedMemPerBlock;
  profile.name = properties.name;

#if defined(NKERNELS_ENABLE_ROCM) || defined(USE_ROCM) || \
    defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  profile.backend = Backend::kRocm;
  profile.vendor = Vendor::kAMD;
  profile.native_arch_name = properties.gcnArchName;
  profile.arch = parse_gfx_arch(profile.native_arch_name);
  profile.family = classify_rocm_family(profile.arch);
  profile.shared_memory_per_multiprocessor = 0;
#else
  profile.backend = Backend::kCuda;
  profile.vendor = Vendor::kNVIDIA;
  profile.major = properties.major;
  profile.minor = properties.minor;
  profile.arch = profile.major * 10 + profile.minor;
  profile.family = classify_cuda_family(profile.arch);
  profile.native_arch_name = "sm" + std::to_string(profile.arch);
  profile.shared_memory_per_multiprocessor =
      properties.sharedMemPerMultiprocessor;
#endif

  profile.features = infer_features(profile);
  return profile;
}

}  // namespace nkernels::hardware_architecture
