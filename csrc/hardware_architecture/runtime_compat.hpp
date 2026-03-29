#pragma once

#if defined(NKERNELS_ENABLE_ROCM) || defined(USE_ROCM) || \
    defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>

namespace nkernels::hardware_architecture::runtime {

using Error = hipError_t;
using DeviceProperties = hipDeviceProp_t;

constexpr Error kSuccess = hipSuccess;
constexpr bool kIsRocm = true;

inline const char* get_error_string(Error error) {
  return hipGetErrorString(error);
}

inline Error get_device(int* device) { return hipGetDevice(device); }

inline Error get_device_properties(DeviceProperties* properties, int device) {
  return hipGetDeviceProperties(properties, device);
}

inline Error runtime_get_version(int* version) {
  return hipRuntimeGetVersion(version);
}

inline Error driver_get_version(int* version) {
  return hipRuntimeGetVersion(version);
}

}  // namespace nkernels::hardware_architecture::runtime

#else

#include <cuda_runtime.h>

namespace nkernels::hardware_architecture::runtime {

using Error = cudaError_t;
using DeviceProperties = cudaDeviceProp;

constexpr Error kSuccess = cudaSuccess;
constexpr bool kIsRocm = false;

inline const char* get_error_string(Error error) {
  return cudaGetErrorString(error);
}

inline Error get_device(int* device) { return cudaGetDevice(device); }

inline Error get_device_properties(DeviceProperties* properties, int device) {
  return cudaGetDeviceProperties(properties, device);
}

inline Error runtime_get_version(int* version) {
  return cudaRuntimeGetVersion(version);
}

inline Error driver_get_version(int* version) {
  return cudaDriverGetVersion(version);
}

}  // namespace nkernels::hardware_architecture::runtime

#endif
