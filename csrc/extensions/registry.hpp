#pragma once

#include <string>
#include <vector>

#include "hardware_architecture/device_profile.hpp"

namespace nkernels::architecture::compound {
struct ScaledGemmDispatchResult;
struct ScaledGemmRequest;
}  // namespace nkernels::architecture::compound

namespace nkernels::extensions {

using ScaledGemmSupportFn = bool (*)(
    const hardware_architecture::DeviceProfile&,
    const architecture::compound::ScaledGemmRequest&,
    architecture::compound::ScaledGemmDispatchResult*);

struct ScaledGemmExtension {
  std::string symbol;
  ScaledGemmSupportFn support = nullptr;
};

void register_scaled_gemm_extension(ScaledGemmExtension extension);
std::vector<ScaledGemmExtension> scaled_gemm_extensions();

class ScaledGemmExtensionRegistrar {
 public:
  explicit ScaledGemmExtensionRegistrar(ScaledGemmExtension extension);
};

}  // namespace nkernels::extensions

#define NKERNELS_CONCAT_INNER(A, B) A##B
#define NKERNELS_CONCAT(A, B) NKERNELS_CONCAT_INNER(A, B)

#define NKERNELS_REGISTER_SCALED_GEMM_EXTENSION(SYMBOL, FN)               \
  static ::nkernels::extensions::ScaledGemmExtensionRegistrar             \
      NKERNELS_CONCAT(nkernels_scaled_gemm_extension_registrar_, __LINE__)( \
          ::nkernels::extensions::ScaledGemmExtension{SYMBOL, FN})
