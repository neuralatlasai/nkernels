#include "extensions/registry.hpp"

#include <mutex>
#include <utility>

namespace nkernels::extensions {

namespace {

std::mutex& scaled_gemm_registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<ScaledGemmExtension>& scaled_gemm_registry() {
  static std::vector<ScaledGemmExtension> registry;
  return registry;
}

}  // namespace

void register_scaled_gemm_extension(ScaledGemmExtension extension) {
  std::lock_guard<std::mutex> lock(scaled_gemm_registry_mutex());
  scaled_gemm_registry().push_back(std::move(extension));
}

std::vector<ScaledGemmExtension> scaled_gemm_extensions() {
  std::lock_guard<std::mutex> lock(scaled_gemm_registry_mutex());
  return scaled_gemm_registry();
}

ScaledGemmExtensionRegistrar::ScaledGemmExtensionRegistrar(
    ScaledGemmExtension extension) {
  register_scaled_gemm_extension(std::move(extension));
}

}  // namespace nkernels::extensions
