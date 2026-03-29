#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nkernels::extensions {

enum class ApiPortState : unsigned char {
  kLegacyBacked = 0,
  kStructuredWrapper,
  kNativeNkernels,
};

struct ApiMethodDescriptor {
  std::string canonical_name;
  std::string legacy_name;
  std::string domain;
  ApiPortState state = ApiPortState::kLegacyBacked;
  bool zero_data_loss_guard = true;
};

struct ApiCoverageSummary {
  std::size_t total_methods = 0;
  std::size_t unique_legacy_methods = 0;
  std::size_t unique_canonical_methods = 0;
  std::size_t structured_methods = 0;
  std::size_t native_nkernels_methods = 0;
  bool zero_method_loss = false;
  bool duplicate_legacy_names = false;
  bool duplicate_canonical_names = false;
};

constexpr const char* canonical_api_namespace() { return "nkernels"; }

constexpr const char* to_string(ApiPortState state) {
  switch (state) {
    case ApiPortState::kLegacyBacked:
      return "legacy_backed";
    case ApiPortState::kStructuredWrapper:
      return "structured_wrapper";
    case ApiPortState::kNativeNkernels:
      return "native_nkernels";
    default:
      return "legacy_backed";
  }
}

std::vector<ApiMethodDescriptor> legacy_api_surface();
ApiCoverageSummary summarize_legacy_api_surface();
std::optional<ApiMethodDescriptor> find_legacy_method(
    std::string_view legacy_name, std::string_view domain = {});

}  // namespace nkernels::extensions
