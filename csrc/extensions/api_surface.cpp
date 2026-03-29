#include "extensions/api_surface.hpp"

#include <unordered_set>

namespace nkernels::extensions {

namespace {

std::vector<ApiMethodDescriptor> build_legacy_api_surface() {
  std::vector<ApiMethodDescriptor> methods;
  methods.reserve(192);

#define NKERNELS_API_METHOD(DOMAIN, LEGACY_NAME)                              \
  methods.push_back(ApiMethodDescriptor{                                      \
      std::string(canonical_api_namespace()) + "." + std::string(DOMAIN) + \
          "." + std::string(LEGACY_NAME),                                   \
      LEGACY_NAME, DOMAIN, ApiPortState::kLegacyBacked, true});
#include "extensions/generated_legacy_api_manifest.inc"
#undef NKERNELS_API_METHOD

  return methods;
}

}  // namespace

std::vector<ApiMethodDescriptor> legacy_api_surface() {
  return build_legacy_api_surface();
}

ApiCoverageSummary summarize_legacy_api_surface() {
  ApiCoverageSummary summary{};
  std::unordered_set<std::string> unique_legacy_names;
  std::unordered_set<std::string> unique_canonical_names;
  bool every_method_guarded = true;

  for (const auto& method : build_legacy_api_surface()) {
    ++summary.total_methods;
    unique_legacy_names.insert(method.legacy_name);
    unique_canonical_names.insert(method.canonical_name);
    summary.structured_methods +=
        method.state == ApiPortState::kStructuredWrapper;
    summary.native_nkernels_methods +=
        method.state == ApiPortState::kNativeNkernels;
    every_method_guarded = every_method_guarded && method.zero_data_loss_guard;
  }

  summary.unique_legacy_methods = unique_legacy_names.size();
  summary.unique_canonical_methods = unique_canonical_names.size();
  summary.duplicate_legacy_names =
      summary.unique_legacy_methods != summary.total_methods;
  summary.duplicate_canonical_names =
      summary.unique_canonical_methods != summary.total_methods;
  summary.zero_method_loss =
      summary.total_methods > 0 && every_method_guarded &&
      !summary.duplicate_canonical_names;
  return summary;
}

std::optional<ApiMethodDescriptor> find_legacy_method(std::string_view legacy_name,
                                                      std::string_view domain) {
  for (const auto& method : build_legacy_api_surface()) {
    if (method.legacy_name == legacy_name &&
        (domain.empty() || method.domain == domain)) {
      return method;
    }
  }
  return std::nullopt;
}

}  // namespace nkernels::extensions
