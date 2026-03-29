#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace nkernels::core {

class Error final : public std::runtime_error {
 public:
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

[[noreturn]] inline void fail(std::string_view component,
                              std::string_view message) {
  throw Error(std::string(component) + ": " + std::string(message));
}

inline void check(bool condition, std::string_view component,
                  std::string_view message) {
  if (!condition) {
    fail(component, message);
  }
}

}  // namespace nkernels::core
