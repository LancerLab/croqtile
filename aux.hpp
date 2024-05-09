#ifndef __CHOREO_AUX_HPP__
#define __CHOREO_AUX_HPP__

#include <cassert>
#include <iostream>
#include <sstream>

#if 0
[[noreturn]] inline void choreo_unreachable_impl(
    const char* file, int line, const char* msg = "Unreachable code reached") {
  std::cerr << file << ":" << line << ": Assertion failed: ";
  std::cerr << msg << std::endl;
  std::abort();
}
#endif

[[noreturn]] inline void choreo_unreachable_impl(
    const char* file, int line,
    const std::string& msg = "Unreachable code reached") {
  std::cerr << file << ":" << line << ": Assertion failed: ";
  std::cerr << msg << std::endl;
  std::abort();
}

// Note: __VA_OPT__ requires C++20. But it is avaliable starting from gcc-8.1
// and clang-6. This pre-requisition should be satisfied.

#if defined(__GNUC__) && !defined(__clang__)
    #if (__GNUC__ < 8) || (__GNUC__ == 8 && __GNUC_MINOR__ < 1)
        #error "GCC version must be at least 8.1"
    #endif
#endif

#if defined(__clang__)
    #if (__clang_major__ < 6)
        #error "Clang version must be at least 6"
    #endif
#endif

// Macro that captures the file and line
#define choreo_unreachable(...) \
  choreo_unreachable_impl(__FILE__, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

template <typename T>
inline std::string DelimitedString(const std::vector<T>& v) {
  std::ostringstream iss;
  if (v.size() > 0) {
    iss << v[0];
    for (size_t i = 1; i < v.size(); ++i) {
      if constexpr (std::is_same_v<T, std::string>)
        iss << ", " << v[i];
      else
        iss << ", " << std::to_string(v[i]);
    }
  }
  return iss.str();
}

#endif  // __CHOREO_AUX_HPP__
