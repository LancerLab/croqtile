#ifndef __CHOREO_AUX_HPP__
#define __CHOREO_AUX_HPP__

#include <cassert>
#include <iostream>
#include <sstream>

[[noreturn]] inline void choreo_unreachable(
    const char* msg = "Unreachable code reached", const char* file = __FILE__,
    int line = __LINE__) {
  std::cerr << "Assertion failed: " << msg << ", file " << file << ", line "
            << line << std::endl;
  std::abort();
}

[[noreturn]] inline void choreo_unreachable(const std::string& msg,
                                            const char* file = __FILE__,
                                            int line = __LINE__) {
  std::cerr << "Assertion failed: " << msg << ", file " << file << ", line "
            << line << std::endl;
  std::abort();
}

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
