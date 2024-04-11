#ifndef __CHOREO_AUX_HPP__
#define __CHOREO_AUX_HPP__

#include <cassert>
#include <iostream>

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

#endif  // __CHOREO_AUX_HPP__
