#ifndef __CHOREO_CODEGEN_CUDA_TYPES_HPP__
#define __CHOREO_CODEGEN_CUDA_TYPES_HPP__

#include <string>
#include <type_traits>

#include "types.hpp"


using namespace Choreo;

namespace Choreo {

namespace CUDA {

// we need util functions to stringify target related string for codegen
// thus we need new utilities (not STR/PSTR)
std::string stringify(const BaseType& type);
std::string stringify(const FundamentalType& t);
std::string stringify(const Type& ty); // this is abstract type, must use ref
std::string stringify(const Storage& mspec);
std::string stringify(const ValueList &vl);

// safe version for pointers
// we still use stringify name for simplification
template <typename T>
inline static std::string stringify(T* pt) {
  if (!pt) return "invalid";
  return stringify(*pt);
}

template <typename T>
inline static std::string stringify(const ptr<T>& pt) {
  if (!pt) return "invalid";
  return stringify(*pt);
}

} // namespace CUDA

} // namespace Choreo

#endif // __CHOREO_CODEGEN_CUDA_TYPES_HPP__
