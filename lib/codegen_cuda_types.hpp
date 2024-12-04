#ifndef CHOREO_CODEGEN_CUDA_TYPES_HPP_
#define CHOREO_CODEGEN_CUDA_TYPES_HPP_

#include <string>
#include <type_traits>

#include "ast.hpp"
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
std::string stringify(const ValueList& vl);
std::string stringify(const Shape& sp);

// utils for stringify AST NODES
std::string stringify(const AST::MultiDimSpans& sp);

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

template <>
std::string stringify(const ptr<AST::Node>& pt) {
  if (auto pt_new = cast<AST::MultiDimSpans>(pt)) return stringify(*pt_new);
  return "invalid";
}

std::string size_expr_of(const Shape& sp);

} // namespace CUDA

} // namespace Choreo

#endif // CHOREO_CODEGEN_CUDA_TYPES_HPP_
