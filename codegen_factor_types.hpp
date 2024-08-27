#ifndef CODEGEN_FACTOR_TYPES_H_
#define CODEGEN_FACTOR_TYPES_H_

#include <string>
#include <type_traits>

#include "types.hpp"


using namespace Choreo;

namespace Choreo {

namespace Factor {
// enums and types declare
//
enum class FactorType {
  I32,
  U32,
  I16,
  U16,
  I8,
  U8,
  F32,
  F16,
  BF16,
  F64,
  BOOL,
  VOID,
};

// Memory specifier for CUDA
enum class FactorMemSpec {
  GLOBAL = (int)Choreo::Storage::GLOBAL,
  SHARED = (int)Choreo::Storage::SHARED,
  LOCAL = (int)Choreo::Storage::LOCAL,
};

// utils funcs for stringfify cudatypes
std::string stringify(const FactorType& type);
std::string stringify(const BaseType& type);
std::string stringify(const FundamentalType& t);
std::string stringify(const Type& ty); // this is abstract type, must use ref
std::string stringify(const FactorMemSpec& mspec);
std::string stringify(const Storage& mspec);
std::string stringify(const ValueList &vl);

} // namespace Factor 

} // namespace Choreo

#endif // CODEGEN_FACTOR_TYPES_H_
