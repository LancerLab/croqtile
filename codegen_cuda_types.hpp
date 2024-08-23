#ifndef CODEGEN_TYPES_H_
#define CODEGEN_TYPES_H_

#include <string>
#include <type_traits>

#include "types.hpp"


using namespace Choreo;

namespace Choreo {

namespace CUDA {
// enums and types declare
//
enum class CudaType {
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
enum class CudaMemSpec {
  GLOBAL = (int)Choreo::Storage::GLOBAL,
  SHARED = (int)Choreo::Storage::SHARED,
  LOCAL = (int)Choreo::Storage::LOCAL,
};

// utils funcs for stringfify cudatypes
std::string stringify(const CudaType& type);
std::string stringify(const BaseType& type);
std::string stringify(const FundamentalType& t);
std::string stringify(const Type& ty); // this is abstract type, must use ref

std::string stringify(const CudaMemSpec& mspec);
std::string stringify(const Storage& mspec);

} // namespace CUDA

} // namespace Choreo

#endif // CODEGEN_TYPES_H_
