#include "codegen_cuda_types.hpp"

using namespace Choreo;

namespace Choreo::CUDA {

std::string stringify(const CudaType& type) {
  switch (type) {
    case CudaType::I32: 
      return "int32_t";
      break;
    case CudaType::U32: 
      return "uint32_t";
      break;
    case CudaType::I16: 
      return "int16_t";
      break;
    case CudaType::U16: 
      return "uint16_t";
      break;
    case CudaType::I8: 
      return "int8_t";
      break;
    case CudaType::U8: 
      return "uint8_t";
      break;
    case CudaType::F32: 
      return "float";
      break;
    case CudaType::F16: 
      return "half";
      break;
    case CudaType::BF16: 
      return "__nv_bfloat16";
      break;
    case CudaType::BOOL:
      return "bool";
      break;
    case CudaType::F64: 
      return "double";
      break;
    case CudaType::VOID: 
      return "void";
      break;
    default:
      choreo_unreachable("This CudaType is not supported.");
  }
}

CudaType fromBaseType(BaseType type) {
  switch (type) {
    case BaseType::F32:
      return CudaType::F32;
      break;
    case BaseType::F16:
      return CudaType::F16;
      break;
    case BaseType::BF16:
      return CudaType::BF16;
      break;
    case BaseType::U32:
      return CudaType::U32;
      break;
    case BaseType::S32:
      return CudaType::I32;
      break;
    case BaseType::U16:
      return CudaType::U16;
      break;
    case BaseType::S16:
      return CudaType::I16;
      break;
    case BaseType::U8:
      return CudaType::U8;
      break;
    case BaseType::S8:
      return CudaType::I8;
      break;
    // should it be passed in?
    case BaseType::INT:
      return CudaType::I32;
      break;
    case BaseType::BOOL:
      return CudaType::BOOL;
      break;
    default:
      choreo_unreachable("BaseType '" + STR(type) + "' is not supported.");
  }
}

std::string stringify(const BaseType& type) {
  CudaType cudatype = fromBaseType(type);
  return stringify(cudatype);
}

std::string stringify(const FundamentalType& t) {
  switch (t) {
    case FundamentalType::F32:
      return "float";
      break;
    case FundamentalType::F16:
      return "half";
      break;
    case FundamentalType::BF16:
      return "__nv_bfloat16";
      break;
    case FundamentalType::U32:
      return "uint32_t";
      break;
    case FundamentalType::S32:
      return "int32_t";
      break;
    case FundamentalType::U16:
      return "uint16_t";
      break;
    case FundamentalType::S16:
      return "int16_t";
      break;
    case FundamentalType::U8:
      return "uint8_t";
      break;
    case FundamentalType::S8:
      return "int8_t";
      break;
    default:
      choreo_unreachable("Type '" + STR(t) + "' is not supported.");
  }
}

std::string stringify(const Type& ty) {
  if (isa<VoidType>(&ty)) return "void";
  if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (isa<BoundedIntegerType>(&ty))
    return "int";
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return stringify(t->f_type);
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
  return 0;
}

  
std::string stringify(const CudaMemSpec& mspec) {
  switch (mspec) {
    case CudaMemSpec::GLOBAL: return "__device__";
    case CudaMemSpec::SHARED: return "__shared__";
    case CudaMemSpec::LOCAL: return "";
  }
  assert(false && "Unknown memlevel\n");
  return "";
}

std::string stringify(const Storage& sto) {
  auto mspec = static_cast<CudaMemSpec>(static_cast<int>(sto));
  return stringify(mspec);
}

} // namespace Choreo::CUDA

