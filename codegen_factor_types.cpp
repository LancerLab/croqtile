#include "codegen_factor_types.hpp"

using namespace Choreo;

namespace Choreo::Factor {

std::string stringify(const FactorType& type) {
  switch (type) {
    case FactorType::I32: 
      return "int32_t";
      break;
    case FactorType::U32: 
      return "uint32_t";
      break;
    case FactorType::I16: 
      return "int16_t";
      break;
    case FactorType::U16: 
      return "uint16_t";
      break;
    case FactorType::I8: 
      return "int8_t";
      break;
    case FactorType::U8: 
      return "uint8_t";
      break;
    case FactorType::F32: 
      return "float";
      break;
    case FactorType::F16: 
      return "half";
      break;
    case FactorType::BF16: 
      return "__nv_bfloat16";
      break;
    case FactorType::BOOL:
      return "bool";
      break;
    case FactorType::F64: 
      return "double";
      break;
    case FactorType::VOID: 
      return "void";
      break;
    default:
      choreo_unreachable("This FactorType is not supported.");
  }
}

FactorType fromBaseType(BaseType type) {
  switch (type) {
    case BaseType::F32:
      return FactorType::F32;
      break;
    case BaseType::F16:
      return FactorType::F16;
      break;
    case BaseType::BF16:
      return FactorType::BF16;
      break;
    case BaseType::U32:
      return FactorType::U32;
      break;
    case BaseType::S32:
      return FactorType::I32;
      break;
    case BaseType::U16:
      return FactorType::U16;
      break;
    case BaseType::S16:
      return FactorType::I16;
      break;
    case BaseType::U8:
      return FactorType::U8;
      break;
    case BaseType::S8:
      return FactorType::I8;
      break;
    // should it be passed in?
    case BaseType::INT:
      return FactorType::I32;
      break;
    case BaseType::BOOL:
      return FactorType::BOOL;
      break;
    default:
      choreo_unreachable("BaseType '" + STR(type) + "' is not supported.");
  }
}

std::string stringify(const BaseType& type) {
  switch (type) {
    case BaseType::F32:
      return "FloatType(32)";
      break;
    case BaseType::F16:
      return "FloatType(16)";
      break;
    case BaseType::BF16:
      return "BFloatType(16)";
      break;
    case BaseType::U32:
    case BaseType::S32:
      return "IntType(32)";
      break;
    case BaseType::U16:
    case BaseType::S16:
      return "IntType(16)";
      break;
    case BaseType::U8:
    case BaseType::S8:
      return "IntType(8)";
      break;
    // should it be passed in?
    case BaseType::INT:
      return "IntType(32)";
      break;
    case BaseType::BOOL:
      return "BoolType(32)";
      break;
    default:
      choreo_unreachable("Type '" + STR(type) + "' is not supported.");
  }
}

std::string stringify(const FundamentalType& t) {
  switch (t) {
    case FundamentalType::F32:
      return "FloatType(32)";
      break;
    case FundamentalType::F16:
      return "FloatType(16)";
      break;
    case FundamentalType::BF16:
      return "BFloatType(16)";
      break;
    case FundamentalType::U32:
    case FundamentalType::S32:
      return "IntType(32)";
      break;
    case FundamentalType::U16:
    case FundamentalType::S16:
      return "IntType(16)";
      break;
    case FundamentalType::U8:
    case FundamentalType::S8:
      return "IntType(8)";
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

  
std::string stringify(const FactorMemSpec& mspec) {
  switch (mspec) {
    case FactorMemSpec::GLOBAL: return "DRAMType";
    case FactorMemSpec::SHARED: return "SRAMType";
    case FactorMemSpec::LOCAL: return "L1Type";
    default:
      choreo_unreachable();
  }
}

std::string stringify(const Storage& sto) {
  auto mspec = static_cast<FactorMemSpec>(static_cast<int>(sto));
  return stringify(mspec);
}

std::string stringify(const ValueList &vl) {
  std::ostringstream oss;
  auto print_variant = [&oss](const ValueItem &vle) {
    if (vle.index() == 0)
      oss << std::get<0>(vle);
    else
      oss << std::get<1>(vle);
  };
  oss << "{";
  if (!vl.empty()) {
    print_variant(vl[0]);
    for (unsigned i = 1; i < vl.size(); ++i) {
      oss << ", ";
      print_variant(vl[i]);
    }
  }
  oss << "}";
  return oss.str();
}

} // namespace Choreo::Factor

