#include "codegen_cuda_types.hpp"

using namespace Choreo;

namespace Choreo::CUDA {

namespace __internal__ {

inline static std::string ToString(BaseType dataType) {
  static const std::unordered_map<BaseType, std::string> enumToString = {
      {BaseType::F32, "float"},          {BaseType::F16, "half"},
      {BaseType::BF16, "__nv_bfloat16"}, {BaseType::U32, "uint32_t"},
      {BaseType::S32, "int32_t"},        {BaseType::U16, "uint16_t"},
      {BaseType::S16, "int16_t"},        {BaseType::U8, "uint8_t"},
      {BaseType::S8, "int8_t"},          {BaseType::INT, "int32_t"},
      {BaseType::BOOL, "bool"},
  };

  auto it = enumToString.find(dataType);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

inline static std::string ToString(Storage st) {
  static const std::unordered_map<Storage, std::string> enumToString = {
      {Storage::LOCAL, "__local__"},
      {Storage::GLOBAL, "__device__"},
      {Storage::SHARED, "__shared__"},
  };

  auto it = enumToString.find(st);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

} // end namespace __internal__

std::string stringify(const BaseType& type) {
  return __internal__::ToString(type);
}

std::string stringify(const FundamentalType& type) {
  return __internal__::ToString((BaseType)type);
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

std::string stringify(const Storage& sto) {
  return __internal__::ToString(sto);
}

std::string stringify(const ValueList& vl) {
  std::ostringstream oss;
  oss << "{";
  if (!vl.empty()) {
    oss << PSTR(vl[0]);
    for (unsigned i = 1; i < vl.size(); ++i) oss << ", " << PSTR(vl[i]);
  }
  oss << "}";
  return oss.str();
}

// >> [4096, 4096]
std::string stringify(const Shape& s) {
  std::ostringstream oss;
  s.PrintPlain(oss);
  return oss.str();
}

// >> [4096 * 4096]
std::string size_expr_of(const Shape& s) {
  std::ostringstream oss;
  oss << "[";
  s.PrintSizeExpr(oss);
  oss << "]";
  return oss.str();
}

std::string stringify(const AST::MultiDimSpans& span) {
  std::ostringstream oss;
  cast<AST::MultiValues>(span.list)->InlinePrint(oss);
  return oss.str();
}

} // namespace Choreo::CUDA
