#include "factor_codegen_types.hpp"

using namespace Choreo;

namespace Choreo::Factor {

namespace __internal__ {

inline static std::string ToString(BaseType dataType) {
  static const std::unordered_map<BaseType, std::string> enumToString = {
      {BaseType::F32, "FloatType(32)"}, {BaseType::F16, "FloatType(16)"},
      {BaseType::BF16, "BFloatType()"}, {BaseType::U32, "IntType(32)"},
      {BaseType::S32, "IntType(32)"},   {BaseType::U16, "IntType(16)"},
      {BaseType::S16, "IntType(16)"},   {BaseType::U8, "IntType(8)"},
      {BaseType::S8, "IntType(8)"},     {BaseType::BOOL, "BoolType(32)"},
  };

  auto it = enumToString.find(dataType);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

inline static std::string ToString(Storage st) {
  static const std::unordered_map<Storage, std::string> enumToString = {
      {Storage::LOCAL, "L1Type"},
      {Storage::GLOBAL, "DRAMType"},
      {Storage::SHARED, "SRAMType"},
  };

  auto it = enumToString.find(st);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

} // end namespace __internal__

std::string stringify(const BaseType& t) { return __internal__::ToString(t); }

std::string stringify(const Type& ty) {
  if (isa<VoidType>(&ty)) return "void";
  if (isa<IntegerType>(&ty))
    return "IntType(32)";
  else if (isa<BooleanType>(&ty))
    return "BoolType(32)";
  else if (isa<BoundedIntegerType>(&ty))
    return "IntType(32)";
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return stringify(t->e_type);
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

} // namespace Choreo::Factor
