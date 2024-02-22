#ifndef __CHOREO_SYMTAB_H__
#define __CHOREO_SYMTAB_H__

#include <string>
#include <unordered_map>

namespace AST {

// For types like f32, f16, etc.
enum class BaseType { F32, F16, BF16, U32, S32, U16, S16, U8, S8, INT };

#if 0
enum class SymbolType {
  MDSPANS,
  MINDS,
  F32,
  F16,
  BF16,
  U16,
  S16,
  U8,
  S8,
  U32,
  S32,
  INT,
};
#endif

class Symbol {
 public:
  std::string name;  // The identifier's name
  BaseType type;     // The identifier's type
  bool aggregate = false;

  // Constructor
  Symbol(const std::string& n, BaseType t, bool a = false) : name(n), type(t), aggregate(a) {}
  Symbol() {}

  BaseType getType() const { return type; }
  bool isAggregate() const { return aggregate; }
};

inline static BaseType getTypeFromString(const std::string& input) {
  static const std::unordered_map<std::string, BaseType> typeMap = {
      {"f32", BaseType::F32}, {"f16", BaseType::F16}, {"bf16", BaseType::BF16},
      {"u32", BaseType::U32}, {"s32", BaseType::S32}, {"u16", BaseType::U16},
      {"s16", BaseType::S16}, {"u8", BaseType::U8},   {"s8", BaseType::S8},
      {"int", BaseType::INT}};

  auto it = typeMap.find(input);
  if (it != typeMap.end()) return it->second;

  assert(0 && "incorrect type string");
}

inline static std::string getStringFrom(BaseType dataType) {
  static const std::unordered_map<BaseType, std::string> enumToString = {
      {BaseType::F32, "f32"}, {BaseType::F16, "f16"}, {BaseType::BF16, "bf16"},
      {BaseType::U32, "u32"}, {BaseType::S32, "s32"}, {BaseType::U16, "u16"},
      {BaseType::S16, "s16"}, {BaseType::U8, "u8"},   {BaseType::S8, "s8"},
      {BaseType::INT, "int"}};

  auto it = enumToString.find(dataType);
  if (it != enumToString.end()) return it->second;

  assert(0 && "unsupported type.");
}

class SymbolTable {
 private:
  std::unordered_map<std::string, Symbol> table;

 public:
  // Add a symbol to the symbol table
  void addSymbol(const std::string& name, BaseType type, bool aggr = false) {
    table[name] = Symbol(name, type, aggr);
  }

  // Retrieve a symbol from the symbol table
  Symbol* getSymbol(const std::string& name) {
    if (table.find(name) != table.end()) {
      return &table[name];
    }
    return nullptr;
  }

  // Check if a symbol with the given name exists in the symbol table
  bool exists(const std::string& name) {
    return table.find(name) != table.end();
  }

 public:
  static std::string getAnonName() {
    return "anon_" + std::to_string(anonymous_count++);
  }
  static int anonymous_count;
};

}  // end of namespace AST

#endif  // __CHOREO_SYMTAB_H__
