#ifndef __CHOREO_SYMTAB_H__
#define __CHOREO_SYMTAB_H__

#include <string>
#include <unordered_map>
#include <vector>

namespace AST {

// BaseType, FundamentalType, and ScalarType
enum class BaseType { F32, F16, BF16, U32, S32, U16, S16, U8, S8, INT, BOOL, ITUPLE };

enum class FundamentalType {
  F32  = (int)BaseType::F32,
  F16  = (int)BaseType::F16,
  BF16 = (int)BaseType::BF16,
  U32  = (int)BaseType::U32,
  U16  = (int)BaseType::U16,
  U8   = (int)BaseType::U8,
  S32  = (int)BaseType::S32,
  S16  = (int)BaseType::S16,
  S8   = (int)BaseType::S8,
};

enum class ScalarType {
  INT  = (int)BaseType::INT,
  BOOL = (int)BaseType::BOOL,
};

enum class Storage { LOCAL, SHARED, GLOBAL, DEFAULT, NONE };


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

struct SpanType {
  std::vector<int> values;

  SpanType(std::initializer_list<int> init) {
    for (auto itr = init.begin(); itr != init.end(); ++itr)
      values.push_back(*itr);
  }

  SpanType() {}
};

class Symbol {
 public:
  std::string name;  // The identifier's name
  BaseType type;     // The identifier's base type
  SpanType s_type;   // The identifier's span type
  bool spanned = false;

  // Constructor
  Symbol(const std::string& n, BaseType t)
      : name(n), type(t), s_type(), spanned(false) {}
  Symbol(const std::string& n, BaseType t, SpanType st)
      : name(n), type(t), s_type(st), spanned(true) {}
  Symbol() {}

  BaseType baseType() const { return type; }
  bool isSpanned() const { return spanned; }
};

inline static BaseType getTypeFromString(const std::string& input) {
  static const std::unordered_map<std::string, BaseType> typeMap = {
      {"f32", BaseType::F32}, {"f16", BaseType::F16},  {"bf16", BaseType::BF16},
      {"u32", BaseType::U32}, {"s32", BaseType::S32},  {"u16", BaseType::U16},
      {"s16", BaseType::S16}, {"u8", BaseType::U8},    {"s8", BaseType::S8},
      {"int", BaseType::INT}, {"bool", BaseType::BOOL}, {"ituple", BaseType::ITUPLE},
  };

  auto it = typeMap.find(input);
  if (it != typeMap.end()) return it->second;

  assert(0 && "incorrect type string");
}

inline static std::string getStringFrom(BaseType dataType) {
  static const std::unordered_map<BaseType, std::string> enumToString = {
      {BaseType::F32, "f32"}, {BaseType::F16, "f16"},  {BaseType::BF16, "bf16"},
      {BaseType::U32, "u32"}, {BaseType::S32, "s32"},  {BaseType::U16, "u16"},
      {BaseType::S16, "s16"}, {BaseType::U8, "u8"},    {BaseType::S8, "s8"},
      {BaseType::INT, "int"}, {BaseType::BOOL, "bool"}, {BaseType::ITUPLE, "ituple"}
  };

  auto it = enumToString.find(dataType);
  if (it != enumToString.end()) return it->second;

  assert(0 && "unsupported type.");
}

struct PartialType {
  std::string name;  // The type's name
                     // anything more?
  PartialType(const std::string& n) : name(n) {}
  PartialType() : name() {}
};

class PartialTypeTable {
 private:
  std::unordered_map<std::string, PartialType> table;

 public:
  // Add a partial type to the partial type table
  void addType(const std::string& name) {
    table.emplace(name, name);
  }

  // Retrieve a partial type from the table
  PartialType * getType(const std::string& name) {
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

class SymbolTable {
 private:
  std::unordered_map<std::string, Symbol> table;

 public:
  // Add a symbol to the symbol table
  void addSymbol(const std::string& name, BaseType type) {
    table[name] = Symbol(name, type);
  }

  void addSymbol(const std::string& name, BaseType type, SpanType s_type) {
    table[name] = Symbol(name, type, s_type);
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
