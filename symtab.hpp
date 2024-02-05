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
  Symbol(const std::string& n, BaseType t) : name(n), type(t) {}
  Symbol() {}

  BaseType getType() const { return type; }
  bool isAggregate() const { return aggregate; }
};

class SymbolTable {
 private:
  std::unordered_map<std::string, Symbol> table;

 public:
  // Add a symbol to the symbol table
  void addSymbol(const std::string& name, BaseType type) {
    table[name] = Symbol(name, type);
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
