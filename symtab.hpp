#ifndef __CHOREO_SYMTAB_H__
#define __CHOREO_SYMTAB_H__

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace Choreo {

class Symbol {
 public:
  std::string name;  // The identifier's name
  ptr<Type> type;    // type associated

  // Constructor
  explicit Symbol(const std::string& n, const ptr<Type>& t) : name(n), type(t) {
    assert(t && "Invalid type.");
  }
  // Symbol() {}

  const ptr<Type>& GetType() const { return type; }
  TypeCategory GetTypeCategory() const { return type->Category(); }
  bool IsComposite() const {
    return GetTypeCategory() == TypeCategory::SPANNED;
  }
};

class SymbolTable {
 private:
  std::unordered_map<std::string, Symbol> table;

 public:
  // Add a symbol to the symbol table
  void AddSymbol(const std::string& name, const ptr<Type>& ty) {
    table.emplace(name, Symbol{name, ty});
  }

  // Retrieve a symbol from the symbol table
  Symbol* GetSymbol(const std::string& name) {
    if (table.find(name) != table.end()) return &table.at(name);

    return nullptr;
  }

  // Check if a symbol with the given name exists in the symbol table
  bool Exists(const std::string& name) {
    return table.find(name) != table.end();
  }

  void Reset() { table.clear(); }

 public:
  static std::string GetAnonName() {
    return "anon_" + std::to_string(anonymous_count++);
  }
  static std::string GetAnonTypeName() {
    return "anon_t_" + std::to_string(anon_type_count++);
  }

  static unsigned anonymous_count;
  static unsigned anon_type_count;
};

}  // end of namespace Choreo

#endif  // __CHOREO_SYMTAB_H__
