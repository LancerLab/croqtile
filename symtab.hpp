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
  void addType(const std::string& name) { table.emplace(name, name); }

  // Retrieve a partial type from the table
  PartialType* getType(const std::string& name) {
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

 public:
  static std::string getAnonName() {
    return "anon_" + std::to_string(anonymous_count++);
  }
  static int anonymous_count;
};

}  // end of namespace Choreo

#endif  // __CHOREO_SYMTAB_H__
