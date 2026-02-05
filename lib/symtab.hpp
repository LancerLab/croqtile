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
  std::string name; // The identifier's name
  ptr<Type> type;   // type associated

  // Constructor
  explicit Symbol(const std::string& n, const ptr<Type>& t) : name(n), type(t) {
    assert(t && "Invalid type.");
  }
  // Symbol() {}

  const ptr<Type>& GetType() const { return type; }
  void SetType(const ptr<Type>& ty) { type = ty; }
  BaseType GetBaseType() const { return type->GetBaseType(); }
  bool IsComposite() const { return GetBaseType() == BaseType::SPANNED; }
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
  static std::string GetAnonPBName() {
    return "anon_pb_" + std::to_string(anon_pb_count++);
  }

  static unsigned anonymous_count;
  static unsigned anon_type_count;
  static unsigned anon_pb_count;

  void Print(std::ostream& os) const {
    for (auto item : table)
      os << "symbol: " << item.first
         << ", type: " << STR(*item.second.GetType()) << "\n";
  }

  void Dump() const { Print(dbgs()); }
};

// TODO(albert): remove this when other utils ready
// currently, we need this for working-around on emit referred ids/types/vars
// however, some objects are not printable in Factor form.
class StringifyTable {
private:
  std::unordered_map<std::string, std::string> type_string_table;
  std::unordered_map<std::string, std::string> type_sym_table;
  std::vector<std::string> syms;

public:
  // Add a symbol to the symbol table
  // emittable = 'a'
  // type_symbol = 'a_type'
  // emitted = 'DRAMType(FloatType(32), {1, 2})'
  void AddSymbol(const std::string& emittable, const std::string& symname,
                 const std::string& emitted) {
    type_string_table.emplace(emittable, emitted);
    type_sym_table.emplace(emittable, symname);
    syms.emplace_back(emittable);
  }

  // Retrieve typename of a symbol
  std::string GetTypeName(const std::string& emittable) {
    if (type_sym_table.find(emittable) != type_sym_table.end())
      return type_sym_table.at(emittable);
    return "";
  }

  // Retrieve a symbol from the symbol table
  std::string GetTypeString(const std::string& emittable) {
    if (type_string_table.find(emittable) != type_string_table.end())
      return type_string_table.at(emittable);
    return "";
  }

  int GetSymbolIndex(const std::string& emittable) {
    /*
    TODO: hardcode now
    if `emittable` == output, should not replace it with args[]
    eg. f = dma.copy.async output.chunkat(p, y, x) => local;
    should generate
      async_load_(f, output, f_buffer, {...});
    rather than
      async_load_(f, args[...], f_buffer, {...});
    Examples of practical applications:
      d = dma.copy.async l1_out => output.span_as([4, 3, 64]).chunkat(a,b,c);
    */
    if (emittable == "output") return -1;
    int syms_num = syms.size();
    for (int idx = 0; idx < syms_num; idx++) {
      if (emittable == syms[idx]) return idx;
    }
    return -1;
  }

  // Check if a symbol with the given name exists in the symbol table
  bool Exists(const std::string& emittable) {
    return type_sym_table.find(emittable) != type_sym_table.end();
  }

  void Reset() {
    type_sym_table.clear();
    syms.clear();
  }

  void Print(std::ostream& os) {
    os << "\n";
    for (auto symbol : syms)
      os << "symbol: " << symbol
         << ", name: " << type_sym_table.find(symbol)->second
         << ", type: " << type_string_table.find(symbol)->second << "\n";
  }
};

// This is the scoped symbol table
class ScopedSymbolTable {
  std::vector<std::unordered_map<std::string, ptr<Type>>>
      scoped_symtab;                    // k: symbol name, v: type
  std::vector<std::string> scope_names; // k: scope-index, v: scope-name

  // global symbol table: set it when required
  ptr<SymbolTable> symtab = nullptr;

public:
  ScopedSymbolTable(const ptr<SymbolTable>& s_tab) : symtab(s_tab) {
    if (symtab == nullptr) symtab = std::make_shared<SymbolTable>();
  }

  ~ScopedSymbolTable() {}

  // produce the global symbol table
  const ptr<SymbolTable>& GlobalSymbolTable() const { return symtab; }

  void UpdateGlobal(const ptr<SymbolTable>& s_tab) { symtab = s_tab; }
  size_t ScopeDepth() const { return scoped_symtab.size(); }

  void EnterScope(const std::string& name = "") {
    scoped_symtab.emplace_back(); // Push a new scope
    scope_names.emplace_back(name);
  }

  void LeaveScope() {
    if (!scoped_symtab.empty()) {
      scoped_symtab.pop_back(); // Pop the last scope
      scope_names.pop_back();
    }
  }

  // declare in current scope
  bool DeclaredInScope(const std::string& sym_name) const {
    return scoped_symtab.back().count(sym_name) != 0;
  }

  bool IsDeclared(const std::string& sym_name) const {
    // Iterate in reverse order to simulate stack behavior
    for (auto it = scoped_symtab.rbegin(); it != scoped_symtab.rend(); ++it) {
      if (it->count(sym_name))
        return true; // Found sym_name in the current or an enclosing scope
    }
    return false; // sym_name not found in any scope
  }

  bool DefineSymbol(const std::string& n, const ptr<Type> ty) {
    if (scoped_symtab.empty()) {
      choreo_unreachable("internal error: symtab is empty (@ insertion of `" +
                         n + "').");
      return false;
    }

    if (scoped_symtab.back().count(n) == 0) {
      // Insert into the current (top) scope and global symtab
      scoped_symtab.back().emplace(n, ty);
      // synchronize the record inside global symtab
      if (symtab) symtab->AddSymbol(InScopeName(n), ty);

      if (CCtx().DebugSymTab())
        dbgs() << "Add Symbol: " << InScopeName(n) << ", type: " << PSTR(ty)
               << "\n";

      return true;
    }

    choreo_unreachable("Symbol `" + n + "' has been defined.");
    return false;
  }

  const ptr<Type> LookupSymbol(const std::string& n) const {
    for (auto it = scoped_symtab.rbegin(); it != scoped_symtab.rend(); ++it) {
      if (it->count(n)) return it->at(n);
    }
    return nullptr;
  }

  bool ModifySymbolType(const std::string& n, const ptr<Type>& ty) {
    for (auto it = scoped_symtab.rbegin(); it != scoped_symtab.rend(); ++it) {
      if (it->count(n)) {
        (*it)[n] = ty;
        if (symtab) {
          auto sym = symtab->GetSymbol(InScopeName(n));
          sym->SetType(ty);

          if (CCtx().DebugSymTab())
            dbgs() << "Modify Symbol: " << InScopeName(n)
                   << ", type: " << PSTR(ty) << "\n";
        }
        return true;
      }
    }
    return false;
  }

  // Suppose the symbol provided is scoped, e.g., "::foo::bar::..."
  // Note: requires symbol to have a valid scope
  bool ModifyScopedSymbolType(const std::string& n, const ptr<Type>& ty) {
    assert(symtab && "requires symbol table exist.");
    assert(PrefixedWith(n, "::") && "symbol must be prefixed with ::");

    if (!symtab->Exists(n)) return false;

    // try to strip the scope names and find the symbol
    auto symbol = n.substr(2);
    size_t scope_index = 1;
    while (scope_index < scope_names.size()) {
      auto pos = symbol.find_first_of("::");
      if (pos == std::string::npos) {
        // symbols' scope are stripped
        break;
      }
      auto sname = symbol.substr(0, pos);

      if (sname != scope_names.at(scope_index)) {
        choreo_unreachable("unexpected symbol with un-matched scope.");
      }

      symbol = symbol.substr(pos + 2);
      ++scope_index;
    }

    auto& sstab = scoped_symtab[scope_index - 1];
    if (!sstab.count(symbol))
      return false; // the symbol does not appears in this scope

    // update both the scoped symbol type and global symbol type
    sstab[symbol] = ty;
    symtab->GetSymbol(n)->SetType(ty);

    return true;
  }

public:
  // utility functions
  const std::string UnScopedName(const std::string& name) const {
    size_t pos = name.find_last_of("::");
    if (pos != std::string::npos) {
      // If found, return the substring after the last "::"
      return name.substr(pos + 1); // skip the "::"
    }
    return name; // Return the original string if ":" is not found
  }

  // get the current scope name
  // consider rename to getScope, more intuitive, not ambiguis to ScopedName
  std::string ScopeName() const {
    std::string name;
    for (auto it = scope_names.begin(); it != scope_names.end(); ++it)
      name += *it + "::";
    return name;
  }

  std::string getParentScope() const {
    // Ensure the input is not empty and ends with "::"
    if (ScopeName().empty() || ScopeName().size() <= 2) {
      return ""; // Invalid or empty scope
    }

    // Find the last occurrence of "::"
    size_t lastPos = ScopeName().rfind("::");

    // Ensure the "::" is found and it's not the first or only "::"
    if (lastPos != std::string::npos && lastPos > 2) {
      // Now find the second-to-last occurrence of "::"
      size_t secondLastPos = ScopeName().rfind("::", lastPos - 1);

      // If second-to-last "::" is found, return the substring up to that point
      if (secondLastPos != std::string::npos) {
        return ScopeName().substr(0, secondLastPos +
                                         2); // Include "::" in the result
      }
    }

    return "";
  }

  // get the name when the symbol is assumed to be defined in current scope
  std::string ScopedName(const std::string& name) const {
    return ScopeName() + name;
  }

  const std::string GetScope(const std::string& name) const {
    if (PrefixedWith(name, "::")) {
      auto pos = name.find_last_of(":");
      if (pos != std::string::npos) return name.substr(0, pos + 1);
    } else if (auto n = NameInScopeOrNull(name))
      return GetScope(*n);

    return name;
  }

  const std::string InScopeName(const std::string& name) const {
    auto n = NameInScopeOrNull(name);
    if (!n) choreo_unreachable("symbol `" + name + "' is not found in scope");
    return *n;
  }

  // If the variable is declared in (multi-level) scopes, retrievd the scoped
  // name. Or else nothing
  std::optional<std::string> NameInScopeOrNull(const std::string& name) const {
    std::string scoped_name;
    auto it = scoped_symtab.rbegin();
    auto in = scope_names.rbegin();
    for (; it != scoped_symtab.rend(); ++it, ++in) {
      if (it->count(name) == 0) continue;

      for (; in != scope_names.rend(); ++in)
        scoped_name = *in + "::" + scoped_name;
      return scoped_name + name;
    }
    return {};
  }

  // Dump current status
  void Dump() const {
    // Iterate in reverse order to simulate stack behavior
    size_t count = 0;
    std::string indent = "";
    for (auto it = scoped_symtab.rbegin(); it != scoped_symtab.rend(); ++it) {
      dbgs() << indent << "<" << scope_names[count] << ">\n";
      for (auto item : *it) {
        dbgs() << indent << " - sym: " << item.first
               << ", type: " << STR(*item.second) << "\n";
      }
      ++count;
      indent = indent + " ";
    }
  }
};

} // end of namespace Choreo

#endif // __CHOREO_SYMTAB_H__
