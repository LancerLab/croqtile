#ifndef __CHOREO_VISITOR_HPP__
#define __CHOREO_VISITOR_HPP__

#include <cstring>
#include <iostream>
#include <optional>
#include <unordered_set>

#include "ast.hpp"
#include "location.hh"

namespace Choreo {

struct Visitor {
  // virtual bool Visit(AST::Node&) = 0;
  virtual bool BeforeVisit(AST::Node&) { return true; }
  virtual bool AfterVisit(AST::Node&) { return true; }

  // For any visitor, it should implement all the necessary steps
  virtual bool Visit(AST::MultiNodes&) = 0;
  virtual bool Visit(AST::MultiValues&) = 0;
  virtual bool Visit(AST::IntLiteral&) = 0;
  virtual bool Visit(AST::Expr&) = 0;
  virtual bool Visit(AST::MultiDimSpans&) = 0;
  virtual bool Visit(AST::NamedTypeDecl&) = 0;
  virtual bool Visit(AST::NamedVariableDecl&) = 0;
  virtual bool Visit(AST::IntTuple&) = 0;
  virtual bool Visit(AST::Assignment&) = 0;
  virtual bool Visit(AST::IntIndex&) = 0;
  virtual bool Visit(AST::DataType&) = 0;
  virtual bool Visit(AST::Identifier&) = 0;
  virtual bool Visit(AST::Parameter&) = 0;
  virtual bool Visit(AST::ParamList&) = 0;
  virtual bool Visit(AST::ParallelBy&) = 0;
  virtual bool Visit(AST::RequireBind&) = 0;
  virtual bool Visit(AST::WithIn&) = 0;
  virtual bool Visit(AST::WithBlock&) = 0;
  virtual bool Visit(AST::Memory&) = 0;
  virtual bool Visit(AST::DMA&) = 0;
  virtual bool Visit(AST::ChunkAt&) = 0;
  virtual bool Visit(AST::Wait&) = 0;
  virtual bool Visit(AST::Call&) = 0;
  virtual bool Visit(AST::Return&) = 0;
  virtual bool Visit(AST::ForeachBlock&) = 0;
  virtual bool Visit(AST::FunctionDecl&) = 0;
  virtual bool Visit(AST::ChoreoFunction&) = 0;
  virtual bool Visit(AST::CppSourceCode&) = 0;
  virtual bool Visit(AST::Program&) = 0;

 public:
  // general scoped variable handling
  std::vector<std::unordered_map<std::string, Symbol>> scopeStack;
  std::vector<std::string> scopeNames;

  virtual void EnterScope(const std::string& name = "") {
    scopeStack.emplace_back();  // Push a new scope
    scopeNames.emplace_back(name);
  }

  virtual void LeaveScope() {
    if (!scopeStack.empty()) {
      scopeStack.pop_back();  // Pop the last scope
      scopeNames.pop_back();
    }
  }

  virtual bool IsDeclared(const std::string& sym_name) {
    // Iterate in reverse order to simulate stack behavior
    for (auto it = scopeStack.rbegin(); it != scopeStack.rend(); ++it) {
      if (it->count(sym_name))
        return true;  // Found sym_name in the current or an enclosing scope
    }
    return false;  // sym_name not found in any scope
  }

  virtual bool DefineSymbol(const std::string& n, const ptr<Type> ty) {
    if (!scopeStack.empty()) {
      if (scopeStack.back().count(n) == 0) {
        scopeStack.back().emplace(
            n, Symbol(n, ty));  // Insert into the current (top) scope
        return true;
      }
    }
    return false;
  }

  virtual const Symbol* LookupSymbol(const std::string& n) {
    for (auto it = scopeStack.rbegin(); it != scopeStack.rend(); ++it) {
      if (it->count(n)) return &(*it).at(n);
    }
    return nullptr;
  }

  virtual std::string UnscopedName(const std::string& name) {
    size_t pos = name.find_last_of("::");
    if (pos != std::string::npos) {
      // If found, return the substring after the last "::"
      return name.substr(pos + 2);  // +2 to skip the "::" itself
    }
    return name;  // Return the original string if "::" is not found
  }

  virtual size_t ScopeDepth() const { return scopeStack.size(); }

  virtual std::string ScopeName() const {
    std::string name;
    for (auto it = scopeNames.begin(); it != scopeNames.end(); ++it)
      name += *it + "::";
    return name;
  }

  // If the variable is declared in (multi-level) scopes, retrievd the scoped
  // name. Or else nothing
  virtual std::string ScopedName(const std::string& name) const {
    return ScopeName() + name;
  }

  // If the variable is declared in (multi-level) scopes, retrievd the scoped
  // name. Or else nothing
  virtual std::optional<std::string> InScopeName(
      const std::string& name) const {
    std::string scoped_name;
    auto it = scopeStack.rbegin();
    auto in = scopeNames.rbegin();
    for (; it != scopeStack.rend(); ++it, ++in) {
      if (it->count(name) == 0) continue;

      for (; in != scopeNames.rend(); ++in)
        scoped_name = *in + "::" + scoped_name;
      return scoped_name + name;
    }
    return {};
  }

 public:
  void Error(const location& loc, const std::string& message) {
    static const char* red = "\033[31m";
    static const char* reset = "\033[0m";

    auto shell_supports_colors = [&]() {
      const char* term = getenv("TERM");
      return term && (strcmp(term, "xterm-256color") == 0 ||
                      strcmp(term, "xterm") == 0);
    };

    std::cerr << loc << ": ";

    if (shell_supports_colors())
      std::cerr << red << "Error: " << reset;
    else
      std::cerr << "Error: ";

    std::cerr << message << std::endl;
  }

  void Warning(const location& loc, const std::string& message) {
    static const char* yellow = "\033[33m";
    static const char* reset = "\033[0m";

    auto shell_supports_colors = [&]() {
      const char* term = getenv("TERM");
      return term && (strcmp(term, "xterm-256color") == 0 ||
                      strcmp(term, "xterm") == 0);
    };

    std::cerr << loc << ": ";

    if (shell_supports_colors())
      std::cerr << yellow << "Warning: " << reset;
    else
      std::cerr << "Warning: ";

    std::cerr << message << std::endl;
  }
};

}  // end namespace Choreo

#endif  // __CHOREO_VISITOR_HPP__
