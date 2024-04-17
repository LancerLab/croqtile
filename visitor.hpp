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

 private:
  // scoped variable handling
  ScopedSymbolTable scoped_symtab;

 public:
  Visitor(const ptr<SymbolTable>& s_tab = nullptr) : scoped_symtab(s_tab) {}
  virtual ~Visitor() {}

  // simple reference to the symbol table
  virtual ScopedSymbolTable& SSTab() { return scoped_symtab; }

  virtual const ptr<SymbolTable> SymTab() const {
    if (auto st = scoped_symtab.GlobalSymbolTable()) return st;
    choreo_unreachable("Retrieving an invalid symbol table.");
    return nullptr;
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

// This accepts static symbol table and provide symbol lookup capability
struct VisitorWithSymTab : public Visitor {
 protected:
  // for the derived classes
  virtual bool BeforeVisitImpl(AST::Node& n) = 0;
  virtual bool AfterVisitImpl(AST::Node& n) = 0;

  std::string InScopeName(const std::string& sym) {
    auto removeLastLevel = [](const std::string& input) -> std::string {
      size_t lastPos = input.rfind("::");
      if (lastPos == std::string::npos)
        return input;  // No "::" found, return the original string
      // Find the second-to-last "::" by searching up to the last found position
      size_t secondLastPos = input.rfind("::", lastPos - 1);
      if (secondLastPos == std::string::npos) return input;
      return input.substr(0,
                          secondLastPos + 2);  // Include the "::" in the result
    };
    std::string scope_name = SSTab().ScopeName();
    while (true) {
      std::string scoped_name = scope_name + sym;
      if (SymTab()->Exists(scoped_name)) return scoped_name;
      std::string stripped_scope = removeLastLevel(scope_name);
      if (stripped_scope == scope_name) break;
      scope_name = stripped_scope;
    }

    choreo_unreachable("unable to find symbol `" + sym +
                       "' in the symbol table.");
    return "";
  }

  virtual ptr<Type> GetSymbolType(const std::string& n) {
    return SymTab()->GetSymbol(InScopeName(n))->GetType();
  }

 private:
  int pb_count = 0;  // counting for parallel_by
  int wi_count = 0;  // counting for with_in
  int fe_count = 0;  // counting for foreach

  void Reset() {
    pb_count = 0;
    wi_count = 0;
    fe_count = 0;
  }

 public:
  bool BeforeVisit(AST::Node& n) final {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      SSTab().EnterScope(f->name);
    } else if (isa<AST::ParallelBy>(&n)) {
      SSTab().EnterScope("paraby_" + std::to_string(pb_count++));
    } else if (isa<AST::WithBlock>(&n)) {
      SSTab().EnterScope("within_" + std::to_string(wi_count++));
    } else if (isa<AST::ForeachBlock>(&n)) {
      SSTab().EnterScope("foreach_" + std::to_string(fe_count++));
    }
    BeforeVisitImpl(n);  // derived class to customize
    return true;
  }

  bool AfterVisit(AST::Node& n) final {
    if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
        isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
      SSTab().LeaveScope();
    }
    AfterVisitImpl(n);  // derived class to customize
    return true;
  }

 public:
  VisitorWithSymTab(const ptr<SymbolTable>& s_tab) : Visitor(s_tab) { Reset(); }
  ~VisitorWithSymTab() {}
};

}  // end namespace Choreo

#endif  // __CHOREO_VISITOR_HPP__
