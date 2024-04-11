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
  Visitor(const ptr<SymbolTable>& s_tab = nullptr)
      : scoped_symtab(s_tab) {}
  virtual ~Visitor() {}

  // simple reference to the symbol table
  virtual ScopedSymbolTable& SSTab() { return scoped_symtab; }

  virtual const ptr<SymbolTable> SymTab() const {
    if (auto st = scoped_symtab.GlobalSymbolTable())
      return st;
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

}  // end namespace Choreo

#endif  // __CHOREO_VISITOR_HPP__
