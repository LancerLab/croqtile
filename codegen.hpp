#ifndef __CHOREO_CODEGEN_HPP__
#define __CHOREO_CODEGEN_HPP__

#include "visitor.hpp"

namespace Choreo {

struct CodeGenerator : public Visitor {
  std::ostream &os;

  // derived class must call this to incorporate with symbol table
  bool BeforeVisit(AST::Node &n) override {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      SSTab().EnterScope(f->name);
    } else if (isa<AST::ParallelBy>(&n)) {
      static size_t count = 0;
      SSTab().EnterScope("paraby_" + std::to_string(count++));
    } else if (isa<AST::WithBlock>(&n)) {
      static size_t count = 0;
      SSTab().EnterScope("within_" + std::to_string(count++));
    } else if (isa<AST::ForeachBlock>(&n)) {
      static size_t count = 0;
      SSTab().EnterScope("foreach_" + std::to_string(count++));
    }
    return true;
  }

  bool AfterVisit(AST::Node &n) override {
    if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
        isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
      SSTab().LeaveScope();
    }
    return true;
  }

  virtual std::string InScopeName(const std::string & sym) {
    auto removeLastLevel = [](const std::string &input) -> std::string {
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

  virtual ptr<Type> GetSymbolType(const std::string &n) {
    assert(SymTab()->Exists(InScopeName(n)) && "symbol is not declared.");
    return SymTab()->GetSymbol(InScopeName(n))->GetType();
  }

  CodeGenerator(std::ostream &o, const ptr<SymbolTable> &symtab)
      : Visitor(symtab), os(o) {
    if (symtab == nullptr)
      choreo_unreachable("symbol table must be initialized.");
  }
};

struct FactorCodeGen : public CodeGenerator {
  // TODO: should the pointer be replaced?
  std::string current_fn = "";
  std::string indent = "";
  std::vector<AST::ptr<AST::Parameter>> *cur_params = nullptr;
  AST::ptr<AST::DataType> current_output = nullptr;

  FactorCodeGen(std::ostream &os, const ptr<SymbolTable> &symtab)
      : CodeGenerator(os, symtab) {}

  bool BeforeVisit(AST::Node&) override;
  bool AfterVisit(AST::Node&) override;

  // bool Visit(AST::Node&) override;
  bool Visit(AST::MultiNodes &) override;
  bool Visit(AST::MultiValues &) override;
  bool Visit(AST::IntLiteral &) override;
  bool Visit(AST::Expr &) override;
  bool Visit(AST::MultiDimSpans &) override;
  bool Visit(AST::NamedTypeDecl &) override;
  bool Visit(AST::NamedVariableDecl &) override;
  bool Visit(AST::IntTuple &) override;
  bool Visit(AST::Assignment &) override;
  bool Visit(AST::IntIndex &) override;
  bool Visit(AST::DataType &) override;
  bool Visit(AST::Identifier &) override;
  bool Visit(AST::Parameter &) override;
  bool Visit(AST::ParamList &) override;
  bool Visit(AST::ParallelBy &) override;
  bool Visit(AST::RequireBind &) override;
  bool Visit(AST::WithIn &) override;
  bool Visit(AST::WithBlock &) override;
  bool Visit(AST::Memory &) override;
  bool Visit(AST::DMA &) override;
  bool Visit(AST::ChunkAt &) override;
  bool Visit(AST::Wait &) override;
  bool Visit(AST::Call &) override;
  bool Visit(AST::Return &) override;
  bool Visit(AST::ForeachBlock &) override;
  bool Visit(AST::FunctionDecl &) override;
  bool Visit(AST::ChoreoFunction &) override;
  bool Visit(AST::CppSourceCode &) override;
  bool Visit(AST::Program &) override;

  // common utils
  void incrementIndent() {
    this->indent += "  ";
  }

  void decrementIndent() {
    if (this->indent.size() >= 2) this->indent = this->indent.substr(0, this->indent.size() - 2);
  }
};

struct TopsccCodeGen : public CodeGenerator {
  // bool Visit(AST::Node&) override;

  bool Visit(AST::MultiNodes &) override { return true; };
  bool Visit(AST::MultiValues &) override { return true; };
  bool Visit(AST::IntLiteral &) override { return true; };
  bool Visit(AST::Expr &) override { return true; };
  bool Visit(AST::MultiDimSpans &) override { return true; };
  bool Visit(AST::NamedTypeDecl &) override { return true; };
  bool Visit(AST::NamedVariableDecl &) override { return true; };
  bool Visit(AST::IntTuple &) override { return true; };
  bool Visit(AST::Assignment &) override { return true; };
  bool Visit(AST::IntIndex &) override { return true; };
  bool Visit(AST::DataType &) override { return true; };
  bool Visit(AST::Identifier &) override { return true; };
  bool Visit(AST::Parameter &) override { return true; };
  bool Visit(AST::ParamList &) override { return true; };
  bool Visit(AST::ParallelBy &) override { return true; };
  bool Visit(AST::RequireBind &) override { return true; };
  bool Visit(AST::WithIn &) override { return true; };
  bool Visit(AST::WithBlock &) override { return true; };
  bool Visit(AST::Memory &) override { return true; };
  bool Visit(AST::DMA &) override { return true; };
  bool Visit(AST::ChunkAt &) override { return true; };
  bool Visit(AST::Wait &) override { return true; };
  bool Visit(AST::Call &) override { return true; };
  bool Visit(AST::Return &) override { return true; };
  bool Visit(AST::ForeachBlock &) override { return true; };
  bool Visit(AST::FunctionDecl &) override { return true; };
  bool Visit(AST::ChoreoFunction &) override { return true; };
  bool Visit(AST::CppSourceCode &) override { return true; };
  bool Visit(AST::Program &) override { return true; };
};

}  // end namespace Choreo

#endif  // __CHOREO_CODEGEN_HPP__
