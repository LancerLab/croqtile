#ifndef __CHOREO_CODEGEN_HPP__
#define __CHOREO_CODEGEN_HPP__

#include "visitor.hpp"

namespace Choreo {

struct CodeGenerator : public VisitorWithSymTab {
  std::ostream &os;

  // some default method for the derived classes that do not want to override.
  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node&) override { return true; }

  CodeGenerator(std::ostream &o, const ptr<SymbolTable> &symtab)
      : VisitorWithSymTab(symtab), os(o) {
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

  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

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
