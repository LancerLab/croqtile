#ifndef CHOREO_CODEGEN_TOPSCC_HPP_
#define CHOREO_CODEGEN_TOPSCC_HPP_

#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

#include "ast.hpp"
// #include "choreo_topscc_header.inc"
#include "codegen.hpp"
// #include "topscc_script.inc"
#include "types.hpp"

namespace Choreo {

namespace Topscc {

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
  bool Visit(AST::WhereBind &) override { return true; };
  bool Visit(AST::WithIn &) override { return true; };
  bool Visit(AST::WithBlock &) override { return true; };
  bool Visit(AST::Memory &) override { return true; };
  bool Visit(AST::DMA &) override { return true; };
  bool Visit(AST::ChunkAt &) override { return true; };
  bool Visit(AST::Wait &) override { return true; };
  bool Visit(AST::Call &) override { return true; };
  bool Visit(AST::Swap &) override { return true; };
  bool Visit(AST::Select &) override { return true; };
  bool Visit(AST::Return &) override { return true; };
  bool Visit(AST::LoopRange &) override { return true; };
  bool Visit(AST::ForeachBlock &) override { return true; };
  bool Visit(AST::FunctionDecl &) override { return true; };
  bool Visit(AST::ChoreoFunction &) override { return true; };
  bool Visit(AST::CppSourceCode &) override { return true; };
  bool Visit(AST::Program &) override { return true; };
};


}  // end namespace CUDA

}  // end namespace Choreo

#endif // CHOREO_CODEGEN_TOPSCC_HPP_
