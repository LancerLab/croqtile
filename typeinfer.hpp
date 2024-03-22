#ifndef __CHOREO_TYPE_INFERENCE_HPP__
#define __CHOREO_TYPE_INFERENCE_HPP__

#include <iostream>
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

struct TypeInference : public Visitor {
  bool Dump = false;
  std::ostream & os;

  TypeInference(bool d, std::ostream & o = std::cout) : Dump(d), os(o) {}

  ptr<Type> cur_type = nullptr;

  bool Visit(AST::MultiNodes&) override { return true; };
  bool Visit(AST::IntLiteral&) override { return true; };
  bool Visit(AST::SValList&) override { return true; };
  bool Visit(AST::Expr&) override { return true; };
  bool Visit(AST::MultiDimSpans&) override { return true; };
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::IntTuple&) override { return true; };
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::IntIndex&) override { return true; };
  bool Visit(AST::NthBound&) override { return true; };
  bool Visit(AST::IntIndexList&) override { return true; };
  bool Visit(AST::DataType&) override;
  bool Visit(AST::Identifier&) override { return true; };
  bool Visit(AST::Parameter&) override { return true; };
  bool Visit(AST::ParamList&) override { return true; };
  bool Visit(AST::ParallelBy&) override { return true; };
  bool Visit(AST::RequireBind&) override { return true; };
  bool Visit(AST::WithIn&) override { return true; };
  bool Visit(AST::WithBlock&) override { return true; };
  bool Visit(AST::Memory&) override { return true; };
  bool Visit(AST::DMA&) override { return true; };
  bool Visit(AST::ChunkAt&) override { return true; };
  bool Visit(AST::Wait&) override { return true; };
  bool Visit(AST::Call&) override { return true; };
  bool Visit(AST::ForeachBlock&) override { return true; };
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override { return true; };
  bool Visit(AST::CppSourceCode&) override { return true; };
  bool Visit(AST::Program&) override { return true; };
};

}  // end namespace Choreo

#endif  // __CHOREO_TYPE_INFERENCE_HPP__
