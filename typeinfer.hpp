#ifndef __CHOREO_TYPE_INFERENCE_HPP__
#define __CHOREO_TYPE_INFERENCE_HPP__

#include <iostream>

#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

struct TypeInference : public Visitor {
 private:
  bool Dump = false;
  std::ostream &os;

 private:
  ptr<Type> cur_type = nullptr;
  std::vector<ptr<Type>> cur_param_types;
  MDSpanValue cur_mdspan_value;

 public:
  TypeInference(bool d, std::ostream &o = std::cout) : Dump(d), os(o) {}

  bool Visit(AST::MultiNodes &) override { return true; };
  bool Visit(AST::IntLiteral &) override { return true; };
  bool Visit(AST::SValList &) override;
  bool Visit(AST::Expr &) override;
  bool Visit(AST::MultiDimSpans &) override;
  bool Visit(AST::NamedTypeDecl &) override;
  bool Visit(AST::NamedVariableDecl &) override;
  bool Visit(AST::IntTuple &) override;
  bool Visit(AST::Assignment &) override;
  bool Visit(AST::IntIndex &) override { return true; };
  bool Visit(AST::IntIndexList &) override { return true; };
  bool Visit(AST::DataType &) override;
  bool Visit(AST::Identifier &) override { return true; };
  bool Visit(AST::Parameter &) override;
  bool Visit(AST::ParamList &) override;
  bool Visit(AST::ParallelBy &) override { return true; };
  bool Visit(AST::RequireBind &) override { return true; };
  bool Visit(AST::WithIn &) override { return true; };
  bool Visit(AST::WithBlock &) override { return true; };
  bool Visit(AST::Memory &) override { return true; };
  bool Visit(AST::DMA &) override { return true; };
  bool Visit(AST::ChunkAt &) override { return true; };
  bool Visit(AST::Wait &) override { return true; };
  bool Visit(AST::Call &) override { return true; };
  bool Visit(AST::ForeachBlock &) override { return true; };
  bool Visit(AST::FunctionDecl &) override;
  bool Visit(AST::ChoreoFunction &) override { return true; };
  bool Visit(AST::CppSourceCode &) override { return true; };
  bool Visit(AST::Program &) override { return true; };

 private:
  bool SetCurrentType(AST::Node &, const std::string &);
};

}  // end namespace Choreo

#endif  // __CHOREO_TYPE_INFERENCE_HPP__
