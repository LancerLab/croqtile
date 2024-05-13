#ifndef __CHOREO_DYNAMIC_SHAPE_INFO_HPP__
#define __CHOREO_DYNAMIC_SHAPE_INFO_HPP__

// This apply the type check and symbol table generation

#include "visitor.hpp"

namespace Choreo {

  struct RuntimeShapeInfo {

  };

struct ShapeDynamics : public VisitorWithSymTab {
 private:
  std::ostream &os;
  bool trace_visit = false;  // for debugging purpose only
  size_t error_count = 0;

  std::unordered_map<std::string, AST::Parameter *> cur_params;

 private:
  bool BeforeVisitImpl(AST::Node &) { return true; }
  bool AfterVisitImpl(AST::Node &) { return true; }

 public:
  ShapeDynamics(const ptr<SymbolTable> s_tab, std::ostream &o = std::cout)
      : VisitorWithSymTab(s_tab),
        os(o),
        trace_visit(std::getenv("TRACE_DYNSHAPE")) {}
  ~ShapeDynamics() {}

  bool Visit(AST::MultiNodes &) { return true; }
  bool Visit(AST::MultiValues &) { return true; }
  bool Visit(AST::IntLiteral &) { return true; }
  bool Visit(AST::Expr &) { return true; }
  bool Visit(AST::MultiDimSpans &) { return true; }
  bool Visit(AST::NamedTypeDecl &) { return true; }
  bool Visit(AST::NamedVariableDecl &) { return true; }
  bool Visit(AST::IntTuple &) { return true; }
  bool Visit(AST::Assignment &) { return true; }
  bool Visit(AST::IntIndex &) { return true; }
  bool Visit(AST::DataType &) { return true; }
  bool Visit(AST::Identifier &) { return true; }
  bool Visit(AST::Parameter &) { return true; }
  bool Visit(AST::ParamList &) { return true; }
  bool Visit(AST::ParallelBy &) { return true; }
  bool Visit(AST::WhereBind &) { return true; }
  bool Visit(AST::WithIn &) { return true; }
  bool Visit(AST::WithBlock &) { return true; }
  bool Visit(AST::Memory &) { return true; }
  bool Visit(AST::DMA &) { return true; }
  bool Visit(AST::ChunkAt &) { return true; }
  bool Visit(AST::Wait &) { return true; }
  bool Visit(AST::Call &) { return true; }
  bool Visit(AST::Return &) { return true; }
  bool Visit(AST::ForeachBlock &) { return true; }
  bool Visit(AST::FunctionDecl &) { return true; }
  bool Visit(AST::ChoreoFunction &) { return true; }
  bool Visit(AST::CppSourceCode &) { return true; }
  bool Visit(AST::Program &) { return true; }

  bool HasError() { return false; }
};

}  // end namespace Choreo

#endif  // __CHOREO_DYNAMIC_SHAPE_INFO_HPP__
