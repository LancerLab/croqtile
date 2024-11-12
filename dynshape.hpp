#ifndef __CHOREO_DYNAMIC_SHAPE_INFO_HPP__
#define __CHOREO_DYNAMIC_SHAPE_INFO_HPP__

// This apply the type check and symbol table generation

#include "codegen.hpp"

namespace Choreo {

struct ShapeDynamics : public VisitorWithSymTab {
private:
  std::ostream& os;
  size_t error_count = 0;

  std::unordered_map<std::string, AST::Parameter*> cur_params;

  std::string fname; // current function name
  ptr<FutureBufferMap> fut_buf = nullptr;

private:
  bool BeforeVisitImpl(AST::Node& n) {
    if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
      assert(fut_buf->count(f->name) == 0);
      fut_buf->insert({f->name, {}});
      fname = f->name;
    } else if (auto dma = dyn_cast<AST::DMA>(&n)) {
      // associate a future with its only buffer
      if (!dma->future.empty() && (dma->operation != ".any")) {
        auto buf_name = cast<AST::ChunkAt>(dma->to)->RefSymbol();
        (*fut_buf)[fname].emplace(dma->future, buf_name);
        VST_DEBUG(os << "associate " << dma->future << " with " << buf_name
                     << "\n");
      }
    }
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) {
    if (isa<AST::ChoreoFunction>(&n)) { fname = ""; }
    return true;
  }

public:
  ShapeDynamics(const ptr<SymbolTable> s_tab, std::ostream& o = std::cout)
      : VisitorWithSymTab("dynshape", s_tab), os(o) {
    fut_buf = std::make_shared<FutureBufferMap>();
  }
  ~ShapeDynamics() {}

  const ptr<FutureBufferMap> FBInfo() { return fut_buf; }

  bool Visit(AST::MultiNodes&) { return true; }
  bool Visit(AST::MultiValues&) { return true; }
  bool Visit(AST::IntLiteral&) { return true; }
  bool Visit(AST::Boolean&) { return true; }
  bool Visit(AST::Expr&) { return true; }
  bool Visit(AST::MultiDimSpans&) { return true; }
  bool Visit(AST::NamedTypeDecl&) { return true; }
  bool Visit(AST::NamedVariableDecl&) { return true; }
  bool Visit(AST::IntTuple&) { return true; }
  bool Visit(AST::Assignment&) { return true; }
  bool Visit(AST::IntIndex&) { return true; }
  bool Visit(AST::DataType&) { return true; }
  bool Visit(AST::Identifier&) { return true; }
  bool Visit(AST::Parameter&) { return true; }
  bool Visit(AST::ParamList&) { return true; }
  bool Visit(AST::ParallelBy&) { return true; }
  bool Visit(AST::WhereBind&) { return true; }
  bool Visit(AST::WithIn&) { return true; }
  bool Visit(AST::WithBlock&) { return true; }
  bool Visit(AST::Memory&) { return true; }
  bool Visit(AST::SpanAs&) { return true; }
  bool Visit(AST::DMA&) { return true; }
  bool Visit(AST::ChunkAt&) { return true; }
  bool Visit(AST::Wait&) { return true; }
  bool Visit(AST::Call&) { return true; }
  bool Visit(AST::Rotate&) { return true; }
  bool Visit(AST::Select&) { return true; }
  bool Visit(AST::Return&) { return true; }
  bool Visit(AST::LoopRange&) { return true; }
  bool Visit(AST::ForeachBlock&) { return true; }
  bool Visit(AST::FunctionDecl&) { return true; }
  bool Visit(AST::ChoreoFunction&) { return true; }
  bool Visit(AST::CppSourceCode&) { return true; }
  bool Visit(AST::Program&) { return true; }

  bool HasError() { return false; }
};

} // end namespace Choreo

#endif // __CHOREO_DYNAMIC_SHAPE_INFO_HPP__
