#ifndef __CHOREO_GCU_CHECK_INFO_HPP__
#define __CHOREO_GCU_CHECK_INFO_HPP__

// This apply the GCU target specific check and information annotation

#include "visitor.hpp"

namespace Choreo {

struct GCUCheck : public VisitorWithSymTab {
 private:
  std::ostream &os;
  bool trace_visit = false;  // for debugging purpose only
  size_t error_count = 0;

  std::unordered_map<std::string, AST::Parameter *> cur_params;

 private:
  bool BeforeVisitImpl(AST::Node &n) {
    if (isa<AST::ChoreoFunction>(&n)) cur_params.clear();
    return true;
  }
  bool AfterVisitImpl(AST::Node &) { return true; }

 public:
  GCUCheck(const ptr<SymbolTable> s_tab, std::ostream &o = std::cout)
      : VisitorWithSymTab(s_tab),
        os(o),
        trace_visit(std::getenv("TRACE_GCU")) {}
  ~GCUCheck() {}

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
  bool Visit(AST::Parameter &n) {
    if (n.sym) cur_params.emplace(InScopeName(n.sym->name), &n);
    return true;
  }
  bool Visit(AST::ParamList &) { return true; }
  bool Visit(AST::ParallelBy &) { return true; }
  bool Visit(AST::RequireBind &) { return true; }
  bool Visit(AST::WithIn &) { return true; }
  bool Visit(AST::WithBlock &) { return true; }
  bool Visit(AST::Memory &) { return true; }
  bool Visit(AST::DMA &n) {
    // The user does not have to explicitly claim a global memory that requires
    // direct copy from host to device. Here Choreo judge if a spanned memory is
    // shadowed from the data movement. Later, codegen handles such a shadow.
    if (!isa<AST::ChunkAt>(n.from)) return true;
    auto f_name = cast<AST::ChunkAt>(n.from)->RefSymbol();
    if (cast<SpannedType>(GetSymbolType(f_name))->GetStorage() !=
        Storage::DEFAULT)
      return true;

    // not referencing the parameter
    if (!cur_params.count(InScopeName(f_name))) return true;

    auto annotate_by_storage = [this, &f_name](Storage st) {
      switch (st) {
        case Storage::GLOBAL:
        case Storage::SHARED:
        case Storage::LOCAL: {
          auto p = cur_params[InScopeName(f_name)];
          p->attr = ATT_SHADOW_TO_GLOBAL;
          break;
        }
        default:
          break;
      }
    };
    if (auto to = dyn_cast<AST::ChunkAt>(n.to))
      annotate_by_storage(
          cast<SpannedType>(GetSymbolType(to->RefSymbol()))->GetStorage());
    else if (auto m = dyn_cast<AST::Memory>(n.to))
      annotate_by_storage(m->st);

    return true;
  }
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

#endif  // __CHOREO_GCU_CHECK_INFO_HPP__
