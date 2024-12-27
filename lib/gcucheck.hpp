#ifndef __CHOREO_GCU_CHECK_INFO_HPP__
#define __CHOREO_GCU_CHECK_INFO_HPP__

// This apply the GCU target specific check and information annotation

#include "ast.hpp"
#include "visitor.hpp"

namespace Choreo {

struct GCUCheck : public VisitorWithSymTab {
private:
  std::unordered_map<std::string, AST::Parameter*> cur_params;
  int parallel_level = 0;
  int max_parallel_level = 0;
  int local_level = 0;

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(pre)");
    if (isa<AST::ChoreoFunction>(&n)) {
      local_level = 0;
      cur_params.clear();
    } else if (isa<AST::ParallelBy>(&n)) {
      parallel_level++;
      assert((parallel_level < 3) && "unexpected parallel level.");
      max_parallel_level = parallel_level;
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(post)");
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      std::string append_note =
          ":" + std::to_string(max_parallel_level - parallel_level);
      auto pty = cast<BoundedITupleType>(NodeType(n));
      pty->AppendNote(append_note);
      for (auto& symbol : pb->iv_symbols->AllValues())
        cast<BoundedITupleType>(NodeType(*symbol))->AppendNote(append_note);

      parallel_level--;
      assert(parallel_level >= 0 && "Unexpected parallel level");
      if (parallel_level == 0) max_parallel_level = 0;
    }
    return true;
  }

  void TraceEachVisit(AST::Node& n, std::string sup = "") {
    if (trace_visit) dbgs() << n.TypeNameString() << sup << "\n";
  }

public:
  GCUCheck() : VisitorWithSymTab("gcu", CCtx().GetGlobalSymbolTable()) {}
  ~GCUCheck() {}

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiValues& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntLiteral& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Boolean& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Expr& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiDimSpans& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedTypeDecl& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedVariableDecl& n) override {
    TraceEachVisit(n);
    auto ty = GetSymbolType(n.name_str);
    if (!isa<SpannedType>(ty)) return true;
    auto sty = cast<SpannedType>(ty);
    auto st = sty->GetStorage();
    switch (st) {
    case Storage::GLOBAL:
      if (parallel_level != 0) {
        Error(n.LOC(), "global variable '" + n.name_str +
                           "` mustn't be declared inside parallel-by.");
        error_count++;
      }
      break;
    case Storage::SHARED:
      if (parallel_level != 1) {
        Error(n.LOC(),
              "shared variable '" + n.name_str +
                  "` must be declared inside single level of parallel-by.");
        error_count++;
      } else if (local_level == 1 && parallel_level == 2) {
        // if parallel_level == 1, allow
        // eg. parallel p by 6 { shared; local; }
        Error(n.LOC(), "shared variable '" + n.name_str +
                           "` mustn't be declared within the same level of "
                           "parallel-by as local variables.");
        error_count++;
      }
      if (sty->RuntimeShaped()) {
        Error(n.LOC(), "GCU forbids shared variable '" + n.name_str +
                           "` to be dynamically shaped (by " +
                           STR(sty->GetShape()) + ").");
        error_count++;
      }
      break;
    case Storage::LOCAL:
      if (parallel_level == 0) {
        Error(n.LOC(), "local variable '" + n.name_str +
                           "` must be declared inside parallel-by.");
        error_count++;
      } else if (local_level != 0 && parallel_level != local_level) {
        Error(n.LOC(), "local variable '" + n.name_str +
                           "` must be declared inside a level of parallel-by "
                           "that is identical to other local variables.");
        error_count++;
      } else if (local_level == 0)
        local_level = parallel_level;
      if (sty->RuntimeShaped()) {
        Error(n.LOC(), "GCU forbids local variable '" + n.name_str +
                           "` to be dynamically shaped (by " +
                           STR(sty->GetShape()) + ").");
        error_count++;
      }
      break;
    default:
      Error(n.LOC(), "can not declare variable '" + n.name_str + "` as " +
                         STR(st) + " inside choreo function.");
      error_count++;
      break;
    }
    return true;
  }
  bool Visit(AST::IntTuple& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Assignment& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntIndex& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DataType& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Identifier& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Parameter& n) override {
    TraceEachVisit(n);
    if (n.sym) cur_params.emplace(InScopeName(n.sym->name), &n);
    return true;
  }
  bool Visit(AST::ParamList& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ParallelBy& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WhereBind& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithIn& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithBlock& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Memory& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::SpanAs& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DMA& n) override {
    TraceEachVisit(n);
    // The user does not have to explicitly claim a global memory that requires
    // direct copy from host to device. Here Choreo judge if a spanned memory is
    // shadowed from the data movement. Later, codegen handles such a shadow.
    if (!isa<AST::ChunkAt>(n.from)) return true;
    auto f_name = cast<AST::ChunkAt>(n.from)->RefSymbol();

    auto sty = GetSpannedType(GetSymbolType(f_name));

    // storage level must be specified
    if (sty->GetStorage() == Storage::NONE) return false;

    // not referencing the parameter
    if (!cur_params.count(InScopeName(f_name))) return false;

    auto annotate_by_storage = [this, &f_name](Storage st) {
      switch (st) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        auto p = cur_params[InScopeName(f_name)];
        p->attr = ATT_SHADOW_TO_GLOBAL;
        break;
      }
      default: break;
      }
    };
    if (auto to = dyn_cast<AST::ChunkAt>(n.to))
      annotate_by_storage(
          cast<SpannedType>(GetSymbolType(to->RefSymbol()))->GetStorage());
    else if (auto m = dyn_cast<AST::Memory>(n.to))
      annotate_by_storage(m->st);

    return true;
  }
  bool Visit(AST::ChunkAt& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Wait& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Call& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Rotate& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Select& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Return& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::LoopRange& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ForeachBlock& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::FunctionDecl& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ChoreoFunction& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::CppSourceCode& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Program& n) override {
    TraceEachVisit(n);
    return true;
  }

  bool HasError() override {
    if (error_count)
      dbgs() << "Totally " << error_count << " errors have been detected.\n";
    return error_count != 0;
  }
};

} // end namespace Choreo

#endif // __CHOREO_GCU_CHECK_INFO_HPP__
