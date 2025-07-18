#ifndef __CHOREO_CODEGEN_PREPARE_HPP__
#define __CHOREO_CODEGEN_PREPARE_HPP__

// This apply the type check and symbol table generation

#include "codegen.hpp"

namespace Choreo {

struct CodegenPrepareStage2 : public CodeGenerator {
private:
  ptr<CodeGenInfo> cgi = nullptr;
  int parallel_level = 0;
  int mxpl = 0;

public:
  CodegenPrepareStage2(ptr<CodeGenInfo> c)
      : CodeGenerator("prepare2", CCtx().GetGlobalSymbolTable()), cgi(c) {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      parallel_level = 0;
      mxpl = 0;
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      parallel_level++;
      mxpl = GetMaxParallelLevelFromNote(*pb);
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ParallelBy>(&n)) {
      if (parallel_level == 1) mxpl = 0;
      parallel_level--;
    }
    return true;
  }

  bool Visit(AST::DMA& n) override {
    if (n.future.empty() || (n.operation == ".any")) return true;
    if ((mxpl == 2 || mxpl == 3) && parallel_level == 1) {
      // the DMA is inside block-shared zone
      cgi->GetFunctionSharedFutures(fname).insert(InScopeName(n.future));
      VST_DEBUG(dbgs() << "Shared Future: " << InScopeName(n.future) << "\n");
    }
    if (mxpl == 3 && parallel_level == 2) {
      // the DMA is inside warp-local zone
      cgi->GetFunctionLocalFutures(fname).insert(InScopeName(n.future));
      VST_DEBUG(dbgs() << "Local Future: " << InScopeName(n.future) << "\n");
    }
    return true;
  }
};

struct CodegenPrepare : public CodeGenerator {
private:
  ptr<CodeGenInfo> cgi;
  int parallel_level = 0;
  int max_parallel_level = 0;

  // special case for `return select.data;`
  std::set<std::string> select_syms;

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      parallel_level = 0;
      cgi->GetFunctionTrait(fname).has_parallelby = false;
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (parallel_level == 0 && cgi->GetFunctionTrait(fname).has_parallelby)
        cgi->GetFunctionTrait(fname).multiple_parallelby = true;
      parallel_level++;
      max_parallel_level = std::max(parallel_level, max_parallel_level);

      auto& lcs = cgi->GetFunctionLaunches(fname);

      // Add a new launch config
      if (parallel_level == 1) {
        // represents the index of the current ParallelBy in cgi
        n.Note().insert_or_assign("outer_pb_idx", std::to_string(lcs.size()));
        lcs.push_back({});
      }

      // All the pb in a nested pb is explicitly specified with pb level.
      if (Storage s = n.GetLevel(); s != Storage::NONE) {
        auto& lc = lcs.back();
        switch (s) {
        case Storage::SHARED: lc.SetGridDims(pb->BoundValues()); break;
        case Storage::LOCAL: lc.SetBlockDims(pb->BoundValues()); break;
        case Storage::SUB: lc.SetWarpDims(pb->BoundValues()); break;
        default:
          choreo_unreachable("The explicit parallel-by level " + STR(s) +
                             " is not supported.");
        }
      } else {
        if (parallel_level == 1) {
          lcs.back().SetBlockDims(pb->BoundValues());
        } else if (parallel_level == 2) {
          auto& lc = lcs.back();
          lc.OverwriteGDimsByBDims();
          lc.ResetBDims();
          lc.SetBlockDims(pb->BoundValues());
        } else if (parallel_level == 3) {
          auto& lc = lcs.back();
          lc.ResetWDims();
          lc.SetWarpDims(pb->BoundValues());
        } else
          choreo_unreachable("The parallel-by level " +
                             std::to_string(parallel_level) +
                             " is not supported.");
      }
    }
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "Symbols in " << fname << ":\n");
      VST_DEBUG(for (auto& item : cgi->GetFunctionSymbols(fname)) {
        dbgs() << " |- " << item.name << ", ty: " << PSTR(item.type)
               << ", is_return: "
               << (item.rty_str.empty() ? "no" : "yes(" + item.rty_str + ")")
               << ", index: " << item.p_index << "\n";
      });
    } else if (isa<AST::ParallelBy>(&n)) {
      n.Note().insert_or_assign("mxl", std::to_string(max_parallel_level));
      if (parallel_level == 1) {
        VST_DEBUG(dbgs() << "\tGrid Dims: "
                         << cgi->GetFunctionLaunches(fname).back().grid_dim_x
                         << "\n");
        VST_DEBUG(dbgs() << "\tBlock Dims: "
                         << cgi->GetFunctionLaunches(fname).back().block_dim_x
                         << "\n");
        max_parallel_level = 0;
      }
      parallel_level--;
    }
    return true;
  }

private:
  bool IsHost() const { return parallel_level == 0; }

public:
  CodegenPrepare() : CodeGenerator("prepare", CCtx().GetGlobalSymbolTable()) {
    cgi = std::make_shared<CodeGenInfo>();
  }
  ~CodegenPrepare() {}

  const ptr<CodeGenInfo> GetASTInfo() { return cgi; }

  bool Visit(AST::MultiNodes&) { return true; }
  bool Visit(AST::MultiValues&) { return true; }
  bool Visit(AST::IntLiteral&) { return true; }
  bool Visit(AST::FloatLiteral&) { return true; }
  bool Visit(AST::BoolLiteral&) { return true; }
  bool Visit(AST::Expr&) { return true; }
  bool Visit(AST::MultiDimSpans&) { return true; }
  bool Visit(AST::NamedTypeDecl&) { return true; }

  bool Visit(AST::NamedVariableDecl& n) override {
    auto name = n.name_str;
    bool ref = n.Note().count("ref");
    cgi->AddSymbolDetail(fname, {InScopeName(name), GetSymbolType(name), ref});
    if (isa<AST::Select>(n.init_expr)) select_syms.insert(InScopeName(name));
    return true;
  }

  bool Visit(AST::IntTuple&) { return true; }

  bool Visit(AST::Assignment& n) override {
    if (n.AssignToDataElement()) return true;
    auto name = n.GetName();
    bool ref = n.Note().count("ref");
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value)) {
      cgi->AddSymbolDetail(fname,
                           {InScopeName(name), GetSymbolType(name), ref});
      if (isa<AST::Select>(n.value)) select_syms.insert(InScopeName(name));
    }
    return true;
  }
  bool Visit(AST::IntIndex&) { return true; }
  bool Visit(AST::DataType&) { return true; }
  bool Visit(AST::Identifier&) { return true; }
  bool Visit(AST::Parameter&) { return true; }

  bool Visit(AST::ParamList& n) override {
    int index = 0;
    for (auto param : n.values) {
      cgi->AddSymbolDetail(fname,
                           {InScopeName(param->sym->name), param->GetType(),
                            param->pass_by_ref, index++, param->GetAttr()});
    }
    return true;
  }

  bool Visit(AST::ParallelBy&) override {
    cgi->GetFunctionTrait(fname).has_parallelby = true;
    return true;
  }

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

  bool Visit(AST::Return& n) override {
    std::string ret_name;
    if (auto id = GetIdentifier(*n.value); id) {
      ret_name = id->name;
    } else {
      if (auto expr = dyn_cast<AST::Expr>(n.value);
          expr && expr->op == "dataof") {
        id = cast<AST::Expr>(expr->GetR())->GetSymbol().get();
        assert(id && "Expect a symbol.");
        // `return select.data;` is ignored in cgi.
        if (select_syms.count(InScopeName(id->name))) return true;
        ret_name = id->name + "__buf__";
      } else {
        return true;
      }
    }
    for (auto& item : cgi->GetFunctionSymbols(fname)) {
      if (item.name == InScopeName(ret_name)) {
        if (auto val = FindOrNull(n.Note(), "host-type"))
          item.SetAsReturn(*val);
        else
          item.SetAsReturn("$");
      }
    }

    cgi->SetReturnSymbol(fname, InScopeName(ret_name));

    return true;
  }

  bool Visit(AST::LoopRange&) { return true; }
  bool Visit(AST::ForeachBlock&) { return true; }
  bool Visit(AST::FunctionDecl&) { return true; }
  bool Visit(AST::ChoreoFunction&) { return true; }
  bool Visit(AST::CppSourceCode&) { return true; }
  bool Visit(AST::Program&) { return true; }

  bool RunOnProgram(AST::Node& root) override {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }

    if (prt_visitor) dbgs() << "|- " << GetName() << NewL;

    if (prt_visitor) dbgs() << " |- Prepare" << NewL;
    root.accept(*this);
    if (HasError()) return false;

    if (prt_visitor) dbgs() << " |- Prepare2" << NewL;
    CodegenPrepareStage2 cps2(cgi);
    cps2.RunOnProgram(root);
    if (cps2.HasError()) return false;

    if (abend_after) return false;

    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_PREPARE_HPP__
