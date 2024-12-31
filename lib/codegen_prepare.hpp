#ifndef __CHOREO_CODEGEN_PREPARE_HPP__
#define __CHOREO_CODEGEN_PREPARE_HPP__

// This apply the type check and symbol table generation

#include "codegen.hpp"

namespace Choreo {

struct CodegenPrepare : public CodeGenerator {
private:
  ptr<CodeGenInfo> cgi;
  int parallel_level = 0;

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      parallel_level = 0;
      cgi->GetFunctionTrait(fname).has_parallelby = false;
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      parallel_level++;
      if (CCtx().GetTarget() == CompileTarget::Factor) {
        // for Factor backend
        auto& lcs = cgi->GetFactorFunctionLaunches(fname);
        if (parallel_level == 1) {
          // represents the index of the current ParallelBy in cgi
          n.note += std::to_string(lcs.size()) + ", ";
          lcs.push_back({});
          lcs.back().SetBlockDims(pb->BoundValues());
        } else if (parallel_level == 2) {
          auto& lc = lcs.back();
          lc.OverwriteGDimsByBDims();
          lc.ResetBDims();
          lc.SetBlockDims(pb->BoundValues());
        } else
          choreo_unreachable("The parallel-by level " +
                             std::to_string(parallel_level) +
                             " is not supported.");
      } else {
        auto& lc = cgi->GetFunctionLaunch(fname);
        if (parallel_level == 1)
          lc.SetBlockDims(pb->BoundValues());
        else if (parallel_level == 2) {
          lc.OverwriteGDimsByBDims();
          lc.ResetBDims();
          lc.SetBlockDims(pb->BoundValues());
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
      if (CCtx().GetTarget() == CompileTarget::Factor) {
        if (parallel_level == 1) {
          VST_DEBUG(
              dbgs() << "\tGrid Dims: "
                     << cgi->GetFactorFunctionLaunches(fname).back().grid_dim_x
                     << "\n");
          VST_DEBUG(
              dbgs() << "\tBlock Dims: "
                     << cgi->GetFactorFunctionLaunches(fname).back().block_dim_x
                     << "\n");
        }
        parallel_level--;
      } else {
        parallel_level--;
        VST_DEBUG(dbgs() << "Grid Dims: "
                         << cgi->GetFunctionLaunch(fname).grid_dim_x);
        VST_DEBUG(dbgs() << "Block Dims: "
                         << cgi->GetFunctionLaunch(fname).block_dim_x);
      }
    }
    return true;
  }

public:
  CodegenPrepare() : CodeGenerator("prepare", CCtx().GetGlobalSymbolTable()) {
    cgi = std::make_shared<CodeGenInfo>();
  }
  ~CodegenPrepare() {}

  const ptr<CodeGenInfo> GetASTInfo() { return cgi; }

  bool Visit(AST::MultiNodes&) { return true; }
  bool Visit(AST::MultiValues&) { return true; }
  bool Visit(AST::IntLiteral&) { return true; }
  bool Visit(AST::Boolean&) { return true; }
  bool Visit(AST::Expr&) { return true; }
  bool Visit(AST::MultiDimSpans&) { return true; }
  bool Visit(AST::NamedTypeDecl&) { return true; }

  bool Visit(AST::NamedVariableDecl& n) override {
    auto name = n.name_str;
    bool ref = (n.GetNote().find("ref") != std::string::npos);
    cgi->AddSymbolDetail(fname, {InScopeName(name), GetSymbolType(name), ref});
    return true;
  }

  bool Visit(AST::IntTuple&) { return true; }

  bool Visit(AST::Assignment& n) override {
    auto name = n.name;
    bool ref = (n.GetNote().find("ref") != std::string::npos);
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value)) {
      cgi->AddSymbolDetail(fname,
                           {InScopeName(name), GetSymbolType(name), ref});
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
      cgi->AddSymbolDetail(fname, {InScopeName(param->sym->name),
                                   param->GetType(), false, index++});
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
    auto id = GetIdentifier(*n.value);
    if (!id) return true;
    for (auto& item : cgi->GetFunctionSymbols(fname)) {
      if (item.name == InScopeName(id->name)) {
        auto rty_str = RemovePrefixOrNull("host-type:", n.GetNote());
        if (rty_str.has_value())
          item.SetAsReturn(STR(rty_str.value()));
        else
          item.SetAsReturn("$");
      }
    }

    cgi->SetReturnSymbol(fname, InScopeName(id->name));

    return true;
  }

  bool Visit(AST::LoopRange&) { return true; }
  bool Visit(AST::ForeachBlock&) { return true; }
  bool Visit(AST::FunctionDecl&) { return true; }
  bool Visit(AST::ChoreoFunction&) { return true; }
  bool Visit(AST::CppSourceCode&) { return true; }
  bool Visit(AST::Program&) { return true; }
};

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_PREPARE_HPP__
