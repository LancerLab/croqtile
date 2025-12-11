#ifndef __CHOREO_CODEGEN_PREPARE_HPP__
#define __CHOREO_CODEGEN_PREPARE_HPP__

// This apply the type check and symbol table generation

#include "codegen.hpp"
#include "target_utils.hpp"

namespace Choreo {

struct FutureInfoCollect : public CodeGenerator {
private:
  ParallelLevel level = ParallelLevel::SEQ;

public:
  FutureInfoCollect() : CodeGenerator("cgp_stage_2") {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      level = ParallelLevel::SEQ;
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      assert(pb->GetLevel() - level == 1);
      ++level;
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ParallelBy>(&n)) { --level; }
    return true;
  }

  bool Visit(AST::DMA& n) override {
    if (n.future.empty() || (n.operation == ".any")) return true;
    if (level == ParallelLevel::BLOCK) {
      // the DMA is inside block-shared zone
      cgi.GetFunctionSharedFutures(fname).insert(InScopeName(n.future));
      VST_DEBUG(dbgs() << "Shared Future: " << InScopeName(n.future) << "\n");
    }
    if (level == ParallelLevel::GROUP) {
      // the DMA is inside warp-local zone
      cgi.GetFunctionLocalFutures(fname).insert(InScopeName(n.future));
      VST_DEBUG(dbgs() << "Local Future: " << InScopeName(n.future) << "\n");
    }
    return true;
  }
};

struct CodegenPrepareStage1 : public CodeGenerator {
private:
  int parallel_depth = 0;
  int max_parallel_depth = 0;

  // special case for `return select.data;`
  std::set<std::string> select_syms;

  AST::ParallelBy* cur_device_pb = nullptr;

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      parallel_depth = 0;
      cgi.GetFunctionTrait(fname).has_parallelby = false;
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (pb->IsOuter()) {
        cur_device_pb = pb;
        auto& tma_descs = cgi.GetTMADescs();
        tma_descs.emplace(pb, std::vector<TMADesc>{});
      }
      if (parallel_depth == 0 && cgi.GetFunctionTrait(fname).has_parallelby)
        cgi.GetFunctionTrait(fname).multiple_parallelby = true;
      parallel_depth++;
      assert(parallel_depth > max_parallel_depth);
      max_parallel_depth = parallel_depth;

      auto& lcs = cgi.GetFunctionLaunches(fname);

      // Add a new launch config
      if (parallel_depth == 1) {
        // represents the index of the current ParallelBy in cgi
        n.Note().insert_or_assign("outer_pb_idx", std::to_string(lcs.size()));
        lcs.push_back({});
      }

      // All the pb in a nested pb is explicitly specified with pb level.
      auto& lc = lcs.back();
      switch (pb->GetLevel()) {
      case ParallelLevel::BLOCK: lc.SetBlockCount(pb->BoundValues()); break;
      case ParallelLevel::GROUP: lc.SetGroupCount(pb->BoundValues()); break;
      case ParallelLevel::THREAD: lc.SetThreadCount(pb->BoundValues()); break;
      default:
        choreo_unreachable("The explicit parallel-by level " +
                           STR(pb->GetLevel()) + " is not supported.");
      }
    }
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      VST_DEBUG(dbgs() << "Symbols in " << fname << ":\n");
      VST_DEBUG(for (auto& item : cgi.GetFunctionSymbols(fname)) {
        dbgs() << " |- " << item.name << ", ty: " << PSTR(item.type)
               << ", is_return: "
               << (item.rty_str.empty() ? "no" : "yes(" + item.rty_str + ")")
               << ", index: " << item.p_index << "\n";
      });
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      n.Note().insert_or_assign("mxl", std::to_string(max_parallel_depth));
      VST_DEBUG(dbgs() << "max depth of `"; pb->InlinePrint(dbgs());
                dbgs() << "': " << max_parallel_depth << "\n");
      if (parallel_depth == 1) {
        VST_DEBUG(dbgs() << "\tGrid Dims: "
                         << cgi.GetFunctionLaunches(fname).back().block_count.x
                         << "\n");
        VST_DEBUG(dbgs() << "\tBlock Dims: "
                         << cgi.GetFunctionLaunches(fname).back().thread_count.x
                         << "\n");
        max_parallel_depth = 0;
      }
      parallel_depth--;

      if (pb->IsOuter()) cur_device_pb = nullptr;
    }
    return true;
  }

private:
  bool IsHost() const { return parallel_depth == 0; }

public:
  CodegenPrepareStage1() : CodeGenerator("cgp_stage_1") {}
  ~CodegenPrepareStage1() {}

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
    cgi.AddSymbolDetail(fname, {InScopeName(name), GetSymbolType(name), ref});
    if (isa<AST::Select>(n.init_expr)) select_syms.insert(InScopeName(name));
    return true;
  }

  bool Visit(AST::IntTuple&) { return true; }

  bool Visit(AST::Assignment& n) override {
    if (n.AssignToDataElement()) return true;
    auto name = n.GetName();
    bool ref = n.Note().count("ref");
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value)) {
      cgi.AddSymbolDetail(fname, {InScopeName(name), GetSymbolType(name), ref});
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
      cgi.AddSymbolDetail(fname,
                          {InScopeName(param->sym->name), param->GetType(),
                           param->pass_by_ref, index++, param->GetAttr()});
    }
    return true;
  }

  bool Visit(AST::ParallelBy&) override {
    cgi.GetFunctionTrait(fname).has_parallelby = true;
    return true;
  }

  bool Visit(AST::WhereBind&) { return true; }
  bool Visit(AST::WithIn&) { return true; }
  bool Visit(AST::WithBlock&) { return true; }
  bool Visit(AST::Memory&) { return true; }
  bool Visit(AST::SpanAs&) { return true; }
  bool Visit(AST::DMA& n) {
    if (n.IsTMA()) {
      cgi.GetFunctionTrait(fname).has_tma = true;
      cgi.GetModuleTrait().has_tma = true;
    }

    if (n.IsAsync() && !n.IsTMA())
      cgi.GetFunctionTrait(fname).has_async_dma = true;

    if (!CCtx().TargetSupportTMA()) return true;
    if (!cur_device_pb) return true; // not device dma

    auto fsty = GetSpannedType(n.GetFrom()->GetType());
    auto tsty = GetSpannedType(n.GetTo()->GetType());

    if (n.IsTMA()) {
      if (((fsty->GetStorage() == Storage::GLOBAL ||
            fsty->GetStorage() == Storage::DEFAULT) &&
           tsty->GetStorage() == Storage::SHARED) ||
          (fsty->GetStorage() == Storage::SHARED &&
           (tsty->GetStorage() == Storage::GLOBAL ||
            tsty->GetStorage() == Storage::DEFAULT))) {
        auto& tma_descs = cgi.GetTMADescs();
        tma_descs[cur_device_pb].emplace_back(
            n.GetFrom(), n.GetTo(), InScopeName(n.GetFrom()->RefSymbol()),
            InScopeName(n.GetTo()->RefSymbol()));
      } else
        choreo_unreachable(
            "unsupport TMA direction: " + STR(fsty->GetStorage()) + " => " +
            STR(tsty->GetStorage()) + ".");
    }

    return true;
  }

  bool Visit(AST::MMA& n) {
    auto& op = *n.GetOperation();
    ValueList mma_shape;
    switch (op.Tag()) {
    case AST::MMAOperation::Fill: break;
    case AST::MMAOperation::Load: break;
    case AST::MMAOperation::Exec: {
      auto& a_sym = op.ExecOperand(1);
      auto& b_sym = op.ExecOperand(2);
      auto& c_sym = op.ExecOperand(0);
      auto a_ty = GetSpannedType(GetSymbolType(a_sym));
      auto b_ty = GetSpannedType(GetSymbolType(b_sym));
      auto c_ty = GetSpannedType(GetSymbolType(c_sym));
      auto a_shape = a_ty->GetShape();
      auto b_shape = b_ty->GetShape();
      switch (op.GetMethod()) {
      case AST::MMAOperation::ROW_ROW:
        mma_shape.push_back(a_shape.ValueAt(0));
        mma_shape.push_back(b_shape.ValueAt(0));
        mma_shape.push_back(a_shape.ValueAt(1));
        break;
      case AST::MMAOperation::ROW_COL:
        mma_shape.push_back(a_shape.ValueAt(0));
        mma_shape.push_back(b_shape.ValueAt(1));
        mma_shape.push_back(a_shape.ValueAt(1));
        break;
      case AST::MMAOperation::COL_ROW:
        mma_shape.push_back(a_shape.ValueAt(1));
        mma_shape.push_back(b_shape.ValueAt(0));
        mma_shape.push_back(a_shape.ValueAt(0));
        break;
      case AST::MMAOperation::COL_COL:
        mma_shape.push_back(a_shape.ValueAt(1));
        mma_shape.push_back(b_shape.ValueAt(1));
        mma_shape.push_back(a_shape.ValueAt(0));
        break;
      default: choreo_unreachable("unsupported mma execution method.");
      }
      auto a_ety = a_ty->ElementType();
      auto b_ety = b_ty->ElementType();
      auto acc_ty = c_ty->ElementType();
      if (a_ety == BaseType::F32) a_ety = BaseType::TF32;
      if (b_ety == BaseType::F32) b_ety = BaseType::TF32;
      cgi.AddSymbolMMA(InScopeName(a_sym),
                       MMAInfo{a_ety, mma_shape, MMAInfo::FRAG_A});
      cgi.AddSymbolMMA(InScopeName(b_sym),
                       MMAInfo{b_ety, mma_shape, MMAInfo::FRAG_B});
      cgi.AddSymbolMMA(InScopeName(c_sym),
                       MMAInfo{acc_ty, mma_shape, MMAInfo::FRAG_C});
      VST_DEBUG(dbgs() << "mma type: " << STR(a_ety) << ", " << STR(b_ety)
                       << ", " << STR(acc_ty) << ", shape: " << STR(mma_shape)
                       << " -> " << a_sym << ", " << b_sym << ", " << c_sym
                       << "\n");
    } break;
    case AST::MMAOperation::Store: break;
    default: choreo_unreachable("unsupported mma operation.");
    }
    return true;
  }
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
    for (auto& item : cgi.GetFunctionSymbols(fname)) {
      if (item.name == InScopeName(ret_name)) {
        if (auto val = FindOrNull(n.Note(), "host-type"))
          item.SetAsReturn(*val);
        else
          item.SetAsReturn("$");
      }
    }

    cgi.SetReturnSymbol(fname, InScopeName(ret_name));

    return true;
  }

  bool Visit(AST::LoopRange&) { return true; }
  bool Visit(AST::ForeachBlock&) { return true; }
  bool Visit(AST::FunctionDecl&) { return true; }
  bool Visit(AST::ChoreoFunction&) { return true; }
  bool Visit(AST::CppSourceCode&) { return true; }
  bool Visit(AST::Program&) { return true; }
};

class CodegenPrepare : public VisitorGroup {
private:
  CodegenPrepareStage1 s1;
  FutureInfoCollect s2;

public:
  CodegenPrepare() : VisitorGroup("prepare", s1, s2) {}
};

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_PREPARE_HPP__
