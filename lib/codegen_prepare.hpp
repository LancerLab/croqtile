#ifndef __CHOREO_CODEGEN_PREPARE_HPP__
#define __CHOREO_CODEGEN_PREPARE_HPP__

// This apply the type check and symbol table generation

#include "codegen.hpp"
#include "target_utils.hpp"

namespace Choreo {

struct FutureAnalysis : public CodeGenerator {
private:
  ParallelLevel level = ParallelLevel::SEQ;
  std::vector<ParallelLevel> pl_stack;

public:
  FutureAnalysis() : CodeGenerator("futanly") {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      pl_stack.push_back(ParallelLevel::SEQ);
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      pl_stack.push_back(pb->GetLevel());
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ParallelBy>(&n)) { pl_stack.pop_back(); }
    return true;
  }

  bool Visit(AST::DMA& n) override {
    if (n.future.empty() || (n.operation == ".any")) return true;
    auto level = pl_stack.back();
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

struct CodegenInfoCollect : public CodeGenerator {
private:
  // special case for `return select.data;`
  std::set<std::string> select_syms;

  AST::ParallelBy* block_pb = nullptr;
  ParallelLevel inner_pb_level = ParallelLevel::BLOCK;
  std::vector<AST::ParallelBy*> pb_stack;
  std::stack<AST::InThreadsBlock*> in_thr_block_stack;
  std::set<std::string> block_tma_futures;

private:
  auto Level() const {
    if (pb_stack.empty()) return ParallelLevel::SEQ;
    return pb_stack.back()->GetLevel();
  }

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      pb_stack.clear();
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      auto pb_level = pb->GetLevel();
      if (pb_level == ParallelLevel::CLUSTER) {
        assert(pb_stack.empty());
        auto& tma_descs = cgi.GetTMADescs();
        tma_descs.emplace(pb, std::vector<TMADesc>{});
      } else if (pb_level == ParallelLevel::BLOCK) {
        block_pb = pb;
        assert(pb_stack.empty() ||
               (pb_stack.size() == 1 &&
                pb_stack.back()->GetLevel() == ParallelLevel::CLUSTER));
        auto& tma_descs = cgi.GetTMADescs();
        tma_descs.emplace(pb, std::vector<TMADesc>{});
      } else {
        if (pb_level == ParallelLevel::GROUP ||
            pb_level == ParallelLevel::GROUPx4) {
          if (pb->IsEnforced()) inner_pb_level = pb_level;
        }
        assert(!pb_stack.empty());
      }
      pb_stack.push_back(pb);
      if (pb_stack.size() > 1) {
        cgi.GetPBTree(fname).AddChild(*(pb_stack.rbegin() + 1),
                                      pb_stack.back());
      } else
        cgi.GetPBTree(fname).AddSingle(pb);

      auto& lcs = cgi.GetFunctionLaunches(fname);

      if (pb_level == ParallelLevel::CLUSTER) {
        lcs.push_back({});
      } else if (pb_level == ParallelLevel::BLOCK) {
        if (lcs.empty() || !lcs.back().HasCluster()) lcs.push_back({});
      }

      // Set the launch configure
      auto& lc = lcs.back();
      switch (pb->GetLevel()) {
      case ParallelLevel::CLUSTER: lc.SetClusterCount(pb->BoundValues()); break;
      case ParallelLevel::BLOCK: lc.SetBlockCount(pb->BoundValues()); break;
      case ParallelLevel::GROUP: lc.SetGroupCount(pb->BoundValues()); break;
      case ParallelLevel::GROUPx4: lc.SetGroupx4Count(pb->BoundValues()); break;
      case ParallelLevel::THREAD: lc.SetThreadCount(pb->BoundValues()); break;
      default:
        choreo_unreachable("The explicit parallel-by level " +
                           STR(pb->GetLevel()) + " is not supported.");
      }
    } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
      if (inner_pb_level == ParallelLevel::GROUPx4 ||
          inner_pb_level == ParallelLevel::GROUP)
        in_thr_block_stack.push(it);
      // Mark warpspec pattern when inthreads.async selects threads from
      // a single warpgroup. Uses ActiveWarpGroup() to handle compound
      // predicates like (p1 == 0 && t == 0).
      if (it->async && it->HasScopeThreadMask() &&
          it->inthreads_level == ParallelLevel::GROUPx4) {
        auto wg = it->ActiveWarpGroup();
        if (wg >= 0) cgi.GetFunctionTrait(fname).has_warpspec_pattern = true;
      }
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      cgi.GetFunctionTrait(fname).has_parallelby =
          !cgi.GetPBTree(fname).IsEmpty();
      cgi.GetFunctionTrait(fname).multiple_parallelby =
          (cgi.GetPBTree(fname).GetRootCount() > 1);
      ValidateEventDecls();
      VST_DEBUG(dbgs() << "Symbols in " << fname << ":\n");
      VST_DEBUG(for (auto& item : cgi.GetFunctionSymbols(fname)) {
        dbgs() << " |- " << item.name << ", ty: " << PSTR(item.type)
               << ", is_return: "
               << (item.rty_str.empty() ? "no" : "yes(" + item.rty_str + ")")
               << ", index: " << item.p_index << "\n";
      });
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      auto lvl = pb->GetLevel();
      assert(lvl == Level());
      pb_stack.pop_back();
      if (lvl == ParallelLevel::BLOCK) {
        assert(pb_stack.empty() ||
               (pb_stack.size() == 1 &&
                pb_stack.back()->GetLevel() == ParallelLevel::CLUSTER));
        auto& children = cgi.GetPBTree(fname).GetChildren(pb);
        if (children.size() >= 2) {
          // check if there are multiple compatible branches
          std::vector<AST::ParallelBy*> leaves;
          std::map<AST::ParallelBy*, ValueList> pcs;
          std::deque<AST::ParallelBy*> work_list;
          work_list.push_back(pb);
          while (!work_list.empty()) {
            auto cpb = work_list.back();
            work_list.pop_back();
            pcs.emplace(cpb, cpb->BoundValues());
            if (cgi.GetPBTree(fname).IsLeaf(cpb))
              leaves.push_back(cpb);
            else
              for (auto& child : cgi.GetPBTree(fname).GetChildren(cpb))
                work_list.push_front(child);
          }
          auto xyz = [this, &pcs](AST::ParallelBy* leaf,
                                  AST::ParallelBy* root) {
            assert(leaf != nullptr);
            assert(root != nullptr);
            assert(leaf != root);
            ValueItem acc_x = sbe::nu(1), acc_y = sbe::nu(1),
                      acc_z = sbe::nu(1);
            auto cnode = leaf;
            while (cnode != root) {
              assert(pcs[cnode].size() > 0);
              acc_x = (acc_x * pcs[cnode][0])->Normalize();
              if (pcs[cnode].size() > 1)
                acc_y = (acc_y * pcs[cnode][1])->Normalize();
              if (pcs[cnode].size() > 2)
                acc_z = (acc_z * pcs[cnode][2])->Normalize();
              cnode = cgi.GetPBTree(fname).GetParent(cnode);
            }
            return acc_x * acc_y * acc_z;
          };
          // compatible: the muliplication of launch parameters equals
          assert(!leaves.empty());
          auto bleaf = *leaves.begin();
          auto bcount = xyz(bleaf, pb);
          for (auto itr = leaves.begin() + 1; itr != leaves.end(); ++itr) {
            auto bc = xyz(*itr, pb);
            if (!sbe::ceq(bcount, bc))
              Error1((*itr)->LOC(), "mulitple inner parallel-bys must have "
                                    "compatible block dimension " +
                                        STR(bcount) + " != " + STR(bc) + ".");
          }
        }
        auto& lcs = cgi.GetFunctionLaunches(fname);
        VST_DEBUG(dbgs() << "\tBlock Count: " << lcs.back().block_count
                         << "\n");
        VST_DEBUG(dbgs() << "\tGroup-4 Count: " << lcs.back().group4_count
                         << "\n");
        VST_DEBUG(dbgs() << "\tGroup Count: " << lcs.back().group_count
                         << "\n");
        VST_DEBUG(dbgs() << "\tThread Count: " << lcs.back().thread_count
                         << "\n");
        block_pb = nullptr;
      }
    } else if (isa<AST::InThreadsBlock>(&n)) {
      if (inner_pb_level == ParallelLevel::GROUPx4 ||
          inner_pb_level == ParallelLevel::GROUP)
        in_thr_block_stack.pop();
    }
    return true;
  }

private:
  bool IsHost() const { return Level() == ParallelLevel::SEQ; }

  void ValidateEventDecls() {
    auto& ft = cgi.GetFunctionTrait(fname);
    for (auto& decl : ft.event_decls) {
      auto ep = ft.GetEventParticipation(decl.name);
      if (decl.explicit_tc > 0 && ep > 0 && decl.explicit_tc != ep) {
        Warning(decl.loc,
                "event '" + decl.name + "' has explicit thread count " +
                    std::to_string(decl.explicit_tc) +
                    " but computed participation from usage scopes is " +
                    std::to_string(ep) +
                    "; the explicit count takes precedence.");
      }
      if (ep == 0) {
        Note(decl.loc, "event '" + decl.name +
                           "' has no recorded thread participation; "
                           "it may not be used in any inthreads scope.");
      }
    }
  }

  std::string ExtractEventName(AST::Node& node) const {
    auto expr = dyn_cast<AST::Expr>(&node);
    if (!expr) return "";
    if (expr->op == Op::ElemOf) {
      auto bid = AST::GetArrayBaseSymbol(*expr);
      return bid ? bid->name : "";
    }
    auto sym = expr->GetSymbol();
    return sym ? sym->name : "";
  }

  void RecordEventRef(const std::string& event_name, const location& loc) {
    if (in_thr_block_stack.empty()) return;
    auto* it = in_thr_block_stack.top();
    if (!it->HasScopeThreadMask()) {
      Note(loc, "event '" + event_name +
                    "' is used in a scope with dynamic thread predicate; "
                    "participation count cannot be determined at compile "
                    "time, runtime default will be used for initialization.");
      return;
    }
    cgi.GetFunctionTrait(fname).RecordEventUsage(event_name,
                                                 it->GetScopeThreadMask());
  }

public:
  CodegenInfoCollect() : CodeGenerator("cg_info") {}
  ~CodegenInfoCollect() {}

  bool Visit(AST::NamedVariableDecl& n) override {
    auto name = n.name_str;
    bool ref = n.HasNote("ref");
    cgi.AddSymbolDetail(fname, {InScopeName(name), GetSymbolType(name), ref});
    if (isa<AST::Select>(n.init_expr)) select_syms.insert(InScopeName(name));

    auto sty = GetSymbolType(name);
    if (auto ety = dyn_cast<EventArrayType>(sty))
      cgi.GetFunctionTrait(fname).RecordEventDecl(
          name, ety->event->GetThreadCount(), n.LOC());
    else if (auto ety = dyn_cast<EventType>(sty))
      cgi.GetFunctionTrait(fname).RecordEventDecl(name, ety->GetThreadCount(),
                                                  n.LOC());
    return true;
  }

  bool Visit(AST::Assignment& n) override {
    if (n.AssignToDataElement()) return true;
    auto name = n.GetName();
    bool ref = n.HasNote("ref");
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value)) {
      cgi.AddSymbolDetail(fname, {InScopeName(name), GetSymbolType(name), ref});
      if (isa<AST::Select>(n.value)) select_syms.insert(InScopeName(name));
    }
    return true;
  }

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

  bool Visit(AST::DMA& n) override {
    auto future_name = n.future;

    if (n.IsDummy()) {
      if (n.IsTMA()) {
        auto in_thr_block =
            (in_thr_block_stack.empty() ? nullptr : in_thr_block_stack.top());
        if (!in_thr_block) {
          // this future of tma is define in block scope, so it is shared among
          // threads in the block. record it in block_tma_futures to distinguish
          // with other futures of tma defined in group or warpgroup scope.
          block_tma_futures.insert(future_name);
        }
      }
      return true;
    }

    if (n.IsTMA()) {
      cgi.GetFunctionTrait(fname).has_tma = true;
      cgi.GetModuleTrait().has_tma = true;
    }

    if (n.IsAsync() && !n.IsTMA())
      cgi.GetFunctionTrait(fname).has_async_dma = true;

    if (n.HasEvent()) {
      auto name = ExtractEventName(*n.Event());
      if (!name.empty()) RecordEventRef(name, n.LOC());
    }

    if (!n.IsTMA() && !in_thr_block_stack.empty()) {
      auto* itb = in_thr_block_stack.top();
      if (itb->HasScopeThreadMask()) {
        int64_t unit_size = -1;
        if (inner_pb_level == ParallelLevel::GROUPx4)
          unit_size = 128;
        else if (inner_pb_level == ParallelLevel::GROUP)
          unit_size = 32;
        if (unit_size > 0) {
          auto& mask = itb->GetScopeThreadMask();
          for (int64_t s = 0; s < (int64_t)mask.size(); s += unit_size) {
            int64_t cnt = 0;
            auto e = std::min(s + unit_size, (int64_t)mask.size());
            for (int64_t i = s; i < e; ++i)
              if (mask[i]) ++cnt;
            if (cnt > 0 && cnt != unit_size) {
              Warning(n.LOC(), "DMA inside inthreads has " +
                                   std::to_string(cnt) +
                                   " active threads in a unit, but tiled copy "
                                   "assumes " +
                                   std::to_string(unit_size) +
                                   " threads; data layout may be incorrect.");
              break;
            }
          }
        }
      }
    }

    if (!CCtx().TargetSupportTMA()) return true;
    if (!block_pb) return true; // not device dma

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
        auto tma_desc = TMADesc(n.GetSrc(), n.GetDst(),
                                InScopeName(n.GetSrc()->RefSymbol()),
                                InScopeName(n.GetDst()->RefSymbol()),
                                n.GetSwizzleMode(), inner_pb_level,
                                n.GetL2PromoteBytes());
        auto in_thr_block =
            (in_thr_block_stack.empty() ? nullptr : in_thr_block_stack.top());
        bool is_block_scope = block_tma_futures.count(future_name) > 0;
        if (is_block_scope) {
          // now this tma may enter a inthreads block, but it is still shared
          // among threads in the block, so we still treat it as block scope and
          // set in_thr_block to nullptr.
          in_thr_block = nullptr;
        }
        tma_desc.SetInThreadsBlock(in_thr_block);
        tma_descs.at(block_pb).push_back(tma_desc);
      } else
        choreo_unreachable(
            "unsupport TMA direction: " + STR(fsty->GetStorage()) + " => " +
            STR(tsty->GetStorage()) + ".");
    }

    return true;
  }

  bool Visit(AST::Trigger& n) override {
    for (auto& f : n.GetEvents()) {
      auto name = ExtractEventName(*f);
      if (!name.empty()) RecordEventRef(name, n.LOC());
    }
    return true;
  }

  bool Visit(AST::Wait& n) override {
    for (auto& t : n.GetTargets()) {
      auto name = ExtractEventName(*t);
      if (!name.empty()) RecordEventRef(name, n.LOC());
    }
    return true;
  }

  bool Visit(AST::MMA& n) override {
    auto& op = *n.GetOperation();
    ValueList mma_shape;
    switch (op.Tag()) {
    case AST::MMAOperation::Fill: break;
    case AST::MMAOperation::Load: break;
    case AST::MMAOperation::Exec: {
      auto& a_sym = AST::FragName(op.ExecOperand(1));
      auto& b_sym = AST::FragName(op.ExecOperand(2));
      auto& c_sym = AST::FragName(op.ExecOperand(0));
      auto a_ty = GetSpannedType(GetSymbolType(a_sym));
      auto b_ty = GetSpannedType(GetSymbolType(b_sym));
      auto c_ty = GetSpannedType(GetSymbolType(c_sym));
      auto a_shape = a_ty->GetShape();
      auto b_shape = b_ty->GetShape();
      switch (op.GetMethod()) {
      case AST::MMAOperation::ROW_ROW:
        mma_shape.push_back(a_shape.ValueAt(0));
        mma_shape.push_back(b_shape.ValueAt(0));
        if (op.IsSparse())
          mma_shape.push_back(a_shape.ValueAt(1) * sbe::nu(2));
        else
          mma_shape.push_back(a_shape.ValueAt(1));
        break;
      case AST::MMAOperation::ROW_COL:
        mma_shape.push_back(a_shape.ValueAt(0));
        mma_shape.push_back(b_shape.ValueAt(1));
        if (op.IsSparse())
          mma_shape.push_back(a_shape.ValueAt(1) * sbe::nu(2));
        else
          mma_shape.push_back(a_shape.ValueAt(1));
        break;
      case AST::MMAOperation::COL_ROW:
        mma_shape.push_back(a_shape.ValueAt(1));
        mma_shape.push_back(b_shape.ValueAt(0));
        if (op.IsSparse())
          mma_shape.push_back(a_shape.ValueAt(0) * sbe::nu(2));
        else
          mma_shape.push_back(a_shape.ValueAt(0));
        break;
      case AST::MMAOperation::COL_COL:
        mma_shape.push_back(a_shape.ValueAt(1));
        mma_shape.push_back(b_shape.ValueAt(1));
        if (op.IsSparse())
          mma_shape.push_back(a_shape.ValueAt(0) * sbe::nu(2));
        else
          mma_shape.push_back(a_shape.ValueAt(0));
        break;
      default: choreo_unreachable("unsupported mma execution method.");
      }
      auto a_ety = a_ty->ElementType();
      auto b_ety = b_ty->ElementType();
      auto acc_ty = c_ty->ElementType();
      if (a_ety == BaseType::F32) a_ety = BaseType::TF32;
      if (b_ety == BaseType::F32) b_ety = BaseType::TF32;
      cgi.AddSymbolMMA(
          InScopeName(a_sym),
          MMAInfo{a_ety, mma_shape, MMAInfo::FRAG_A, op.GetMethod()});
      cgi.AddSymbolMMA(
          InScopeName(b_sym),
          MMAInfo{b_ety, mma_shape, MMAInfo::FRAG_B, op.GetMethod()});
      cgi.AddSymbolMMA(
          InScopeName(c_sym),
          MMAInfo{acc_ty, mma_shape, MMAInfo::FRAG_C, op.GetMethod()});

      if (op.IsSparse() && op.ExecOperand(3)) {
        auto e_sym = AST::FragName(op.ExecOperand(3));
        auto e_ty = GetSpannedType(GetSymbolType(e_sym));
        cgi.AddSymbolMMA(InScopeName(e_sym),
                         MMAInfo{e_ty->ElementType(), mma_shape,
                                 MMAInfo::FRAG_E, op.GetMethod()});
      }

      VST_DEBUG(dbgs() << "mma type: " << STR(a_ety) << ", " << STR(b_ety)
                       << ", " << STR(acc_ty) << ", shape: " << STR(mma_shape)
                       << " -> " << a_sym << ", " << b_sym << ", " << c_sym
                       << (op.IsSparse() ? ", " + PSTR(op.ExecOperand(3)) : "")
                       << "\n");
    } break;
    case AST::MMAOperation::Scale: break;
    case AST::MMAOperation::Store: break;
    case AST::MMAOperation::Commit: break;
    case AST::MMAOperation::Wait: break;
    default: choreo_unreachable("unsupported mma operation.");
    }
    return true;
  }

  bool Visit(AST::Return& n) override {
    std::string ret_name;
    if (auto id = GetIdentifier(*n.value); id) {
      ret_name = id->name;
    } else {
      if (auto expr = dyn_cast<AST::Expr>(n.value);
          expr && (expr->op == Op::DataOf || expr->op == Op::MDataOf)) {
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
        if (n.HasNote("host-type"))
          item.SetAsReturn(n.GetNote("host-type"));
        else
          item.SetAsReturn("$");
      }
    }

    cgi.SetReturnSymbol(fname, InScopeName(ret_name));

    return true;
  }
};

class CodegenPrepare : public VisitorGroup {
private:
  CodegenInfoCollect cic;
  FutureAnalysis fa;

public:
  CodegenPrepare() : VisitorGroup("codegen", cic, fa) {}
};

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_PREPARE_HPP__
