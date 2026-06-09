#include "fragment_layout_pass.hpp"
#include "context.hpp"
#include "types.hpp"

namespace Choreo {

// ---------------------------------------------------------------------------
// Thread count helpers
// ---------------------------------------------------------------------------

size_t FragmentLayoutPass::EffectiveThreadCount() const {
  if (current_inthreads_ && current_inthreads_->HasActiveThreads()) {
    auto at = VIInt(current_inthreads_->active_threads);
    return at.value_or(current_thread_count_);
  }
  return current_thread_count_;
}

std::string FragmentLayoutPass::EffectiveThreadCountExpr() const {
  if (current_inthreads_ && current_inthreads_->HasActiveThreads()) {
    auto at = VIInt(current_inthreads_->active_threads);
    if (at.has_value()) return std::to_string(at.value());
  }
  return current_thread_count_expr_;
}

// ---------------------------------------------------------------------------
// Scope tracking
// ---------------------------------------------------------------------------

bool FragmentLayoutPass::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) {
    usages_.clear();
    copy_edges_.clear();
    apply_refs_.clear();
    current_thread_count_ = 0;
    current_thread_count_expr_.clear();
    current_inthreads_ = nullptr;
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pb->GetLevel() == ParallelLevel::THREAD) {
      current_thread_count_ = 0;
      current_thread_count_expr_.clear();
      if (!pb->BoundValues().empty()) {
        if (auto v0 = VIInt(pb->BoundValues()[0])) {
          size_t tc = *v0;
          for (size_t i = 1; i < pb->BoundValues().size(); ++i)
            if (auto vi = VIInt(pb->BoundValues()[i])) tc *= *vi;
          current_thread_count_ = tc;
          current_thread_count_expr_ = std::to_string(tc);
        }
      }
    }
  } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    current_inthreads_ = it;
  }
  return true;
}

bool FragmentLayoutPass::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) {
    AssignLayouts();
    usages_.clear();
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pb->GetLevel() == ParallelLevel::THREAD) {
      current_thread_count_ = 0;
      current_thread_count_expr_.clear();
    }
  } else if (isa<AST::InThreadsBlock>(&n)) {
    current_inthreads_ = nullptr;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Phase 1: COLLECT
// ---------------------------------------------------------------------------

bool FragmentLayoutPass::Visit(AST::NamedVariableDecl& n) {
  if (!n.HasNote("fragment_decl")) return true;

  auto sty = GetSpannedType(GetSymbolType(n.name_str));
  if (!sty) return true;

  auto scoped = InScopeName(n.name_str);
  FragmentUsage usage;
  usage.scoped_name = scoped;
  usage.element_type = sty->ElementType();
  usage.scope_thread_count = EffectiveThreadCount();
  usage.scope_thread_count_expr = EffectiveThreadCountExpr();

  auto& shape = sty->GetShape();
  if (shape.IsValid()) {
    for (size_t d = 0; d < shape.Rank(); ++d) {
      if (auto v = VIInt(shape.ValueAt(d))) usage.shape.push_back(*v);
    }
  }

  usages_[scoped] = usage;
  return true;
}

bool FragmentLayoutPass::Visit(AST::MMA& n) {
  auto& op = *n.operation;

  if (op.IsKind(AST::MMAOperation::Exec)) {
    // ExecOperand order: 0=C(acc), 1=A, 2=B (matches GPUAdaptor convention).
    auto c_sym = InScopeName(AST::FragName(op.ExecOperand(0)));
    auto a_sym = InScopeName(AST::FragName(op.ExecOperand(1)));
    auto b_sym = InScopeName(AST::FragName(op.ExecOperand(2)));

    // MMA accumulators from mma.fill.f32 don't have NamedVariableDecl,
    // so they must be added to usages_ when first seen in an MMA exec.
    if (!usages_.count(c_sym)) {
      FragmentUsage usage;
      usage.scoped_name = c_sym;
      usage.scope_thread_count = EffectiveThreadCount();
      usage.scope_thread_count_expr = EffectiveThreadCountExpr();
      usages_[c_sym] = usage;
    }
    usages_[c_sym].is_mma_acc = true;

    if (usages_.count(a_sym)) usages_[a_sym].is_mma_operand_a = true;
    if (usages_.count(b_sym)) usages_[b_sym].is_mma_operand_b = true;
  }

  return true;
}

bool FragmentLayoutPass::Visit(AST::FragReduce& n) {
  auto dst_scoped = InScopeName(n.DstName());
  auto src_scoped = InScopeName(n.SrcName());

  if (usages_.count(dst_scoped)) {
    usages_[dst_scoped].is_reduce_target = true;
    usages_[dst_scoped].reduce_source = src_scoped;
  }

  return true;
}

bool FragmentLayoutPass::Visit(AST::FragTransfer& n) {
  if (n.op != AST::FragTransferKind::COPY) return true;
  auto dst_scoped = InScopeName(n.DstName());
  auto src_scoped = InScopeName(n.SrcName());
  copy_edges_.emplace_back(dst_scoped, src_scoped);
  return true;
}

void FragmentLayoutPass::CollectBodyRefs(const ptr<AST::Node>& node,
                                         std::vector<std::string>& refs) {
  if (!node) return;
  if (auto da = dyn_cast<AST::DataAccess>(node)) {
    if (da->AccessElement()) {
      auto sc = InScopeNameForRef(da->data->name);
      if (usages_.count(sc)) refs.push_back(sc);
    }
    return;
  }
  if (auto expr = dyn_cast<AST::Expr>(node)) {
    if (expr->GetL()) CollectBodyRefs(expr->GetL(), refs);
    if (expr->GetR()) CollectBodyRefs(expr->GetR(), refs);
    if (expr->GetC()) CollectBodyRefs(expr->GetC(), refs);
    return;
  }
  if (auto call = dyn_cast<AST::Call>(node)) {
    if (call->arguments)
      for (auto& a : call->arguments->AllValues()) CollectBodyRefs(a, refs);
    return;
  }
  if (auto cast_expr = dyn_cast<AST::CastExpr>(node)) {
    if (cast_expr->GetR()) CollectBodyRefs(cast_expr->GetR(), refs);
    return;
  }
}

bool FragmentLayoutPass::Visit(AST::ApplyBlock& n) {
  auto span_scoped = InScopeNameForRef(n.SpanFragmentName());
  std::vector<std::string> body_frags;

  // Walk the multi-statement body collecting all fragment .at() accesses.
  auto CollectFromBody = [&](auto&& self,
                             const ptr<AST::Node>& node) -> void {
    if (!node) return;
    if (auto da = dyn_cast<AST::DataAccess>(node)) {
      if (da->AccessElement()) {
        auto sc = InScopeNameForRef(da->data->name);
        if (usages_.count(sc)) body_frags.push_back(sc);
      }
      return;
    }
    if (auto mn = dyn_cast<AST::MultiNodes>(node)) {
      for (auto& child : mn->values) self(self, child);
      return;
    }
    if (auto asgn = dyn_cast<AST::Assignment>(node)) {
      if (asgn->da) self(self, asgn->da);
      if (asgn->value) self(self, asgn->value);
      return;
    }
    if (auto ie = dyn_cast<AST::IfElseBlock>(node)) {
      if (ie->pred) self(self, ie->pred);
      if (ie->stmts) self(self, ie->stmts);
      if (ie->else_stmts) self(self, ie->else_stmts);
      return;
    }
    if (auto expr = dyn_cast<AST::Expr>(node)) {
      if (expr->GetL()) self(self, expr->GetL());
      if (expr->GetR()) self(self, expr->GetR());
      if (expr->GetC()) self(self, expr->GetC());
      return;
    }
    if (auto call = dyn_cast<AST::Call>(node)) {
      if (call->arguments)
        for (auto& a : call->arguments->AllValues()) self(self, a);
      return;
    }
    if (auto cast_expr = dyn_cast<AST::CastExpr>(node)) {
      if (cast_expr->GetR()) self(self, cast_expr->GetR());
      return;
    }
  };
  CollectFromBody(CollectFromBody, n.body);

  // Register each body fragment as referencing the span fragment.
  // This enables layout propagation: e.g., a 1D fragment accessed inside
  // an apply over a 2D MMA accumulator gets REPLICATED_1D layout.
  for (auto& frag : body_frags) {
    if (frag == span_scoped) continue;
    auto& refs = apply_refs_[frag];
    refs.push_back(span_scoped);
  }

  return true;
}

// ---------------------------------------------------------------------------
// Phase 2 + 3: INFER and LEGALIZE
// ---------------------------------------------------------------------------

void FragmentLayoutPass::AssignLayouts() {
  // Build MMA accumulator layouts first (anchors).
  std::map<std::string, FragmentLayout> anchors;

  for (auto& [scoped, usage] : usages_) {
    if (!usage.is_mma_acc) continue;
    if (!FCtx(fname).FragHasMMAType(scoped)) continue;

    auto& mma_info = cgi.GetSymbolMMA(scoped);
    auto& mma_shape = mma_info.GetShape();
    if (mma_shape.size() < 2) continue;

    auto M_opt = VIInt(mma_shape[0]);
    auto N_opt = VIInt(mma_shape[1]);
    if (!M_opt || !N_opt) continue;
    size_t M = *M_opt, N = *N_opt;

    if (FCtx(fname).FragIsWGMMA(scoped)) {
      anchors[scoped] = FragmentLayout::MakeWGMMA_ACC(M, N);
    } else if (FCtx(fname).FragIsCTMMA(scoped)) {
      anchors[scoped] = FragmentLayout::MakeCTMMA_ACC(M, N);
    }
  }

  // Assign RS A operand layouts.
  for (auto& [scoped, usage] : usages_) {
    if (!usage.is_mma_operand_a) continue;
    if (anchors.count(scoped)) continue;
    if (!FCtx(fname).FragHasMMAType(scoped)) continue;

    if (FCtx(fname).FragIsWGMMA(scoped)) {
      if (usage.shape.size() >= 2) {
        anchors[scoped] =
            FragmentLayout::MakeWGMMA_RS_A(usage.shape[0], usage.shape[1]);
        continue;
      }
      auto& mma_info = cgi.GetSymbolMMA(scoped);
      auto& mma_shape = mma_info.GetShape();
      if (mma_shape.size() < 3) continue;
      auto M_opt = VIInt(mma_shape[0]);
      auto K_opt = VIInt(mma_shape[2]);
      if (!M_opt || !K_opt) continue;
      anchors[scoped] = FragmentLayout::MakeWGMMA_RS_A(*M_opt, *K_opt);
    }
  }

  // Store all inferred layouts.
  // First pass: assign anchors and UNIFORM fallbacks.
  for (auto& [scoped, usage] : usages_) {
    FragmentLayout fl;

    if (anchors.count(scoped)) {
      fl = anchors[scoped];
      fl.anchor_symbol = scoped;
    } else if (usage.is_reduce_target) {
      // Defer -- assigned after source layouts are known.
      continue;
    } else {
      // Fallback: UNIFORM with scope-aware thread count.
      size_t total = 1;
      for (auto d : usage.shape) total *= d;

      size_t tc = usage.scope_thread_count;
      std::string tc_expr = usage.scope_thread_count_expr;
      if (tc == 0) {
        tc = current_thread_count_;
        tc_expr = current_thread_count_expr_;
      }
      if (tc_expr.empty()) tc_expr = "blockDim.x";

      if (usage.shape.size() == 1) {
        fl = FragmentLayout::MakeUniform(total, tc, tc_expr);
      } else if (usage.shape.size() >= 2) {
        fl = FragmentLayout::MakeUniform2D(usage.shape[0], usage.shape[1], tc,
                                           tc_expr);
      } else {
        fl = FragmentLayout::MakeUniform(total, tc, tc_expr);
      }
    }

    FCtx(fname).SetFragmentLayout(scoped, fl);
  }

  // Second pass: assign REPLICATED_1D to reduce targets.
  for (auto& [scoped, usage] : usages_) {
    if (!usage.is_reduce_target) continue;
    if (FCtx(fname).HasFragmentLayout(scoped)) continue;

    auto& src_name = usage.reduce_source;
    if (FCtx(fname).HasFragmentLayout(src_name)) {
      auto& src_fl = FCtx(fname).GetFragmentLayout(src_name);
      auto fl = FragmentLayout::MakeReduceDst(src_fl);
      FCtx(fname).SetFragmentLayout(scoped, fl);
    } else {
      // Source layout unknown; fall back to UNIFORM 1D.
      size_t total = 1;
      for (auto d : usage.shape) total *= d;
      size_t tc = usage.scope_thread_count;
      std::string tc_expr = usage.scope_thread_count_expr;
      if (tc == 0) {
        tc = current_thread_count_;
        tc_expr = current_thread_count_expr_;
      }
      if (tc_expr.empty()) tc_expr = "blockDim.x";
      auto fl = FragmentLayout::MakeUniform(total, tc, tc_expr);
      FCtx(fname).SetFragmentLayout(scoped, fl);
    }
  }

  // Third pass: propagate layouts through frag.copy and frag.apply edges.
  // Fragments that got UNIFORM but should inherit REPLICATED_1D or MMA layout
  // from related fragments are re-assigned here (fixpoint iteration).
  bool changed = true;
  int max_iters = 10;
  while (changed && max_iters-- > 0) {
    changed = false;

    // Propagate through frag.copy: both operands share the same layout.
    for (auto& [dst, src] : copy_edges_) {
      if (!FCtx(fname).HasFragmentLayout(src)) continue;
      if (!FCtx(fname).HasFragmentLayout(dst)) continue;
      auto& src_fl = FCtx(fname).GetFragmentLayout(src);
      auto& dst_fl = FCtx(fname).GetFragmentLayout(dst);
      if (dst_fl.kind == LayoutKind::UNIFORM &&
          src_fl.kind != LayoutKind::UNIFORM) {
        FCtx(fname).SetFragmentLayout(dst, src_fl);
        changed = true;
      } else if (src_fl.kind == LayoutKind::UNIFORM &&
                 dst_fl.kind != LayoutKind::UNIFORM) {
        FCtx(fname).SetFragmentLayout(src, dst_fl);
        changed = true;
      }
    }

    // Propagate through frag.apply body references.
    for (auto& [target, refs] : apply_refs_) {
      if (!FCtx(fname).HasFragmentLayout(target)) continue;
      auto& tgt_fl = FCtx(fname).GetFragmentLayout(target);
      if (tgt_fl.kind != LayoutKind::UNIFORM) continue;

      for (auto& ref : refs) {
        if (ref == target) continue;
        if (!FCtx(fname).HasFragmentLayout(ref)) continue;
        auto& ref_fl = FCtx(fname).GetFragmentLayout(ref);
        if (ref_fl.kind == LayoutKind::UNIFORM) continue;

        // 1D target referencing a 2D MMA acc -> REPLICATED_1D.
        if (usages_.count(target) && usages_[target].shape.size() == 1 &&
            ref_fl.IsMMAAnchored()) {
          auto fl = FragmentLayout::MakeReduceDst(ref_fl);
          FCtx(fname).SetFragmentLayout(target, fl);
          changed = true;
          break;
        }
        // 2D target with same shape as MMA acc -> inherit WGMMA_ACC layout.
        if (usages_.count(target) && usages_[target].shape.size() == 2 &&
            ref_fl.kind == LayoutKind::WGMMA_ACC &&
            usages_[target].shape[0] == (size_t)ref_fl.logical_rows &&
            usages_[target].shape[1] == (size_t)ref_fl.logical_cols) {
          FCtx(fname).SetFragmentLayout(target, ref_fl);
          changed = true;
          break;
        }
        // Same-shape 1D fragments inherit REPLICATED_1D from peers.
        if (ref_fl.kind == LayoutKind::REPLICATED_1D && usages_.count(target) &&
            usages_[target].shape.size() == 1) {
          FCtx(fname).SetFragmentLayout(target, ref_fl);
          changed = true;
          break;
        }
      }
    }
  }

  // Debug logging for all layouts.
  for (auto& [scoped, usage] : usages_) {
    if (!FCtx(fname).HasFragmentLayout(scoped)) continue;
    auto& fl = FCtx(fname).GetFragmentLayout(scoped);

    if (debug_visit) {
      dbgs() << "[frag-layout] " << scoped << ": " << STR(fl.kind)
             << " regs=" << fl.regs_per_thread << " threads=" << fl.thread_count
             << " (" << fl.thread_count_expr << ")"
             << " shape=[" << fl.logical_rows;
      if (fl.logical_cols > 1) dbgs() << "," << fl.logical_cols;
      dbgs() << "]";
      if (fl.vec_width > 1) dbgs() << " vec_width=" << fl.vec_width;
      if (fl.IsMMAAnchored()) dbgs() << " anchor=" << fl.anchor_symbol;
      if (fl.kind == LayoutKind::WGMMA_ACC)
        dbgs() << " rows_per_thread=" << fl.rows_per_thread
               << " threads_per_row=" << fl.threads_per_row;
      if (fl.kind == LayoutKind::REPLICATED_1D)
        dbgs() << " replicate=" << fl.replicate;
      dbgs() << "\n";
    }
  }
}

} // namespace Choreo
