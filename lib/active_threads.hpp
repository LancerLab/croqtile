#ifndef __CHOREO_ACTIVE_THREADS_HPP__
#define __CHOREO_ACTIVE_THREADS_HPP__

#include "visitor.hpp"
#include <map>
#include <optional>
#include <variant>
#include <vector>

namespace Choreo {

// Computes a bool[threadDim] mask for each scope, tracking exactly which
// threads are active. The mask is propagated to all nested Block scopes.
struct ActiveThreadsAnalysis : public VisitorWithScope {
private:
  struct PVInfo {
    std::string name;
    ParallelLevel level;
    int64_t bound; // compile-time bound, -1 if unknown
  };

  struct PBEntry {
    AST::ParallelBy* pb;
    ParallelLevel level;
    std::vector<PVInfo> pvs;
  };

  std::vector<PBEntry> pb_stack;
  // Stack of masks from enclosing inthreads blocks.
  std::vector<std::vector<bool>> mask_stack;
  int64_t thread_dim = -1;

  static constexpr int64_t THREADS_PER_WARPGROUP = 128;
  static constexpr int64_t THREADS_PER_WARP = 32;

  int64_t ThreadsPerUnit(ParallelLevel level) const {
    switch (level) {
    case ParallelLevel::GROUPx4: return THREADS_PER_WARPGROUP;
    case ParallelLevel::GROUP: return THREADS_PER_WARP;
    case ParallelLevel::THREAD: return 1;
    default: return 0;
    }
  }

  bool IsThreadLevel(ParallelLevel level) const {
    return level == ParallelLevel::GROUPx4 || level == ParallelLevel::GROUP ||
           level == ParallelLevel::THREAD;
  }

  // Compute thread_dim from the outermost thread-contributing parallel level.
  void UpdateThreadDim(ParallelLevel level, int64_t bound) {
    if (thread_dim > 0) return; // already set by outer level
    auto tpu = ThreadsPerUnit(level);
    if (tpu > 0 && bound > 0) thread_dim = bound * tpu;
  }

  std::string FindPVInExpr(AST::Node* node) const {
    if (!node) return "";
    if (auto id = dyn_cast<AST::Identifier>(node)) {
      for (auto it = pb_stack.rbegin(); it != pb_stack.rend(); ++it)
        for (auto& pv : it->pvs)
          if (pv.name == id->name) return id->name;
      return "";
    }
    if (auto expr = dyn_cast<AST::Expr>(node))
      if (expr->GetForm() == AST::Expr::Reference)
        return FindPVInExpr(expr->GetR().get());
    return "";
  }

  std::optional<int64_t> ExtractConstant(AST::Node* node) const {
    if (!node) return std::nullopt;
    if (auto lit = dyn_cast<AST::IntLiteral>(node))
      return std::visit(
          [](auto v) -> int64_t { return static_cast<int64_t>(v); },
          lit->value);
    if (auto expr = dyn_cast<AST::Expr>(node)) {
      if (expr->GetForm() == AST::Expr::Reference)
        return ExtractConstant(expr->GetR().get());
      if (expr->Opts().HasVal()) return VIInt(expr->Opts().GetVal());
    }
    return std::nullopt;
  }

  const PVInfo* LookupPV(const std::string& name) const {
    for (auto it = pb_stack.rbegin(); it != pb_stack.rend(); ++it)
      for (auto& pv : it->pvs)
        if (pv.name == name) return &pv;
    return nullptr;
  }

  // Compute the value of parallel variable `pv` for thread index `tid`.
  // pv_value = (tid / ThreadsPerUnit(pv.level)) % pv.bound
  int64_t PVValueForTid(const PVInfo& pv, int64_t tid) const {
    auto stride = ThreadsPerUnit(pv.level);
    if (stride <= 0 || pv.bound <= 0) return -1;
    return (tid / stride) % pv.bound;
  }

  // Evaluate a comparison: lhs_val op rhs_val
  bool EvalCompare(Opcode::Kind op, int64_t lhs, int64_t rhs) const {
    switch (op) {
    case Opcode::Kind::Eq: return lhs == rhs;
    case Opcode::Kind::Ne: return lhs != rhs;
    case Opcode::Kind::Lt: return lhs < rhs;
    case Opcode::Kind::Le: return lhs <= rhs;
    case Opcode::Kind::Gt: return lhs > rhs;
    case Opcode::Kind::Ge: return lhs >= rhs;
    default: return true;
    }
  }

  enum class EvalResult { TRUE, FALSE, UNPREDICTABLE };

  // Evaluate predicate for a specific thread index.
  EvalResult EvaluateForTid(AST::Expr* pred, int64_t tid) const {
    if (!pred) return EvalResult::UNPREDICTABLE;
    auto kind = pred->GetOp().GetKind();

    // Comparison: pv op constant
    if (kind == Opcode::Kind::Eq || kind == Opcode::Kind::Ne ||
        kind == Opcode::Kind::Lt || kind == Opcode::Kind::Le ||
        kind == Opcode::Kind::Gt || kind == Opcode::Kind::Ge) {

      auto* lhs = pred->GetL().get();
      auto* rhs = pred->GetR().get();
      std::string pv_name = FindPVInExpr(lhs);
      auto constant = ExtractConstant(rhs);
      auto actual_kind = kind;

      if (pv_name.empty()) {
        pv_name = FindPVInExpr(rhs);
        constant = ExtractConstant(lhs);
        if (!pv_name.empty() && constant.has_value()) {
          switch (kind) {
          case Opcode::Kind::Lt: actual_kind = Opcode::Kind::Gt; break;
          case Opcode::Kind::Le: actual_kind = Opcode::Kind::Ge; break;
          case Opcode::Kind::Gt: actual_kind = Opcode::Kind::Lt; break;
          case Opcode::Kind::Ge: actual_kind = Opcode::Kind::Le; break;
          default: break;
          }
        }
      }

      if (!pv_name.empty() && constant.has_value()) {
        auto* pv = LookupPV(pv_name);
        if (pv && pv->bound > 0) {
          int64_t pv_val = PVValueForTid(*pv, tid);
          return EvalCompare(actual_kind, pv_val, constant.value())
                     ? EvalResult::TRUE
                     : EvalResult::FALSE;
        }
      }

      // Check for modulo pattern: (pv % M) op C
      auto lhs_expr = dyn_cast<AST::Expr>(lhs);
      if (lhs_expr && lhs_expr->GetOp().GetKind() == Opcode::Kind::Mod) {
        pv_name = FindPVInExpr(lhs_expr->GetL().get());
        auto mod_val = ExtractConstant(lhs_expr->GetR().get());
        constant = ExtractConstant(rhs);
        if (!pv_name.empty() && mod_val.has_value() && constant.has_value() &&
            mod_val.value() > 0) {
          auto* pv = LookupPV(pv_name);
          if (pv && pv->bound > 0) {
            int64_t pv_val = PVValueForTid(*pv, tid);
            int64_t mod_result = pv_val % mod_val.value();
            return EvalCompare(kind, mod_result, constant.value())
                       ? EvalResult::TRUE
                       : EvalResult::FALSE;
          }
        }
      }

      return EvalResult::UNPREDICTABLE;
    }

    // Logical AND
    if (kind == Opcode::Kind::LogicAnd) {
      auto lhs_r = EvaluateForTid(dyn_cast<AST::Expr>(pred->GetL().get()), tid);
      auto rhs_r = EvaluateForTid(dyn_cast<AST::Expr>(pred->GetR().get()), tid);
      if (lhs_r == EvalResult::FALSE || rhs_r == EvalResult::FALSE)
        return EvalResult::FALSE;
      if (lhs_r == EvalResult::TRUE && rhs_r == EvalResult::TRUE)
        return EvalResult::TRUE;
      return EvalResult::UNPREDICTABLE;
    }

    // Logical OR
    if (kind == Opcode::Kind::LogicOr) {
      auto lhs_r = EvaluateForTid(dyn_cast<AST::Expr>(pred->GetL().get()), tid);
      auto rhs_r = EvaluateForTid(dyn_cast<AST::Expr>(pred->GetR().get()), tid);
      if (lhs_r == EvalResult::TRUE || rhs_r == EvalResult::TRUE)
        return EvalResult::TRUE;
      if (lhs_r == EvalResult::FALSE && rhs_r == EvalResult::FALSE)
        return EvalResult::FALSE;
      return EvalResult::UNPREDICTABLE;
    }

    return EvalResult::UNPREDICTABLE;
  }

  // Build a thread mask by evaluating the predicate for each tid.
  // Returns empty vector if the predicate is unpredictable.
  std::vector<bool> BuildMask(AST::Expr* pred) const {
    if (thread_dim <= 0) return {};
    std::vector<bool> mask(thread_dim, false);
    bool any_unpredictable = false;
    for (int64_t tid = 0; tid < thread_dim; ++tid) {
      auto r = EvaluateForTid(pred, tid);
      if (r == EvalResult::UNPREDICTABLE) {
        any_unpredictable = true;
        break;
      }
      mask[tid] = (r == EvalResult::TRUE);
    }
    if (any_unpredictable) return {};

    // AND with parent mask if nested inside another inthreads
    if (!mask_stack.empty()) {
      auto& parent = mask_stack.back();
      for (int64_t i = 0; i < thread_dim; ++i) mask[i] = mask[i] && parent[i];
    }
    return mask;
  }

  // Determine the primary parallel level referenced in a predicate.
  ParallelLevel FindPredLevel(AST::Expr* pred) const {
    if (!pred) return ParallelLevel::NONE;
    auto kind = pred->GetOp().GetKind();
    if (kind == Opcode::Kind::Eq || kind == Opcode::Kind::Ne ||
        kind == Opcode::Kind::Lt || kind == Opcode::Kind::Le ||
        kind == Opcode::Kind::Gt || kind == Opcode::Kind::Ge) {
      std::string pv = FindPVInExpr(pred->GetL().get());
      if (pv.empty()) pv = FindPVInExpr(pred->GetR().get());
      if (!pv.empty()) {
        auto* info = LookupPV(pv);
        if (info) return info->level;
      }
      // Check modulo pattern
      auto lhs = dyn_cast<AST::Expr>(pred->GetL().get());
      if (lhs && lhs->GetOp().GetKind() == Opcode::Kind::Mod) {
        pv = FindPVInExpr(lhs->GetL().get());
        if (!pv.empty()) {
          auto* info = LookupPV(pv);
          if (info) return info->level;
        }
      }
    }
    if (kind == Opcode::Kind::LogicAnd || kind == Opcode::Kind::LogicOr) {
      auto l = FindPredLevel(dyn_cast<AST::Expr>(pred->GetL().get()));
      if (l != ParallelLevel::NONE) return l;
      return FindPredLevel(dyn_cast<AST::Expr>(pred->GetR().get()));
    }
    return ParallelLevel::NONE;
  }

  // Extract selected_unit_index for simple `pv == constant` predicates.
  int64_t FindSelectedUnit(AST::Expr* pred) const {
    if (!pred) return -1;
    if (pred->GetOp().GetKind() != Opcode::Kind::Eq) return -1;
    std::string pv = FindPVInExpr(pred->GetL().get());
    auto cval = ExtractConstant(pred->GetR().get());
    if (pv.empty()) {
      pv = FindPVInExpr(pred->GetR().get());
      cval = ExtractConstant(pred->GetL().get());
    }
    if (!pv.empty() && cval.has_value()) return cval.value();
    return -1;
  }

  void PropagateToBlock(AST::Block* block) {
    if (!block || mask_stack.empty()) return;
    block->SetScopeThreadMask(mask_stack.back());
  }

  bool BeforeVisitImpl(AST::Node& n) override {
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      PBEntry entry;
      entry.pb = pb;
      entry.level = pb->GetLevel();

      if (pb->IsEnforced() && IsThreadLevel(entry.level)) {
        auto bpv = pb->BPV();
        auto bound_vals = pb->BoundValues();

        if (pb->HasSubPVs() && pb->SubPVCount() > 0) {
          for (size_t i = 0; i < pb->SubPVCount(); ++i) {
            auto sub_pv = pb->GetSubPV(i);
            int64_t bv = -1;
            if (i < bound_vals.size()) {
              auto v = VIInt(bound_vals[i]);
              if (v.has_value()) bv = v.value();
            }
            entry.pvs.push_back({sub_pv->name, entry.level, bv});
          }
        }

        if (bpv) {
          int64_t total_bound = -1;
          auto tb = pb->BoundValue();
          if (IsValidValueItem(tb)) {
            auto v = VIInt(tb);
            if (v.has_value()) total_bound = v.value();
          } else if (!bound_vals.empty()) {
            total_bound = 1;
            for (auto& bv : bound_vals) {
              auto v = VIInt(bv);
              if (v.has_value())
                total_bound *= v.value();
              else {
                total_bound = -1;
                break;
              }
            }
          }
          entry.pvs.push_back({bpv->name, entry.level, total_bound});
          UpdateThreadDim(entry.level, total_bound);
        }
      }

      pb_stack.push_back(entry);
      PropagateToBlock(pb);
    } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
      auto pred = it->GetPred();
      if (pred && thread_dim > 0) {
        auto mask = BuildMask(pred.get());
        if (!mask.empty()) {
          int64_t count = 0;
          for (bool b : mask)
            if (b) ++count;
          it->active_threads = sbe::nu(count);
          it->inthreads_level = FindPredLevel(pred.get());
          it->selected_unit_index = FindSelectedUnit(pred.get());
          it->SetScopeThreadMask(mask);
          mask_stack.push_back(mask);
        } else {
          Note(it->LOC(), "inthreads predicate is dynamic; active thread count "
                          "cannot be determined at compile time.");
        }
      } else if (pred && thread_dim <= 0) {
        Note(it->LOC(), "inthreads active thread count unknown: thread "
                        "dimension could not be determined from parallel-by.");
      }
    } else if (auto block = dyn_cast<AST::Block>(&n)) {
      PropagateToBlock(block);
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ParallelBy>(&n)) {
      if (!pb_stack.empty()) {
        auto& entry = pb_stack.back();
        bool was_thread_level = IsThreadLevel(entry.level);
        pb_stack.pop_back();
        // Reset thread_dim when no thread-level parallels remain in the stack.
        if (was_thread_level) {
          bool any_thread_level = false;
          for (auto& e : pb_stack)
            if (IsThreadLevel(e.level)) {
              any_thread_level = true;
              break;
            }
          if (!any_thread_level) thread_dim = -1;
        }
      }
    } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
      if (it->HasScopeThreadMask() && !mask_stack.empty())
        mask_stack.pop_back();
    }
    return true;
  }

public:
  ActiveThreadsAnalysis() : VisitorWithScope("active-threads") {}

  bool Visit(AST::InThreadsBlock&) override { return true; }
  bool Visit(AST::MultiNodes&) override { return true; }
  bool Visit(AST::MultiValues&) override { return true; }
  bool Visit(AST::Nullptr&) override { return true; }
  bool Visit(AST::NoValue&) override { return true; }
  bool Visit(AST::IntLiteral&) override { return true; }
  bool Visit(AST::FloatLiteral&) override { return true; }
  bool Visit(AST::StringLiteral&) override { return true; }
  bool Visit(AST::BoolLiteral&) override { return true; }
  bool Visit(AST::Expr&) override { return true; }
  bool Visit(AST::CastExpr&) override { return true; }
  bool Visit(AST::AttributeExpr&) override { return true; }
  bool Visit(AST::MultiDimSpans&) override { return true; }
  bool Visit(AST::NamedTypeDecl&) override { return true; }
  bool Visit(AST::NamedVariableDecl&) override { return true; }
  bool Visit(AST::IntTuple&) override { return true; }
  bool Visit(AST::DataAccess&) override { return true; }
  bool Visit(AST::Assignment&) override { return true; }
  bool Visit(AST::IntIndex&) override { return true; }
  bool Visit(AST::DataType&) override { return true; }
  bool Visit(AST::Identifier&) override { return true; }
  bool Visit(AST::Parameter&) override { return true; }
  bool Visit(AST::ParamList&) override { return true; }
  bool Visit(AST::ParallelBy&) override { return true; }
  bool Visit(AST::WhereBind&) override { return true; }
  bool Visit(AST::WithIn&) override { return true; }
  bool Visit(AST::WithBlock&) override { return true; }
  bool Visit(AST::Memory&) override { return true; }
  bool Visit(AST::SpanAs&) override { return true; }
  bool Visit(AST::DMA&) override { return true; }
  bool Visit(AST::MMA&) override { return true; }
  bool Visit(AST::ChunkAt&) override { return true; }
  bool Visit(AST::Wait&) override { return true; }
  bool Visit(AST::Trigger&) override { return true; }
  bool Visit(AST::Break&) override { return true; }
  bool Visit(AST::Continue&) override { return true; }
  bool Visit(AST::Call&) override { return true; }
  bool Visit(AST::Rotate&) override { return true; }
  bool Visit(AST::Synchronize&) override { return true; }
  bool Visit(AST::Select&) override { return true; }
  bool Visit(AST::Return&) override { return true; }
  bool Visit(AST::LoopRange&) override { return true; }
  bool Visit(AST::ForeachBlock&) override { return true; }
  bool Visit(AST::WhileBlock&) override { return true; }
  bool Visit(AST::IfElseBlock&) override { return true; }
  bool Visit(AST::FunctionDecl&) override { return true; }
  bool Visit(AST::ChoreoFunction&) override { return true; }
  bool Visit(AST::CppSourceCode&) override { return true; }
  bool Visit(AST::DeviceFunctionDecl&) override { return true; }
  bool Visit(AST::Program&) override { return true; }
};

} // end namespace Choreo

#endif // __CHOREO_ACTIVE_THREADS_HPP__
