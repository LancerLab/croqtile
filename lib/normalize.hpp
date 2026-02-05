#ifndef __CHOREO_NORMALIZATION_HPP__
#define __CHOREO_NORMALIZATION_HPP__

// This applies 'normalization' or 'canonicalization' of AST for easier handling
// in later visiting passes.

#include <tuple>

#include "loop_vectorize.hpp"
#include "pbtree.hpp"
#include "symtab.hpp"
#include "target_utils.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

static inline std::string GenerateLoopName() {
  static int loop_count = 0;
  return "loop" + std::to_string(++loop_count);
}

struct NormBase : public VisitorWithScope {
  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) {
      dbgs() << n.TypeNameString();
      if (!n.IsBlock()) dbgs() << ": " << STR(n);
      dbgs() << "\n";
    }
  }
  NormBase(const std::string& name) : VisitorWithScope(name) {}
};

struct LoopNorm final : public NormBase {
public:
  std::map<std::string, ptr<AST::MultiValues>> matcher_map; // map of with-in

  LoopNorm() : NormBase("loopnorm") {}
  bool BeforeVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "before visiting " << n.TypeNameString() << "\n";
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "after visiting " << n.TypeNameString() << "\n";
    return true;
  }

  bool Visit(AST::WithIn& n) override {
    if (n.with && n.with_matchers) {
      matcher_map[n.with->name] = n.with_matchers;
    }
    return true;
  }

  bool Visit(AST::MultiNodes& n) override {
    for (size_t idx = 0; idx < n.Count(); ++idx) {
      if (auto fb = dyn_cast<AST::ForeachBlock>(n.SubAt(idx))) {
        std::vector<ptr<AST::ForeachBlock>> loops;

        auto rng = cast<AST::LoopRange>(fb->ranges->ValueAt(0));
        auto cname = rng->IVName();
        if (fb->ranges->Count() == 1 && matcher_map.count(cname)) {
          // single range, multiple loops (range dim > 1)
          for (auto matcher : matcher_map[cname]->values) {
            auto matcher_iv = AST::GetIdentifier(matcher);
            auto iv_ty = matcher_iv->GetType();
            auto iv_name = matcher_iv->name;
            auto new_iv = AST::Make<AST::Identifier>(rng->LOC(), iv_name);
            new_iv->SetType(iv_ty);
            auto ranges = AST::Make<AST::MultiValues>(fb->ranges->LOC());
            ranges->Append(AST::Make<AST::LoopRange>(rng->LOC(), new_iv));
            auto stmts = AST::Make<AST::MultiNodes>(fb->stmts->LOC());
            auto new_fb =
                AST::Make<AST::ForeachBlock>(fb->LOC(), ranges, stmts);
            auto loop =
                std::make_shared<Loop>(GenerateLoopName(), iv_ty, fb->LOC());
            new_fb->loop = loop;
            loops.push_back(new_fb);
          }
        } else if (fb->ranges->Count() > 1) {
          // multiple ranges, multiple loops
          const auto& ranges = fb->GetRangeNodes();
          for (size_t i = 0; i < ranges->Count(); ++i) {
            auto rng = cast<AST::LoopRange>(ranges->ValueAt(i));
            auto ranges = AST::Make<AST::MultiValues>(rng->LOC());
            ranges->Append(rng);
            auto stmts = AST::Make<AST::MultiNodes>(fb->stmts->LOC());
            auto new_fb =
                AST::Make<AST::ForeachBlock>(fb->LOC(), ranges, stmts);
            auto loop = AST::Make<Loop>(GenerateLoopName(),
                                        rng->IV()->GetType(), fb->LOC());
            new_fb->loop = loop;
            loops.push_back(new_fb);
          }
        } else if (fb->ranges->Count() == 1 && !matcher_map.count(cname)) {
          // single range, single loop
          auto loop = std::make_shared<Loop>(GenerateLoopName(),
                                             rng->IV()->GetType(), fb->LOC());
          fb->loop = loop;
          continue;
        } else {
          Error1(fb->LOC(), "invalid range of foreach block: " + STR(fb));
          continue;
        }
        // construct the loop hierarchy
        size_t loop_level = 0;
        for (; loop_level < loops.size() - 1; ++loop_level) {
          assert(loops[loop_level]->stmts);
          loops[loop_level]->stmts->Append(loops[loop_level + 1]);
        }
        // apply suffixes to the correct loop level
        auto suffixs = fb->suffixs;
        if (suffixs && suffixs->Count() > 0) {
          for (auto& suffix : suffixs->values) {
            auto attr_expr = dyn_cast<AST::AttributeExpr>(suffix);

            if (attr_expr->AttrName() == "vectorize") {
              auto arg_id = AST::GetIdentifier(attr_expr->AttrValueAt(0));
              assert(arg_id && "expect identifier as the argument.");
              bool found = false;
              for (size_t j = 0; j < loops.size(); ++j) {
                auto sub_fb = loops[j];
                assert(sub_fb->ranges->Count() == 1 &&
                       "expect only one range in the loop hierarchy.");
                if (arg_id->name ==
                    dyn_cast<AST::LoopRange>(sub_fb->GetRanges()[0])
                        ->IVName()) {
                  found = true;
                  sub_fb->suffixs = suffixs;
                }
              }
              // suffixes are not matched to any loop index variable
              if (!found) {
                Error(attr_expr->LOC(),
                      "suffix expr '" + PSTR(suffix) +
                          "' does not match any loop index variable.");
              }
            }
          }
        }

        loops[loop_level]->stmts = fb->stmts;
        // replace the original node with the new loop hierarchy
        n.values[idx] = loops[0];
      }
    }
    return true;
  }

  bool IsAllowed(AST::Node& root) const override {
    if (CCtx().NoVectorize()) return false;

    auto vhc = GetResult<VectorizationHintChecker>(root);
    if (vhc->HasError()) return false;

    return (CCtx().LoopNorm() || CCtx().Vectorize() ||
            vhc->HasVectorizationHint());
  }
};

struct CompoundNorm : public NormBase {
private:
  bool changed = false;

  std::string old;
  size_t count = 0; // name suffix of runtime int values

  // within_map
  std::set<std::string> within_norm_iv;

  bool handle_parameter = false;
  ptr<AST::Expr> list_ref = nullptr;
  void SetListReference(const location& l, const std::string& r) {
    list_ref = AST::MakeIdExpr(l, r);
  }
  void ResetListReference() { list_ref = nullptr; }

  // for node hoisting
  using NodeInsertInfo =
      std::vector<std::tuple<int, ptr<AST::Node>, std::string>>;
  std::stack<AST::MultiNodes*> multi_nodes;
  std::stack<int> cur_node_indices;
  int GetNodeIndex() const { return cur_node_indices.top(); }
  int GetValidNodeIndex() const {
    assert(cur_node_indices.top() != -1 && "invalid node index.");
    return cur_node_indices.top();
  }
  void SetNodeIndex(int i) {
    cur_node_indices.pop();
    cur_node_indices.push(i);
  }
  void PushNodeIndex(int i) { cur_node_indices.push(i); }
  void PopNodeIndex() { cur_node_indices.pop(); }
  std::map<AST::MultiNodes*, NodeInsertInfo> mnodes_insertions;

  void InsertNode(int index, const ptr<AST::Node>& n, const std::string& name) {
    assert(index >= 0);
    mnodes_insertions[multi_nodes.top()].emplace_back(
        std::make_tuple(index, n, name));
  }

  std::string GetInternalValueString() { return "$" + std::to_string(count++); }

  ptr<AST::CastExpr> GenCastExprNode(BaseType to, BaseType from,
                                     ptr<AST::Node> to_cast) {
    auto ce = AST::Make<AST::CastExpr>(to_cast->LOC(), to_cast);
    ce->SetType(MakeScalarType(to, true));
    ce->SetFrom(from);
    ce->SetTo(to);
    return ce;
  }

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) {
      dbgs() << n.TypeNameString();
      if (!n.IsBlock()) dbgs() << ": " << STR(n);
      dbgs() << "\n";
    }
  }

public:
  // it does not require a symbol table
  CompoundNorm() : NormBase("comp_n") {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (auto* b = dyn_cast<AST::MultiDimSpans>(&n)) {
      if (b->ref_name != "") {
        SetListReference(n.LOC(), b->ref_name);
        return true;
      }

      if (b->list) return true;
      if (!handle_parameter) return true;

      assert(IsValidRank(b->Rank()));

      // append the node that have multiple dynamic values
      auto mvals = AST::Make<AST::MultiValues>(n.LOC());
      for (size_t i = 0; i < b->Rank(); ++i)
        mvals->Append(AST::Make<AST::IntLiteral>(n.LOC()));
      b->list = mvals;
      changed = true;
    } else if (auto* b = dyn_cast<AST::IntTuple>(&n)) {
      if (b->ref_name != "") SetListReference(n.LOC(), b->ref_name);
    } else if (auto p = dyn_cast<AST::Parameter>(&n)) {
      handle_parameter = true;
      old = AST::STR(*p->type);
      changed = false;
    } else if (auto m = dyn_cast<AST::MultiNodes>(&n)) {
      multi_nodes.push(m);
      PushNodeIndex(0);
    } else if (isa<AST::DMA>(&n) || isa<AST::NamedVariableDecl>(&n) ||
               isa<AST::Assignment>(&n) || isa<AST::Return>(&n) ||
               isa<AST::ForeachBlock>(&n) || isa<AST::ChunkAt>(&n)) {
      auto idx = multi_nodes.top()->GetIndex(&n);
      assert(idx != -1 && "unexpected node index.");
      SetNodeIndex(idx);
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (auto* b = dyn_cast<AST::MultiDimSpans>(&n)) {
      ResetListReference();
      b->ref_name = ""; // no reference is required
    } else if (isa<AST::IntTuple>(&n)) {
      ResetListReference();
    } else if (auto p = dyn_cast<AST::Parameter>(&n)) {
      handle_parameter = false;
      VST_DEBUG(if (changed) dbgs()
                << "Name dims of `" << STR(*p->sym) << "': " << old << " ---> "
                << STR(*p->type) << "\n");
      old.clear();
      changed = false;
    } else if (isa<AST::ChoreoFunction>(&n)) {
      count = 0;
    } else if (isa<AST::Return>(&n)) {
      SetNodeIndex(-1);
    } else if (auto wb = dyn_cast<AST::WithBlock>(&n)) {
      for (const auto& node : wb->withins->AllSubs()) {
        auto wi = cast<AST::WithIn>(node);
        for (const auto& val : wi->with_matchers->AllValues()) {
          auto iv = AST::GetIdentifier(val);
          if (iv && within_norm_iv.count(iv->name))
            within_norm_iv.erase(iv->name);
        }
      }
    }
    return true;
  }

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);

    // insert the node at the given place
    assert(&n == multi_nodes.top());
    for (auto item : mnodes_insertions[multi_nodes.top()]) {
      auto& index = std::get<0>(item);
      auto& pnode = std::get<1>(item);
      // auto& name = std::get<2>(item);

      n.values.insert(n.values.begin() + index, pnode);
      VST_DEBUG(dbgs() << "Hoisted: " << PSTR(pnode) << "\n");
    }

    mnodes_insertions.erase(&n);
    multi_nodes.pop();
    PopNodeIndex();

    return true;
  }

  bool Visit(AST::MultiValues& n) override {
    TraceEachVisit(n);
    if (list_ref) { // desugar the list reference
      for (size_t i = 0; i < n.values.size(); ++i) {
        if (auto expr = dyn_cast<AST::Expr>(n.values[i])) {
          if (auto ref = expr->GetReference()) {
            if (isa<AST::IntIndex>(ref.get())) {
              // apply desugaring a {(0), 1} -> {a(0), 1}
              auto new_expr = AST::Make<AST::Expr>(
                  expr->LOC(), "dimof", list_ref->Clone(), ref->Clone());
              VST_DEBUG(dbgs() << "Desugar ref: " << STR(*expr) << " ---> "
                               << STR(*new_expr) << "\n");
              new_expr->SetType(MakeIntegerType());
              n.values[i] = new_expr;
            }
          }
        }
      }
    }

    if (handle_parameter) { // make runtime values of "?" to be named
      for (size_t i = 0; i < n.values.size(); ++i) {
        if (auto il = dyn_cast<AST::IntLiteral>(n.values[i])) {
          if (!IsUnKnownInteger(il->Val())) continue;
          auto new_il =
              AST::Make<AST::Identifier>(il->LOC(), GetInternalValueString());

          changed = true;
          n.values[i] = new_il;
        }
      }
    }

    return true;
  }

  bool Visit(AST::Expr& n) override {
    TraceEachVisit(n);
    if (list_ref) { // could be with syntax sugar
      auto Apply = [this](const ptr<AST::Expr>& expr) -> ptr<AST::Expr> {
        if (!expr) return nullptr;
        if (auto ref = expr->GetReference()) {
          if (!isa<AST::IntIndex>(ref.get())) return nullptr;

          // apply desugaring a {(0), 1} -> {a(0), 1}
          auto ret = AST::Make<AST::Expr>(expr->LOC(), "dimof",
                                          list_ref->Clone(), ref->Clone());
          ret->SetType(MakeIntegerType()); // must be integer

          VST_DEBUG(dbgs() << "Desugaring expression node: " << PSTR(expr)
                           << " --->" << PSTR(ret) << "\n";);

          changed = true;

          return ret;
        }
        return nullptr;
      };

      if (auto new_value = Apply(n.GetC())) n.SetC(new_value);
      if (auto lv = dyn_cast<AST::Expr>(n.GetL()))
        if (auto new_value = Apply(lv)) n.SetL(new_value);
      if (auto rv = dyn_cast<AST::Expr>(n.GetR())) {
        if (auto new_value = Apply(rv)) n.SetR(new_value);
      }

      return true;
    }

    if (n.op == "sizeof" && isa<SpannedType>(n.GetR())) {
      auto id = cast<AST::Expr>(n.GetR())->GetSymbol();
      assert(!SuffixedWith(id->name, ".span"));
      VST_DEBUG(dbgs() << "Desugaring sizeof: " << id->name << " ->");
      id->name += ".span";
      VST_DEBUG(dbgs() << id->name << ".\n");
    }

    if (n.IsReference()) {
      auto ref = n.GetReference();
      auto nty = n.GetType();
      if (auto call = dyn_cast<AST::Call>(ref)) {
        auto func_name = call->function->name;
        if (func_name == "__alignup" || func_name == "__aligndown") {
          auto arg0 = call->GetArguments()[0];
          auto arg1 = call->GetArguments()[1];
          ptr<AST::Expr> new_expr = nullptr;
          if (func_name == "__alignup") {
            // __alignup(a, b) -> (a + b - 1) / b * b
            auto il = AST::MakeIntExpr(n.LOC(), 1);
            il->SetType(nty);
            auto add = AST::Make<AST::Expr>(n.LOC(), "+", arg0, arg1);
            add->SetType(nty);
            auto sub = AST::Make<AST::Expr>(n.LOC(), "-", add, il);
            sub->SetType(nty);
            auto div = AST::Make<AST::Expr>(n.LOC(), "/", sub, arg1);
            div->SetType(nty);
            new_expr = AST::Make<AST::Expr>(n.LOC(), "*", div, arg1);
            new_expr->SetType(nty);
          } else if (func_name == "__aligndown") {
            // __aligndown(a, b) -> a / b * b
            auto div = AST::Make<AST::Expr>(n.LOC(), "/", arg0, arg1);
            div->SetType(nty);
            new_expr = AST::Make<AST::Expr>(n.LOC(), "*", div, arg1);
            new_expr->SetType(nty);
          }
          VST_DEBUG(dbgs() << "Desugar " << STR(n) << " -> " << PSTR(new_expr)
                           << "\n");
          n.OverWrite(*new_expr);
        }
      }
    } else if (n.IsBinary() || n.IsTernary()) {
      if (n.IsBinary())
        if (!(n.IsArith() && !n.IsUBArith())) return true;

      auto l = n.GetL();
      auto r = n.GetR();
      auto lty = l->GetType();
      auto rty = r->GetType();

      auto lsty = dyn_cast<ScalarType>(lty);
      auto rsty = dyn_cast<ScalarType>(rty);
      if (!lsty || !rsty) return true;

      auto lbty = lsty->GetBaseType();
      auto rbty = rsty->GetBaseType();

      auto promote_res = PromoteType(lbty, rbty);

      if (lbty != promote_res.lty) {
        n.SetL(GenCastExprNode(promote_res.lty, lbty, l));
        n.SetType(MakeScalarType(promote_res.lty, true));
        VST_DEBUG({
          dbgs() << "Promote '" << PSTR(l) << "' at " << l->LOC() << "\n\t'"
                 << STR(lbty) << "' => '" << STR(promote_res.lty) << "'\n";
        });
      }
      if (rbty != promote_res.rty) {
        n.SetR(GenCastExprNode(promote_res.rty, rbty, r));
        n.SetType(MakeScalarType(promote_res.rty, true));
        VST_DEBUG({
          dbgs() << "Promote '" << PSTR(r) << "' at " << r->LOC() << "\n\t'"
                 << STR(rbty) << "' => '" << STR(promote_res.rty) << "'\n";
        });
      }
    }

    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) override {
    TraceEachVisit(n);

    if (n.mem && (n.mem->Get() == Storage::DEFAULT)) {
      // Should this be set by target?
      n.mem->Set(Storage::GLOBAL);

      VST_DEBUG(dbgs() << "Place storage of '" << n.name_str
                       << "': DEFAULT ---> GLOBAL\n");
    }

    if (!n.init_expr) return true;
    if (!n.type) return true;
    if (isa<AST::Call>(n.init_expr)) {
      auto init_ty = n.init_expr->GetType();
      n.init_expr = AST::Make<AST::Expr>(n.init_expr->LOC(), n.init_expr);
      n.init_expr->SetType(init_ty);
    }
    // insert CastExpr node if needed
    auto r = n.init_expr;
    auto lty = n.type->GetType();
    auto rty = r->GetType();
    auto lsty = dyn_cast<ScalarType>(lty);
    auto rsty = dyn_cast<ScalarType>(rty);
    if (!lsty || !rsty) return true;
    auto lbty = lsty->GetBaseType();
    auto rbty = rsty->GetBaseType();
    if (lbty == rbty) return true;
    // need to do type casting
    auto casted = GenCastExprNode(lbty, rbty, n.init_expr);
    VST_DEBUG({
      dbgs() << "Cast '" << PSTR(n.init_expr) << "' at " << n.init_expr->LOC()
             << "\n\t'" << STR(rbty) << "' => '" << STR(lbty) << "'\n";
    });
    n.init_expr = casted;

    return true;
  }
  bool Visit(AST::IntTuple& n) override {
    bool replace_mv = false;
    auto mv = AST::Make<AST::MultiValues>(n.GetValues()->LOC(), ",");
    for (auto& v : n.GetValues()->AllValues()) {
      if (isa<AST::IntLiteral>(v)) {
        mv->Append(v->Clone());
        continue;
      }
      auto expr = cast<AST::Expr>(v);
      if (auto itt = dyn_cast<ITupleType>(expr->GetType())) {
        VST_DEBUG(dbgs() << "Replace " << PSTR(expr) << " in " << STR(n)
                         << " with:\n");
        for (size_t idx = 0; idx < itt->dim_count; ++idx) {
          auto ii = AST::Make<AST::IntIndex>(
              expr->LOC(), AST::Make<AST::IntLiteral>(expr->LOC(), idx));
          ii->SetType(MakeIndexType());
          auto new_expr =
              AST::Make<AST::Expr>(expr->LOC(), "dimof", expr->Clone(), ii);
          new_expr->SetType(MakeIntegerType());
          mv->Append(new_expr);
          VST_DEBUG(dbgs() << "\t" << PSTR(new_expr) << "\n");
          replace_mv = true;
        }
      } else {
        mv->Append(v);
      }
    }
    if (replace_mv) {
      n.vlist = mv;
      n.SetType(MakeITupleType(n.vlist->Count()));
    }
    return true;
  }
  bool Visit(AST::Assignment& n) override {
    auto l = n.da;
    auto r = n.value;
    auto lty = l->GetType();
    auto rty = r->GetType();

    auto lsty = dyn_cast<ScalarType>(lty);
    auto rsty = dyn_cast<ScalarType>(rty);
    if (!lsty || !rsty) return true;

    auto lbty = lsty->GetBaseType();
    auto rbty = rsty->GetBaseType();

    if (lbty == rbty) return true;

    // need to do type casting

    auto casted = GenCastExprNode(lbty, rbty, n.value);
    VST_DEBUG({
      dbgs() << "Cast '" << PSTR(n.value) << "'\n\t from type '" << STR(rbty)
             << "'\n\t to type '" << STR(lbty) << "'\n";
    });
    n.value = casted;

    return true;
  }

  bool Visit(AST::WithIn& n) override {
    TraceEachVisit(n);

    if (isa<ScalarIntegerType>(n.in->GetType())) {
      auto mv = AST::Make<AST::MultiValues>(n.in->LOC(), ",");
      mv->Append(n.in);
      n.in = AST::Make<AST::MultiDimSpans>(n.in->LOC(), "", mv, 1);
      n.with->SetType(MakeBoundedITupleType(Shape(1)));
    }

    if (n.with_matchers) return true;
    assert(n.with && "must have with statement.");

    auto wty = n.with->GetType();
    assert(isa<BoundedITupleType>(wty) && "expect a bounded ituple type.");

    auto mval = AST::Make<AST::MultiValues>(n.LOC(), ",");
    auto bity = cast<BoundedITupleType>(wty);
    // fill the with-matchers
    for (size_t i = 0; i < wty->Dims(); ++i) {
      auto name = n.with->name + "__elem__" + std::to_string(i);
      if (within_norm_iv.count(name))
        name += "_" + SymbolTable::GetAnonName();
      else
        within_norm_iv.insert(name);
      mval->Append(AST::Make<AST::Identifier>(n.with->LOC(), name));
      if (bity->HasValidBound())
        mval->ValueAt(i)->SetType(
            MakeBoundedIntegerType(bity->GetUpperBound(i)));
      else
        mval->ValueAt(i)->SetType(MakeUnknownBoundedIntegerType());
    }

    n.with_matchers = mval;

    VST_DEBUG(dbgs() << "Generate with-matchers for '" << n.with->name
                     << "': " << STR(mval) << "\n");

    return true;
  }

  bool Visit(AST::DMA& n) override {
    if (n.operation == ".any") return true;

    if (n.operation == ".pad") {
      auto pc = cast<PadConfig>(n.GetConfig());
      auto span_bty = GetUnderlyingType(n.from->GetType());
      auto pv_bty = GetUnderlyingType(pc->GetPadValue()->GetType());
      if (span_bty != pv_bty) {
        auto casted = GenCastExprNode(span_bty, pv_bty, pc->GetPadValue());
        VST_DEBUG({
          dbgs() << "Cast '" << PSTR(pc->GetPadValue()) << "' at "
                 << pc->GetPadValue()->LOC() << "\n\t'" << STR(pv_bty)
                 << "' => '" << STR(span_bty) << "'\n";
        });
        pc->SetPadValue(casted);
      }
      for (auto& mv : {pc->pad_high, pc->pad_low, pc->pad_mid}) {
        int idx = 0;
        for (auto& v : mv->AllValues()) {
          auto pv_bty = GetUnderlyingType(v->GetType());
          if (pv_bty != BaseType::U32)
            if (auto r = AST::Ref(v); r && !AST::IsLiteral(*r)) {
              auto casted = GenCastExprNode(BaseType::U32, pv_bty, v);
              VST_DEBUG({
                dbgs() << "Cast '" << PSTR(v) << "' at " << v->LOC() << "\n\t'"
                       << STR(pv_bty) << "' => '" << STR(BaseType::U32)
                       << "'\n";
              });
              mv->SetValueAt(idx, casted);
            }
          ++idx;
        }
      }
    }

    if (!isa<AST::Select>(n.to)) return true;

    auto anon_sym = SymbolTable::GetAnonName();
    // hoist the select to multinodes
    int index =
        GetValidNodeIndex() + mnodes_insertions[multi_nodes.top()].size();
    cast<AST::Select>(n.to)->inDMA = false;
    auto assign = AST::Make<AST::Assignment>(n.to->LOC(), anon_sym, n.to);
    assign->SetType(n.to->GetType()->Clone());
    assign->da->SetType(n.to->GetType()->Clone());
    assign->SetDecl();
    InsertNode(index, assign, anon_sym);
    VST_DEBUG(dbgs() << n.TypeNameString() << ": replace-0 " << PSTR(n.to)
                     << " with " << anon_sym << "(" << PSTR(assign->GetType())
                     << ".\n");

    n.to = AST::Make<AST::ChunkAt>(
        n.to->LOC(), AST::Make<AST::Identifier>(n.to->LOC(), anon_sym));
    return true;
  }

  bool Visit(AST::ChunkAt& n) override {
    TraceEachVisit(n);

    if (n.sa) {
      // hoist the span_as to multinodes
      int index =
          GetValidNodeIndex() + mnodes_insertions[multi_nodes.top()].size();
      auto assign =
          AST::Make<AST::Assignment>(n.sa->LOC(), n.sa->nid->name, n.sa);
      assign->SetType(n.sa->GetType()->Clone());
      assign->da->SetType(n.sa->GetType()->Clone());
      assign->SetDecl();
      InsertNode(index, assign, n.sa->nid->name);
      VST_DEBUG(dbgs() << n.TypeNameString() << ": replace-1 " << PSTR(n.sa)
                       << " with " << n.sa->nid->name << "("
                       << PSTR(assign->GetType()) << ")\n");
      n.sa.reset();
    }

    if (CCtx().HasFeature(ChoreoFeature::RSTM0) && n.OpCount() > 0 &&
        // special mode: requires to generate 'bitcast' for a reshape
        isa<AST::SOP::Reshape>(n.FirstOp())) {
      int index =
          GetValidNodeIndex() + mnodes_insertions[multi_nodes.top()].size();
      auto nname = SymbolTable::GetAnonName();
      auto so = cast<AST::SOP::Reshape>(n.FirstOp());
      auto id = AST::Make<AST::Identifier>(so->LOC(), nname);
      auto mv = cast<AST::MultiValues>(so->GetNewSpan()->Clone());
      mv->SetDelimiter(", ");
      auto sa = AST::Make<AST::SpanAs>(
          id->LOC(), cast<AST::Identifier>(n.data->Clone()), id, mv);
      auto assign = AST::Make<AST::Assignment>(id->LOC(), nname, sa);
      assign->SetDecl();
      auto sty = cast<SpannedType>(n.GetType());
      auto nty = MakeUnRankedSpannedType(sty->ElementType(), sty->GetStorage());
      sa->SetType(nty);
      assign->SetType(nty);
      InsertNode(index, assign, nname);
      VST_DEBUG(dbgs() << n.TypeNameString() << ": convert " << STR(n)
                       << " as:";
                assign->Print(dbgs(), "   ");
                dbgs() << " (" << PSTR(assign->GetType()) << ")\n");
      n.RemoveOperation(0);
      n.data = cast<AST::Identifier>(id->Clone());
      VST_DEBUG(dbgs() << "   `- " << STR(n) << " (" << PSTR(n.GetType())
                       << ")\n");
    }

    // hoist any arith inside of chunkat positions
    for (auto tsi : n.AllOperations()) {
      std::vector<std::pair<int, ptr<AST::Node>>> repls;
      int i = -1;
      for (auto& v : tsi->IndexNodes()) {
        ++i;
        auto expr = cast<AST::Expr>(v);
        if (expr->op == "getith") {
          // 'getith' must be kept.
          if (auto lexpr = dyn_cast<AST::Expr>(expr->GetL())) {
            if (!lexpr->GetSymbol()) {
              // hoist the non-getith part
              int index = GetValidNodeIndex() +
                          mnodes_insertions[multi_nodes.top()].size();
              auto nname = SymbolTable::GetAnonName();
              auto assign = AST::Make<AST::Assignment>(expr->GetL()->LOC(),
                                                       nname, expr->GetL());
              assign->SetType(expr->GetL()->GetType()->Clone());
              assign->da->SetType(expr->GetL()->GetType()->Clone());
              assign->SetDecl();
              InsertNode(index, assign, nname);
              VST_DEBUG(dbgs()
                        << n.TypeNameString() << ": replace-2 "
                        << PSTR(expr->GetL()) << " with " << nname << "\n");
              expr->SetL(AST::Make<AST::Identifier>(v->LOC(), nname));
              VST_DEBUG(dbgs() << PSTR(expr->GetL()) << ".\n");
            }
          }
          continue;
        } else if (expr->GetSymbol() || expr->GetInt()) {
          // do not hoist symbol or integer reference
          continue;
        }

        // else, hoist the arith out
        int index =
            GetValidNodeIndex() + mnodes_insertions[multi_nodes.top()].size();
        auto nname = SymbolTable::GetAnonName();
        auto assign = AST::Make<AST::Assignment>(v->LOC(), nname, v);
        assign->SetType(v->GetType()->Clone());
        assign->da->SetType(v->GetType()->Clone());
        assign->SetDecl();
        InsertNode(index, assign, nname);
        repls.emplace_back(i, AST::MakeIdExpr(v->LOC(), nname));
        VST_DEBUG(dbgs() << n.TypeNameString() << ": replace-3 " << PSTR(v)
                         << " with " << nname << "(" << PSTR(assign->GetType())
                         << ")\n");
      }
      for (auto& repl : repls) {
        VST_DEBUG(dbgs() << n.TypeNameString() << ": replace "
                         << PSTR(tsi->GetIndices()->ValueAt(repl.first))
                         << " with ");

        tsi->GetIndices()->SetValueAt(repl.first, repl.second);

        VST_DEBUG(dbgs() << PSTR(tsi->GetIndices()->ValueAt(repl.first)) << "("
                         << PSTR(repl.second->GetType()) << ").\n");
      }
    }
    return true;
  }

  bool Visit(AST::Call& n) override {
    TraceEachVisit(n);
    if (n.IsArith() && n.IsBIF()) {
      auto normalized_args =
          AST::Make<AST::MultiValues>(n.arguments->LOC(), ", ");
      bool normalized = false;
      ptr<Type> normalized_ty = nullptr;
      for (size_t i = 0; i < n.arguments->Count(); ++i) {
        auto arg = n.arguments->ValueAt(i);
        auto arg_ty = arg->GetType();

        if (isa<ScalarFloatType>(arg_ty)) {
          if (!normalized_ty) {
            normalized_ty = arg_ty->Clone();
            continue;
          }
          auto bty_f = arg_ty->GetBaseType();
          auto bty_t = normalized_ty->GetBaseType();
          if (bty_f != bty_t && IsLossyCast(bty_f, bty_t))
            normalized_ty = arg_ty->Clone();
        } else
          normalized_ty = MakeScalarFloatType(BaseType::F32);
      }

      assert(normalized_ty && "must have a type to normalize to");
      for (size_t i = 0; i < n.arguments->Count(); ++i) {
        auto arg = n.arguments->ValueAt(i);
        auto arg_ty = arg->GetType();
        if (arg_ty == normalized_ty) {
          normalized_args->Append(arg->Clone());
          continue;
        }
        if (auto casted = GenCastExprNode(normalized_ty->GetBaseType(),
                                          arg_ty->GetBaseType(), arg)) {
          normalized_args->Append(casted);
          normalized = true;
        } else
          choreo_unreachable("unable to normalize argument type: " +
                             STR(*arg_ty) + " to " + STR(*normalized_ty));
      }

      n.SetType(normalized_ty);
      if (normalized) { n.arguments = normalized_args; }
    }
    return true;
  }

  bool Visit(AST::Return& n) override {
    TraceEachVisit(n);

    if (AST::GetIdentifier(*n.value)) return true;

    if (!CCtx().HasFeature(ChoreoFeature::RSTM0)) return true;

    // non-identifier may be normalized
    auto vty = NodeType(*n.value);

    // tricky: we must convert a integer to be 's32 [1] ...' for a specific
    // return value;
    if (CCtx().HasFeature(ChoreoFeature::RSTM0) &&
        isa<ScalarIntegerType>(vty)) {
      auto expr = cast<AST::Expr>(n.value);
      if (auto il = expr->GetInt()) {
        auto& loc = n.value->LOC();
        auto anon_sym = SymbolTable::GetAnonName();

        // compose the named variable decl with initial value
        auto mv = AST::Make<AST::MultiValues>(loc, ",");
        mv->Append(AST::MakeIntExpr(loc, 1));
        auto mds = AST::Make<AST::MultiDimSpans>(loc, "", mv, 1);
        auto dt = AST::Make<AST::DataType>(loc, BaseType::S32, mds);
        auto sto = AST::Make<AST::Memory>(loc, Storage::GLOBAL);
        auto nv = AST::Make<AST::NamedVariableDecl>(loc, anon_sym, dt, sto,
                                                    nullptr, il);
        nv->SetType(vty);

        int index =
            GetValidNodeIndex() + mnodes_insertions[multi_nodes.top()].size();
        InsertNode(index, nv, anon_sym);

        // replace return value now
        VST_DEBUG(dbgs() << "[Norm] Replace " << STR(n) << "\n to be:\n");
        n.value = AST::MakeIdExpr(n.value->LOC(), anon_sym);
        VST_DEBUG(dbgs() << STR(n) << "\n");

        // In host, its return type is still 'int'
        n.AddNote("host-type", "int");
      }
    }

    return true;
  }
  bool Visit(AST::ForeachBlock& n) override {
    auto handle_bounds = [this, &n](auto get_bound, auto set_bound) {
      std::vector<std::pair<int, ptr<AST::Node>>> repls;
      int i = -1;
      for (auto& v : n.GetRanges()) {
        ++i;
        auto lr = cast<AST::LoopRange>(v);
        auto& bound = get_bound(lr);
        if (bound == nullptr) continue;

        auto bound_expr = cast<AST::Expr>(bound);
        if (bound_expr->GetSymbol()) {
          repls.emplace_back(i, bound_expr);
          continue;
        } else if (bound_expr->op == "getith") {
          if (auto lexpr = dyn_cast<AST::Expr>(bound_expr->GetL())) {
            if (!lexpr->GetSymbol()) {
              int index = GetValidNodeIndex() +
                          mnodes_insertions[multi_nodes.top()].size();
              auto nname = SymbolTable::GetAnonName();
              auto assign = AST::Make<AST::Assignment>(
                  bound_expr->GetL()->LOC(), nname, bound_expr->GetL());
              auto lty = bound_expr->GetL()->GetType();
              assign->SetType(lty);
              assign->da->SetType(lty);
              assign->SetDecl();
              InsertNode(index, assign, nname);
              VST_DEBUG(dbgs() << "range - getith: replace "
                               << PSTR(bound_expr->GetL()) << "\n with "
                               << nname << ".\n");
              auto id_expr = AST::MakeIdExpr(v->LOC(), nname);
              id_expr->SetType(lty);
              bound_expr->SetL(id_expr);
              VST_DEBUG(dbgs() << PSTR(bound_expr->GetL()) << ".\n");
            }
          }
          continue;
        }

        int index =
            GetValidNodeIndex() + mnodes_insertions[multi_nodes.top()].size();
        auto nname = SymbolTable::GetAnonName();
        auto assign = AST::Make<AST::Assignment>(v->LOC(), nname, bound_expr);
        auto bty = bound_expr->GetType();
        assign->SetType(bty);
        assign->da->SetType(bty);
        assign->SetDecl();
        InsertNode(index, assign, nname);
        auto id_expr = AST::MakeIdExpr(v->LOC(), nname);
        id_expr->SetType(bty);
        repls.emplace_back(i, id_expr);
        VST_DEBUG(dbgs() << "range - " << bound_expr->op << ": "
                         << "replace " << PSTR(bound_expr->GetL()) << "\n with "
                         << nname << ".\n");
      }

      for (auto& repl : repls) {
        VST_DEBUG(dbgs() << n.TypeNameString() << ": " << "replace "
                         << PSTR(n.GetRangeNodes()->ValueAt(repl.first))
                         << " with ");
        auto lr = cast<AST::LoopRange>(n.GetRangeNodes()->values[repl.first]);
        set_bound(lr, repl.second);
        VST_DEBUG(dbgs() << PSTR(n.GetRangeNodes()->ValueAt(repl.first))
                         << ".\n");
      }
    };

    handle_bounds([](auto lr) -> auto& { return lr->lbound; },
                  [](auto lr, auto val) { lr->lbound = val; });

    handle_bounds([](auto lr) -> auto& { return lr->ubound; },
                  [](auto lr, auto val) { lr->ubound = val; });

    return true;
  }
};

// Try to amend the missing parallel-by levels that are unspecified.
//
// It always fill the parallel-by level to its target.
struct ParaByFiller : public NormBase {
private:
  bool changed = false;

  // analyze the parallel-by structure
  std::vector<AST::ParallelBy*> pb_stack;
  PBTree pb_tree;

  // Note: post-visiting each parallel-by
  //
  //  ...
  //   pb <- literal_depth
  //    pb <- ddepth
  //     ...
  //      pb <- max_depth
  //
  //

  enum FillType { Inner, Outer, AppendInner, LastInner };
  struct FillInfo {
    AST::ParallelBy* pb;
    FillType ft;
    ParallelLevel lvl;
    size_t ubound;
    FillInfo(AST::ParallelBy* p, FillType t, ParallelLevel l, size_t ub = 1)
        : pb(p), ft(t), lvl(l), ubound(ub) {}
  };
  std::vector<FillInfo> fill_info;

private:
  const char* Name(FillType fty) const {
    switch (fty) {
    case Inner: return "Inner";
    case Outer: return "Outer";
    case AppendInner: return "AppendInner";
    case LastInner: return "LastInner";
    default: choreo_unreachable("unsupported fill type.");
    }
    return "";
  }

  bool ExplicitLevel(AST::ParallelBy* pb) const {
    assert(pb != nullptr);
    auto pl = pb->GetLevel();
    assert(pl != ParallelLevel::UNKNOWN);
    return pl != ParallelLevel::NONE;
  }

public:
  // it does not require a symbol table
  ParaByFiller() : NormBase("pbfill") {}

  AST::ParallelBy& InsertInnerLevel(AST::ParallelBy& pb, ParallelLevel pl,
                                    size_t ub) {
    // may fill gap only for a single level
    VST_DEBUG(dbgs() << "Replace `"; pb.InlinePrint(dbgs());
              dbgs() << "` by\n  +-");

    auto new_pb = AST::MakeSimpleParallelBy(pb.LOC(), pb.stmts, ub);
    new_pb->SetOuter(false);
    new_pb->SetLevel(pl);
    new_pb->SetEnforced(false);
    pb.stmts = AST::Make<AST::MultiNodes>(pb.LOC(), new_pb);

    VST_DEBUG(pb.InlinePrint(dbgs()); dbgs() << "\n   +-";
              new_pb->InlinePrint(dbgs()); dbgs() << "\n");

    return *new_pb;
  }

  AST::ParallelBy& AppendInnerLevel(AST::ParallelBy& pb, ParallelLevel pl,
                                    size_t ub) {
    VST_DEBUG(dbgs() << "Replace `"; pb.InlinePrint(dbgs());
              dbgs() << "` by\n  +-");

    auto new_pb = AST::MakeSimpleParallelBy(pb.LOC(), nullptr, ub);
    new_pb->SetLevel(pl);
    new_pb->SetEnforced(false);
    pb.stmts->Append(new_pb);

    VST_DEBUG(pb.InlinePrint(dbgs()));

    return pb;
  }

  AST::ParallelBy& InsertLastInnerLevel(AST::ParallelBy& pb, ParallelLevel pl,
                                        size_t ub) {
    // may fill gap only for a single level
    VST_DEBUG(dbgs() << "Replace `"; pb.InlinePrint(dbgs());
              dbgs() << "` by\n  +-");

    auto lpb = cast<AST::ParallelBy>(pb.stmts->Last());
    auto new_pb = AST::MakeSimpleParallelBy(
        pb.LOC(), AST::Make<AST::MultiNodes>(pb.LOC(), lpb), ub);
    new_pb->SetLevel(pl);
    new_pb->SetEnforced(false);
    pb.stmts->PopBack();
    pb.stmts->Append(new_pb);

    VST_DEBUG(pb.InlinePrint(dbgs()));

    return pb;
  }

  AST::ParallelBy& InsertOuterLevel(AST::ParallelBy& pb, ParallelLevel pl) {
    VST_DEBUG(dbgs() << "Replace `"; pb.InlinePrint(dbgs());
              dbgs() << "` by\n  +-");

    auto new_pb = cast<AST::ParallelBy>(pb.Clone());
    new_pb->SetOuter(false);
    // pb is now the outer level
    pb.SetLevel(pl);
    pb.SetEnforced(false);

    // convert current pb to be simple
    auto anon_sym = SymbolTable::GetAnonPBName();
    auto pv = AST::Make<AST::Identifier>(new_pb->LOC(), anon_sym);
    pv->SetType(MakeBoundedITupleType(Shape(1, 1)));
    pb.SetPV(pv);

    // elements
    auto spv = AST::Make<AST::MultiValues>(new_pb->LOC(), ", ");
    auto epv =
        AST::Make<AST::Identifier>(new_pb->LOC(), anon_sym + "__elem__x");
    epv->SetType(MakeBoundedIntegerType(sbe::nu(1)));
    spv->Append(epv);
    pb.SetSubPVs(spv);

    // bound
    auto p_bound = AST::MakeIntExpr(new_pb->LOC(), 1);
    p_bound->SetType(MakeIntegerType());
    pb.SetBoundExpr(p_bound);

    // element-bounds
    auto spv_bounds = AST::Make<AST::MultiValues>(new_pb->LOC(), ", ");
    spv_bounds->Append(p_bound->Clone());
    spv_bounds->SetType(MakeITupleType(1));
    pb.SetBoundExprs(spv_bounds);

    // add the pb level
    pb.stmts = AST::Make<AST::MultiNodes>(pb.LOC(), new_pb);

    VST_DEBUG(pb.InlinePrint(dbgs()); dbgs() << "\n   +-";
              new_pb->InlinePrint(dbgs()); dbgs() << "\n");

    return pb;
  }

  void Reset() { pb_tree.Clear(); }

  void FillPB(AST::ParallelBy* pb, FillType ty, ParallelLevel pl, int v = 1) {
    VST_DEBUG(dbgs() << "[PB-Fill] " << Name(ty) << "(" << STR(pl) << "): ";
              pb->InlinePrint(dbgs()); dbgs() << "\n");
    fill_info.emplace_back(pb, ty, pl, v);
  }

  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      Reset();
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      pb_stack.push_back(pb);
      if (pb_stack.size() > 1)
        pb_tree.AddChild(*(pb_stack.rbegin() + 1), pb_stack.back());
      else
        pb_tree.AddSingle(pb);
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      for (auto fi : fill_info) {
        if (fi.ft == Outer)
          InsertOuterLevel(*fi.pb, fi.lvl);
        else if (fi.ft == Inner)
          InsertInnerLevel(*fi.pb, fi.lvl, fi.ubound);
        else if (fi.ft == AppendInner)
          AppendInnerLevel(*fi.pb, fi.lvl, fi.ubound);
        else {
          assert(fi.ft == LastInner);
          InsertLastInnerLevel(*fi.pb, fi.lvl, fi.ubound);
        }
      }
      fill_info.clear();
      return true;
    }
    if (!isa<AST::ParallelBy>(&n)) return true;
    pb_stack.pop_back();
    auto pb = cast<AST::ParallelBy>(&n);

    if (ExplicitLevel(pb) && !TargetHasLevel(pb->GetLevel()))
      Error1(pb->LOC(),
             STR(pb->GetLevel()) +
                 " level is not supported by the target architecture: " +
                 CCtx().GetArch() + ".");

    auto literal_depth = pb_tree.GetDepth(pb) + 1;
    if (literal_depth > (size_t)TargetMaxDepth()) {
      Error1(pb->LOC(),
             "too many parallel-by levels: " + std::to_string(literal_depth) +
                 " > " + std::to_string(TargetMaxDepth()) + ".");
    }

    if (!ExplicitLevel(pb)) InferImplicitLevel(pb);

    FillMissingLevels(pb);

    return true;
  }

  bool InferImplicitLevel(AST::ParallelBy* pb) const {
    auto InferAs = [this, &pb](ParallelLevel pl) {
      VST_DEBUG(dbgs() << "[PB-Infer] " << STR(pl) << ": ";
                pb->InlinePrint(dbgs()); dbgs() << "\n";);
      pb->SetLevel(pl);
    };

    if (pb_tree.IsLeaf(pb)) {
      // if not annotated, the inner-most is always thread level
      InferAs(ParallelLevel::THREAD);
      return true;
    }

    assert(pb_tree.GetDepth(pb) < (size_t)TargetMaxDepth());
    auto literal_depth = pb_tree.GetDepth(pb) + 1;

    auto& children = pb_tree.GetChildren(pb);
    auto& child = children.front();

    auto child_level = child->GetLevel();
    assert(child_level != ParallelLevel::UNKNOWN);
    assert(child_level != ParallelLevel::NONE);

    auto max_literal_depth = pb_tree.GetDepth(pb) + pb_tree.GetHeight(pb);

    // Maintain this table for easier code check:
    // +--------------------------+------------------------+
    // | LEVEL and LITERAL DEPTH  |     PARENT LEVELS      |
    // +--------------+-----+-----+-------+-------+--------+
    // |    CHILD     | MAX | P.. | N = 2 | N = 3 | N = 4  |
    // +----------+---+-----+-----+-------+-------+--------+
    // | THREAD   | 2 |  2  |  1  | BLOCK | BLOCK | BLOCK  |
    // | GROUP    | 2 |  2  |  1  | error | BLOCK | BLOCK  |
    // | GROUPx4  | 2 |  2  |  1  | error | BLOCK | BLOCK  |
    // | BLOCK    | 2 |  2  |  1  | error | error | error  |
    // +----------+---+-----+-----+-------+-------+--------+
    // | THREAD   | 2 |  3  |  1  |   -   | error | error  |
    // | GROUP    | 2 |  3  |  1  |   -   | BLOCK | BLOCK  |
    // | GROUPx4  | 2 |  3  |  1  |   -   | BLOCK | BLOCK  |
    // | BLOCK    | 2 |  3  |  1  |   -   | error | error  |
    // +----------+---+-----+-----+-------+-------+--------+
    // | THREAD   | 3 |  3  |  2  |   -   | GROUP | GROUP  |
    // | GROUP    | 3 |  3  |  2  |   -   | error | GROUPx4|
    // | GROUPx4  | 3 |  3  |  2  |   -   | error | error  |
    // | BLOCK    | 3 |  3  |  2  |   -   | error | error  |
    // +----------+---+-----+-----+-------+-------+--------+
    // | THREAD   | 2 |  4  |  1  |   -   |   -   | error  |
    // | GROUP    | 2 |  4  |  1  |   -   |   -   | error  |
    // | GROUPx4  | 2 |  4  |  1  |   -   |   -   | BLOCK  |
    // | BLOCK    | 2 |  4  |  1  |   -   |   -   | error  |
    // +----------+---+-----+-----+-------+-------+--------+
    // | THREAD   | 3 |  4  |  2  |   -   |   -   | error  |
    // | GROUP    | 3 |  4  |  2  |   -   |   -   | GROUPx4|
    // | GROUPx4  | 3 |  4  |  2  |   -   |   -   | error  |
    // | BLOCK    | 3 |  4  |  2  |   -   |   -   | error  |
    // +----------+---+-----+-----+-------+-------+--------+
    // | THREAD   | 4 |  4  |  3  |   -   |   -   | GROUP  |
    // | GROUP    | 4 |  4  |  3  |   -   |   -   | error  |
    // | GROUPx4  | 4 |  4  |  3  |   -   |   -   | error  |
    // | BLOCK    | 4 |  4  |  3  |   -   |   -   | error  |
    // +----------+---+-----+-----+-------+-------+--------+

    if (max_literal_depth == 2) {
      assert(literal_depth == 1);
      switch (child_level) {
      case ParallelLevel::THREAD: {
        InferAs(ParallelLevel::BLOCK);
      } break;
      case ParallelLevel::GROUP: {
        if (TargetMaxDepth() == 2)
          Error1(pb->LOC(), "the parallel level can not be inferred.");
        else
          InferAs(ParallelLevel::BLOCK);
      } break;
      case ParallelLevel::GROUPx4: {
        if (TargetMaxDepth() == 2)
          Error1(pb->LOC(), "the parallel level can not be inferred.");
        else
          InferAs(ParallelLevel::BLOCK);
      } break;
      case ParallelLevel::BLOCK: {
        Error1(pb->LOC(), "the parallel level can not be inferred.");
      } break;
      default: choreo_unreachable("unsupported parallel level.");
      }
    } else if (max_literal_depth == 3) {
      if (literal_depth == 1) {
        switch (child_level) {
        case ParallelLevel::THREAD: {
          if (TargetMaxDepth() == 3 || TargetMaxDepth() == 4)
            Error1(pb->LOC(), "the parallel level can not be inferred.");
          else
            InferAs(ParallelLevel::BLOCK);
        } break;
        case ParallelLevel::GROUP: {
          InferAs(ParallelLevel::BLOCK);
        } break;
        case ParallelLevel::GROUPx4: {
          InferAs(ParallelLevel::BLOCK);
        } break;
        case ParallelLevel::BLOCK: {
          Error1(pb->LOC(), "the parallel level can not be inferred.");
        } break;
        default: choreo_unreachable("unsupported parallel level.");
        }
      } else if (literal_depth == 2) {
        switch (child_level) {
        case ParallelLevel::THREAD: {
          if (TargetMaxDepth() == 2)
            Error1(pb->LOC(), "the parallel level can not be inferred.");
          else
            InferAs(ParallelLevel::GROUP);
        } break;
        case ParallelLevel::GROUP: {
          if (TargetMaxDepth() <= 3)
            Error1(pb->LOC(), "the parallel level can not be inferred.");
          else
            InferAs(ParallelLevel::GROUPx4);
        } break;
        case ParallelLevel::GROUPx4: {
          InferAs(ParallelLevel::BLOCK);
        } break;
        case ParallelLevel::BLOCK: {
          Error1(pb->LOC(), "the parallel level can not be inferred.");
        } break;
        default: choreo_unreachable("unsupported parallel level.");
        }
      } else
        choreo_unreachable("unsupported literal depth: " +
                           std::to_string(literal_depth));
    } else if (max_literal_depth == 4) {
      if (child_level == ParallelLevel::GROUPx4 && literal_depth == 1)
        InferAs(ParallelLevel::BLOCK);
      else if (child_level == ParallelLevel::GROUP && literal_depth == 2)
        InferAs(ParallelLevel::GROUPx4);
      else if (child_level == ParallelLevel::THREAD && literal_depth == 3)
        InferAs(ParallelLevel::GROUP);
      else
        Error1(pb->LOC(), "the parallel level can not be inferred.");
    }
    return true;
  }

  bool FillMissingLevels(AST::ParallelBy* pb) {
    for (auto& child : pb_tree.GetChildren(pb)) {
      auto cd = TargetDepth(child->GetLevel());
      auto pd = TargetDepth(pb->GetLevel());
      if (pd >= cd) {
        Error1(pb->LOC(), "unable to infer the parallel-by level(s) between " +
                              STR(pb->GetLevel()) + " and " +
                              STR(child->GetLevel()) + ".");
      } else if (cd - pd > 1)
        FillMissingBetweens(pb, child);
    }

    if (pb_tree.IsLeaf(pb)) FillMissingInners(pb);
    if (pb_tree.IsRoot(pb)) FillMissingOuters(pb);

    return true;
  }

  // Fill the missing levels between parent and child
  bool FillMissingBetweens(AST::ParallelBy* pb, AST::ParallelBy* child) {
    bool support_group = TargetHasLevel(ParallelLevel::GROUP);
    bool support_4x_group = TargetHasLevel(ParallelLevel::GROUPx4);

    auto pl = pb->GetLevel();
    auto cl = child->GetLevel();

    assert(pl != ParallelLevel::UNKNOWN);
    assert(cl != ParallelLevel::UNKNOWN);
    assert(TargetDepth(pl) < TargetDepth(cl));

    switch (cl) {
    case ParallelLevel::THREAD: {
      switch (pl) {
      case ParallelLevel::BLOCK: {
        if (support_group) FillPB(child, Outer, ParallelLevel::GROUP);
        if (support_4x_group) FillPB(child, Outer, ParallelLevel::GROUPx4);
      } break;
      case ParallelLevel::GROUPx4: {
        if (support_group) FillPB(child, Outer, ParallelLevel::GROUP);
      } break;
      case ParallelLevel::GROUP:
        // no-fill
        break;
      case ParallelLevel::THREAD:
      default:
        Error1(pb->LOC(), "unable to infer the parallel-by level(s) between  " +
                              STR(pl) + " and " + STR(cl) + ".");
        break;
      }
    } break;
    case ParallelLevel::GROUP: {
      switch (pl) {
      case ParallelLevel::BLOCK: {
        if (support_4x_group) FillPB(child, Outer, ParallelLevel::GROUPx4);
      } break;
      case ParallelLevel::GROUPx4: {
        // no-fill
      } break;
      case ParallelLevel::GROUP:
      case ParallelLevel::THREAD:
      default:
        Error1(pb->LOC(), "unable to infer the parallel-by level(s) between  " +
                              STR(pl) + " and " + STR(cl) + ".");
        break;
      }
    } break;
    case ParallelLevel::GROUPx4: {
      switch (pl) {
      case ParallelLevel::BLOCK: {
        // no-fill
      } break;
      case ParallelLevel::GROUPx4:
      case ParallelLevel::GROUP:
      case ParallelLevel::THREAD:
      default:
        Error1(pb->LOC(), "unable to infer the parallel-by level(s) between  " +
                              STR(cl) + " and " + STR(pl) + ".");
        break;
      }
    } break;
    case ParallelLevel::BLOCK:
    default:
      Error1(pb->LOC(), "unable to infer the parallel-by level(s) between  " +
                            STR(pl) + " and " + STR(cl) + ".");
    }
    return true;
  }

  bool FillMissingInners(AST::ParallelBy* pb) {
    assert(pb_tree.IsLeaf(pb) && "can not fill non-leaf inners.");
    auto level = pb->GetLevel();
    assert(level != ParallelLevel::UNKNOWN);
    bool support_group = TargetHasLevel(ParallelLevel::GROUP);
    bool support_4x_group = TargetHasLevel(ParallelLevel::GROUPx4);

    switch (level) {
    case ParallelLevel::BLOCK: {
      FillPB(pb, AppendInner, ParallelLevel::THREAD);
      if (support_4x_group) {
        FillPB(pb, LastInner, ParallelLevel::GROUP);
        FillPB(pb, LastInner, ParallelLevel::GROUPx4);
      } else if (support_group)
        FillPB(pb, LastInner, ParallelLevel::GROUP);
    } break;
    case ParallelLevel::GROUPx4: {
      FillPB(pb, AppendInner, ParallelLevel::THREAD,
             CCtx().GetMinGroupDim() * 4);
      if (support_group) FillPB(pb, LastInner, ParallelLevel::GROUP);
    } break;
    case ParallelLevel::GROUP: {
      FillPB(pb, AppendInner, ParallelLevel::THREAD, CCtx().GetMinGroupDim());
    } break;
    case ParallelLevel::THREAD: break;
    default:
      Error1(pb->LOC(), "unable to infer the parallel-by level(s) inside " +
                            STR(level) + ".");
      break;
    }
    return true;
  }

  bool FillMissingOuters(AST::ParallelBy* pb,
                         ParallelLevel min_lvl = ParallelLevel::SEQ) {
    auto level = pb->GetLevel();
    assert(level != ParallelLevel::UNKNOWN);
    bool support_group = TargetHasLevel(ParallelLevel::GROUP);
    bool support_4x_group = TargetHasLevel(ParallelLevel::GROUPx4);
    switch (level) {
    case ParallelLevel::BLOCK: {
    } break;
    case ParallelLevel::GROUPx4: {
      if (TargetDepth(min_lvl) < TargetDepth(ParallelLevel::BLOCK))
        FillPB(pb, Outer, ParallelLevel::BLOCK);
    } break;
    case ParallelLevel::GROUP: {
      if (support_4x_group &&
          (TargetDepth(min_lvl) < TargetDepth(ParallelLevel::GROUPx4)))
        FillPB(pb, Outer, ParallelLevel::GROUPx4);
      if (TargetDepth(min_lvl) < TargetDepth(ParallelLevel::BLOCK))
        FillPB(pb, Outer, ParallelLevel::BLOCK);
    } break;
    case ParallelLevel::THREAD: {
      if (support_group &&
          (TargetDepth(min_lvl) < TargetDepth(ParallelLevel::GROUP)))
        FillPB(pb, Outer, ParallelLevel::GROUP);
      if (support_4x_group &&
          (TargetDepth(min_lvl) < TargetDepth(ParallelLevel::GROUPx4)))
        FillPB(pb, Outer, ParallelLevel::GROUPx4);
      if (TargetDepth(min_lvl) < TargetDepth(ParallelLevel::BLOCK))
        FillPB(pb, Outer, ParallelLevel::BLOCK);
    } break;
    default:
      Error1(pb->LOC(),
             "unable to infer parallel-by level outside " + STR(level) + ".");
      break;
    }

    return true;
  }

  bool NormPB(AST::ParallelBy& n) {
    // fill the sub elements
    if (!n.HasSubPVs()) {
      // `parallel p by 2`  ==> `parallel p={p__elem__x} by [2]`
      auto spv = AST::Make<AST::MultiValues>(n.LOC(), ", ");
      spv->Append(AST::Make<AST::Identifier>(n.BPV()->LOC(),
                                             n.BPV()->name + "__elem__x"));
      n.SetSubPVs(spv);
      auto sub = AST::Make<AST::MultiValues>(n.LOC(), ", ");
      sub->Append(n.BoundExpr()->Clone());
      n.SetBoundExprs(sub);
      n.SubPVs()->ValueAt(0)->SetType(NodeType(*n.BPV()));
      n.SubPVs()->SetType(NodeType(*n.BPV()));
      VST_DEBUG(dbgs() << "Generate cmpt_bpvs in parallelby for '"
                       << PSTR(n.BPV()) << "': " << STR(n.SubPVs()) << "\n");
    }
    return true;
  }

  bool Visit(AST::ParallelBy& pb) override { return NormPB(pb); }
};

class Normalizer : public VisitorGroup {
private:
  CompoundNorm comp;
  ParaByFiller filler;
  LoopNorm ln;

public:
  Normalizer() : VisitorGroup("norm", comp, filler, ln) {}
};

} // end namespace Choreo

#endif // __CHOREO_NORMALIZATION_HPP__
