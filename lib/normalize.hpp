#ifndef __CHOREO_NORMALIZATION_HPP__
#define __CHOREO_NORMALIZATION_HPP__

// This applies 'normalization' or 'canonicalization' of AST for easier handling
// in later visiting passes.

#include <iostream>
#include <tuple>

#include "symtab.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

struct Normalizer : public VisitorWithScope {
private:
  bool changed = false;

  std::string old;
  size_t count = 0; // name suffix of runtime int values

  bool handle_parameter = false;
  ptr<AST::Expr> list_ref = nullptr;
  void SetListReference(const location& l, const std::string& r) {
    list_ref = AST::Make<AST::Expr>(l, AST::Make<AST::Identifier>(l, r));
  }
  void ResetListReference() { list_ref = nullptr; }

  // for node hoisting
  using NodeInsertInfo =
      std::vector<std::tuple<int, ptr<AST::Node>, std::string>>;
  std::stack<AST::MultiNodes*> multi_nodes;
  int cur_node_index = -1;
  std::map<AST::MultiNodes*, NodeInsertInfo> mnodes_insertions;

  std::string GetInternalValueString() { return "$" + std::to_string(count++); }

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) {
      dbgs() << n.TypeNameString();
      if (!n.IsBlock()) dbgs() << ": " << STR(n);
      dbgs() << "\n";
    }
  }

public:
  // it does not require a symbol table
  Normalizer() : VisitorWithScope("norm") {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "before visiting " << n.TypeNameString() << "\n";

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
    } else if (auto d = dyn_cast<AST::DMA>(&n)) {
      cur_node_index = multi_nodes.top()->GetIndex(d);
      assert(cur_node_index != -1 && "unexpected node index.");
    } else if (auto d = dyn_cast<AST::Return>(&n)) {
      cur_node_index = multi_nodes.top()->GetIndex(d);
      assert(cur_node_index != -1 && "unexpected node index.");
    } else if (auto f = dyn_cast<AST::ForeachBlock>(&n)) {
      cur_node_index = multi_nodes.top()->GetIndex(f);
      assert(cur_node_index != -1 && "unexpected node index.");
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "after visiting " << n.TypeNameString() << "\n";

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
      cur_node_index = -1;
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
    cur_node_index = -1;

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
              auto new_expr =
                  AST::Make<AST::Expr>(expr->LOC(), "dimof", list_ref, ref);
              VST_DEBUG(dbgs() << "Desugar ref: " << STR(*expr) << " ---> "
                               << STR(*new_expr) << "\n");
              n.values[i] = new_expr;
            }
          }
        }
      }
    }

    if (handle_parameter) { // make runtime values of "?" to be named
      for (size_t i = 0; i < n.values.size(); ++i) {
        if (auto il = dyn_cast<AST::IntLiteral>(n.values[i])) {
          if (!IsUnKnownInteger(il->value)) continue;
          auto new_il =
              AST::Make<AST::Identifier>(il->LOC(), GetInternalValueString());

          changed = true;
          n.values[i] = new_il;
        }
      }
    }

    return true;
  }

  bool Visit(AST::IntLiteral&) override { return true; }
  bool Visit(AST::FloatLiteral&) override { return true; }
  bool Visit(AST::StringLiteral&) override { return true; }
  bool Visit(AST::Boolean&) override { return true; }
  bool Visit(AST::Expr& n) override {
    TraceEachVisit(n);
    if (list_ref) { // could be with syntax sugar
      auto Apply = [this](const ptr<AST::Expr>& expr) -> ptr<AST::Expr> {
        if (!expr) return nullptr;
        if (auto ref = expr->GetReference()) {
          if (!isa<AST::IntIndex>(ref.get())) return nullptr;

          // apply desugaring a {(0), 1} -> {a(0), 1}
          auto ret = AST::Make<AST::Expr>(expr->LOC(), "dimof", list_ref, ref);

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

    return true;
  }

  bool Visit(AST::MultiDimSpans&) override { return true; }
  bool Visit(AST::NamedTypeDecl&) override { return true; }
  bool Visit(AST::NamedVariableDecl& n) override {
    TraceEachVisit(n);
    if (n.mem && (n.mem->Get() == Storage::DEFAULT)) {
      // Should this be set by target?
      n.mem->Set(Storage::GLOBAL);

      VST_DEBUG(dbgs() << "Place storage of '" << n.name_str
                       << "': DEFAULT ---> GLOBAL\n");
    }

    return true;
  }
  bool Visit(AST::IntTuple& n) override {
    bool replace_mv = false;
    auto mv = AST::Make<AST::MultiValues>(n.GetValues()->LOC(), ",");
    for (auto& v : n.GetValues()->AllValues()) {
      if (isa<AST::IntLiteral>(v)) {
        mv->Append(v);
        continue;
      }
      auto expr = cast<AST::Expr>(v);
      if (auto itt = dyn_cast<ITupleType>(expr->GetType())) {
        VST_DEBUG(dbgs() << "Replace " << PSTR(expr) << " in " << STR(n)
                         << " with:\n");
        for (size_t idx = 0; idx < itt->dim_count; ++idx) {
          auto ii = AST::Make<AST::IntIndex>(
              expr->LOC(), AST::Make<AST::IntLiteral>(expr->LOC(), idx));
          auto new_expr = AST::Make<AST::Expr>(expr->LOC(), "dimof", expr, ii);
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
  bool Visit(AST::DataAccess&) override { return true; }
  bool Visit(AST::Assignment&) override { return true; }
  bool Visit(AST::IntIndex&) override { return true; }
  bool Visit(AST::DataType&) override { return true; }
  bool Visit(AST::Identifier&) override { return true; }
  bool Visit(AST::Parameter&) override { return true; }
  bool Visit(AST::ParamList&) override { return true; }
  bool Visit(AST::ParallelBy& n) override {
    if (n.HasBPV()) {
      if (n.cmpt_bpvs == nullptr) {
        // `parallel p by 2`  ==> `parallel p={p__elem__x} by [2]`
        n.cmpt_bpvs = AST::Make<AST::MultiValues>(n.LOC(), ", ");
        n.cmpt_bpvs->Append(
            AST::Make<AST::Identifier>(n.LOC(), n.bpv->name + "__elem__x"));
        n.cmpt_bounds = AST::Make<AST::MultiValues>(n.LOC(), ", ");
        if (auto b = VIInt(n.GetBound()))
          n.cmpt_bounds->Append(AST::Make<AST::IntLiteral>(n.LOC(), *b));
        else
          n.cmpt_bounds->Append(AST::Make<AST::Identifier>(
              n.LOC(), ValueItemAsString(n.GetBound())));
        n.cmpt_bpvs->ValueAt(0)->SetType(MakeBoundedIntegerType(n.GetBound()));
        n.bpv->SetType(MakeBoundedIntegerType(n.bpv->name));
        VST_DEBUG(dbgs() << "Generate cmpt_bpvs in parallelby for '"
                         << PSTR(n.bpv) << "': " << STR(n.cmpt_bpvs) << "\n");
      }
    } else {
      assert(n.cmpt_bpvs && "At least one bpv and cmpt_bpvs should exist!");
      // `parallel {px} by [2]`  ==> `parallel anon={px} by [2]`
      n.bpv = AST::Make<AST::Identifier>(n.LOC(), SymbolTable::GetAnonName());
      n.bpv->SetType(MakeBoundedITupleType(Shape(n.SubCount()), "pv"));
    }
    return true;
  }
  bool Visit(AST::WhereBind&) override { return true; }

  bool Visit(AST::WithIn& n) override {
    TraceEachVisit(n);

    if (isa<IntegerType>(n.in->GetType())) {
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
      mval->Append(AST::Make<AST::Identifier>(
          n.with->LOC(), n.with->name + "__elem__" + std::to_string(i)));
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

  bool Visit(AST::WithBlock&) override { return true; }
  bool Visit(AST::Memory&) override { return true; }
  bool Visit(AST::SpanAs&) override { return true; }

  bool Visit(AST::DMA& n) override {
    if (n.operation == ".any") return true;
    if (!isa<AST::Select>(n.to)) return true;

    auto anon_sym = SymbolTable::GetAnonName();
    assert(cur_node_index != -1);
    // hoist the select to multinodes
    int index = cur_node_index + mnodes_insertions[multi_nodes.top()].size();
    cast<AST::Select>(n.to)->inDMA = false;
    auto assign = AST::Make<AST::Assignment>(n.to->LOC(), anon_sym, n.to);
    assign->SetType(n.to->GetType());
    assign->da->SetType(n.to->GetType());
    mnodes_insertions[multi_nodes.top()].emplace_back(
        std::make_tuple(index, assign, anon_sym));
    VST_DEBUG(dbgs() << n.TypeNameString() << ": replace " << PSTR(n.to)
                     << " with " << anon_sym << ".\n");

    n.to = AST::Make<AST::ChunkAt>(
        n.to->LOC(), AST::Make<AST::Identifier>(n.to->LOC(), anon_sym));
    return true;
  }

  bool Visit(AST::ChunkAt& n) override {
    TraceEachVisit(n);

    if (n.sa) {
      assert(cur_node_index != -1);
      // hoist the span_as to multinodes
      int index = cur_node_index + mnodes_insertions[multi_nodes.top()].size();
      auto assign =
          AST::Make<AST::Assignment>(n.sa->LOC(), n.sa->nid->name, n.sa);
      assign->SetType(n.sa->GetType());
      assign->da->SetType(n.sa->GetType());
      mnodes_insertions[multi_nodes.top()].emplace_back(
          std::make_tuple(index, assign, n.sa->nid->name));
      VST_DEBUG(dbgs() << n.TypeNameString() << ": replace " << PSTR(n.sa)
                       << " with " << n.sa->nid->name << "\n");
      n.sa.reset();
    }

    // hoist any arith inside of chunkat positions
    for (auto tsi : n.AllTSInfo()) {
      std::vector<std::pair<int, ptr<AST::Node>>> repls;
      int i = -1;
      for (auto& v : tsi->GetIndices()) {
        ++i;
        auto expr = cast<AST::Expr>(v);
        if (expr->GetSymbol()) {
          // replace expr reference by the id node
          repls.emplace_back(i, expr->GetReference());
          continue;
        } else if (expr->op == "getith") {
          // 'getith' must be kept.
          if (auto lexpr = dyn_cast<AST::Expr>(expr->GetL())) {
            if (!lexpr->GetSymbol()) {
              // hoist the non-getith part
              int index =
                  cur_node_index + mnodes_insertions[multi_nodes.top()].size();
              auto nname = SymbolTable::GetAnonName();
              auto assign = AST::Make<AST::Assignment>(expr->GetL()->LOC(),
                                                       nname, expr->GetL());
              assign->SetType(expr->GetL()->GetType());
              assign->da->SetType(expr->GetL()->GetType());
              mnodes_insertions[multi_nodes.top()].emplace_back(
                  std::make_tuple(index, assign, nname));
              VST_DEBUG(dbgs()
                        << n.TypeNameString() << ": replace "
                        << PSTR(expr->GetL()) << " with " << nname << "\n");
              expr->SetL(AST::Make<AST::Identifier>(v->LOC(), nname));
              VST_DEBUG(dbgs() << PSTR(expr->GetL()) << ".\n");
            }
          }
          continue;
        }

        // else, hoist the arith out
        int index =
            cur_node_index + mnodes_insertions[multi_nodes.top()].size();
        auto nname = SymbolTable::GetAnonName();
        auto assign = AST::Make<AST::Assignment>(v->LOC(), nname, v);
        assign->SetType(v->GetType());
        assign->da->SetType(v->GetType());
        mnodes_insertions[multi_nodes.top()].emplace_back(
            std::make_tuple(index, assign, nname));
        repls.emplace_back(i, AST::Make<AST::Identifier>(v->LOC(), nname));
        VST_DEBUG(dbgs() << n.TypeNameString() << ": replace " << PSTR(v)
                         << " with " << nname << "\n");
      }
      for (auto& repl : repls) {
        VST_DEBUG(dbgs() << n.TypeNameString() << ": replace "
                         << PSTR(tsi->Positions()->ValueAt(repl.first))
                         << " with ");

        tsi->Positions()->values[repl.first] = repl.second;

        VST_DEBUG(dbgs() << PSTR(tsi->Positions()->ValueAt(repl.first))
                         << ".\n");
      }
    }
    return true;
  }
  bool Visit(AST::Wait&) override { return true; }
  bool Visit(AST::Trigger&) override { return true; }
  bool Visit(AST::Call&) override { return true; }
  bool Visit(AST::Rotate&) override { return true; }
  bool Visit(AST::Synchronize&) override { return true; }
  bool Visit(AST::Select&) override { return true; }
  bool Visit(AST::Return& n) override {
    TraceEachVisit(n);

    if (AST::GetIdentifier(*n.value)) return true;

    if (CCtx().GetTarget() != CompileTarget::Factor) return true;

    // non-identifier may be normalized
    auto vty = NodeType(*n.value);

    // tricky: we must convert a integer to be 's32 [1] ...' for a factor return
    // value;
    if (isa<IntegerType>(vty)) {
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
        auto nv = AST::Make<AST::NamedVariableDecl>(
            loc, anon_sym, dt, sto, nullptr, std::vector<size_t>{}, il);
        nv->SetType(vty);

        assert(cur_node_index != -1);
        int index =
            cur_node_index + mnodes_insertions[multi_nodes.top()].size();
        mnodes_insertions[multi_nodes.top()].emplace_back(
            std::make_tuple(index, nv, anon_sym));

        // replace return value now
        VST_DEBUG(dbgs() << "[Norm] Replace " << STR(n) << "\n to be:\n");
        n.value = AST::MakeIdExpr(n.value->LOC(), anon_sym);
        VST_DEBUG(dbgs() << STR(n) << "\n");

        // In host, its return type is still 'int'
        n.SetNote("host-type:int");
      }
    }

    return true;
  }
  bool Visit(AST::LoopRange&) override { return true; }
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
              int index =
                  cur_node_index + mnodes_insertions[multi_nodes.top()].size();
              auto nname = SymbolTable::GetAnonName();
              auto assign = AST::Make<AST::Assignment>(
                  bound_expr->GetL()->LOC(), nname, bound_expr->GetL());
              auto lty = bound_expr->GetL()->GetType();
              assign->SetType(lty);
              assign->da->SetType(lty);
              mnodes_insertions[multi_nodes.top()].emplace_back(
                  std::make_tuple(index, assign, nname));
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
            cur_node_index + mnodes_insertions[multi_nodes.top()].size();
        auto nname = SymbolTable::GetAnonName();
        auto assign = AST::Make<AST::Assignment>(v->LOC(), nname, bound_expr);
        auto bty = bound_expr->GetType();
        assign->SetType(bty);
        assign->da->SetType(bty);
        mnodes_insertions[multi_nodes.top()].emplace_back(
            std::make_tuple(index, assign, nname));
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
  bool Visit(AST::InThreadsBlock&) override { return true; }
  bool Visit(AST::IfElseBlock&) override { return true; }
  bool Visit(AST::IncrementBlock&) override { return true; }
  bool Visit(AST::FunctionDecl&) override { return true; }
  bool Visit(AST::ChoreoFunction&) override { return true; }
  bool Visit(AST::CppSourceCode&) override { return true; }
  bool Visit(AST::Program&) override { return true; }
};

} // end namespace Choreo

#endif // __CHOREO_NORMALIZATION_HPP__
