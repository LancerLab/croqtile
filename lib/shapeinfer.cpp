#include "shapeinfer.hpp"

using namespace Choreo::valno;

namespace Choreo {

void ShapeInference::InvalidateVisitorValNOs() {
  InvalidateVN(cur_vn);
  InvalidateVN(cur_mdspan_vn);
  InvalidateVN(cur_ub_vn);
}

void ShapeInference::TraceEachVisit(AST::Node& n, bool detail,
                                    const std::string& m) const {
  if (!trace_visit && !debug_visit) return;

  if (debug_visit) dbgs() << "[";

  if (detail)
    dbgs() << m << STR(n);
  else
    dbgs() << m << n.TypeNameString();

  if (debug_visit)
    dbgs() << "]\t "
           << (ValidVN(cur_vn) ? ("#" + std::to_string(cur_vn)) : "nil")
           << "(vn),\t "
           << (ValidVN(cur_mdspan_vn) ? ("#" + std::to_string(cur_mdspan_vn))
                                      : "nil")
           << "(mds),\t "
           << (ValidVN(cur_ub_vn) ? ("#" + std::to_string(cur_ub_vn)) : "nil")
           << "(ub)";

  dbgs() << "\n";
}

bool ShapeInference::BeforeVisitImpl(AST::Node& n) {
  TraceEachVisit(n, false, "before ");

  if (isa<AST::Program>(&n)) {
    vn.EnterScope(); // global scope
    ast_vn.EnterScope();
  } else if (isa<AST::ChoreoFunction>(&n)) {
    vn.EnterScope();
    ast_vn.EnterScope();
    cannot_proceed = false; // recover state when starting a new function
    NumTy valno = vn.GetOrGenValueNumberFromSignature("const_1");
    ValNoAliasSign(InScopeName("@__choreo_no_tiling__"), valno);
    vn.GetOrGenValueNumberFromSignature(InScopeName("__choreo_no_tiling__"));
    InvalidateVisitorValNOs();
  } else if (isa<AST::ParallelBy>(&n)) {
    vn.EnterScope();
    ast_vn.EnterScope();
  } else if (isa<AST::WithBlock>(&n) || isa<AST::InThreadsBlock>(&n) ||
             isa<AST::IfElseBlock>(&n)) {
    vn.EnterScope();
    ast_vn.EnterScope();
  } else if (isa<AST::ForeachBlock>(&n) || isa<AST::IncrementBlock>(&n)) {
    vn.EnterScope();
    ast_vn.EnterScope();
    gen_values = false; // disable valno on range expressions
  } else if (auto* b = dyn_cast<AST::MultiDimSpans>(&n)) {
    if (b->ref_name != "") {
      if (auto n = SSTab().NameInScopeOrNull(b->ref_name))
        vn.SetListReference(n.value());
      else
        choreo_unreachable("variable `" + b->ref_name +
                           "' is not found in scopes.");
    }
  } else if (auto* b = dyn_cast<AST::IntTuple>(&n)) {
    if (b->ref_name != "") {
      if (auto n = SSTab().NameInScopeOrNull(b->ref_name))
        vn.SetListReference(n.value());
      else
        choreo_unreachable("variable `" + b->ref_name +
                           "' is not found in scopes.");
    }
  } else if (isa<AST::Wait>(&n) || isa<AST::Rotate>(&n) ||
             isa<AST::Select>(&n) || isa<AST::Trigger>(&n) ||
             isa<AST::Call>(&n) || isa<AST::DataAccess>(&n)) {
    gen_values = false;
  } else if (isa<AST::Parameter>(&n)) {
    allow_named_dim = true;
  } else if (isa<AST::MultiNodes>(&n))
    InvalidateVisitorValNOs();

  return true;
}

bool ShapeInference::InMidVisitImpl(AST::Node& n) {
  if (isa<AST::IfElseBlock>(&n)) {
    vn.LeaveScope(); // must clear the vn inside if-scope
    ast_vn.LeaveScope();
    vn.EnterScope();
    ast_vn.EnterScope();
  }
  return true;
}

bool ShapeInference::AfterVisitImpl(AST::Node& n) {
  TraceEachVisit(n, false, "after ");
  if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
      isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n)) {
    vn.LeaveScope();
    ast_vn.LeaveScope();
  } else if (isa<AST::ForeachBlock>(&n) || isa<AST::InThreadsBlock>(&n) ||
             isa<AST::IfElseBlock>(&n) || isa<AST::IncrementBlock>(&n)) {
    vn.LeaveScope();
    ast_vn.LeaveScope();
  } else if (isa<AST::MultiDimSpans>(&n) || isa<AST::IntTuple>(&n)) {
    vn.ResetListReference();
  } else if (isa<AST::Wait>(&n) || isa<AST::Call>(&n) || isa<AST::Rotate>(&n) ||
             isa<AST::Select>(&n) || isa<AST::Trigger>(&n) ||
             isa<AST::DataAccess>(&n)) {
    gen_values = true;
  } else if (isa<AST::Parameter>(&n)) {
    allow_named_dim = false;
  }

  return true;
}

ptr<Type> ShapeInference::NodeType(const AST::Node& n) const {
  if (auto id = dyn_cast<AST::Identifier>(&n)) {
    if (!SSTab().IsDeclared(id->name)) {
      return n.GetType();
    } else {
      return GetSymbolType(id->name);
    }
  } else if (auto expr = dyn_cast<AST::Expr>(&n)) {
    if (auto sym = expr->GetSymbol()) return GetSymbolType(sym->name);
    return expr->GetType();
  }
  return VisitorWithScope::NodeType(n);
}

bool ShapeInference::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;
  return true;
}

void ShapeInference::CollapseMultiValues(const AST::MultiValues& mv) {
  auto GetValNoList = [this](std::vector<NumTy>& vs, NumTy valno) {
    std::deque<NumTy> work_list;
    work_list.push_back(valno);
    while (!work_list.empty()) {
      auto val_no = work_list.front();
      work_list.pop_front();

      auto val_sign = SignValNo(val_no);
      auto vn_count = CountElementsInSignature(val_sign);

      assert(vn_count >= 1);

      if (vn_count == 1) {
        vs.push_back(val_no);
        continue;
      }

      for (int i = vn_count - 1; i >= 0; --i)
        work_list.push_front(vn.GetNthValNo(val_sign, i));
    }
  };

  std::vector<NumTy> vvs; // values
  std::vector<NumTy> uvs; // ubounds
  for (auto v : mv.AllValues()) {
    // 'getith' is specific: it does not affect the ubound valno
    auto gv = v;
    if (auto e = dyn_cast<AST::Expr>(v); e && (e->op == "getith"))
      gv = cast<AST::Expr>(e->GetL())->GetSymbol();

    // bounded ituples variable can be collapsed to be multiple variables
    if (auto id = AST::GetIdentifier(gv);
        id && bv_map.count(SSTab().InScopeName(id->name)) &&
        bv_map[SSTab().InScopeName(id->name)].size() > 1) {
      for (auto& name : bv_map[SSTab().InScopeName(id->name)]) {
        vvs.push_back(vn.GetValueNumberOfSignature(name));

        auto uname = SSTab().GetScope(name) + "@" + SSTab().UnScopedName(name);
        uvs.push_back(vn.GetValueNumberOfSignature(uname));
      }
    } else {
      if (HasValNo(*gv, VNKind::VNK_UBOUND))
        GetValNoList(uvs, GetValNo(*v, VNKind::VNK_UBOUND));

      if (HasValNo(*v, VNKind::VNK_MDSPAN))
        GetValNoList(vvs,
                     GetValNo(*v, VNKind::VNK_MDSPAN)); // flatten the mdspan
      else
        GetValNoList(vvs, GetValNo(*v, VNKind::VNK_VALUE));
    }
  }

  if (!uvs.empty()) assert(uvs.size() == vvs.size());

  auto SetMultiValNo = [this, &mv](const std::vector<NumTy> vs, VNKind vnt) {
    assert(!ast_vn.Hit(&mv, vnt));
    assert(!vs.empty());

    NumTy valno = GetInvalidValueNumber();
    if (vs.size() == 1) {
      valno = vs[0];
    } else {
      std::ostringstream oss;
      for (size_t i = 0; i < vs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "#" << vs[i];
      }
      valno = vn.GetOrGenValueNumberFromSignature(oss.str());
    }
    ast_vn.Update(&mv, valno, vnt);
  };

  // note: multivalues may represent a mdspan but does not have a VNK_MDSPAN
  // valno.
  if (!vvs.empty()) SetMultiValNo(vvs, VNKind::VNK_VALUE);
  if (!uvs.empty()) SetMultiValNo(uvs, VNKind::VNK_UBOUND);
}

bool ShapeInference::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);

  if (cannot_proceed || !CanBeValueNumbered(&n)) {
    InvalidateVN(cur_vn);
    return true;
  }

  if (gen_values) CollapseMultiValues(n);

  InvalidateVN(cur_vn);

  return true;
}

bool ShapeInference::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;
  cur_vn = GenValNo(n);
  return true;
}

bool ShapeInference::Visit(AST::FloatLiteral& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;
  cur_vn = GenValNo(n);
  return true;
}

bool ShapeInference::Visit(AST::StringLiteral& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;
  InvalidateVN(cur_vn);
  SetNodeType(n, MakeStringType());
  return true;
}

bool ShapeInference::Visit(AST::BoolLiteral& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;
  cur_vn = GenValNo(n);
  return true;
}

bool ShapeInference::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;

  if (!CanBeValueNumbered(&n)) {
    InvalidateVN(cur_vn);
    return true;
  }

  auto nty = NodeType(n);
  // std::cout << "Node: " << STR(n) << ", type: " << PSTR(nty) << "\n";
  cur_vn = GenValNo(n);

  auto ShouldOpt = [](const ValueList& vl) -> bool {
    if (!IsValidValueList(vl)) return false;
    // factor should always use expression when symbolic
    if (CCtx().GetTarget() == CompileTarget::Factor)
      return IsValueListNumericOrBool(vl);
    return true;
  };
  // decide the optimized values
  switch (NodeValNoKind(n)) {
  case VNKind::VNK_VALUE: {
    auto vl = vn.GenValueListFromSignature(GetSign(n));
    if (ShouldOpt(vl) && !n.ContainsNote("diverge")) n.Opts().SetVals(vl);
    VST_DEBUG(dbgs() << " |-<exprval> <" << PSTR(nty) << "> " << STR(n) << ": "
                     << STR(vl) << "\n");
  } break;
  case VNKind::VNK_UBOUND: {
    auto vl = vn.GenValueListFromSignature(GetSign(n));
    if (ShouldOpt(vl) && !n.ContainsNote("diverge")) n.Opts().SetVals(vl);
    VST_DEBUG(dbgs() << " |-<exprval> <" << PSTR(nty) << "> " << STR(n) << ": "
                     << STR(vl) << "\n");
    auto ub_vl = vn.GenValueListFromSignature(GetSign(n, VNKind::VNK_UBOUND));
    if (ShouldOpt(ub_vl)) n.Opts().SetUBounds(ub_vl);
    VST_DEBUG(dbgs() << " |-<exprbound> <" << PSTR(nty) << "> " << STR(n)
                     << ": " << STR(ub_vl) << "\n");
  } break;
  case VNKind::VNK_MDSPAN: {
    auto mds_valno = GetValNo(n, VNKind::VNK_MDSPAN);
    if (!ValidVN(mds_valno)) return true; // drop all the work
    auto vl = vn.GenValueListFromSignature(SignValNo(mds_valno));
    if (ShouldOpt(vl)) n.Opts().SetVals(vl);
    VST_DEBUG(dbgs() << " |-<exprspan> <" << PSTR(nty) << "> " << STR(n) << ": "
                     << STR(vl) << "\n");
    auto sz = MultiplyAll(vl);
    n.Opts().SetSize(sz);
    VST_DEBUG(dbgs() << " |-<exprsize> <" << PSTR(nty) << "> " << STR(n) << ": "
                     << STR(sz) << "\n");
  } break;
  default: choreo_unreachable("unsupported valno kind.");
  }

  // fullfil the incomplete shaped types when necessary
  n.s = GenShapeFromSignature(SignValNo(cur_vn));
  VST_DEBUG(dbgs() << " |-<exprshape> <" << PSTR(nty) << "> " << STR(n) << ": "
                   << STR(n.s) << "\n");

  if (isa<BoundedITupleType>(nty)) {
    SetNodeType(n, MakeBoundedITupleType(n.s));
  } else if (isa<MDSpanType>(nty)) {
    auto mty = cast<MDSpanType>(nty->Clone());
    mty->SetShape(n.s);
    SetNodeType(n, mty);
    cur_mdspan_vn = cur_vn;
    // InvalidateVN(cur_vn);
  }

  if (IsActualBoundedIntegerType(NodeType(n))) {
    cur_ub_vn = cur_vn;
    InvalidateVN(cur_vn);
  }

  return true;
}

bool ShapeInference::Visit(AST::MultiDimSpans& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (n.list) {
    // TODO: should normalize as expr?
    auto vnt =
        (isa<AST::Expr>(n.list)) ? VNKind::VNK_MDSPAN : VNKind::VNK_VALUE;
    auto sign = GetSign(*n.list, vnt);

    // The Shape now can be deduced from the value number.
    // Update the type detail accordingly.
    auto vl = GenShapeFromSignature(sign);
    SetMdsShape(n, vl);

    if (!IsValidRank(n.Rank())) n.SetRank(vl.Rank());

    // pass the list value number over
    cur_vn = ValNoSign(sign);
    ast_vn.Update(&n, cur_vn, VNKind::VNK_MDSPAN);
    cur_mdspan_vn = cur_vn;
  } else if (n.Rank() > 0) {
    std::string unknown_spans = "#" + std::to_string(UnknownValue());
    for (size_t i = 1; i < n.Rank(); ++i)
      unknown_spans = unknown_spans + ",#" + std::to_string(UnknownValue());
    cur_mdspan_vn = vn.GetOrGenValueNumberFromSignature(unknown_spans);
    SetMdsShape(n, GenShapeFromSignature(unknown_spans));
    ast_vn.Update(&n, cur_mdspan_vn, VNKind::VNK_MDSPAN);
  } else {
    SetUnknownVN(cur_mdspan_vn); // failed to deduce the type detail
  }

  InvalidateVN(cur_vn);
  return true;
}

bool ShapeInference::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  auto name = n.name_str;
  if (n.init_expr) {
    assert(ValidVN(cur_mdspan_vn) &&
           "invalid value number for the named type.");
    DefineASymbol(name, n.GetType());

    ValNoAliasSign(SSTab().ScopedName(name), cur_mdspan_vn);

    InvalidateVN(cur_mdspan_vn); // consumes the mdspan
  }
  return true;
}

bool ShapeInference::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  auto name = n.name_str;

  if (!CanBeValueNumbered(&n)) {
    DefineASymbol(name, NodeType(n));
    return true; // mutables and events are not valno-able
  }

  Storage sto = Storage::NONE;
  if (auto sel = dyn_cast<AST::Select>(n.init_expr)) {
    if (auto sty = dyn_cast<SpannedType>(sel->GetType())) {
      assert(!n.mem);
      sto = sty->GetStorage();
    }
  }

  if (n.mem) sto = n.mem->st;

  ptr<Type> nty = nullptr;
  if (n.init_expr) {
    nty = NodeType(*n.init_expr);
    if (GetSpannedType(nty)) {
      cur_mdspan_vn = GetValNo(*n.init_expr, VNKind::VNK_MDSPAN);
      assert(ValidVN(cur_mdspan_vn) && "expecting a valid mdspan valno.");
      ValNoAliasSign(SSTab().ScopedName(name + ".span"), cur_mdspan_vn);
    } else if (IsActualBoundedIntegerType(nty)) {
      cur_ub_vn = GetValNo(*n.init_expr, VNKind::VNK_UBOUND);
      assert(ValidVN(cur_ub_vn));
      DefineASymbol("@" + name, MakeBoundedIntegerType(cur_ub_vn));
      ValNoAliasSign(SSTab().ScopedName("@" + name), cur_ub_vn);
      Shape s =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_ub_vn));
      nty = MakeBoundedITupleType(s);

      ValNoAliasSign(SSTab().ScopedName(name), GetValNo(*n.init_expr));
      InvalidateVN(cur_ub_vn);
    } else {
      if (!isa<PlaceHolderType>(nty)) {
        assert(ValidVN(cur_vn) &&
               "cur_mdspan_vn and cur_vn must be exclusive.");
        ValNoAliasSign(SSTab().ScopedName(name), cur_vn);
      }
    }
  } else {
    // obtain the types from declaration
    if (ValidVN(cur_mdspan_vn)) {
      ValNoAliasSign(SSTab().ScopedName(name + ".span"), cur_mdspan_vn);
      auto mds_value =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_mdspan_vn));
      if (n.IsArray())
        nty = MakeSpannedArrayType(n.type->base_type, mds_value,
                                   n.ArrayDimensions(), sto);
      else
        nty = MakeSpannedType(n.type->base_type, mds_value, sto);
    } else if (ValidVN(cur_vn)) {
      ValNoAliasSign(SSTab().ScopedName(name), cur_vn);
      nty = NodeType(*n.type);
    } else
      nty = NodeType(*n.type);
  }

  // fill-up the symbol table
  assert(nty);
  DefineASymbol(name, nty);
  SetNodeType(n, nty);

  // TODO(wsj): BooleanType? HalfType...?
  if ((isa<FloatType>(nty) || isa<DoubleType>(nty) || isa<IntegerType>(nty) ||
       isa<HalfType>(nty) || isa<Half8Type>(nty)) &&
      ValidVN(cur_vn)) {
    auto shape = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
    assert(shape.DimCount() == 1);
    VST_DEBUG(dbgs() << " |-<symval> " << InScopeName(name) << ": "
                     << STR(shape.ValueAt(0)) << "\n");
    SymVal(InScopeName(name)).SetVal(shape.ValueAt(0));
  } else if (isa<ITupleType>(nty)) {
    SymVal(InScopeName(name)).SetVals(vn.GenValueListFromValueNumber(cur_vn));
    VST_DEBUG(dbgs() << " |-<symval> " << InScopeName(name) << ": "
                     << STR(SymVal(InScopeName(name)).GetVals()) << "\n");
  }

  if (isa<FutureType>(n.GetType()) || isa<SpannedType>(n.GetType()))
    DefineASymbol(name + ".span", GetSpannedType(n.GetType())->GetMDSpanType());

  InvalidateVN(cur_mdspan_vn); // stop propagation
  InvalidateVN(cur_vn);

  return true;
}

bool ShapeInference::Visit(AST::IntTuple& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  auto valno = GenValNo(n);
  auto sign = SignValNo(valno);
  auto cnt = CountElementsInSignature(sign);
  // cur_ituple_vn = cur_vn;
  SetNodeType(n, MakeITupleType(cnt));

#if 0
  if (cnt > 1) {
    // set alias expressions with proper value numbers
    ForeachValueNumber(sign, [this, &sign](int valno, size_t index) {
      if (UnknownVN(valno)) return; // do not associate it with vn of "?"
      vn.GetOrGenValueNumberFromSignature("index_const_" +
                                             std::to_string(index));
      ValNoAliasSign(
          sign + "(" + std::to_string(index) + ")", valno);
    });
  }
#endif

  InvalidateVN(cur_vn); // Currently cut off value numbering
  return true;
}

bool ShapeInference::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);
  return true;
}

bool ShapeInference::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (IsMutable(*NodeType(n))) return true; // mutables are not valno-able

  if (SSTab().IsDeclared(n.GetName())) return true;

  // this is the un-type-annotated declaration
  auto nty = n.value->GetType();
  DefineASymbol(n.GetName(), nty);

  auto name = n.GetName();
  if (auto san = dyn_cast<AST::SpanAs>(n.value))
    assert((name == san->nid->name) && "inconsistent span_as variable name.");

  if (auto sty = GetSpannedType(nty)) {
    name += ".span";
    DefineASymbol(name, sty->GetMDSpanType());
    cur_mdspan_vn = GetValNo(*n.value, VNKind::VNK_MDSPAN);
    assert(ValidVN(cur_mdspan_vn) && "expected a valid current value number.");
    ValNoAliasSign(SSTab().ScopedName(name), cur_mdspan_vn);
    return true;
  } else if (isa<ITupleType>(nty)) {
    cur_mdspan_vn = GetValNo(*n.value, VNKind::VNK_MDSPAN);
    ValNoAliasSign(SSTab().ScopedName(name), cur_mdspan_vn);
    return true;
  } else if (isa<BoundedType>(nty)) {
    auto uname = "@" + name;
    cur_ub_vn = GetValNo(*n.value, VNKind::VNK_UBOUND);
    assert(ValidVN(cur_ub_vn));
    DefineASymbol(uname, MakeIntegerType());
    ValNoAliasSign(SSTab().ScopedName(uname), cur_ub_vn);
    InvalidateVN(cur_ub_vn);
  }

  cur_vn = GetValNo(*n.value);
  assert(ValidVN(cur_vn) && "expected a valid current value number.");
  ValNoAliasSign(SSTab().ScopedName(name), cur_vn);

  if (isa<IntegerType>(nty) && ValidVN(cur_vn)) {
    auto shape = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
    assert(shape.DimCount() == 1);
    VST_DEBUG(dbgs() << " |-<symval> " << SSTab().ScopedName(name) << ": "
                     << STR(shape.ValueAt(0)) << "\n");
    SymVal(SSTab().ScopedName(name)).SetVal(shape.ValueAt(0));
  }

  return true;
}

bool ShapeInference::Visit(AST::IntIndex& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (!HasValNo(n)) GenValNo(n);
  cur_vn = GetValNo(n);

  return true;
}

bool ShapeInference::Visit(AST::DataType& n) {
  TraceEachVisit(n);

  allow_named_dim = false;

  if (cannot_proceed) return true;

  if (ValidVN(cur_mdspan_vn)) { cur_vn = cur_mdspan_vn; }

  if (n.mdspan_type)
    ast_vn.Update(&n, GetValNo(*n.mdspan_type, VNKind::VNK_MDSPAN),
                  VNKind::VNK_MDSPAN);

  return true;
}

bool ShapeInference::Visit(AST::Identifier& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (!CanBeValueNumbered(&n)) { return false; }

  auto nty = NodeType(n);

  if (SSTab().IsDeclared(n.name)) {
    // Not neccessary to generate new valno. Instead utlize existing valno
    // passed from the symbols
    switch (NodeValNoKind(n)) {
    case VNKind::VNK_UBOUND: {
      if (!vn.HasValueNumberOfSignature(SSTab().InScopeName("@" + n.name)))
        choreo_unreachable("value number of `" + SSTab().InScopeName(n.name) +
                           "' has not been generated.");
      cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName("@" + n.name));
      ast_vn.Update(&n, cur_vn, VNKind::VNK_UBOUND);
      if (!vn.HasValueNumberOfSignature(SSTab().InScopeName(n.name)))
        choreo_unreachable("value number of `" + SSTab().InScopeName(n.name) +
                           "' has not been generated.");
      auto valno = vn.GetValueNumberOfSignature(SSTab().InScopeName(n.name));
      ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    } break;
    case VNKind::VNK_MDSPAN: {
      auto name = VNSymbolName(n);
      if (!vn.HasValueNumberOfSignature(SSTab().InScopeName(name)))
        choreo_unreachable("value number of `" + SSTab().InScopeName(name) +
                           "' has not been generated.");
      cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(name));
      ast_vn.Update(&n, cur_vn, VNKind::VNK_MDSPAN);
    } break;
    case VNKind::VNK_VALUE: {
      // it is a reference
      if (!vn.HasValueNumberOfSignature(SSTab().InScopeName(n.name)))
        choreo_unreachable("value number of `" + SSTab().InScopeName(n.name) +
                           "' has not been generated.");
      cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(n.name));
      ast_vn.Update(&n, cur_vn, VNKind::VNK_VALUE);
    } break;
    default: choreo_unreachable("unsupport valno kind.");
    }

    return true;
  }

  if (!gen_values) return false;

  if (allow_named_dim) { // for named dims in parameters
    if (!SSTab().DeclaredInScope(n.name)) {
      DefineASymbol(n.name, MakeIntegerType());
      cur_vn = vn.GenerateValueNumberFromSignature(SSTab().InScopeName(n.name));
    } else {
      cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(n.name));
    }
    ast_vn.Update(&n, cur_vn, VNKind::VNK_VALUE);
    return true;
  }

  if (HasValNo(n)) {
    Error(n.LOC(), "value number has been generated for `" + n.name + "'.");
    error_count++;
    return false;
  }

  // symbolic and bounded symbol requires valno
  if (CanYieldAnInteger(nty)) cur_vn = GenValNo(n);

  return true;
}

bool ShapeInference::Visit(AST::Parameter& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (n.type->isSpanned()) {
    assert(isa<AST::MultiDimSpans>(n.type->mdspan_type.get()) &&
           "Invalid mdspan.");
    auto span = cast<AST::MultiDimSpans>(n.type->mdspan_type.get());
    if (span->list) {
      cur_mdspan_vn = GetValNo(*span->list);
      assert(ValidVN(cur_mdspan_vn) && "unexpected value number for mdspan.");

      // Put alias names of mdspan into the value number table
      ValNoAliasSign(SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
      SetNodeType(*n.type,
                  MakeSpannedType(n.type->base_type, span->GetTypeDetail()));
    } else if (IsValidRank(span->Rank())) {
      cur_mdspan_vn = GetValNo(*span->list, VNKind::VNK_MDSPAN);
      assert(ValidVN(cur_mdspan_vn) && "unexpected value number for mdspan.");
      // Put alias names of mdspan into the value number table
      ValNoAliasSign(SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
      SetNodeType(*n.type,
                  MakeSpannedType(n.type->base_type, span->GetTypeDetail()));
    } else {
      // the value number is unknown at compile time
      Error(n.LOC(), "The type can not be inference at compile time.");
      error_count++;
      return false;
    }

    if (n.sym) {
      auto span_name = n.sym->name + ".span";
      DefineASymbol(span_name,
                    cast<SpannedType>(n.type->GetType())->GetMDSpanType());

#if 0
      // generate all aliases
      auto sign = SignValNo(cur_mdspan_vn);
      if (CountElementsInSignature(sign) == 1) sign = "#" + STR(ValNoSign(sign));
      ForeachValueNumber(sign, [this, &span_name, &n](int valno, size_t index) {
        auto sname = SSTab().ScopedName(span_name) + "(" + std::to_string(index) + ")";
        ValNoAliasSign(sname, valno);
      });
#endif

      DefineASymbol(n.sym->name, n.type->GetType());
    }

    InvalidateVisitorValNOs();
    return true;
  }

  if (n.sym && n.type->isScalar()) {
    assert(!ValidVN(cur_mdspan_vn) && "unexpected current mdspan value.");

    // get the value number and make it defined
    vn.GenerateValueNumberFromSignature(SSTab().ScopedName(n.sym->name));
    DefineASymbol(n.sym->name, n.GetType());

    InvalidateVisitorValNOs();
    return true;
  }

  InvalidateVisitorValNOs();
  return true;
}

bool ShapeInference::Visit(AST::ParamList& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  // handle bounds
  auto b_sign =
      (n.HasSubPVs()) ? GetSign(*n.BoundExprs()) : GetSign(*n.BoundExpr());
  auto b_valno = ValNoSign(b_sign);
  ast_vn.Update(n.BPV().get(), b_valno, VNKind::VNK_UBOUND);
  ast_vn.Update(n.SubPVs().get(), b_valno, VNKind::VNK_UBOUND);

  Shape s = GenShapeFromSignature(b_sign);
  SetNodeType(n, MakeMDSpanType(s)); // useful for the sema check

  std::string iv_name = SSTab().ScopedName("@" + n.BPV()->name);
  ValNoAliasSign(iv_name, b_valno);
  SetNodeType(*n.BPV(), MakeBoundedITupleType(s, "pv"));
  DefineASymbol("@" + n.BPV()->name, MakeMDSpanType(s));
  VST_DEBUG(dbgs() << " |-<pvbound> " << n.BPV()->name << ": " << STR(s)
                   << "\n");

  DefineASymbol(n.BPV()->name, n.BPV()->GetType());
  assert(!ast_vn.Hit(n.BPV().get(), VNKind::VNK_VALUE) &&
         "The value number has already been generated.");
  auto sname = SSTab().ScopedName(n.BPV()->name);
  auto vv = vn.GetOrGenValueNumberFromSignature(sname);
  ast_vn.Update(n.BPV().get(), vv, VNKind::VNK_VALUE);
  n.BoundExpr()->Opts().SetVals(vn.GenValueListFromValueNumber(vv));

  assert(n.HasSubPVs() && "normalization failed.");

  auto sign_cnt = CountElementsInSignature(b_sign);
  assert((size_t)sign_cnt == n.SubPVCount());
  auto sign = b_sign;
  if (sign_cnt == 1) sign = "#" + STR(ValNoSign(b_sign));
  std::string idx2dim[] = {"x", "y", "z"};
  ForeachValueNumber(sign, [this, &n, &idx2dim](int valno, size_t index) {
    auto id = n.GetSubPV(index);
    std::string pv_name = SSTab().ScopedName("@" + id->name);
    ValNoAliasSign(pv_name, valno);
    Shape s = GenShapeFromSignature(SignValNo(valno));
    SetNodeType(*id, MakeBoundedITupleType(s, "pi:" + idx2dim[index]));
    DefineASymbol("@" + id->name, MakeMDSpanType(s));

    assert(!ast_vn.Hit(id.get(), VNKind::VNK_VALUE) &&
           "The value number has already been generated.");
    auto sname = SSTab().ScopedName(id->name);
    auto vv = vn.GetOrGenValueNumberFromSignature(sname);
    ast_vn.Update(id.get(), vv, VNKind::VNK_VALUE);
    DefineASymbol(id->name, id->GetType());

    // update the bound opt value
    auto ubvi = vn.GenValueItemFromValueNumber(valno);
    n.BoundExprAt(index)->Opts().SetVal(ubvi);
  });

  VST_DEBUG(dbgs() << " |-<pv-bounds> {" << DelimitedSTR(n.AllSubPVs())
                   << "} : " << STR(GenShapeFromSignature(b_sign)) << "\n");

  return true;
}

bool ShapeInference::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  assert(isa<AST::Identifier>(n.lhs) &&
         "non-id is not supported in where bind.");
  assert(isa<AST::Identifier>(n.rhs) &&
         "non-id is not supported in where bind.");

  auto l_id = cast<AST::Identifier>(n.lhs);
  auto r_id = cast<AST::Identifier>(n.rhs);
  auto l_vn =
      vn.GetValueNumberOfSignature(SSTab().ScopedName("@" + l_id->name));
  auto r_vn =
      vn.GetValueNumberOfSignature(SSTab().ScopedName("@" + r_id->name));

  // TODO: sometimes the lhs would have same valno with existing one, which is
  // allowed. However, for runtime valued bound, they may have different
  // bound. The problem here is how to judge if the upper bound of bounded
  // variables are actually illegal? (e.g, different static upper bound)
  vn.BindValueNumbers(l_vn, r_vn);
  VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Bind> VN #" << l_vn << " <-> VN #"
                   << r_vn << "\n");
  return true;
}

bool ShapeInference::Visit(AST::WithIn& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  cur_mdspan_vn = GetValNo(*n.in, VNKind::VNK_MDSPAN);
  auto mds_sign = SignValNo(cur_mdspan_vn);

  // requires the elements inside mdspan to be non-zero values
  bool found_zero = false;
  ForeachValueNumber(mds_sign,
                     [this, &mds_sign, &n, &found_zero](int valno, size_t) {
                       auto sig = SignValNo(valno);
                       if (sig == "const_0") { found_zero = true; }
                     });

  // we have to abend early here since it can not process further shape
  // inference
  if (found_zero) {
    Error(n.LOC(),
          "zero value is deduced for the mdspan inside the with-in statement.");
    error_count++;
    cannot_proceed = true;
    Error(n.LOC(),
          "unable to apply shape inference for function '" + fname + "'.");
    return false;
  }

  auto GenSignatureAndDoValno = [this, &n](int valno, size_t index) {
    if (UnknownVN(valno)) return; // do not associate it with vn of "?"
    if (n.with) {
      std::string name = SSTab().ScopedName("@" + n.with->name) + "(" +
                         std::to_string(index) + ")";
      ValNoAliasSign(name, valno);
    }

    if (n.with_matchers) {
      // set the node type
      auto sym = cast<AST::Identifier>(n.with_matchers->ValueAt(index));
      Shape s = GenShapeFromSignature(SignValNo(valno));
      SetNodeType(*sym, MakeBoundedITupleType(s));

      // generate the ubound symbol and associate its valno
      DefineASymbol("@" + sym->name, MakeMDSpanType(s));
      auto ub_name = SSTab().ScopedName("@" + sym->name);
      ValNoAliasSign(ub_name, valno);
      ast_vn.Update(&n, valno, VNKind::VNK_UBOUND);

      // generate the symbol but only generate the symbolic node value
      DefineASymbol(sym->name, sym->GetType());
      assert(!ast_vn.Hit(sym.get(), VNKind::VNK_VALUE) &&
             "The value number has already been generated.");
      auto sname = SSTab().ScopedName(sym->name);
      auto vv = vn.GetOrGenValueNumberFromSignature(sname);
      ast_vn.Update(sym.get(), vv, VNKind::VNK_VALUE);
    }
  };

  auto sign = mds_sign;
  if (CountElementsInSignature(mds_sign) == 1)
    sign = "#" + STR(ValNoSign(mds_sign));

  ForeachValueNumber(sign, GenSignatureAndDoValno);

  if (n.with) {
    // generate symbolic node valno
    DefineASymbol(n.with->name, n.with->GetType());
    assert(!ast_vn.Hit(n.with.get(), VNKind::VNK_VALUE) &&
           "The value number has already been generated.");
    auto sname = SSTab().ScopedName(n.with->name);
    auto vv = vn.GetOrGenValueNumberFromSignature(sname);
    ast_vn.Update(n.with.get(), vv, VNKind::VNK_VALUE);

    // upper-bound valnos
    ValNoAliasSign(SSTab().ScopedName("@" + n.with->name), cur_mdspan_vn);
    Shape s = GenShapeFromSignature(mds_sign);
    SetNodeType(*n.with, MakeBoundedITupleType(s));
    DefineASymbol("@" + n.with->name, MakeMDSpanType(s));
  }

  InvalidateVN(cur_mdspan_vn);

  return true;
}

bool ShapeInference::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::Memory& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::SpanAs& n) {
  TraceEachVisit(n);

  cur_mdspan_vn = GetValNo(*n.list);
  auto pty = SSTab().LookupSymbol(n.id->name);
  assert((isa<SpannedType>(pty) || isa<FutureType>(pty)) &&
         "unexpected data type.");

  auto sty = GetSpannedType(pty);
  if (!sty) {
    Error(n.LOC(), "internal error: span_as operates on non-spanned type.");
    return false;
  }

#if 0
  ValNoAliasSign(
      SSTab().ScopedName(n.nid->name + ".span"), cur_mdspan_vn);
#endif

  auto shape =
      GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_mdspan_vn));
  auto nty = MakeSpannedType(sty->ElementType(), shape, sty->GetStorage());

  SetNodeType(n, nty);
  ast_vn.Update(&n, cur_mdspan_vn, VNKind::VNK_MDSPAN);

  return true;
}

bool ShapeInference::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (n.operation == ".any") {
    assert(!n.future.empty() && "unexpected: the future is empty.");
    DefineASymbol(n.future, MakePlaceHolderFutureType());
    DefineASymbol(n.future + ".span", MakePlaceHolderMDSpanType());
    vn.AssociateSignatureWithInvalidValueNumber(
        SSTab().ScopedName(n.future + ".span"));
    InvalidateVN(cur_vn);
    return true;
  }

  assert(ValidVN(cur_vn) &&
         "unexpected current value number for shape inference of dma.");

  if (auto pcfg = dyn_cast<PadConfig>(n.config)) {
    size_t size = pcfg->pad_high.size();
    std::vector<size_t> all_pads(size);
    std::fill_n(all_pads.begin(), size, 0);
    for (size_t i = 0; i < size; ++i)
      all_pads[i] += pcfg->pad_high[i] + pcfg->pad_low[i] + pcfg->pad_mid[i];
    // now generate signature for original signature plus padding values
    auto ElementSignature = [this](size_t n) {
      std::string cv = "const_" + std::to_string(n);
      return "#" + std::to_string(vn.GetOrGenValueNumberFromSignature(cv));
    };
    std::string sig;
    if (size > 1) {
      sig = ElementSignature(all_pads[0]);
      for (size_t i = 1; i < size; ++i)
        sig += "," + ElementSignature(all_pads[i]);
    } else {
      sig = "const_" + std::to_string(all_pads[0]);
    }
    std::string add_sig = vn.SignBinaryCompositeValues(
        n.LOC(), "+", vn.GetSignatureFromValueNumber(cur_vn), sig);
    // update the cur_vn
    cur_vn = vn.GetOrGenValueNumberFromSignature(add_sig);
  } else if (auto tcfg = dyn_cast<TransposeConfig>(n.config)) {
    // gen new vn if and only if n.to is AST::Memory
    if (isa<AST::Memory>(n.to)) {
      auto& dim_values = tcfg->dim_values;
      auto orig_sig = vn.GetSignatureFromValueNumber(cur_vn);
      auto shape_components = SplitStringByDelimiter(orig_sig);
      auto sig = shape_components[dim_values[0]];
      for (size_t i = 1; i < dim_values.size(); ++i)
        sig += "," + shape_components[dim_values[i]];
      cur_vn = vn.GetOrGenValueNumberFromSignature(sig);
    }
  }

  // annotate the shape on AST for later type inference
  auto s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
  SetNodeType(n, MakeShapedFutureType(s, n.async));

  if (n.future.empty()) {
    InvalidateVN(cur_vn);
    return true;
  }

  if (SSTab().IsDeclared(n.future)) {
    assert(cast<PlaceHolderType>(SSTab().LookupSymbol(n.future))->Category() ==
           TypeCategory::FUTURE);
    vn.RebindSignatureWithValueNumber(SSTab().InScopeName(n.future) + ".span",
                                      cur_vn);
    SSTab().ModifySymbolType(n.future, n.GetType());
    SSTab().ModifySymbolType(n.future + ".span", MakeMDSpanType(s));
  } else {
    std::string f_span = n.future + ".span";
    ValNoAliasSign(SSTab().ScopedName(f_span), cur_vn);
    DefineASymbol(n.future, n.GetType());
    DefineASymbol(f_span, MakeMDSpanType(s)); // implicit symbol
  }

  auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);
#if 0
  // set alias expressions with proper value numbers
  if (CountElementsInSignature(vn_sig) > 1) {
    ForeachValueNumber(vn_sig, [this, &vn_sig](int valno, size_t index) {
      if (UnknownVN(valno)) return; // do not associate it with vn of "?"
      vn.GetOrGenValueNumberFromSignature("index_const_" +
                                             std::to_string(index));
      auto elem_sig = vn_sig + "(" + std::to_string(index) + ")";
      if (!vn.HasValueNumberOfSignature(elem_sig))
        ValNoAliasSign(
            vn_sig + "(" + std::to_string(index) + ")", valno);
    });
  }
#endif

  InvalidateVN(cur_vn);
  return true;
}

bool ShapeInference::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  int ca_valno = GetInvalidValueNumber();

  auto pty = SSTab().LookupSymbol(n.data->name);
  assert((isa<SpannedType>(pty) || isa<FutureType>(pty)) &&
         "unexpected data type.");

  auto span_name = RemoveSuffix(n.data->name, ".data") + ".span";
  auto sty = GetSpannedType(pty);

  if (n.NoTile()) {
    // it is just a symbol reference
    ca_valno = vn.GetValueNumberOfSignature(SSTab().InScopeName(span_name));
    auto future_shape =
        GenShapeFromSignature(vn.GetSignatureFromValueNumber(ca_valno));
    // set the chunkat's type
    SetNodeType(n,
                MakeSpannedType(sty->f_type, future_shape, sty->GetStorage()));
    n.s = future_shape;
    assert(n.s.IsValid());

    cur_vn = ca_valno;
    return true;
  }

  std::string data_sig = vn.SignatureOfSymbol(SSTab().InScopeName(span_name));

  auto AddValno = [this](int valno, std::string& sign) {
    // and append the value number as
    if (!sign.empty()) sign += ",";
    sign += "#" + std::to_string(valno);
  };

  auto SignatureOfBinOp = [this, &AddValno, &n](const std::string& op,
                                                int dividend_vn,
                                                int divisor_vn) {
    // the signature without optimize
    std::string res_sig = op + ":#" + std::to_string(dividend_vn) + ":#" +
                          std::to_string(divisor_vn);

    if (auto quotient = vn.TryToSimplifyBinary(
            n.LOC(), op, vn.GetSignatureFromValueNumber(dividend_vn),
            vn.GetSignatureFromValueNumber(divisor_vn), true))
      res_sig = quotient.value();

    return res_sig;
  };

  auto data_vns = vn.Flatten(vn.GetValueNumberOfSignature(data_sig));

  std::vector<int> mod_vns;
  std::vector<int> res_vns;

  bool is_modspan = false;
  auto cur_vns = data_vns;

  // handle all chunkat expressions iteratively
  for (auto tsi : n.AllTSInfo()) {
    mod_vns.clear();
    res_vns.clear();

    auto old_gv = gen_values;
    gen_values = true;
    // make sure all expressions get the value numbers
    tsi->Positions()->accept(*this);
    if (tsi->MultipleExprs()) tsi->GetTFSSExpr()->accept(*this);
    gen_values = old_gv;

    std::vector<NumTy> tfs_vns;
    std::vector<NumTy> pos_vns;

    auto tsi_vn = GetValNo(*tsi->Positions(), VNKind::VNK_UBOUND);
    pos_vns = vn.AsVector(tsi_vn);
    assert(cur_vns.size() == pos_vns.size());

    if (tsi->MultipleExprs()) {
      // when the code provides explicit tiling factors or subspan
      tfs_vns = vn.AsVector(GetValNo(*tsi->GetTFSSExpr()));
      assert(tfs_vns.size() == pos_vns.size());
    }

    for (size_t index = 0; index < pos_vns.size(); ++index) {
      if (tsi->HasSubSpanExpr()) {
        // block.span = subspan
        auto lvi = vn.GenValueItemFromValueNumber(cur_vns[index]);
        auto rvi = vn.GenValueItemFromValueNumber(tfs_vns[index]);
        if (sbe::clt(lvi, rvi)) {
          Error(tsi->LOC(),
                "the subspan dimension (dim: " + std::to_string(index) +
                    ") is larger than original (" + STR(rvi) + " > " +
                    STR(lvi) + ").");
          error_count++;
        }
        res_vns.push_back(tfs_vns[index]);
      } else if (tsi->HasModSpanExpr()) {
        // block.span = data.span % tiling_factor
        auto lvi = vn.GenValueItemFromValueNumber(cur_vns[index]);
        auto rvi = vn.GenValueItemFromValueNumber(tfs_vns[index]);
        if (sbe::clt(lvi, rvi)) {
          Error(tsi->LOC(),
                "the subspan dimension (dim: " + std::to_string(index) +
                    ") is larger than the data (" + STR(rvi) + " > " +
                    PSTR(lvi) + ").");
          error_count++;
        }
        auto mod_sig = SignatureOfBinOp("%", cur_vns[index], tfs_vns[index]);
        mod_vns.push_back(vn.GetOrGenValueNumberFromSignature(mod_sig));
        res_vns.push_back(tfs_vns[index]);
      } else if (tsi->HasTilingExpr()) {
        // block.span = data.span / tiling_factor
        auto lvi = vn.GenValueItemFromValueNumber(cur_vns[index]);
        auto rvi = vn.GenValueItemFromValueNumber(tfs_vns[index]);
        if (sbe::clt(lvi, rvi)) {
          Error(tsi->LOC(), "the tiling factor (dim: " + std::to_string(index) +
                                ") is larger than the data dimension (" +
                                STR(rvi) + " > " + PSTR(lvi) + ").");
          error_count++;
        }
        auto res_sig = SignatureOfBinOp("/", cur_vns[index], tfs_vns[index]);
        res_vns.push_back(vn.GetOrGenValueNumberFromSignature(res_sig));
      } else {
        // block.span = data.span / #pos
        auto lvi = vn.GenValueItemFromValueNumber(cur_vns[index]);
        auto rvi = vn.GenValueItemFromValueNumber(pos_vns[index]);
        if (sbe::clt(lvi, rvi)) {
          Error(tsi->LOC(), "the tiling factor (dim: " + std::to_string(index) +
                                ") is larger than the data dimension (" +
                                STR(rvi) + " > " + STR(lvi) + ").");
          error_count++;
        }
        auto res_sig = SignatureOfBinOp("/", cur_vns[index], pos_vns[index]);
        res_vns.push_back(vn.GetOrGenValueNumberFromSignature(res_sig));
      }
    }

    if (tsi->HasModSpanExpr()) {
      cur_vns = mod_vns;
      is_modspan = true;
    } else {
      cur_vns = res_vns;
      is_modspan = false;
    }
    std::string sig; // signature of the sub-block
    for (auto rvn : res_vns) AddValno(rvn, sig);
    tsi->SetBlockShape(GenShapeFromSignature(sig));
  }

  assert(res_vns.size() == data_vns.size());

  Shape block_shape;
  if (res_vns.size() == 1) {
    if (is_modspan) {
      ca_valno = mod_vns[0];
      block_shape =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(res_vns[0]));
    } else
      ca_valno = res_vns[0];
  } else {
    // generate signature for multi-valnos
    std::string b_sign; // signature of the sub-block
    for (auto rvn : res_vns) AddValno(rvn, b_sign);

    if (is_modspan) {     // remainder value as the current shape value
      std::string m_sign; // specific for modspan operation
      for (auto mvn : mod_vns) AddValno(mvn, m_sign);
      ca_valno = vn.GetOrGenValueNumberFromSignature(m_sign);
      auto b_valno = vn.GetOrGenValueNumberFromSignature(b_sign);
      block_shape =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(b_valno));
    } else
      ca_valno = vn.GetOrGenValueNumberFromSignature(b_sign);
  }

  auto future_shape =
      GenShapeFromSignature(vn.GetSignatureFromValueNumber(ca_valno));

  // set the chunkat's type
  SetNodeType(n, MakeSpannedType(sty->f_type, future_shape, sty->GetStorage()));

  if (is_modspan)
    n.s = block_shape;
  else
    n.s = future_shape;

  assert(n.s.IsValid());

  cur_vn = ca_valno;

  return true;
}

bool ShapeInference::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  InvalidateVisitorValNOs();
  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::Trigger& n) {
  TraceEachVisit(n);
  InvalidateVisitorValNOs();
  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::Call& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  // value the scalars
  for (auto& s : n.GetArguments()) {
    if (!CanBeValueNumbered(s.get())) continue;
    if (isa<IntegerType>(NodeType(*s))) {
      auto expr = cast<AST::Expr>(s);
      expr->s = GenShapeFromSignature(GetSign(*s));
      VST_DEBUG(dbgs() << "[ExprShape] Shape for " << PSTR(s) << ": "
                       << STR(expr->s) << "\n");
      assert(expr->s.DimCount() == 1);
      expr->Opts().SetVal(expr->s.ValueAt(0));
      VST_DEBUG(dbgs() << "[ExprVal] Value for " << PSTR(expr) << ": "
                       << STR(expr->s.ValueAt(0)) << "\n");
    }
  }

  InvalidateVisitorValNOs();

  if (!(n.IsBIF() && n.CompileTimeEval())) return true;

  const auto func_name = n.function->name;
  // compile-time print
  if (func_name == "print" || func_name == "println") {
    for (const auto& arg : n.GetArguments()) {
      const auto nty = NodeType(*arg);
      auto e = cast<AST::Expr>(arg);
      if (auto sl = e->GetString()) {
        dbgs() << sl->Val();
      } else if (auto fl = e->GetFloat()) {
        if (fl->IsFloat32())
          dbgs() << fl->Val_f32();
        else if (fl->IsFloat64())
          dbgs() << fl->Val_f64();
      } else if (auto bl = e->GetBoolean()) {
        dbgs() << bl->Val();
      } else if (ConvertibleToInt(nty)) {
        if (e->Opts().HasVal())
          dbgs() << STR(e->Opts().GetVal());
        else
          dbgs() << "unknown"; // TODO
      } else if (isa<ScalarType>(nty)) {
        dbgs() << "unknown"; // TODO: opt values
      } else if (isa<ITupleType>(nty)) {
        //        dbgs() << STR(e->Opts().GetVals());
        PrintValueList(e->Opts().GetVals(), dbgs(), "{", "}");
      } else if (isa<MDSpanType>(nty)) {
        dbgs() << STR(GetShape(nty));
      } else if (isa<BoundedType>(nty) || isa<SpannedType>(nty) ||
                 isa<AsyncType>(nty)) {
        dbgs() << "rt-val";
      } else
        choreo_unreachable("unsupported type for print: " +
                           AST::TYPE_STR(*arg) + "\n\targ: " + PSTR(arg));
    }
    if (func_name == "println") dbgs() << "\n";
  }
  return true;
}

bool ShapeInference::Visit(AST::Rotate& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  auto rty = type_equals.ResolveEqualFutures(*n.ids, true);

  if (!rty) {
    Error(n.LOC(), "Failed to resolve future types.");
    error_count++;
    return false;
  }

  // do not care about placeholders
  if (isa<PlaceHolderType>(rty)) return true;

  int valno = GetOnlyValueNumberFromMultiValues(*n.ids);

  if (!ValidVN(valno)) {
    Error(n.LOC(), "failed to find a valid value number inside ROTATE.");
    error_count++;
    cannot_proceed = true;
    return false;
  }

  for (auto v : n.GetIds()) ast_vn.Update(v.get(), valno, NodeValNoKind(*v));

  // now update the valnos
  UpdateValueNumberForMultiValues(*n.ids, valno);

  InvalidateVisitorValNOs();

  return true;
}

bool ShapeInference::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);
  return true;
}

bool ShapeInference::Visit(AST::Select& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  assert(!n.inDMA);
  if (auto sty = dyn_cast<SpannedType>(NodeType(n))) {
    auto s0 = cast<AST::Expr>(n.expr_list->ValueAt(0));
    auto s0ty = NodeType(*s0);
    if (s0ty && s0ty->HasSufficientInfo()) {
      SetNodeType(n, s0ty);
    } else {
      // handle dataof expr (TODO: any better idea?)
      if (!s0->s.IsValid()) {
        Error(n.LOC(), "Failed to decide the type of Select." + STR(n) +
                           ", type0: " + PSTR(s0ty));
        error_count++;
        return false;
      }
      auto nty = MakeSpannedType(sty->f_type, s0->s, sty->GetStorage());
      SetNodeType(n, nty);
    }
    ast_vn.Update(&n, GetValNo(*s0, VNKind::VNK_MDSPAN), VNKind::VNK_MDSPAN);
    cur_mdspan_vn = GetValNo(n, VNKind::VNK_MDSPAN);
    InvalidateVN(cur_vn); // used for variable def
  } else if (GeneralFutureType(NodeType(n))) {
    InvalidateVN(cur_vn);

    auto fty = type_equals.ResolveEqualFutures(*n.expr_list, true);
    if (!fty) {
      Error(n.LOC(), "Failed to resolve future types.");
      error_count++;
      return false;
    }
    SetNodeType(n, fty);
    if (isa<PlaceHolderType>(fty)) return true;

    cur_mdspan_vn = GetOnlyValueNumberFromMultiValues(*n.expr_list);
    if (!ValidVN(cur_mdspan_vn)) {
      Error(n.LOC(), "no valid value number is found for a SELECT expression.");
      error_count++;
      cannot_proceed = true;
      return false;
    }

    // now update the valnos
    UpdateValueNumberForMultiValues(*n.expr_list, cur_mdspan_vn);
    ast_vn.Update(&n, cur_mdspan_vn, VNKind::VNK_MDSPAN);
  } else
    choreo_unreachable("unsupported type.");

  return true;
}

bool ShapeInference::Visit(AST::Return& n) {
  TraceEachVisit(n);
  InvalidateVisitorValNOs();
  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::LoopRange& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);

  gen_values = true; // allow generate values for statements

  // invalidate any current value generated
  InvalidateVisitorValNOs();

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);

  // invalidate any current value generated
  InvalidateVisitorValNOs();

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);

  // invalidate any current value generated
  InvalidateVisitorValNOs();

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::IncrementBlock& n) {
  TraceEachVisit(n);

  gen_values = true; // allow generate values for statements

  // invalidate any current value generated
  InvalidateVisitorValNOs();

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::Program& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  return true;
}

// TODO: should be recursive
Shape ShapeInference::GenShapeFromSignature(const std::string& input) {
  ValueList result;

  std::istringstream stream(input);
  std::string component;
  while (std::getline(stream, component, ',')) {
    // Trim whitespace
    component.erase(remove_if(component.begin(), component.end(), isspace),
                    component.end());

    assert(!component.empty() && "unexpected component.");

    if (auto vi = vn.GenValueItemFromSignature(component)) {
      result.push_back(vi);
    } else {
      if (debug_visit)
        dbgs() << "failed to generate value item for " << component << ".\n";
      // TODO: remove the legacy method totally
      auto expr = GenerateExpression(component);
      int int_val;
      auto [ptr, ec] =
          std::from_chars(expr.data(), expr.data() + expr.size(), int_val);
      if (ec == std::errc() && ptr == expr.data() + expr.size()) {
        result.emplace_back(sbe::nu(int_val));
      } else
        result.emplace_back(sbe::sym(expr));
    }
  }

  return {result.size(), result};
}

std::string ShapeInference::GenerateExpression(const std::string& sig) {
  if (auto digit = RemovePrefixOrNull("const_", sig)) return *digit;

  // a value number reference
  if (auto digit = RemovePrefixOrNull("#", sig);
      digit && !PrefixedWith(sig, "#:"))
    return GenerateExpression(
        vn.GetSignatureFromValueNumber(std::stoi(*digit)));

  // unary expressions
  if (PrefixedWith(sig, "!:")) {
    return "!" + GenerateExpression(sig.substr(2));
  }
  if (PrefixedWith(sig, "sizeof:")) {
    auto all = vn.GetSignatureFromValueNumber(std::stoi(sig.substr(8)));
    auto parts = SplitStringByDelimiter(all, ",");
    std::string res;
    for (auto& p : parts) {
      if (res != "") res += "*";
      res += "(" + GenerateExpression(p) + ")";
    }
    return res;
  }

  // binary expressions
  if (PrefixedWith(sig, "cdiv:")) {
    auto parts = SplitStringByDelimiter(sig.substr(5), ":");
    assert(parts.size() == 2);
    auto lhs = GenerateExpression(parts[0]);
    auto rhs = GenerateExpression(parts[1]);
    return "((" + lhs + ")+(" + rhs + ")-1)/(" + rhs + ")";
  }
  if (PrefixedWith(sig, "#:")) {
    auto parts = SplitStringByDelimiter(sig.substr(2), ":");
    assert(parts.size() == 2);
    std::string res;
    for (auto& p : parts) {
      if (res != "") res += "*";
      res += "(" + GenerateExpression(p) + ")";
    }
    return res;
  }
  static const std::initializer_list<std::string> prefixes = {
      "+:",  "-:", "*:", "/:",  "%:",  ">=:", "||:",
      "&&:", "<:", ">:", "==:", "!=:", "<=:", ">=:"};
  if (std::any_of(prefixes.begin(), prefixes.end(),
                  [&](const std::string& prefix) {
                    return PrefixedWith(sig, prefix);
                  })) {
    std::istringstream stream(sig);
    std::vector<std::string> parts;
    std::string part;

    while (std::getline(stream, part, ':')) parts.push_back(part);
    assert(parts.size() == 3);
    return "(" + GenerateExpression(parts[1]) + ")" + parts[0] + "(" +
           GenerateExpression(parts[2]) + ")";
  }

  // ternary expressions
  if (PrefixedWith(sig, "?:")) {
    auto parts = SplitStringByDelimiter(sig.substr(2), ":");
    assert(parts.size() == 3);
    std::string res = "(" + GenerateExpression(parts[0]) + ")?" + "(" +
                      GenerateExpression(parts[1]) + "):(" +
                      GenerateExpression(parts[2]) + ")";
    return res;
  }

  // this is a symbol
  return sig;
}

int ShapeInference::GetOnlyValueNumberFromMultiValues(
    const AST::MultiValues& mv) {
  int valno = GetInvalidValueNumber();
  for (auto& v : mv.AllValues()) {
    auto id = AST::GetIdentifier(*v);
    if (!id) choreo_unreachable("expect an identifier.\n");
    auto ln = SSTab().InScopeName(VNSymbolName(*id));

    if (!ValidVN(valno)) {
      valno = vn.GetValueNumberOfSignature(ln);
      continue;
    }

    // Check for consistence between different values
    if (vn.HasValidValueNumberOfSignature(ln)) {
      if (valno != vn.GetValueNumberOfSignature(ln)) {
// currently some equivalence cannot be detected, drop the check
#if 0
        Error(mv.LOC(), "value number does not match.");
        cannot_proceed = true;
        return GetInvalidValueNumber();
#endif
      }
    }
  }
  return valno;
}

void ShapeInference::UpdateValueNumberForMultiValues(const AST::MultiValues& mv,
                                                     int valno) {
  for (auto& v : mv.AllValues()) {
    if (auto id = AST::GetIdentifier(*v)) {
      auto symbol = SSTab().InScopeName(VNSymbolName(*id));
      // the VN is considered to be identical if none exist
      if (!ValidVN(vn.GetValueNumberOfSignature(symbol))) {
        vn.RebindSignatureWithValueNumber(symbol, valno);
        auto equals = type_equals.GetEquals(SSTab().InScopeName(id->name));
        for (auto& e : equals.value()) {
          auto asym = e + ".span";
          if (!vn.HasValidValueNumberOfSignature(asym))
            vn.RebindSignatureWithValueNumber(asym, valno);
          else
            assert(valno == vn.GetValueNumberOfSignature(asym));
        }
        ast_vn.Update(id, valno, NodeValNoKind(*id));
      }
    } else
      choreo_unreachable("expect an identifier.");
  }
}

bool ShapeInference::CanBeValueNumbered(AST::Node* n) const {
  if (!n) return true;

  if (auto mv = dyn_cast<AST::MultiValues>(n)) {
    if (mv->None()) return false;
    for (auto v : mv->AllValues())
      if (!CanBeValueNumbered(v.get())) return false;
    return true;
  }

  assert(!n->IsBlock() && "do not pass in block node.");

  if (isa<AST::ChunkAt>(n)) return false;
  if (isa<AST::StringLiteral>(n)) return false;
  if (isa<AST::DataAccess>(n)) return false;
  auto nty = NodeType(*n);
  if (!nty) {
    // sometimes the symbol is yet to define, simply make it work.
    return true;
  }
  if (IsMutable(*nty)) return false;
  if (isa<EventType>(nty)) return false;
  if (isa<StringType>(nty)) return false;

  if (auto e = dyn_cast<AST::Expr>(n)) {
    if (e->op == "elemof") return false;
    if (e->op == "addrof") return false;
    return CanBeValueNumbered(e->GetR().get()) &&
           CanBeValueNumbered(e->GetL().get()) &&
           CanBeValueNumbered(e->GetC().get());
  }
  return true; // could be id/int/...
}

void ShapeInference::DefineASymbol(const std::string& name,
                                   const ptr<Type>& ty) {
  // assert(!SSTab().IsDeclared(name) && "symbol has been declared.");
  SSTab().DefineSymbol(name, ty);
  if (debug_visit)
    dbgs() << " |-<symtab> add: " << SSTab().InScopeName(name)
           << ", type: " << PSTR(ty) << "\n";
}

const std::string ShapeInference::SignSpan(const AST::Node& n) {
  // std::cout << "sign span: " << STR(n) << "\n";
  if (auto* id = dyn_cast<AST::Identifier>(&n)) {
    // only cares about value inside the mdspan
    auto name = RemoveSuffix(id->name, ".span") + ".span";

    if (auto sname = SSTab().NameInScopeOrNull(name)) {
      if (vn.HasValueNumberOfSignature(*sname))
        return *sname;
      else
        choreo_unreachable("symbol `" + *sname +
                           "' is not associated with a value number.");
    }
    // or else, it is a new name definition
    return SSTab().ScopedName(name);
  } else if (auto e = dyn_cast<AST::Expr>(&n); e && e->op == "dataof") {
    return GetSign(*e->GetR(), VNKind::VNK_MDSPAN); // simple propagate
  } else if (auto* s = dyn_cast<AST::Select>(&n)) {
    // any one could have the same span
    return GetSign(*s->expr_list->ValueAt(0), VNKind::VNK_MDSPAN);
  } else if (auto* it = dyn_cast<AST::IntTuple>(&n)) {
    // turn multivalues to be the spanned value
    return GetSign(*(it->GetValues()));
  } else if (auto* mds = dyn_cast<AST::MultiDimSpans>(&n)) {
    return GetSign(*(mds->list));
  } else if (auto* sa = dyn_cast<AST::SpanAs>(&n)) {
    return GetSign(*(sa->list));
  } else if (auto* e = dyn_cast<AST::Expr>(&n)) {
    if (e->op == "sizeof") {
      // TODO: use valuelist directly
      auto s = GetShape(NodeType(*e->GetR()));
      return vn.ValueItemToSignature(s.ElementCountValue(), true);
    }
    auto signature = e->op;
    if (e->GetC()) signature += ":#" + STR(GetValNo(*e->GetC()));
    if (e->GetL()) {
      if (HasValNo(*e->GetL(), VNKind::VNK_MDSPAN))
        signature += ":#" + STR(GetValNo(*e->GetL(), VNKind::VNK_MDSPAN));
      else
        signature += ":#" + STR(GetValNo(*e->GetL()));
    }
    assert(e->GetR() && "expression is invalid.");
    if (HasValNo(*e->GetR(), VNKind::VNK_MDSPAN))
      signature += ":#" + STR(GetValNo(*e->GetR(), VNKind::VNK_MDSPAN));
    else
      signature += ":#" + STR(GetValNo(*e->GetR()));
    auto sign = vn.SimplifySignature(e->LOC(), signature);
    if (sign != signature) {
      VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Simplify> '" << STR(n) << ": '"
                       << signature << "' to '" << sign << "'\n");
    }
    return sign;
  } else
    choreo_unreachable("sign mdspan failed on " + STR(n) + ": " +
                       n.TypeNameString() + ".");
  return "";
}

const std::string ShapeInference::SignNode(const AST::Node& n) {
  // std::cout << "sign node: " << STR(n) << "\n";
  if (auto* id = dyn_cast<AST::Identifier>(&n)) {
    auto name = id->name;
    if (auto sname = SSTab().NameInScopeOrNull(name)) {
      if (vn.HasValueNumberOfSignature(*sname))
        return *sname;
      else
        choreo_unreachable("symbol `" + *sname +
                           "' is not associated with a value number.");
    }
    // or else, it is a new name definition
    return SSTab().ScopedName(name);
  } else if (auto* e = dyn_cast<AST::Expr>(&n)) {
    if (e->op == "sizeof") {
      auto esign = GetSign(*e->GetR(), VNKind::VNK_MDSPAN);
      auto vl = vn.GenValueListFromSignature(esign);
      auto sz = MultiplyAll(vl);
      return vn.ValueItemToSignature(sz);
    }
    if (e->IsReference()) return GetSign(n);
    if (e->op == "ubound") return GetSign(*e->GetR(), VNKind::VNK_UBOUND);
    auto signature = e->op;
    if (e->GetC()) signature += ":#" + STR(GetValNo(*e->GetC()));
    if (e->GetL()) signature += ":#" + STR(GetValNo(*e->GetL()));
    assert(e->GetR() && "expression is invalid.");
    signature += ":#" + STR(GetValNo(*e->GetR()));
    auto sign = vn.SimplifySignature(e->LOC(), signature);
    if (sign != signature) {
      VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Simplify> '" << STR(n) << ": '"
                       << signature << "' to '" << sign << "'\n");
    }
    return sign;
  } else if (auto* ii = dyn_cast<AST::IntIndex>(&n)) {
    return "index_" + GetSign(*ii->value);
  }

  if (debug_visit)
    Warning(n.LOC(), "invalid signature for expression `" + AST::STR(n) +
                         "': " + n.TypeNameString() + ".");

  return ""; // invalid value
}

std::pair<const std::string, const std::string>
ShapeInference::SignBounded(const AST::Node& n) {
  // std::cout << "sign bounded: " << STR(n) << "\n";
  assert(!ast_vn.Hit(&n, VNKind::VNK_UBOUND));

  std::string v_sign = SignNode(n);
  std::string ub_sign;
  if (auto id = dyn_cast<AST::Identifier>(&n)) {
    if (SSTab().NameInScopeOrNull("@" + id->name)) {
      ub_sign = SSTab().InScopeName("@" + id->name);
    } else {
      // first time encounter
      ub_sign = SSTab().ScopedName("@" + id->name);
    }
  } else if (auto e = dyn_cast<AST::Expr>(&n)) {
    if (e->IsReference()) {
      ub_sign = GetSign(*e->GetReference(), VNKind::VNK_UBOUND);
    } else if (e->IsBinary()) {
      auto& lhs = *e->GetL();
      auto& rhs = *e->GetR();

      if (e->IsArith() && !e->IsUBArith()) {
        if (isa<BoundedType>(lhs.GetType()) && lhs.GetType()->Dims() == 1 &&
            CanYieldAnInteger(NodeType(rhs))) {
          ub_sign = GetSign(lhs, VNKind::VNK_UBOUND); // ubound does not change
        } else if (isa<BoundedType>(rhs.GetType()) &&
                   (rhs.GetType()->Dims() == 1) &&
                   CanYieldAnInteger(NodeType(lhs))) {
          ub_sign = GetSign(rhs, VNKind::VNK_UBOUND); // ubound does not change
        } else if (isa<BoundedITupleType>(lhs.GetType()) &&
                   (isa<ITupleType>(rhs.GetType())))
          ub_sign = GetSign(lhs, VNKind::VNK_UBOUND);
        else if (isa<BoundedITupleType>(rhs.GetType()) &&
                 (isa<ITupleType>(lhs.GetType())))
          ub_sign = GetSign(rhs, VNKind::VNK_UBOUND);
        else
          choreo_unreachable("operation '" + e->op +
                             "' is not permitted for '" + STR(lhs) + "(" +
                             PSTR(lhs.GetType()) + ")' and '" + STR(rhs) + "(" +
                             PSTR(rhs.GetType()) + ")'.");
      } else if (e->op == "#") {
        if (IsActualBoundedIntegerType(lhs.GetType()) &&
            IsActualBoundedIntegerType(rhs.GetType())) {
          auto lvn = GetValNo(lhs, VNKind::VNK_UBOUND);
          auto rvn = GetValNo(rhs, VNKind::VNK_UBOUND);
          ub_sign = vn.SimplifySignature(e->LOC(),
                                         "*:#" + STR(lvn) + ":#" + STR(rvn));
          v_sign[0] = '@';
        } else
          choreo_unreachable("operation is not permitted.");
      } else if (e->op == "#+" || e->op == "#-") {
        if (IsActualBoundedIntegerType(lhs.GetType()) &&
            isa<IntegerType>(rhs.GetType())) {
          auto lvn = GetValNo(lhs, VNKind::VNK_UBOUND);
          auto rvn = GetValNo(rhs, VNKind::VNK_VALUE);
          ub_sign = vn.SimplifySignature(
              e->LOC(), e->op.substr(1) + ":#" + STR(lvn) + ":#" + STR(rvn));
          v_sign[0] = '@';
        } else
          choreo_unreachable("operation is not permitted.");
      } else
        choreo_unreachable("operation is not supported for bounded variables.");
    } else
      choreo_unreachable("unexpected expression for bounded variables.");
  } else
    choreo_unreachable("unable to sign bounded: " + STR(n) + "(" +
                       n.TypeNameString() + ").");

  VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Bounded> '" << STR(n)
                   << "'s ubound: '" << ub_sign << "'\n");

  return {v_sign, ub_sign};
}

bool ShapeInference::HasValNo(const AST::Node& n, VNKind vnt) const {
  return ast_vn.Hit(&n, vnt);
}

// Note:
// the valno related to a node could either be:
//   1. integer value of the expression
//   2. associated span value of the expression
//   3. associated upper bound value of the expression
int ShapeInference::GetValNo(const AST::Node& n, VNKind vnt) const {
  if (vnt == VNKind::VNK_UBOUND) assert(isa<BoundedType>(NodeType(n)));
  return ast_vn.Get(&n, vnt);
}

int ShapeInference::GenValNo(const AST::Node& n) {
  // if (auto v = GetOrNull(n, NodeValNoKind(n))) return *v;

  auto Generate = [this, &n](const std::string& nsign, VNKind vnt) {
    // only generate values at the first time
    if (ast_vn.Hit(&n, vnt))
      choreo_unreachable("The '" + STR(vnt) +
                         "' valno has already been generated for " + STR(n) +
                         ".");

    if (nsign == "")
      Error(n.LOC(), "failed to generate " + STR(vnt) +
                         " signature for expression `" + STR(n) + "'.");

    // Different expression could have the same signature, which implies
    // duplicated computation that requires optimization.
    int val_no = vn.GetOrGenValueNumberFromSignature(nsign);

    ast_vn.Update(&n, val_no, vnt);

    return val_no;
  };

  // directly get those with multiple values (already generated)
  if (auto* il = dyn_cast<AST::IntLiteral>(&n)) {
    if (HasValNo(n)) return GetValNo(n);
    SignTy ilsign;
    if (IsUnKnownInteger(il->Val()))
      ilsign = "?";
    else
      ilsign = "const_" + std::to_string(il->Val());
    NumTy valno = vn.GetOrGenValueNumberFromSignature(ilsign);
    ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    return valno;
  } else if (auto* fl = dyn_cast<AST::FloatLiteral>(&n)) {
    if (HasValNo(n)) return GetValNo(n);
    SignTy flsign;
    if (fl->IsFloat32()) {
      auto f32 = fl->Val_f32();
      if (IsUnKnownFloatPoint(f32))
        flsign = "?"; // why?
      else
        flsign = "const_" + std::to_string(f32) + "f";
    } else if (fl->IsFloat64()) {
      auto f64 = fl->Val_f64();
      if (IsUnKnownFloatPoint(f64))
        flsign = "?"; // why?
      else
        flsign = "const_" + std::to_string(f64);
    } else
      choreo_unreachable("unexpected float point type.");
    NumTy valno = vn.GetOrGenValueNumberFromSignature(flsign);
    ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    return valno;
  } else if (auto* bl = dyn_cast<AST::BoolLiteral>(&n)) {
    if (HasValNo(n)) return GetValNo(n);
    NumTy valno = vn.GetOrGenValueNumberFromSignature(PSTR(bl));
    ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    return valno;
  } else if (auto* it = dyn_cast<AST::IntTuple>(&n)) {
    NumTy valno = GetValNo(*(it->GetValues()));
    ast_vn.Update(&n, valno, VNKind::VNK_MDSPAN);
    return valno;
  } else if (auto mds = dyn_cast<AST::MultiDimSpans>(&n)) {
    NumTy valno = GetValNo(*(mds->list));
    ast_vn.Update(&n, valno, VNKind::VNK_MDSPAN);
    return valno;
  } else if (auto sa = dyn_cast<AST::SpanAs>(&n)) {
    NumTy valno = GetValNo(*(sa->list));
    ast_vn.Update(&n, valno, VNKind::VNK_MDSPAN);
    return valno;
  } else if (auto e = dyn_cast<AST::Expr>(&n)) {
    if (auto r = e->GetReference()) {
      ast_vn.Copy(r.get(), e);
      return ast_vn.Get(e, NodeValNoKind(n));
    } else if (e->op == "ubound") {
      auto valno = GetValNo(*e->GetR(), VNKind::VNK_UBOUND);
      ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
      if (isa<ITupleType>(e->GetType()))
        ast_vn.Update(&n, valno, VNKind::VNK_MDSPAN);
      return valno;
    } else if (e->op == "dimof") {
      auto cv = RemovePrefixOrNull("index_const_", GetSign(*e->GetR()));
      assert(cv && "indexing of mdspan can not be evaluated.");
      auto index = std::stoi(*cv);
      if (ast_vn.Hit(e->GetL().get(), VNKind::VNK_MDSPAN)) {
        assert(!ast_vn.Hit(e, VNKind::VNK_VALUE));

        SignTy msign = GetSign(*e->GetL(), VNKind::VNK_MDSPAN);
        assert((index < CountElementsInSignature(msign)) &&
               "out of bound in 'dimof'.");
        NumTy valno = vn.GetNthValNo(msign, index);
        ast_vn.Update(e, valno, VNKind::VNK_VALUE);
        return valno;
      } else if (ast_vn.Hit(e->GetL().get(), VNKind::VNK_VALUE)) {
        assert(!ast_vn.Hit(e, VNKind::VNK_VALUE));
        SignTy msign = GetSign(*e->GetL(), VNKind::VNK_VALUE);
        assert((index < CountElementsInSignature(msign)) &&
               "out of bound in 'dimof'.");
        NumTy valno = vn.GetNthValNo(msign, index);
        ast_vn.Update(e, valno, VNKind::VNK_VALUE);
        return valno;
      } else if (ast_vn.Hit(e->GetL().get(), VNKind::VNK_UBOUND)) {
        SignTy msign = GetSign(*e->GetL(), VNKind::VNK_VALUE);
        assert((index < CountElementsInSignature(msign)) &&
               "out of bound in 'dimof'.");
        NumTy valno = vn.GetNthValNo(msign, index);
        ast_vn.Update(e, valno, VNKind::VNK_VALUE);

        SignTy usign = GetSign(*e->GetL(), VNKind::VNK_UBOUND);
        NumTy uvalno = vn.GetNthValNo(usign, index);
        ast_vn.Update(e, uvalno, VNKind::VNK_UBOUND);
        return uvalno;
      } else
        choreo_unreachable("unsupported dimof valno generation.");
    } else if (e->op == "getith") {
      Generate(SignNode(n), VNKind::VNK_VALUE);
      // the upper bound is not changed
      NumTy uvn = GetValNo(*e->GetL(), VNKind::VNK_UBOUND);
      ast_vn.Update(e, uvn, VNKind::VNK_UBOUND);
      return uvn;
    } else if (e->op == "?") {
      auto cond = GetSign(*e->GetC());
      if (cond == "true") {
        ast_vn.Copy(e->GetL().get(), e);
        return ast_vn.Get(e, NodeValNoKind(n));
      } else if (cond == "false") {
        ast_vn.Copy(e->GetR().get(), e);
        return ast_vn.Get(e, NodeValNoKind(n));
      }
    }
  }

  // generate both the valno and the ubound valno when required
  switch (NodeValNoKind(n)) {
  case VNKind::VNK_UBOUND: {
    // always generate two value numbers for the bounded type
    auto [v_sign, ubsign] = SignBounded(n);
    Generate(v_sign, VNKind::VNK_VALUE);
    return Generate(ubsign, VNKind::VNK_UBOUND);
  } break;
  case VNKind::VNK_MDSPAN: {
    // only generate spanned value for now.
    // TODO: shall we also generate node valno
    auto l_sign = SignSpan(n);
    return Generate(l_sign, VNKind::VNK_MDSPAN);
  } break;
  case VNKind::VNK_VALUE: {
    auto signature = SignNode(n);
    return Generate(signature, VNKind::VNK_VALUE);
  } break;
  default: choreo_unreachable("unsupported valno kind.");
  }
  return GetInvalidValueNumber();
}

const std::string
ShapeInference::VNSymbolName(const AST::Identifier& id) const {
  auto sig = id.name;
  auto pty = NodeType(id);
  if (isa<SpannedType>(pty) || GeneralFutureType(pty)) {
    sig = RemoveSuffix(sig, ".span") +
          ".span"; // only cares about value inside the mdspan
  }
  return sig;
}

} // end namespace Choreo
