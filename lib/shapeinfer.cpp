#include "shapeinfer.hpp"

using namespace Choreo::valno;

namespace Choreo {

void ShapeInference::InvalidateVisitorValNOs() {
  cur_vn.Invalidate();
  cur_mdspan_vn.Invalidate();
  cur_ub_vn.Invalidate();
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
    dbgs() << "]\t " << cur_vn.ToString(true) << "(vn),\t "
           << cur_mdspan_vn.ToString(true) << "(mds),\t "
           << cur_ub_vn.ToString(true) << "(ub)";

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
    SymbolAliasNum(InScopeName("@__choreo_no_tiling__"),
                   GetOrGenValNum(c_sn(1)));
    SymbolAliasNum(InScopeName("__choreo_no_tiling__"),
                   GetOrGenValNum(c_sn(0)));
    GetOrGenValNum(s_sn(InScopeName("__choreo_parent_dim__")));
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
    if (auto call = dyn_cast<AST::Call>(&n)) {
      if (call->template_args)
        in_template_param = true; // enter template param visit
    }
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
    if (auto sym = expr->GetSymbol()) {
      if (SSTab().IsDeclared(sym->name)) return GetSymbolType(sym->name);
    }
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

      if (val_sign->Count() == 1) {
        vs.push_back(val_no);
        continue;
      }

      auto msn = MSign(val_sign);
      assert(msn);
      for (int i = msn->Count() - 1; i >= 0; --i)
        work_list.push_front(GetValNum(msn->At(i)));
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
        vvs.push_back(GetValNum(name));

        auto uname = SSTab().GetScope(name) + "@" + SSTab().UnScopedName(name);
        uvs.push_back(GetValNum(uname));
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

  // for spanat, it allows mixup of bounded and non-bounded
  if (!uvs.empty()) assert(uvs.size() <= vvs.size());

  auto SetMultiValNo = [this, &mv](const std::vector<NumTy> vs, VNKind vnt) {
    assert(!ast_vn.Hit(&mv, vnt));
    assert(!vs.empty());

    auto mss = m_sn();
    for (auto& n : vs) mss->Append(vn.SignNum(n));

    auto valno = GetOrGenValNum(vn.Simplify(mss));
    ast_vn.Update(&mv, valno, vnt);
  };

  // Note: a multi-value may represent a mdspan but does not have a VNK_MDSPAN
  // valno.
  if (!vvs.empty()) SetMultiValNo(vvs, VNKind::VNK_VALUE);

  // the multi-value has a ubound only when all values have ubounds
  if (!uvs.empty() && uvs.size() == vvs.size())
    SetMultiValNo(uvs, VNKind::VNK_UBOUND);
}

bool ShapeInference::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);

  if (in_template_param) in_template_param = false; // exit template param visit
  if (cannot_proceed || !CanBeValueNumbered(&n)) {
    cur_vn.Invalidate();
    return true;
  }

  if (gen_values) {
    CollapseMultiValues(n);
    n.Opts().SetVals(vn.GenValueListFromValueNumber(ast_vn.Get(&n)));
    VST_DEBUG(dbgs() << " |-<exprval> values of multivalues: ["
                     << STR(n.Opts().GetVals()) << "]\n");
  }

  if (n.HasNote("array_dims")) {
    // apply the inferred shape to DataType node.
    auto nty = NodeType(n);
    assert(isa<ArrayType>(nty));
    cast<ArrayType>(nty)->dims = GenShape(GetValNo(n)).Value();
    SetNodeType(n, nty);
  }

  cur_vn.Invalidate();

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
  cur_vn.Invalidate();
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
    cur_vn.Invalidate();
    return true;
  }

  if (n.IsReference()) SetNodeType(n, NodeType(*n.GetReference()));
  auto nty = NodeType(n);
  cur_vn = GenValNo(n);

  auto ShouldOpt = [](const ValueList& vl) -> bool {
    if (!IsValidValueList(vl)) return false;
    // restricted mode: should always use expression when symbolic
    if (CCtx().HasFeature(ChoreoFeature::RSTM0))
      return IsValueListNumericOrBool(vl);
    return true;
  };
  // decide the optimized values
  switch (NodeValNoKind(n)) {
  case VNKind::VNK_VALUE: {
    auto vl = vn.GenValueListFromSignature(GetSign(n));
    if (ShouldOpt(vl)) {
      n.Opts().SetVals(vl);
      VST_DEBUG(dbgs() << " |-<exprval> <" << PSTR(nty) << "> " << STR(n)
                       << ": " << STR(vl) << "\n");
    }
  } break;
  case VNKind::VNK_UBOUND: {
    auto vl = vn.GenValueListFromSignature(GetSign(n));
    if (ShouldOpt(vl)) {
      // If there is symval of `id`, use it first (`id` is the with in
      // AST::WithIn). For example, `id` = `within_0::index`, then `vl` is just
      // the symbolic expr `within_0::index`, which is useless. While the symval
      // is `within_0::index__elem__0, within_0::index__elem__1`, which is
      // useful when generating offset in spanned operation.
      if (auto id = AST::GetIdentifier(n);
          id && SymVal(InScopeName(id->name)).HasVals())
        n.Opts().SetVals(SymVal(InScopeName(id->name)).GetVals());
      else
        n.Opts().SetVals(vl);
      VST_DEBUG(dbgs() << " |-<exprval> <" << PSTR(nty) << "> " << STR(n)
                       << ": " << STR(n.Opts().GetVals()) << "\n");
    }
    auto ub_vl = vn.GenValueListFromSignature(GetSign(n, VNKind::VNK_UBOUND));
    if (ShouldOpt(ub_vl)) n.Opts().SetUBounds(ub_vl);
    VST_DEBUG(dbgs() << " |-<exprbound> <" << PSTR(nty) << "> " << STR(n)
                     << ": " << STR(ub_vl) << "\n");
  } break;
  case VNKind::VNK_MDSPAN: {
    auto mds_valno = GetValNo(n, VNKind::VNK_MDSPAN);
    if (!mds_valno.IsValid()) return true; // drop all the work
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

  if (isa<BoundedITupleType>(nty)) {
    // fullfil the incomplete shaped types when necessary
    n.s = GenShape(SignValNo(cur_vn));
    VST_DEBUG(dbgs() << " |-<exprshape> <" << PSTR(nty) << "> " << STR(n)
                     << ": " << STR(n.s) << "\n");
    SetNodeType(n, MakeBoundedITupleType(n.s));
  } else if (isa<MDSpanType>(nty)) {
    // fullfil the incomplete shaped types when necessary
    n.s = GenShape(SignValNo(cur_vn));
    VST_DEBUG(dbgs() << " |-<exprshape> <" << PSTR(nty) << "> " << STR(n)
                     << ": " << STR(n.s) << "\n");
    auto mty = cast<MDSpanType>(nty->Clone());
    mty->SetShape(n.s);
    SetNodeType(n, mty);
    cur_mdspan_vn = cur_vn;
    // cur_vn.Invalidate();
  }

  if (IsActualBoundedIntegerType(NodeType(n))) {
    cur_ub_vn = cur_vn;
    cur_vn.Invalidate();
  }

  return true;
}

bool ShapeInference::Visit(AST::CastExpr& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;
  if (HasValNo(*n.GetR()))
    ast_vn.Update(&n, GetValNo(*n.GetR()), VNKind::VNK_VALUE);
  return true;
}

bool ShapeInference::Visit(AST::MultiDimSpans& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (n.list) {
    auto sign = GetSign(*n.list, VNKind::VNK_VALUE);
    auto rank = sign->Count();

    // The Shape now can be deduced from the value number.
    // Update the type detail accordingly.
    auto vl = GenShape(sign);
    SetMdsShape(n, vl);

    if (!IsValidRank(n.Rank())) n.SetRank(rank);

    // pass the list value number over
    cur_vn = ValNoSign(sign);
    ast_vn.Update(&n, cur_vn, VNKind::VNK_VALUE);
    cur_mdspan_vn = cur_vn;

  } else if (n.Rank() > 0) {
    auto unknown_spans = m_sn(unk_sn(), n.Rank());
    cur_mdspan_vn = GetOrGenValNum(unknown_spans);
    ast_vn.Update(&n, cur_mdspan_vn, VNKind::VNK_VALUE);
  } else {
    cur_mdspan_vn = NumTy::Unknown(); // failed to deduce the type detail
  }

  cur_vn.Invalidate();
  return true;
}

bool ShapeInference::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  auto name = n.name_str;
  if (n.init_expr) {
    assert(cur_mdspan_vn.IsValid() &&
           "invalid value number for the named type.");

    DefineASymbol(name, n.GetType());

    SymbolAliasNum(SSTab().ScopedName(name), cur_mdspan_vn);

    auto vl = vn.GenValueListFromValueNumber(cur_mdspan_vn);
    if (IsValidValueList(vl) && !IsComputable(vl))
      Note(n.init_expr->LOC(), "mdspan contains unspecific value.");

    cur_mdspan_vn.Invalidate(); // consumes the mdspan
  }
  return true;
}

bool ShapeInference::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  auto& name = n.name_str;

  if (n.IsArray()) {
    auto array_vn = GetValNo(*n.array_dims);
    // apply the inferred shape to DataType node.
    n.type->array_dims = GenShape(array_vn).Value();
    // special case for event var. Because `CanBeValueNumbered(event)` is false.
    if (auto eaty = dyn_cast<EventArrayType>(NodeType(*n.type)); eaty) {
      eaty->dims = n.type->array_dims;
      SetNodeType(n, eaty);
    }
  }

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

  if (n.init_expr && !isa<AST::Call>(n.init_expr)) {
    if (!CanBeValueNumbered(n.init_expr.get())) {
      GenValNum(SSTab().ScopedName(name));
      DefineASymbol(name, NodeType(n));
      return true;
    }
    nty = NodeType(n);
    if (GetSpannedType(nty)) {
      cur_mdspan_vn = GetValNo(*n.init_expr, VNKind::VNK_MDSPAN);
      assert(cur_mdspan_vn.IsValid() && "expecting a valid mdspan valno.");
      auto vl = vn.GenValueListFromValueNumber(cur_mdspan_vn);
      if (IsValidValueList(vl) && !IsComputable(vl))
        Note(n.init_expr->LOC(),
             "`" + n.name_str + "'s upper-bound value can not be evaluated.");
      SymbolAliasNum(SSTab().ScopedName(name + ".span"), cur_mdspan_vn);
    } else if (IsActualBoundedIntegerType(nty)) {
      cur_ub_vn = GetValNo(*n.init_expr, VNKind::VNK_UBOUND);
      assert(cur_ub_vn.IsValid());
      DefineASymbol("@" + name, MakeBoundedIntegerType(cur_ub_vn.Value()));
      SymbolAliasNum(SSTab().ScopedName("@" + name), cur_ub_vn);
      Shape s = GenShape(cur_ub_vn);
      nty = MakeBoundedITupleType(s);

      auto vl = vn.GenValueListFromValueNumber(cur_ub_vn);
      if (IsValidValueList(vl) && !IsComputable(vl))
        Note(n.init_expr->LOC(),
             "`" + n.name_str + "'s upper-bound value can not be evaluated.");

      SymbolAliasNum(SSTab().ScopedName(name), GetValNo(*n.init_expr));
      cur_ub_vn.Invalidate();
    } else {
      if (!isa<PlaceHolderType>(nty)) {
        cur_vn = GetValNo(*n.init_expr, VNKind::VNK_VALUE);
        assert(cur_vn.IsValid() &&
               "cur_mdspan_vn and cur_vn must be exclusive.");
        auto vl = vn.GenValueListFromValueNumber(cur_vn);
        if (IsValidValueList(vl) && !IsComputable(vl))
          Note(n.init_expr->LOC(),
               "`" + n.name_str + "'s value can not be evaluated.");
        if (IsMutable(*nty))
          GenValNum(SSTab().ScopedName(name));
        else
          SymbolAliasNum(SSTab().ScopedName(name), cur_vn);
      }
    }
  } else {
    // obtain the types from declaration.
    if (cur_mdspan_vn.IsValid()) {
      SymbolAliasNum(SSTab().ScopedName(name + ".span"), cur_mdspan_vn);
      auto mds_value = GenShape(cur_mdspan_vn);
      if (n.IsArray())
        nty =
            MakeDenseSpannedArrayType(n.type->base_type, mds_value,
                                      MakeValueList(n.ArrayDimensions()), sto);
      else
        nty = MakeDenseSpannedType(n.type->base_type, mds_value, sto);
    } else if (cur_vn.IsValid()) {
      SymbolAliasNum(SSTab().ScopedName(name), cur_vn);
      nty = NodeType(*n.type);
    } else {
      nty = NodeType(*n.type);
      if (auto sty = dyn_cast<ScalarType>(nty); sty && sty->IsMutable())
        cur_vn = GenValNum(SSTab().ScopedName(name));
    }
  }

  // fill-up the symbol table
  assert(nty);
  DefineASymbol(name, nty);
  SetNodeType(n, nty);

  if ((isa<ScalarIntegerType>(nty) || isa<ScalarFloatType>(nty)) &&
      cur_vn.IsValid()) {
    // mutables or vars of float type do not have constant values
    if (IsMutable(*nty) || isa<ScalarFloatType>(nty)) {
      auto vi = sbe::sym(InScopeName(name));
      VST_DEBUG(dbgs() << " |-<symval> " << InScopeName(name) << ": " << STR(vi)
                       << "\n");
      SymVal(InScopeName(name)).SetVal(vi);
    } else {
      auto shape = GenShape(cur_vn);
      assert(shape.DimCount() == 1);
      VST_DEBUG(dbgs() << " |-<symval> " << InScopeName(name) << ": "
                       << STR(shape.ValueAt(0)) << "\n");
      SymVal(InScopeName(name)).SetVal(shape.ValueAt(0));
    }
  } else if (isa<ITupleType>(nty)) {
    SymVal(InScopeName(name)).SetVals(vn.GenValueListFromValueNumber(cur_vn));
    VST_DEBUG(dbgs() << " |-<symval> " << InScopeName(name) << ": "
                     << STR(SymVal(InScopeName(name)).GetVals()) << "\n");
  }

  if (auto sty = GetSpannedType(n.GetType()))
    DefineASymbol(name + ".span", sty->GetMDSpanType());

  cur_mdspan_vn.Invalidate(); // stop propagation
  cur_vn.Invalidate();

  return true;
}

bool ShapeInference::Visit(AST::IntTuple& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  auto sign = SignValNo(GenValNo(n));
  auto cnt = sign->Count();
  // cur_ituple_vn = cur_vn;
  SetNodeType(n, MakeITupleType(cnt));

  cur_vn.Invalidate(); // Currently cut off value numbering
  return true;
}

bool ShapeInference::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);
  return true;
}

bool ShapeInference::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (n.da->AccessElement()) return true;
  // if defined in current scope, do not re-define
  if (SSTab().DeclaredInScope(n.GetName())) return true;
  // if assigned to a mutable variable, do not re-define
  if (IsMutable(*NodeType(*n.da->data))) return true;

  // this is the un-type-annotated declaration
  auto nty = n.value->GetType();
  DefineASymbol(n.GetName(), nty);

  auto name = n.GetName();
  if (auto san = dyn_cast<AST::SpanAs>(n.value))
    assert((name == san->nid->name) && "inconsistent span_as variable name.");

  if (auto sty = GetSpannedType(nty)) {
    name = RemoveSuffix(name, ".span") + ".span";
    DefineASymbol(name, sty->GetMDSpanType());
    cur_mdspan_vn = GetValNo(*n.value, VNKind::VNK_MDSPAN);
    assert(cur_mdspan_vn.IsValid() && "expected a valid current value number.");
    SymbolAliasNum(SSTab().ScopedName(name), cur_mdspan_vn);
    return true;
  } else if (isa<ITupleType>(nty)) {
    cur_mdspan_vn = GetValNo(*n.value, VNKind::VNK_VALUE);
    SymbolAliasNum(SSTab().ScopedName(name), cur_mdspan_vn);
    return true;
  } else if (auto sty = dyn_cast<ScalarType>(nty); sty && sty->IsMutable()) {
    // we need to generate a valno for mutable names
    cur_vn = GetOrGenValNum(s_sn(SSTab().ScopedName(name)));
    return true;
  } else if (isa<BoundedType>(nty)) {
    auto uname = "@" + name;
    cur_ub_vn = GetValNo(*n.value, VNKind::VNK_UBOUND);
    assert(cur_ub_vn.IsValid());
    DefineASymbol(uname, MakeIntegerType());
    SymbolAliasNum(SSTab().ScopedName(uname), cur_ub_vn);
    cur_ub_vn.Invalidate();
  }

  if (IsMutable(*NodeType(n))) return true; // other mutables are not valno-able

  cur_vn = GetValNo(*n.value);
  assert(cur_vn.IsValid() && "expected a valid current value number.");
  SymbolAliasNum(SSTab().ScopedName(name), cur_vn);

  if (isa<ScalarIntegerType>(nty) && cur_vn.IsValid()) {
    auto shape = GenShape(cur_vn);
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

  if (cur_mdspan_vn.IsValid()) { cur_vn = cur_mdspan_vn; }

  if (n.mdspan_type) ast_vn.Copy(n.mdspan_type.get(), &n, VNKind::VNK_VALUE);

  return true;
}

bool ShapeInference::Visit(AST::NoValue& n) {
  TraceEachVisit(n);
  ast_vn.Update(&n, NumTy::None(), VNKind::VNK_VALUE);
  return true;
}

bool ShapeInference::Visit(AST::Identifier& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (!CanBeValueNumbered(&n)) { return false; }

  auto nty = NodeType(n);
  if (in_template_param && !SSTab().IsDeclared(n.name)) {
    DefineASymbol(n.name, MakeIntegerType());
    GenValNum(SSTab().InScopeName(n.name));
    cur_vn = GenValNo(n);
    return true;
  }

  if (SSTab().IsDeclared(n.name)) {
    // Not neccessary to generate new valno. Instead utlize existing valno
    // passed from the symbols
    switch (NodeValNoKind(n)) {
    case VNKind::VNK_UBOUND: {
      if (!vn.HasValueNumberOfSignature(
              s_sn(SSTab().InScopeName("@" + n.name))))
        choreo_unreachable("value number of `" + SSTab().InScopeName(n.name) +
                           "' has not been generated.");
      cur_vn = GetValNum(SSTab().InScopeName("@" + n.name));
      ast_vn.Update(&n, cur_vn, VNKind::VNK_UBOUND);
      if (!vn.HasValueNumberOfSignature(s_sn(SSTab().InScopeName(n.name))))
        choreo_unreachable("value number of `" + SSTab().InScopeName(n.name) +
                           "' has not been generated.");
      auto valno = GetValNum(SSTab().InScopeName(n.name));
      ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    } break;
    case VNKind::VNK_MDSPAN: {
      auto name =
          RemoveSuffix(RemoveSuffix(n.name, ".span"), ".data") + ".span";
      if (!vn.HasValueNumberOfSignature(s_sn(SSTab().InScopeName(name))))
        choreo_unreachable("value number of `" + SSTab().InScopeName(name) +
                           "' has not been generated.");
      cur_vn = GetValNum(SSTab().InScopeName(name));
      ast_vn.Update(&n, cur_vn, VNKind::VNK_MDSPAN);
    } break;
    case VNKind::VNK_VALUE: {
      // it is a reference
      if (!vn.HasValueNumberOfSignature(s_sn(SSTab().InScopeName(n.name))))
        choreo_unreachable("value number of `" + SSTab().InScopeName(n.name) +
                           "' has not been generated.");
      cur_vn = GetValNum(SSTab().InScopeName(n.name));
      ast_vn.Update(&n, cur_vn, VNKind::VNK_VALUE);
    } break;
    default: choreo_unreachable("unsupport valno kind.");
    }

    return true;
  }

  if (!gen_values) return false;

  if (allow_named_dim) { // for named dims in parameters
    if (isa<NoValueType>(nty)) {
      cur_vn = NumTy::None();
    } else if (!SSTab().DeclaredInScope(n.name)) {
      DefineASymbol(n.name, MakeIntegerType());
      cur_vn = GenValNum(SSTab().InScopeName(n.name));
    } else {
      cur_vn = GetValNum(SSTab().InScopeName(n.name));
    }
    ast_vn.Update(&n, cur_vn, VNKind::VNK_VALUE);
    return true;
  }

  if (HasValNo(n)) {
    Error1(n.LOC(), "value number has been generated for `" + n.name + "'.");
    return false;
  }

  // symbolic and bounded symbol requires valno
  if (CanYieldAnInteger(nty)) cur_vn = GenValNo(n);

  return true;
}

bool ShapeInference::Visit(AST::Parameter& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (n.type->ExplicitSpanned()) {
    assert(isa<AST::MultiDimSpans>(n.type->mdspan_type.get()) &&
           "Invalid mdspan.");
    auto span = cast<AST::MultiDimSpans>(n.type->mdspan_type.get());
    if (span->list) {
      cur_mdspan_vn = GetValNo(*span->list);
      assert(cur_mdspan_vn.IsValid() && "unexpected value number for mdspan.");

      // Put alias names of mdspan into the value number table
      SymbolAliasNum(SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
      SetNodeType(*n.type, MakeDenseSpannedType(n.type->base_type,
                                                span->GetTypeDetail()));
    } else if (IsValidRank(span->Rank())) {
      cur_mdspan_vn = GetValNo(*span->list, VNKind::VNK_MDSPAN);
      assert(cur_mdspan_vn.IsValid() && "unexpected value number for mdspan.");
      // Put alias names of mdspan into the value number table
      SymbolAliasNum(SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
      SetNodeType(*n.type, MakeDenseSpannedType(n.type->base_type,
                                                span->GetTypeDetail()));
    } else {
      // the value number is unknown at compile time
      Error1(n.LOC(), "The type can not be inferred at compile time.");
      return false;
    }

    if (n.sym) {
      auto span_name = n.sym->name + ".span";
      DefineASymbol(span_name,
                    cast<SpannedType>(n.type->GetType())->GetMDSpanType());

      DefineASymbol(n.sym->name, n.type->GetType());
    }

    InvalidateVisitorValNOs();
    return true;
  }

  if (n.sym && n.type->IsScalar()) {
    assert(!cur_mdspan_vn.IsValid() && "unexpected current mdspan value.");

    // get the value number and make it defined
    GenValNum(SSTab().ScopedName(n.sym->name));
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

  auto vl = vn.GenValueListFromSignature(b_sign);
  if (IsValidValueList(vl) && !IsComputable(vl))
    Error1(n.BoundExpr()->LOC(), "The parallel count can not be evaluated.");

  Shape s = GenShape(b_sign);
  SetNodeType(n, MakeMDSpanType(s)); // useful for the sema check

  std::string iv_name = SSTab().ScopedName("@" + n.BPV()->name);
  SymbolAliasNum(iv_name, b_valno);
  SetNodeType(*n.BPV(), MakeBoundedITupleType(s));
  DefineASymbol("@" + n.BPV()->name, MakeMDSpanType(s));
  VST_DEBUG(dbgs() << " |-<pvbound> " << n.BPV()->name << ": " << STR(s)
                   << "\n");

  DefineASymbol(n.BPV()->name, n.BPV()->GetType());
  assert(!ast_vn.Hit(n.BPV().get(), VNKind::VNK_VALUE) &&
         "The value number has already been generated.");
  auto sname = SSTab().ScopedName(n.BPV()->name);
  auto vv = GetOrGenValNum(sname);
  ast_vn.Update(n.BPV().get(), vv, VNKind::VNK_VALUE);
  n.BoundExpr()->Opts().SetVals(s.Value());

  assert(n.HasSubPVs() && "normalization failed.");

  [[maybe_unused]] auto sign_cnt = b_sign->Count();
  assert((size_t)sign_cnt == n.SubPVCount());
  auto msn = vn.ToMSign(b_sign);
  std::string idx2dim[] = {"x", "y", "z"};
  for (size_t index = 0; index < msn->Count(); ++index) {
    auto valno = GetValNum(msn->At(index));
    auto id = n.GetSubPV(index);
    std::string pv_name = SSTab().ScopedName("@" + id->name);
    SymbolAliasNum(pv_name, valno);
    Shape s = GenShape(valno);
    SetNodeType(*id, MakeBoundedITupleType(s, "pi", idx2dim[index]));
    DefineASymbol("@" + id->name, MakeMDSpanType(s));

    assert(!ast_vn.Hit(id.get(), VNKind::VNK_VALUE) &&
           "The value number has already been generated.");
    auto sname = SSTab().ScopedName(id->name);
    auto vv = GetOrGenValNum(sname);
    ast_vn.Update(id.get(), vv, VNKind::VNK_VALUE);
    DefineASymbol(id->name, id->GetType());

    // update the bound opt value
    auto ubvi = vn.GenValueItemFromValueNumber(valno);
    n.BoundExprAt(index)->Opts().SetVal(ubvi);
  }

  VST_DEBUG(dbgs() << " |-<pv-bounds> {" << DelimitedSTR(n.AllSubPVs())
                   << "} : " << STR(GenShape(b_sign)) << "\n");

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
  auto l_vn = GetValNum(SSTab().ScopedName("@" + l_id->name));
  auto r_vn = GetValNum(SSTab().ScopedName("@" + r_id->name));

  // TODO: sometimes the lhs would have same valno with existing one, which is
  // allowed. However, for runtime valued bound, they may have different
  // bound. The problem here is how to judge if the upper bound of bounded
  // variables are actually illegal? (e.g, different static upper bound)
  vn.BindValueNumbers(l_vn, r_vn);
  VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Bind> VN " << l_vn << " <-> VN "
                   << r_vn << "\n");
  return true;
}

bool ShapeInference::Visit(AST::WithIn& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  cur_mdspan_vn = GetValNo(*n.in, VNKind::VNK_VALUE);
  auto mds_sign = vn.ToMSign(SignValNo(cur_mdspan_vn));

  // requires the elements inside mdspan to be non-zero values
  // we have to abend early here since it blocks further shape inference
  if (vn.ContainsZero(mds_sign)) {
    Error1(
        n.LOC(),
        "zero value is deduced for the mdspan inside the with-in statement.");
    cannot_proceed = true;
    Error(n.LOC(),
          "unable to apply shape inference for function '" + fname + "'.");
    return false;
  }

  auto sign = vn.ToMSign(mds_sign);
  for (size_t index = 0; index < sign->Count(); ++index) {
    const NumTy& valno = GetValNum(sign->At(index));
    if (valno.IsUnknown()) continue; // do not associate it with vn of "?"
    if (n.with) {
      std::string name = SSTab().ScopedName("@" + n.with->name) + "(" +
                         std::to_string(index) + ")";
      SymbolAliasNum(name, valno);
    }

    if (n.with_matchers) {
      // set the node type
      auto sym = cast<AST::Identifier>(n.with_matchers->ValueAt(index));
      Shape s = GenShape(SignValNo(valno));
      SetNodeType(*sym, MakeBoundedITupleType(s));

      // generate the ubound symbol and associate its valno
      DefineASymbol("@" + sym->name, MakeMDSpanType(s));
      auto ub_name = SSTab().ScopedName("@" + sym->name);
      SymbolAliasNum(ub_name, valno);
      ast_vn.Update(&n, valno, VNKind::VNK_UBOUND);

      // generate the symbol but only generate the symbolic node value
      DefineASymbol(sym->name, sym->GetType());
      assert(!ast_vn.Hit(sym.get(), VNKind::VNK_VALUE) &&
             "The value number has already been generated.");
      auto sname = SSTab().ScopedName(sym->name);
      auto vv = GetOrGenValNum(sname);
      ast_vn.Update(sym.get(), vv, VNKind::VNK_VALUE);
    }
  }

  if (n.with) {
    // generate symbolic node valno
    DefineASymbol(n.with->name, n.with->GetType());
    assert(!ast_vn.Hit(n.with.get(), VNKind::VNK_VALUE) &&
           "The value number has already been generated.");
    auto sname = SSTab().ScopedName(n.with->name);
    auto vv = GetOrGenValNum(sname);
    ast_vn.Update(n.with.get(), vv, VNKind::VNK_VALUE);

    // fill the detail values
    if (n.with_matchers) CollapseMultiValues(*n.with_matchers);
    SymVal(sname).SetVals(
        vn.GenValueListFromValueNumber(ast_vn.Get(n.with_matchers.get())));
    VST_DEBUG(dbgs() << " |-<symval> " << sname << ": "
                     << STR(SymVal(sname).GetVals()) << "\n");

    // upper-bound valnos
    SymbolAliasNum(SSTab().ScopedName("@" + n.with->name), cur_mdspan_vn);
    Shape s = GenShape(mds_sign);
    SetNodeType(*n.with, MakeBoundedITupleType(s));
    DefineASymbol("@" + n.with->name, MakeMDSpanType(s));
  }

  cur_mdspan_vn.Invalidate();

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

  auto shape = GenShape(cur_mdspan_vn);
  auto nty = MakeDenseSpannedType(sty->ElementType(), shape, sty->GetStorage());

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
        s_sn(SSTab().ScopedName(n.future + ".span")));
    cur_vn.Invalidate();
    return true;
  }

  // try to report more errors
  if (!cur_vn.IsValid() && error_count > 0) return false;

  if (auto pcfg = dyn_cast<PadConfig>(n.config)) {
    size_t size = pcfg->pad_high->Count();

    auto from_vn = GetValNo(*n.GetFrom(), VNKind::VNK_MDSPAN);
    auto s_cnt = vn.Flatten(from_vn).size();
    if (s_cnt != size) {
      Error1(n.LOC(), "rank mismatch: padding config requires " +
                          std::to_string(size) + ", but got data ranked " +
                          std::to_string(s_cnt) + ".");
      return false;
    }

    auto mss = m_sn();
    for (size_t i = 0; i < size; ++i) {
      auto h_l_sig = vn.MakeOpSign("+", GetSign(*pcfg->pad_high->ValueAt(i)),
                                   GetSign(*pcfg->pad_low->ValueAt(i)));
      // now generate signature for original signature plus padding values
      mss->Append(
          vn.MakeOpSign("+", h_l_sig, GetSign(*pcfg->pad_mid->ValueAt(i))));
    }
    // update the cur_vn
    cur_vn = vn.MakeOpNum("+", from_vn, GetOrGenValNum(mss));
  } else if (auto tcfg = dyn_cast<TransposeConfig>(n.config)) {
    auto size = tcfg->dim_values.size();
    auto from_vn = GetValNo(*n.GetFrom(), VNKind::VNK_MDSPAN);
    auto s_cnt = vn.Flatten(from_vn).size();
    if (s_cnt != size) {
      Error1(n.LOC(), "rank mismatch: transpose config requires " +
                          std::to_string(size) + ", but got data ranked " +
                          std::to_string(s_cnt) + ".");
      return false;
    }

    // gen new vn if and only if n.to is AST::Memory
    if (isa<AST::Memory>(n.to)) {
      auto& dim_values = tcfg->dim_values;
      auto shape_components = vn.NumVector(vn.SignNum(cur_vn));
      std::vector<NumTy> mss;
      for (auto dim : dim_values) mss.push_back(shape_components[dim]);
      cur_vn = vn.MakePluralNum(mss);
    }
  }

  auto s = GenShape(cur_vn);
  // annotate the shape on AST for later type inference
  if (n.IsDstInferred()) {
    auto fsty = GetSpannedType(n.GetFrom()->GetType());
    auto tsty = MakeDenseSpannedType(fsty->ElementType(), s,
                                     cast<AST::Memory>(n.GetTo())->Get());
    SetNodeType(*n.GetTo(), tsty);
    SetNodeType(n, MakeFutureType(CloneP(tsty), n.IsAsync()));
  } else {
    auto tsty = GetSpannedType(n.GetTo()->GetType());
    assert(tsty);
    SetNodeType(n, MakeShapedFutureType(s, n.IsAsync(), tsty->GetStrides(),
                                        tsty->ElementType()));
  }

  if (n.future.empty()) {
    cur_vn.Invalidate();
    return true;
  }

  if (SSTab().IsDeclared(n.future)) {
    assert(
        cast<PlaceHolderType>(SSTab().LookupSymbol(n.future))->GetBaseType() ==
        BaseType::FUTURE);
    SymbolRebindNum(SSTab().InScopeName(n.future) + ".span", cur_vn);
    SSTab().ModifySymbolType(n.future, n.GetType());
    SSTab().ModifySymbolType(n.future + ".span", MakeMDSpanType(s));
  } else {
    std::string f_span = n.future + ".span";
    SymbolAliasNum(SSTab().ScopedName(f_span), cur_vn);
    DefineASymbol(n.future, n.GetType());
    DefineASymbol(f_span, MakeMDSpanType(s)); // implicit symbol
  }

  cur_vn.Invalidate();
  return true;
}

bool ShapeInference::Visit(AST::MMA& n) {
  TraceEachVisit(n);
  // NOTE: The node type maybe differ from symbol type!
  auto& op = *n.GetOperation();
  switch (op.Tag()) {
  case AST::MMAOperation::Fill: {
    if (op.FillingIsDecl()) {
      NumTy array_vn;
      if (op.FillingArrayDims()) array_vn = GetValNo(*op.FillingArrayDims());
      std::string fill_sym = AST::FragName(op.FillingTo());
      auto fill_ty = op.FillingType();
      if (fill_ty != BaseType::UNKSCALAR) {
        if (op.FillingArrayDims())
          DefineASymbol(fill_sym,
                        MakeUnRankedSpannedArrayType(
                            fill_ty, GenShape(array_vn).Value(), Storage::REG));
        else
          DefineASymbol(fill_sym,
                        MakeUnRankedSpannedType(fill_ty, Storage::REG));
      } else {
        if (op.FillingArrayDims())
          DefineASymbol(fill_sym,
                        MakeDummySpannedArrayType(GenShape(array_vn).Value()));
        else
          DefineASymbol(fill_sym, MakeDummySpannedType());
      }
      DefineASymbol(fill_sym + ".span", MakeUninitMDSpanType());
      SymbolAliasNoNum(SSTab().InScopeName(fill_sym) +
                       ".span"); // valno is yet invalid
    }
  } break;
  case AST::MMAOperation::Load: {
    std::string load_to_sym = AST::FragName(op.LoadTo());
    auto fty = cast<SpannedType>(op.LoadFrom()->GetType());
    auto f_span = load_to_sym + ".span";
    SymbolAliasNum(SSTab().ScopedName(f_span), cur_vn);
    auto s = MakeDenseSpannedType(fty->ElementType(), GenShape(cur_vn),
                                  Storage::REG);
    auto f = MakeFutureType(s, op.IsAsync());
    DefineASymbol(load_to_sym, f);
    DefineASymbol(f_span, s->Clone());
    SetNodeType(n, f);
  } break;
  case AST::MMAOperation::Exec: {
    std::string op0_sym = AST::FragName(op.ExecOperand(0)); // mc
    std::string op1_sym = AST::FragName(op.ExecOperand(1)); // ma
    std::string op2_sym = AST::FragName(op.ExecOperand(2)); // mb
    auto fty = GetSpannedType(GetSymbolType(op1_sym));
    assert(fty);
    auto lspan = RemoveSuffix(SSTab().InScopeName(op1_sym), ".data") + ".span";
    auto rspan = RemoveSuffix(SSTab().InScopeName(op2_sym), ".data") + ".span";
    auto lsig = cast<MultiSigns>(SymbolSign(lspan));
    auto rsig = cast<MultiSigns>(SymbolSign(rspan));
    auto asig = m_sn();
    switch (op.GetMethod()) {
    case AST::MMAOperation::ROW_ROW:
      asig->Append(lsig->At(0));
      asig->Append(rsig->At(0));
      break;
    case AST::MMAOperation::ROW_COL:
      asig->Append(lsig->At(0));
      asig->Append(rsig->At(1));
      break;
    case AST::MMAOperation::COL_ROW:
      asig->Append(lsig->At(1));
      asig->Append(rsig->At(0));
      break;
    case AST::MMAOperation::COL_COL:
      asig->Append(lsig->At(1));
      asig->Append(rsig->At(1));
      break;
    default: choreo_unreachable("unsupported mma execution method.");
    }
    cur_vn = GetOrGenValNum(asig);
    auto mdsym = SSTab().InScopeName(op0_sym) + ".span";
    if (!vn.HasValidValueNumberOfSignature(s_sn(mdsym)))
      SymbolAliasNum(mdsym, cur_vn);
    auto mty = MakeMDSpanType(GenShape(cur_vn));
    auto c_sty = GetSpannedType(GetSymbolType(op0_sym));
    auto c_elem = (c_sty && c_sty->ElementType() != BaseType::UNKSCALAR)
                      ? c_sty->ElementType()
                      : fty->ElementType();
    if (op.IsSparse()) {
      auto a_elem = fty->ElementType();
      bool a_is_fp8 =
          a_elem == BaseType::F8_E4M3 || a_elem == BaseType::F8_E5M2 ||
          a_elem == BaseType::F8_UE4M3 || a_elem == BaseType::F8_UE8M0;
      bool c_is_fp8 =
          c_elem == BaseType::F8_E4M3 || c_elem == BaseType::F8_E5M2 ||
          c_elem == BaseType::F8_UE4M3 || c_elem == BaseType::F8_UE8M0;
      if (a_is_fp8 && c_is_fp8) { c_elem = BaseType::F32; }
    }
    if (AST::FragIsArrayElem(op.ExecOperand(0))) {
      auto pty = SSTab().LookupSymbol(op0_sym);
      assert(isa<ArrayType>(pty));
      auto sty = MakeDenseSpannedArrayType(
          c_elem, GenShape(cur_vn), GetArrayDimensions(pty), Storage::REG);
      UpdateSymbolType(op0_sym, sty);
    } else {
      auto sty = MakeDenseSpannedType(c_elem, GenShape(cur_vn), Storage::REG);
      UpdateSymbolType(op0_sym, sty);
    }
    UpdateSymbolType(op0_sym + ".span", mty);

    // Metadata handling for sparse MMA (operand 3)
    if (op.IsSparse() && op.ExecOperand(3)) {
      std::string mdata_sym = AST::FragName(op.ExecOperand(3));
      auto mdata_span =
          RemoveSuffix(SSTab().InScopeName(mdata_sym), ".data") + ".span";
      // We don't necessarily update the result shape based on E,
      // but we ensure it's visited and registered in the valno table if needed.
    }
    auto sym_ty = GetSymbolType(op0_sym);
    // always set the node type to spannedtype in exec.
    SetNodeType(n, GetSpannedType(sym_ty)->Clone());
  } break;
  case AST::MMAOperation::Store: {
  } break;
  default: break;
  }
  return true;
}

bool ShapeInference::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  auto pty = SSTab().LookupSymbol(n.data->name);
  assert((isa<SpannedType>(pty) || isa<FutureType>(pty)) &&
         "unexpected data type.");

  auto span_name = RemoveSuffix(n.data->name, ".data") + ".span";

  auto sty = GetSpannedType(pty);

  cur_vn = GetValNum(SSTab().InScopeName(span_name));

  if (n.NoOperation()) {
    // it is just a symbol reference
    auto vl = vn.GenValueListFromValueNumber(cur_vn);
    if (IsValidValueList(vl) && !IsComputable(vl))
      Error1(n.LOC(), "The destination block shape can not be evaluated.");

    auto res_shape = GenShape(cur_vn);

    // set the chunkat's type
    auto nty = MakeSpannedType(sty->e_type, res_shape, sty->GetStrides(),
                               sty->GetStorage());
    SetNodeType(n, nty);

    n.SetBlockShape(nty->GetShape());

    assert(n.GetBlockShape().IsValid());
    ast_vn.Update(&n, cur_vn, VNKind::VNK_MDSPAN);

    return true;
  }

  auto data_sig = SymbolSign(SSTab().InScopeName(span_name));
  auto data_vns = vn.Flatten(GetValNum(data_sig));

  auto data_strd = sty->GetStrides();
  if (data_vns.size() > 1)
    assert(!data_strd.empty() && "strides are not obtained.");

  auto cur_vns = data_vns;
  auto cur_strd = data_strd;

  { // make sure all expressions get the value numbers
    VST_DEBUG(dbgs() << "+--[" << n.TypeNameString() << ": begin sub-nodes]\n");
    auto old_gv = gen_values;
    gen_values = true;
    for (auto op : n.AllOperations()) op->accept(*this);
    gen_values = old_gv;
    VST_DEBUG(dbgs() << "+--[" << n.TypeNameString() << ": end sub-nodes]\n");
  }

  // handle all spanned expressions iteratively
  for (auto op : n.AllOperations()) {

    std::vector<NumTy> tfs_vns;  // value number of tiling factors
    std::vector<NumTy> sbs_vns;  // value number of subspan
    std::vector<NumTy> idx_vns;  // value number of indices
    std::vector<NumTy> off_vns;  // value number of offsets
    std::vector<NumTy> stp_vns;  // value number of steps
    std::vector<NumTy> strd_vns; // value number of strides

    if (auto rop = dyn_cast<AST::SOP::Reshape>(op)) {
      auto rshp_vn = GetValNo(*rop->GetNewSpan());
      auto rvi = vn.GenValueListFromValueNumber(rshp_vn);
      auto cvi = vn.GenValueListFromSignature(vn.MakePluralSign(cur_vns));
      // check if the mutilplicant is equal
      auto r_count = MultiplyAll(rvi);
      auto c_count = MultiplyAll(cvi);
      if (sbe::cne(r_count, c_count)) {
        if (VIIsInt(r_count) && VIIsInt(c_count)) {
          Error1(op->LOC(), "can not apply span_as to reshape from [" +
                                STR(cvi) + "](" + STR(c_count) + ") to [" +
                                STR(rvi) + "](" + STR(r_count) + ").");
        } else if (VIIsNil(r_count) || VIIsNil(c_count)) {
          Error1(op->LOC(), "can not apply span_as to a mdspan with infinite a "
                            "dimension value.");
        } else {
          Warning(op->LOC(), "can not prove equality of mdspan [" + STR(cvi) +
                                 "] and mdspan [" + STR(rvi) +
                                 "] at compile-time.");
          // TODO: (at semacheck) emit runtime check
        }
      }
      cur_vns = vn.NumVector(SignValNo(rshp_vn));
      auto bshape = GenShape(vn.MakePluralSign(cur_vns));
      op->SetBlockShape(bshape);
      cur_strd = bshape.GenDenseStrides();
      op->SetBlockStrides(cur_strd);
      VST_DEBUG(dbgs() << " |-<sop: " << PSTR(op)
                       << "> block shape: " << STR(bshape)
                       << ", strides: " << STR(cur_strd) << "\n");
    } else {
      // when the code provides explicit tiling factors or subspan
      auto tfs = op->GetTilingFactors();
      auto sbs = op->GetSubSpan();
      auto idx = op->GetIndices();
      auto off = op->GetOffsets();
      auto stp = op->GetSteps();
      auto strd = op->GetStrides();
      if (tfs) tfs_vns = vn.Flatten(GetValNo(*tfs));
      if (sbs) sbs_vns = vn.Flatten(GetValNo(*sbs));
      if (idx) idx_vns = vn.Flatten(GetValNo(*idx));
      if (off) off_vns = vn.Flatten(GetValNo(*off));
      if (stp) stp_vns = vn.Flatten(GetValNo(*stp));
      if (strd) strd_vns = vn.Flatten(GetValNo(*strd));

      if (sbs)
        for (size_t index = 0; index < sbs_vns.size(); ++index) {
          auto sbi = vn.GenValueItemFromValueNumber(sbs_vns[index]);
          if (STR(sbi) == "::__choreo_parent_dim__") {
            if (isa<AST::SOP::ModSpan>(op))
              continue;
            else
              sbs_vns[index] = cur_vns[index];
          }
        }

      if (tfs) {
        assert(!sbs && "defining both tilling-factors and subspan.");
        assert(idx && "not defining indexing.");
        if (cur_vns.size() != tfs_vns.size()) {
          Error1(op->LOC(),
                 "data rank (" + std::to_string(cur_vns.size()) +
                     ") must be consistent with tiling factor count (" +
                     std::to_string(tfs_vns.size()) + ").");
          return false;
        }
        for (size_t index = 0; index < cur_vns.size(); ++index) {
          auto shi = vn.GenValueItemFromValueNumber(cur_vns[index]);
          auto tfi = vn.GenValueItemFromValueNumber(tfs_vns[index]);
          if (sbe::ceq(tfi, sbe::nu(0))) {
            Error1(tfs->ValueAt(index)->LOC(),
                   "tiling factor can not be zero for dimension " +
                       std::to_string(index) + ".");
            return false; // can not continue
          } else if (sbe::clt(shi, tfi))
            Error1(tfs->LOC(), "tiling factor exceeds data size (" + STR(tfi) +
                                   " > " + PSTR(shi) + ") in dimension " +
                                   std::to_string(index) + ".");
          sbs_vns.push_back(vn.MakeOpNum("/", cur_vns[index], tfs_vns[index]));
        }
      } else if (sbs) {
        assert(!tfs && "defining both tilling-factors and subspan.");
        if (cur_vns.size() != sbs_vns.size()) {
          Error1(op->LOC(), "data rank (" + std::to_string(cur_vns.size()) +
                                ") must be consistent with subspan rank (" +
                                std::to_string(sbs_vns.size()) + ").");
          return false;
        }
        for (size_t index = 0; index < cur_vns.size(); ++index) {
          auto shi = vn.GenValueItemFromValueNumber(cur_vns[index]);
          auto sbi = vn.GenValueItemFromValueNumber(sbs_vns[index]);
          if (sbe::ceq(sbi, sbe::nu(0))) {
            Error1(sbs->LOC(), "subspan dimension " + std::to_string(index) +
                                   " can not be zero.");
            return false; // can not continue
          } else if (sbe::clt(shi, sbi))
            Error1(sbs->LOC(), "subspan too large for dimension " +
                                   std::to_string(index) + " (" + STR(sbi) +
                                   " > " + PSTR(shi) + ").");
          tfs_vns.push_back(vn.MakeOpNum("/", cur_vns[index], sbs_vns[index]));
        }
      }

      if (idx) {
        if (cur_vns.size() != idx_vns.size()) {
          Error1(op->LOC(), "data rank (" + std::to_string(cur_vns.size()) +
                                ") must be consistent with index count (" +
                                std::to_string(idx_vns.size()) + ").");
          return false;
        }
        for (size_t index = 0; index < cur_vns.size(); ++index) {
          auto tfi = vn.GenValueItemFromValueNumber(tfs_vns[index]);
          auto idi = vn.GenValueItemFromValueNumber(idx_vns[index]);
          if (sbe::clt(tfi, idi))
            Error1(idx->LOC(), "index out of bounds for dimension " +
                                   std::to_string(index) + " (" + STR(idi) +
                                   " >= " + PSTR(tfi) + ").");
          // TODO: index upper-bound check
        }
      }

      if (off) {
        if (cur_vns.size() != off_vns.size()) {
          Error1(op->LOC(),
                 "data rank (" + std::to_string(cur_vns.size()) +
                     ") must be consistent with offset index count (" +
                     std::to_string(off_vns.size()) + ").");
          return false;
        }
        for (size_t index = 0; index < cur_vns.size(); ++index) {
          auto shi = vn.GenValueItemFromValueNumber(cur_vns[index]);
          auto ofi = vn.GenValueItemFromValueNumber(off_vns[index]);
          if (sbe::clt(shi, ofi))
            Error1(off->LOC(), "offset out of bounds for dimension " +
                                   std::to_string(index) + " (" + STR(ofi) +
                                   " >= " + PSTR(shi) + ").");
        }
      }

      if (stp) {
        if (stp_vns.size() != cur_strd.size()) {
          Error1(strd->LOC(), "stepping value count (" +
                                  std::to_string(strd_vns.size()) +
                                  ") must be consistent with data (" +
                                  std::to_string(cur_strd.size()) + ").");
          return false;
        }

        for (size_t index = 0; index < stp_vns.size(); ++index) {
          auto sti = vn.GenValueItemFromValueNumber(stp_vns[index]);
          if (sbe::ceq(sti, sbe::nu(0)))
            Note(stp->LOC(), "zero step may be unexpected unless use it "
                             "intentionally for repeated data access.");
          cur_strd[index] = cur_strd[index] * sti;
        }
      }

      if (strd) {
        if (strd_vns.size() != cur_strd.size()) {
          Error1(strd->LOC(), "stride value count (" +
                                  std::to_string(strd_vns.size()) +
                                  ") must be consistent with data (" +
                                  std::to_string(cur_strd.size()) + ").");
          return false;
        }

        for (size_t index = 0; index < strd_vns.size(); ++index) {
          auto sti = vn.GenValueItemFromValueNumber(strd_vns[index]);
          if (sbe::ceq(sti, sbe::nu(0)))
            Note(strd->LOC(), "zero stride may be unexpected unless use it "
                              "intentionally for repeated data access.");
          cur_strd[index] = sti;
        }
      }
      // update shape and value numbers
      if (auto mop = dyn_cast<AST::SOP::ModSpan>(op)) {
        std::vector<NumTy> mod_vns;
        for (size_t index = 0; index < cur_vns.size(); ++index) {
          auto shi = vn.GenValueItemFromValueNumber(cur_vns[index]);
          auto sbi = vn.GenValueItemFromValueNumber(sbs_vns[index]);
          if (STR(sbi) == "::__choreo_parent_dim__") {
            mod_vns.push_back(cur_vns[index]); // special handling
            sbs_vns[index] = cur_vns[index];
          } else if (sbe::ceq((shi % sbi), sbe::nu(0)))
            mod_vns.push_back(GetOrGenValNum(c_sn(1))); // avoid 0-dim
          else
            mod_vns.push_back(
                vn.MakeOpNum("%", cur_vns[index], sbs_vns[index]));
        }
        // calculate and append the offset when not specified
        if (!mop->GetIndices()) {
          auto mv = AST::Make<AST::MultiValues>(mop->LOC());
          ValueList ovl;
          for (size_t i = 0; i < tfs_vns.size(); ++i) {
            auto tfis = vn.GenValueListFromValueNumber(tfs_vns[i]);
            for (auto tfi : tfis) {
              auto v = AST::MakeIntExpr(mop->LOC(), -1);
              v->SetType(MakeIntegerType());
              v->Opts().SetVal(tfi);
              mv->Append(v);
              ovl.push_back(tfi);
            }
          }
          mop->SetIndexNodes(mv);
          mv->Opts().SetVals(ovl);
          VST_DEBUG(dbgs() << " |-<exprval> values of modspan offset: ["
                           << STR(mv->Opts().GetVals()) << "]\n");
        }
        cur_vns = mod_vns;
      } else {
        if (auto sop = dyn_cast<AST::SOP::SubSpan>(op)) {
          if (!sop->GetIndices()) {
            auto mv = AST::Make<AST::MultiValues>(sop->LOC());
            ValueList ovl;
            for (auto ssp : sop->SubSpanNodes()) {
              auto sse = cast<AST::Expr>(ssp);
              assert(sse->Opts().HasVals());
              for (auto vi : sse->Opts().GetVals()) {
                auto zero = AST::MakeIntExpr(sop->LOC(), 0);
                zero->SetType(MakeIntegerType());
                zero->Opts().SetVal(sbe::nu(0));
                mv->Append(zero);
                ovl.push_back(sbe::nu(0));
              }
            }
            sop->SetIndexNodes(mv);
            mv->Opts().SetVals(ovl);
            VST_DEBUG(dbgs() << " |-<exprval> values of subspan offset: ["
                             << STR(mv->Opts().GetVals()) << "]\n");
          }
        }
        cur_vns = sbs_vns;
      }
      auto block_shape = GenShape(vn.MakePluralSign(sbs_vns));
      op->SetBlockShape(block_shape);
      op->SetBlockStrides(cur_strd);
      VST_DEBUG(dbgs() << " |-<sop: " << PSTR(op)
                       << "> block shape: " << STR(block_shape)
                       << ", strides: " << STR(cur_strd) << "\n");
    }
  }

  auto psn = vn.MakePluralSign(cur_vns);
  cur_vn = GetOrGenValNum(psn);
  auto vl = vn.GenValueListFromValueNumber(cur_vn);
  if (IsValidValueList(vl) && !IsComputable(vl))
    Error1(n.LOC(), "The destination block shape can not be evaluated.");

  auto res_shape = GenShape(psn);

  // set the chunkat's type
  SetNodeType(
      n, MakeSpannedType(sty->e_type, res_shape, cur_strd, sty->GetStorage()));
  n.SetBlockShape(res_shape); // TODO: this is redudant

  VST_DEBUG(dbgs() << " |-<output> shape: " << STR(res_shape) << "\n");

  assert(n.GetBlockShape().IsValid());

  ast_vn.Update(&n, cur_vn, VNKind::VNK_MDSPAN);

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

// the opts val of expr have been processed when visiting AST::Expr
#if 0
  // value the scalars
  for (auto& arg : n.GetArguments()) {
    if (!CanBeValueNumbered(arg.get())) continue;
    auto expr = cast<AST::Expr>(arg);
    if (auto sty = dyn_cast<ScalarIntegerType>(NodeType(*arg))) {
      if (!sty->IsMutable()) {
        expr->s = GenShape(GetSign(*arg));
        VST_DEBUG(dbgs() << " |-<exprshape> Shape for " << PSTR(arg) << ": "
                         << STR(expr->s) << "\n");
        assert(expr->s.DimCount() == 1);
        expr->Opts().SetVal(expr->s.ValueAt(0));
      } else {
        expr->Opts().SetVal(sbe::sym(GetSign(*arg)->ToString()));
      }
      VST_DEBUG(dbgs() << " |-<exprval> Value for " << PSTR(expr) << ": "
                       << STR(expr->Opts().GetVal()) << "\n");
    }
  }
#endif

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
    Error1(n.LOC(), "Failed to resolve future types.");
    return false;
  }

  // do not care about placeholders
  if (isa<PlaceHolderType>(rty)) return true;

  NumTy valno = GetOnlyValueNumberFromMultiValues(*n.ids);

  if (!valno.IsValid()) {
    Error1(n.LOC(), "failed to find a valid value number inside ROTATE.");
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
      cur_mdspan_vn = GetOnlyValueNumber(*n.expr_list, VNKind::VNK_MDSPAN);
      // handle dataof expr (TODO: any better idea?)
      if (!cur_mdspan_vn.IsValid()) {
        Error1(n.LOC(), "Failed to decide the type of Select." + STR(n) +
                            ", type0: " + PSTR(s0ty));
        return false;
      }
      auto nty = NodeType(*n.expr_list->ValueAt(0))->Clone();
      SetNodeType(n, nty);
    }
    ast_vn.Copy(s0.get(), &n, VNKind::VNK_MDSPAN);
    cur_mdspan_vn = GetValNo(n, VNKind::VNK_MDSPAN);
    cur_vn.Invalidate(); // used for variable def
  } else if (GeneralFutureType(NodeType(n))) {
    cur_vn.Invalidate();

    auto fty = type_equals.ResolveEqualFutures(*n.expr_list, true);
    if (!fty) {
      Error1(n.LOC(), "Failed to resolve future types.");
      return false;
    }
    SetNodeType(n, fty);
    if (isa<PlaceHolderType>(fty)) return true;

    cur_mdspan_vn = GetOnlyValueNumberFromMultiValues(*n.expr_list);
    if (!cur_mdspan_vn.IsValid()) {
      Error1(n.LOC(),
             "no valid value number is found for a SELECT expression.");
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

  for (auto r : n.GetRanges()) {
    auto rng = cast<AST::LoopRange>(r);
    auto valno = GetValNo(*rng->IV(), VNKind::VNK_UBOUND);
    auto vl = vn.GenValueListFromValueNumber(valno);
    if (IsValidValueList(vl) && !IsComputable(vl))
      Error1(rng->IV()->LOC(), "The upper bound(s) of '" + rng->IVName() +
                                   "' can not be evaluated.");
  }

  // invalidate any current value generated
  InvalidateVisitorValNOs();

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);

  if (!isa<AST::Call>(n.GetPred()) && HasValNo(*n.GetPred())) {
    auto valno = GetValNo(*n.GetPred());
    auto vl = vn.GenValueListFromValueNumber(valno);
    if (IsValidValueList(vl) && !IsComputable(vl))
      Error1(n.GetPred()->LOC(),
             "The inthreads-condition can not be evaluated.");
  }

  // invalidate any current value generated
  InvalidateVisitorValNOs();

  if (cannot_proceed) return true;

  return true;
}

bool ShapeInference::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);

  if (!isa<AST::Call>(n.GetPred()) && HasValNo(*n.GetPred())) {
    auto valno = GetValNo(*n.GetPred());
    auto vl = vn.GenValueListFromValueNumber(valno);
    if (IsValidValueList(vl) && !IsComputable(vl))
      Error1(n.GetPred()->LOC(), "The if-condition can not be evaluated.");
  }

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

const NumTy ShapeInference::GetOnlyValueNumber(const AST::MultiValues& mv,
                                               VNKind vnt) {
  NumTy valno = GetInvalidValueNumber();
  for (auto& v : mv.AllValues()) {
    auto num = GetValNo(*v, vnt);

    if (!valno.IsValid()) {
      valno = num;
      continue;
    } else if (valno != num) {
      Error(mv.LOC(), "value number does not match.");
      cannot_proceed = true;
      return GetInvalidValueNumber();
    }
  }
  assert(valno.IsValid());
  return valno;
}

const NumTy
ShapeInference::GetOnlyValueNumberFromMultiValues(const AST::MultiValues& mv) {
  NumTy valno = GetInvalidValueNumber();
  for (auto& v : mv.AllValues()) {
    auto id = AST::GetIdentifier(*v);
    if (!id) choreo_unreachable("expect an identifier.\n");
    auto ln = s_sn(SSTab().InScopeName(VNSymbolName(*id)));

    if (!valno.IsValid()) {
      valno = GetValNum(ln);
      continue;
    }

    // Check for consistence between different values
    if (vn.HasValidValueNumberOfSignature(ln)) {
      if (valno != GetValNum(ln)) {
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
                                                     const NumTy& valno) {
  for (auto& v : mv.AllValues()) {
    if (auto id = AST::GetIdentifier(*v)) {
      auto symbol = SSTab().InScopeName(VNSymbolName(*id));
      // the VN is considered to be identical if none exist
      if (!GetValNum(symbol).IsValid()) {
        SymbolRebindNum(symbol, valno);
        auto equals = type_equals.GetEquals(SSTab().InScopeName(id->name));
        for (auto& e : equals.value()) {
          auto asym = e + ".span";
          if (!vn.HasValidValueNumberOfSignature(s_sn(asym)))
            SymbolRebindNum(asym, valno);
          else
            assert(valno == GetValNum(asym));
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

  // if (isa<AST::ChunkAt>(n)) return false;
  if (isa<AST::StringLiteral>(n)) return false;
  if (isa<AST::DataAccess>(n)) return false;
  if (isa<AST::Call>(n)) return false;
  if (isa<AST::DataType>(n)) return false;
  auto nty = NodeType(*n);
  if (!nty) {
    // sometimes the symbol is yet to define, simply make it work.
    return true;
  }
  // mutable integers can now be valued
  // if (isa<ScalarIntegerType>(nty) && IsMutable(*nty)) return true;
  if (isa<EventType>(nty)) return false;
  if (isa<StringType>(nty)) return false;
  if (isa<VoidType>(nty)) return false;
  if (isa<StreamType>(nty)) return false;

  if (auto e = dyn_cast<AST::Expr>(n)) {
    if (e->op == "elemof") return false;
    if (e->op == "addrof") return false;
    if (e->op == "cast") return false;
    return CanBeValueNumbered(e->GetR().get()) &&
           CanBeValueNumbered(e->GetL().get()) &&
           CanBeValueNumbered(e->GetC().get());
  }
  return true; // could be id/int/...
}

void ShapeInference::DefineASymbol(const std::string& name,
                                   const ptr<Type>& ty) {
  // assert(!SSTab().IsDeclared(name) && "symbol has been declared.");
  if (isa<ArrayType>(ty))
    SSTab().DefineSymbol(name, ty);
  else
    SSTab().DefineSymbol(name, ty->Clone());
  VST_DEBUG(dbgs() << " |-<symtab> add: " << SSTab().InScopeName(name)
                   << ", type: " << PSTR(ty) << "\n");
}

const SignTy ShapeInference::SignSpan(const AST::Node& n) {
  if (auto* id = dyn_cast<AST::Identifier>(&n)) {
    // only cares about value inside the mdspan
    auto name = RemoveSuffix(id->name, ".span") + ".span";

    if (auto sname = SSTab().NameInScopeOrNull(name)) {
      auto nsn = s_sn(*sname);
      if (vn.HasValueNumberOfSignature(nsn))
        return nsn;
      else
        choreo_unreachable("symbol `" + *sname +
                           "' is not associated with a value number.");
    }
    // or else, it is a new name definition
    return s_sn(SSTab().ScopedName(name));
  } else if (auto e = dyn_cast<AST::Expr>(&n);
             e && (e->op == "dataof" || e->op == "mdataof")) {
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
    auto signature = o_sn(e->op);
    if (e->GetC()) signature->Append(GetSign(*e->GetC()));
    if (e->GetL()) {
      if (HasValNo(*e->GetL(), VNKind::VNK_MDSPAN))
        signature->Append(GetSign(*e->GetL(), VNKind::VNK_MDSPAN));
      else
        signature->Append(GetSign(*e->GetL()));
    }
    assert(e->GetR() && "expression is invalid.");
    if (HasValNo(*e->GetR(), VNKind::VNK_MDSPAN))
      signature->Append(GetSign(*e->GetR(), VNKind::VNK_MDSPAN));
    else
      signature->Append(GetSign(*e->GetR()));
    auto sign = vn.Simplify(signature);
    if (sign != signature) {
      VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Simplify> '" << STR(n) << ": '"
                       << STR(signature) << "' to '" << STR(sign) << "'\n");
    }
    return sign;
  } else
    choreo_unreachable("sign mdspan failed on " + STR(n) + ": " +
                       n.TypeNameString() + ".");
  return unk_sn();
}

const SignTy ShapeInference::SignNode(const AST::Node& n) {
  if (auto* id = dyn_cast<AST::Identifier>(&n)) {
    auto name = id->name;
    if (auto sname = SSTab().NameInScopeOrNull(name)) {
      auto nsn = s_sn(*sname);
      if (vn.HasValueNumberOfSignature(nsn))
        return nsn;
      else
        choreo_unreachable("symbol `" + *sname +
                           "' is not associated with a value number.");
    }
    // or else, it is a new name definition
    return s_sn(SSTab().ScopedName(name));
  } else if (auto* e = dyn_cast<AST::Expr>(&n)) {
    if (e->op == "sizeof") {
      auto esign = GetSign(*e->GetR(), NodeValNoKind(*e->GetR()));
      auto vl = vn.GenValueListFromSignature(esign);
      auto sz = MultiplyAll(vl);
      return vn.ValueItemToSignature(sz);
    } else if (e->op == "getith") {
      auto ii = cast<AST::IntIndex>(e->GetR());
      // negative number must add the ubound
      if (ii->IsNegative()) {
        auto signature = o_sn("+");
        signature->Append(GetSign(*e->GetL(), VNKind::VNK_UBOUND));
        signature->Append(GetSign(*ii->Val()));
        auto sign = vn.Simplify(signature);
        if (sign != signature) {
          VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Simplify> '" << STR(n)
                           << ": '" << STR(signature) << "' to '" << STR(sign)
                           << "'\n");
        }
        return sign;
      } else
        return GetSign(*ii->Val());
    } else if (e->op == "#") {
      auto sign0 = o_sn("*", GetSign(*e->GetL()),
                        GetSign(*e->GetR(), VNKind::VNK_UBOUND));
      auto s0 = vn.Simplify(sign0);
      if (s0 != sign0) {
        VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Simplify> '" << STR(n)
                         << ": '" << STR(sign0) << "' to '" << STR(s0)
                         << "'\n");
      }
      GetOrGenValNum(s0); // force to generate the valno
      auto signature = o_sn("+");
      signature->Append(s0);
      signature->Append(GetSign(*e->GetR()));
      auto sign = vn.Simplify(signature);
      if (sign != signature) {
        VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Simplify> '" << STR(n)
                         << ": '" << STR(signature) << "' to '" << STR(sign)
                         << "'\n");
      }
      return sign;
    } else if (e->op == "#+" || e->op == "#-") {
      // bound is mutated without value change
      return GetSign(*e->GetL());
    }
    if (e->IsReference()) return GetSign(n);
    if (e->op == "ubound") return GetSign(*e->GetR(), VNKind::VNK_UBOUND);
    auto signature = o_sn(e->op);
    if (e->GetC()) signature->Append(GetSign(*e->GetC()));
    if (e->GetL()) signature->Append(GetSign(*e->GetL()));
    assert(e->GetR() && "expression is invalid.");
    signature->Append(GetSign(*e->GetR()));
    auto sign = vn.Simplify(signature);
    if (sign != signature) {
      VST_DEBUG(dbgs() << vn.ScopeIndent() << "<Simplify> '" << STR(n) << ": '"
                       << STR(signature) << "' to '" << STR(sign) << "'\n");
    }
    return sign;
  } else if (auto* ii = dyn_cast<AST::IntIndex>(&n)) {
    return GetSign(*ii->value);
  }

  if (debug_visit)
    Warning(n.LOC(), "invalid signature for expression `" + AST::STR(n) +
                         "': " + n.TypeNameString() + ".");

  return unk_sn(); // invalid value
}

std::pair<const SignTy, const SignTy>
ShapeInference::SignBounded(const AST::Node& n) {
  assert(!ast_vn.Hit(&n, VNKind::VNK_UBOUND));

  SignTy v_sign = SignNode(n);
  SignTy ub_sign;
  if (auto id = dyn_cast<AST::Identifier>(&n)) {
    if (SSTab().NameInScopeOrNull("@" + id->name))
      ub_sign = s_sn(SSTab().InScopeName("@" + id->name));
    else {
      // first time encounter
      ub_sign = s_sn(SSTab().ScopedName("@" + id->name));
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
          auto lsn = GetSign(lhs, VNKind::VNK_UBOUND);
          auto rsn = GetSign(rhs, VNKind::VNK_UBOUND);
          ub_sign = vn.Simplify(o_sn("*", lsn, rsn));
        } else
          choreo_unreachable("operation is not permitted.");
      } else if (e->op == "#+" || e->op == "#-") {
        if (IsActualBoundedIntegerType(lhs.GetType()) &&
            isa<ScalarIntegerType>(rhs.GetType())) {
          auto lsn = GetSign(lhs, VNKind::VNK_UBOUND);
          auto rsn = GetSign(rhs, VNKind::VNK_VALUE);
          ub_sign = vn.Simplify(o_sn(e->op.substr(1), lsn, rsn));
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
const NumTy ShapeInference::GetValNo(const AST::Node& n, VNKind vnt) const {
  if (vnt == VNKind::VNK_UBOUND) assert(isa<BoundedType>(NodeType(n)));
  return ast_vn.Get(&n, vnt);
}

const NumTy ShapeInference::GenValNo(const AST::Node& n) {
  // if (auto v = GetOrNull(n, NodeValNoKind(n))) return *v;

  auto Generate = [this, &n](const SignTy& nsign, VNKind vnt) {
    assert(IsValid(nsign));
    // only generate values at the first time
    if (ast_vn.Hit(&n, vnt))
      choreo_unreachable("The '" + STR(vnt) +
                         "' valno has already been generated for " + STR(n) +
                         ".");

    // Different expression could have the same signature, which implies
    // duplicated computation that requires optimization.
    NumTy val_no = GetOrGenValNum(nsign);

    ast_vn.Update(&n, val_no, vnt);

    return val_no;
  };

  // directly get those with multiple values (already generated)
  if (auto* il = dyn_cast<AST::IntLiteral>(&n)) {
    if (HasValNo(n)) return GetValNo(n);
    SignTy ilsign;
    if (IsUnKnownInteger(il->Val()))
      ilsign = unk_sn();
    else
      ilsign = c_sn(il->Val());
    NumTy valno = GetOrGenValNum(ilsign);
    ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    return valno;
  } else if (auto* fl = dyn_cast<AST::FloatLiteral>(&n)) {
    if (HasValNo(n)) return GetValNo(n);
    SignTy flsign;
    if (fl->IsFloat32())
      flsign = c_sn(fl->Val_f32());
    else if (fl->IsFloat64())
      flsign = c_sn(fl->Val_f64());
    else
      choreo_unreachable("unexpected float point type.");
    NumTy valno = GetOrGenValNum(flsign);
    ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    return valno;
  } else if (auto* bl = dyn_cast<AST::BoolLiteral>(&n)) {
    if (HasValNo(n)) return GetValNo(n);
    NumTy valno = GetOrGenValNum(c_sn(bl->Val()));
    ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    return valno;
  } else if (auto* it = dyn_cast<AST::IntTuple>(&n)) {
    NumTy valno = GetValNo(*(it->GetValues()));
    ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
    return valno;
  } else if (auto mds = dyn_cast<AST::MultiDimSpans>(&n)) {
    NumTy valno = GetValNo(*(mds->list));
    ast_vn.Update(&n, valno, VNKind::VNK_VALUE);
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
      return valno;
    } else if (e->op == "dimof") {
      auto cv = CSign(GetSign(*e->GetR()));
      assert(cv && "expect a const signature.");
      auto index = cv->GetInt();
      if (ast_vn.Hit(e->GetL().get(), VNKind::VNK_MDSPAN)) {
        assert(!ast_vn.Hit(e, VNKind::VNK_VALUE));

        SignTy msign = GetSign(*e->GetL(), VNKind::VNK_MDSPAN);

        if ((size_t)index >= msign->Count())
          Error1(n.LOC(), "out of bound in 'dimof'.");

        NumTy valno = GetValNum(vn.ToMSign(msign)->At(index));
        ast_vn.Update(e, valno, VNKind::VNK_VALUE);
        return valno;
      } else if (ast_vn.Hit(e->GetL().get(), VNKind::VNK_VALUE)) {
        assert(!ast_vn.Hit(e, VNKind::VNK_VALUE));
        SignTy msign = GetSign(*e->GetL(), VNKind::VNK_VALUE);

        if ((size_t)index >= msign->Count())
          Error1(n.LOC(), "out of bound in 'dimof'.");

        NumTy valno = GetValNum(vn.ToMSign(msign)->At(index));
        ast_vn.Update(e, valno, VNKind::VNK_VALUE);
        return valno;
      } else if (ast_vn.Hit(e->GetL().get(), VNKind::VNK_UBOUND)) {
        SignTy msign = GetSign(*e->GetL(), VNKind::VNK_VALUE);

        if ((size_t)index >= msign->Count())
          Error1(n.LOC(), "out of bound in 'dimof'.");

        NumTy valno = GetValNum(vn.ToMSign(msign)->At(index));
        ast_vn.Update(e, valno, VNKind::VNK_VALUE);

        SignTy usign = GetSign(*e->GetL(), VNKind::VNK_UBOUND);
        NumTy uvalno = GetValNum(vn.ToMSign(usign)->At(index));
        ast_vn.Update(e, uvalno, VNKind::VNK_UBOUND);
        return uvalno;
      } else
        choreo_unreachable("unsupported dimof valno generation.");
    } else if (e->op == "getith") {
      NumTy vvn = Generate(SignNode(n), VNKind::VNK_VALUE);
      // the upper bound is not changed
      NumTy uvn = GetValNo(*e->GetL(), VNKind::VNK_UBOUND);
      ast_vn.Update(e, vvn, VNKind::VNK_VALUE);
      ast_vn.Update(e, uvn, VNKind::VNK_UBOUND);
      return uvn;
    } else if (e->op == "?") {
      if (auto cond = CSign(GetSign(*e->GetC()))) {
        if (cond->GetBool() == true) {
          ast_vn.Copy(e->GetL().get(), e);
          return ast_vn.Get(e, NodeValNoKind(n));
        } else if (cond->GetBool() == false) {
          ast_vn.Copy(e->GetR().get(), e);
          return ast_vn.Get(e, NodeValNoKind(n));
        }
      }
    } else if (e->op == "cast") {
      assert(false);
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
