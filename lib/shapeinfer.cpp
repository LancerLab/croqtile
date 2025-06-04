#include "shapeinfer.hpp"

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
  } else if (isa<AST::ChoreoFunction>(&n)) {
    vn.EnterScope();
    cannot_proceed = false; // recover state when starting a new function
    int valno = vn.GetOrInsertValueNumberFromSignature("const_1");
    vn.AssociateSignatureWithValueNumber(InScopeName("@__choreo_no_tiling__"),
                                         valno);
    InvalidateVisitorValNOs();
  } else if (isa<AST::ParallelBy>(&n)) {
    vn.EnterScope();
  } else if (isa<AST::WithBlock>(&n) || isa<AST::InThreadsBlock>(&n) ||
             isa<AST::IfElseBlock>(&n)) {
    vn.EnterScope();
  } else if (isa<AST::ForeachBlock>(&n) || isa<AST::IncrementBlock>(&n)) {
    vn.EnterScope();
    gen_values = false; // disable valno on range expressions
  } else if (auto* b = dyn_cast<AST::MultiDimSpans>(&n)) {
    if (b->ref_name != "") {
      auto n = SSTab().NameInScopeOrNull(b->ref_name);
      if (!n)
        choreo_unreachable("variable `" + b->ref_name +
                           "' is not found in scopes.");
      vn.SetListReference(n.value());
    }
  } else if (auto* b = dyn_cast<AST::IntTuple>(&n)) {
    if (b->ref_name != "") {
      auto n = SSTab().NameInScopeOrNull(b->ref_name);
      if (!n)
        choreo_unreachable(
            ("variable `" + b->ref_name + "' is not found in scopes.").c_str());
      vn.SetListReference(n.value());
    }
  } else if (isa<AST::Wait>(&n) || isa<AST::Call>(&n) || isa<AST::Rotate>(&n) ||
             isa<AST::Select>(&n) || isa<AST::Trigger>(&n) ||
             isa<AST::DataAccess>(&n)) {
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
    vn.EnterScope();
  }
  return true;
}

bool ShapeInference::AfterVisitImpl(AST::Node& n) {
  TraceEachVisit(n, false, "after ");
  if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
      isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n)) {
    vn.LeaveScope();
  } else if (isa<AST::ForeachBlock>(&n) || isa<AST::InThreadsBlock>(&n) ||
             isa<AST::IfElseBlock>(&n) || isa<AST::IncrementBlock>(&n)) {
    vn.LeaveScope();
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

std::vector<int> ShapeInference::Collapse(const ptr<AST::MultiValues>& mv,
                                          bool handle_getith) {
  if (!mv) choreo_unreachable("expect a valid multivalues.");
  std::vector<int> mvn;
  for (auto v : mv->AllValues()) {
    int valno = GetInvalidValueNumber();
    if (handle_getith) {
      auto e = dyn_cast<AST::Expr>(v);
      if (e && (e->op == "getith")) v = cast<AST::Expr>(e->GetL())->GetSymbol();
    }
    if (vn.HasValueNumberForNode(*v)) {
      valno = vn.GetValueNumberForNode(*v);
    } else
      valno = vn.GenerateValueNumberForNode(*v);

    std::deque<int> work_list;
    work_list.push_back(valno);
    while (!work_list.empty()) {
      auto val_no = work_list.front();
      work_list.pop_front();

      auto valsign = vn.GetSignatureFromValueNumber(val_no);
      auto vn_count = CountElementsInSignature(valsign);

      assert(vn_count >= 1);

      if (vn_count == 1) {
        mvn.push_back(val_no);
        continue;
      }

      for (int i = vn_count - 1; i >= 0; --i)
        work_list.push_front(vn.GetNthValNo(valsign, i));
    }
  }
  assert(mvn.size() > 0);
  return mvn;
}
bool ShapeInference::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;
  return true;
}

bool ShapeInference::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);

  if (cannot_proceed || !CanBeValueNumbered(&n)) {
    InvalidateVN(cur_vn);
    return true;
  }

  if (gen_values) {
    int valNo = vn.GenerateValueNumberForNode(n);
    cur_vn = valNo;
    cur_mdspan_vn = cur_vn;
  } else
    InvalidateVN(cur_vn);

  return true;
}

bool ShapeInference::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;
  int valNo = vn.GenerateValueNumberForNode(n);
  cur_vn = valNo;
  return true;
}

bool ShapeInference::Visit(AST::FloatLiteral& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;
  int valNo = vn.GenerateValueNumberForNode(n);
  cur_vn = valNo;
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
  int valNo = vn.GenerateValueNumberForNode(n);
  cur_vn = valNo;
  return true;
}

bool ShapeInference::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  if (cannot_proceed) return true;

  if (!CanBeValueNumbered(&n)) {
    InvalidateVN(cur_vn);
    return true;
  }

  if (auto id = n.GetSymbol()) {
    auto name = vn.VNSymbolName(*id);
    if (SSTab().IsDeclared(name)) {
      if (vn.HasValueNumberOfSignature(SSTab().InScopeName(name))) {
        cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(name));
        auto nty = NodeType(n);
        if (isa<MDSpanType>(SSTab().LookupSymbol(name))) {
          n.s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
          cur_mdspan_vn = cur_vn;
          InvalidateVN(cur_vn);
        }
        if (ValidVN(cur_vn)) {
          n.s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
          if (ConvertibleToInt(NodeType(n))) {
            assert(n.s.DimCount() == 1);
            if (!n.s.IsDynamic()) {
              n.Opts().SetVal(n.s.ValueAt(0));
              VST_DEBUG(dbgs() << "[ExprVal] " << STR(n) << ": "
                               << STR(n.s.ValueAt(0)) << "\n");
            }
          }
        }
      } else {
        // no value number is obtained
        InvalidateVN(cur_vn);
      }

      if (isa<ITupleType>(SSTab().LookupSymbol(name))) {
        n.Opts().SetVals(SymVal(SSTab().ScopedName(name)).GetVals());
        VST_DEBUG(dbgs() << "[ExprVal] " << STR(n) << ": "
                         << STR(n.Opts().GetVals()) << "\n");
      }

      return true;
    }
  } else if (n.op == "dataof") {
    // fill the mdspan type of this node
    auto id = cast<AST::Expr>(n.GetR())->GetSymbol();
    assert(!SuffixedWith(id->name, ".span"));
    auto name = id->name + ".span";
    assert(SSTab().IsDeclared(name));
    cur_mdspan_vn = vn.GetValueNumberOfSignature(InScopeName(name));
    assert(ValidVN(cur_mdspan_vn));
    n.s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_mdspan_vn));
    VST_DEBUG(dbgs() << "[ExprShape] Shape for " << STR(n) << ": " << STR(n.s)
                     << "\n");
    InvalidateVN(cur_vn); // a spanned data does not have a value number
    return true;
  }

  // the expression could be mdspan/ituple. record the information for later
  // type inference
  cur_vn = vn.GenerateValueNumberForNode(n);
  auto cur_sign = vn.GetSignatureFromValueNumber(cur_vn);
  n.s = GenShapeFromSignature(cur_sign);

  if (n.IsUBArith()) {
    SetNodeType(n, MakeBoundedITupleType(n.s));
    n.Opts().SetUBounds(vn.GenValueListFromSignature(cur_sign));
  }
  if (IsActualBoundedIntegerType(NodeType(n))) {
    cur_ub_vn = cur_vn;
    InvalidateVN(cur_vn);
  }

  if (ConvertibleToInt(NodeType(n))) {
    assert(n.s.DimCount() == 1);
    if (!n.s.IsDynamic()) {
      n.Opts().SetVal(n.s.ValueAt(0));
      VST_DEBUG(dbgs() << "[ExprVal] " << STR(n) << ": " << STR(n.s.ValueAt(0))
                       << "\n");
    }
  } else if (isa<ITupleType>(NodeType(n))) {
    n.Opts().SetVals(vn.GenValueListFromSignature(cur_sign));
    VST_DEBUG(dbgs() << "[ExprVal] " << STR(n) << ": "
                     << STR(n.Opts().GetVals()) << "\n");
  }

  if (AST::istypeof<MDSpanType>(&n)) {
    cur_mdspan_vn = cur_vn;
    auto vn_sig = vn.GetSignatureFromValueNumber(cur_mdspan_vn);
    cast<MDSpanType>(n.GetType())->SetShape(GenShapeFromSignature(vn_sig));
    if (CountElementsInSignature(vn_sig) > 1) {
      // set alias expressions with proper value numbers
      ProcessValueNumberString(
          vn_sig, [this, &vn_sig](int valno, size_t index) {
            if (UnknownVN(valno)) return; // do not associate it with vn of "?"
            vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                   std::to_string(index));
            vn.AssociateSignatureWithValueNumber(
                vn_sig + "(" + std::to_string(index) + ")", valno);
          });
    }
    //      InvalidateVN(cur_vn);
  } else if ((n.op == "sizeof") && n.s.IsValid()) {
    n.Opts().SetSize(n.s.ElementCountValue());
  } else if (n.op == "#") {
    if (IsActualBoundedIntegerType(n.GetL()->GetType()) &&
        IsActualBoundedIntegerType(n.GetR()->GetType())) {
      assert(n.s.DimCount() == 1);
      SetNodeType(n, MakeBoundedIntegerType(n.s.ValueAt(0)));
    }
  } else if (AST::istypeof<ITupleType>(&n)) {
    auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

    if (CountElementsInSignature(vn_sig) > 1) {
      // set alias expressions with proper value numbers
      ProcessValueNumberString(
          vn_sig, [this, &vn_sig](int valno, size_t index) {
            if (UnknownVN(valno)) return; // do not associate it with vn of "?"
            vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                   std::to_string(index));
            vn.AssociateSignatureWithValueNumber(
                vn_sig + "(" + std::to_string(index) + ")", valno);
          });
    }
  }

  return true;
}

bool ShapeInference::Visit(AST::MultiDimSpans& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (n.list) {
    // The Shape now can be deduced from the value number.
    // Update the type detail accordingly.
    auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

    // set alias expressions with proper value numbers
    if (CountElementsInSignature(vn_sig) > 1) {
      ProcessValueNumberString(
          vn_sig, [this, &vn_sig](int valno, size_t index) {
            if (UnknownVN(valno)) return; // do not associate it with vn of "?"
            vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                                   std::to_string(index));
            auto elem_sig = vn_sig + "(" + std::to_string(index) + ")";
            if (!vn.HasValueNumberOfSignature(elem_sig))
              vn.AssociateSignatureWithValueNumber(
                  vn_sig + "(" + std::to_string(index) + ")", valno);
          });
    }

    auto vl = GenShapeFromSignature(vn_sig);
    SetMdsShape(n, vl);

    if (IsValidRank(n.Rank())) {
#if 0
      if (vl.Dims() != n.Rank())
        Error(n.LOC(),
              "mdspan's dimension is inconsistent with its initialization "
              "expression: " +
                  std::to_string(vl.Dims()) + " vs. " +
                  std::to_string(n.Rank()) + ".");
#endif
    } else
      n.SetRank(vl.Rank());

    // pass the list value number over
    cur_mdspan_vn = cur_vn;
  } else if (n.Rank() > 0) {
    std::string unknown_spans = "#" + std::to_string(UnknownValue());
    for (size_t i = 1; i < n.Rank(); ++i)
      unknown_spans = unknown_spans + ",#" + std::to_string(UnknownValue());
    cur_mdspan_vn = vn.GetOrInsertValueNumberFromSignature(unknown_spans);
    SetMdsShape(n, GenShapeFromSignature(unknown_spans));
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

    vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name),
                                         cur_mdspan_vn);

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
      assert(ValidVN(cur_mdspan_vn) && "expecting a valid mdspan valno.");
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name + ".span"),
                                           cur_mdspan_vn);
    } else if (IsActualBoundedIntegerType(nty)) {
      assert(ValidVN(cur_ub_vn));
      DefineASymbol("@" + name, MakeBoundedIntegerType(cur_ub_vn));
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName("@" + name),
                                           cur_ub_vn);
      Shape s =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_ub_vn));
      nty = MakeBoundedITupleType(s);
      vn.AssociateSignatureWithValueNumber(name, cur_ub_vn);

      InvalidateVN(cur_ub_vn);
    } else {
      if (!isa<PlaceHolderType>(nty)) {
        assert(ValidVN(cur_vn) &&
               "cur_mdspan_vn and cur_vn must be exclusive.");
        vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_vn);
      }
    }
  } else {
    // obtain the types from declaration
    if (ValidVN(cur_mdspan_vn)) {
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name + ".span"),
                                           cur_mdspan_vn);
      auto mds_value =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_mdspan_vn));
      if (n.IsArray())
        nty = MakeSpannedArrayType(n.type->base_type, mds_value,
                                   n.ArrayDimensions(), sto);
      else
        nty = MakeSpannedType(n.type->base_type, mds_value, sto);
    } else if (ValidVN(cur_vn)) {
      vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_vn);
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
    VST_DEBUG(dbgs() << "[SymVal] " << InScopeName(name) << ": "
                     << STR(shape.ValueAt(0)) << "\n");
    SymVal(InScopeName(name)).SetVal(shape.ValueAt(0));
  } else if (isa<ITupleType>(nty)) {
    SymVal(InScopeName(name)).SetVals(vn.GenValueListFromValueNumber(cur_vn));
    VST_DEBUG(dbgs() << "[SymVal] " << InScopeName(name) << ": "
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

  auto mvals = n.GetValues();
  // cur_ituple_vn = cur_vn;
  SetNodeType(n, MakeITupleType(mvals->Count()));

  auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);

  if (CountElementsInSignature(vn_sig) > 1) {
    // set alias expressions with proper value numbers
    ProcessValueNumberString(vn_sig, [this, &vn_sig](int valno, size_t index) {
      if (UnknownVN(valno)) return; // do not associate it with vn of "?"
      vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                             std::to_string(index));
      vn.AssociateSignatureWithValueNumber(
          vn_sig + "(" + std::to_string(index) + ")", valno);
    });
  }
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
    assert(ValidVN(cur_mdspan_vn) && "expected a valid current value number.");
    vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name),
                                         cur_mdspan_vn);
    return true;
  }

  if (IsActualBoundedIntegerType(nty)) {
    name = "@" + name;
    assert(ValidVN(cur_ub_vn));
    DefineASymbol(name, MakeBoundedIntegerType(cur_ub_vn));
    vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_ub_vn);
    InvalidateVN(cur_ub_vn);
  } else {
    assert(ValidVN(cur_vn) && "expected a valid current value number.");
    vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(name), cur_vn);
  }

  if (isa<IntegerType>(nty) && ValidVN(cur_vn)) {
    auto shape = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
    assert(shape.DimCount() == 1);
    VST_DEBUG(dbgs() << "[SymVal] " << SSTab().ScopedName(name) << ": "
                     << STR(shape.ValueAt(0)) << "\n");
    SymVal(SSTab().ScopedName(name)).SetVal(shape.ValueAt(0));
  }

  return true;
}

bool ShapeInference::Visit(AST::IntIndex& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  cur_vn = vn.GenerateValueNumberForNode(n);

  return true;
}

bool ShapeInference::Visit(AST::DataType& n) {
  TraceEachVisit(n);

  allow_named_dim = false;

  if (cannot_proceed) return true;

  if (ValidVN(cur_mdspan_vn)) { cur_vn = cur_mdspan_vn; }

  return true;
}

bool ShapeInference::Visit(AST::Identifier& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (!gen_values) return false;
  if (SSTab().IsDeclared(n.name))
    if (!CanBeValueNumbered(&n)) { return false; }

  auto name = vn.VNSymbolName(n);
  if (SSTab().IsDeclared(name)) {
    // it is a reference
    if (!vn.HasValueNumberOfSignature(SSTab().InScopeName(name)))
      choreo_unreachable("value number of `" + SSTab().InScopeName(name) +
                         "' has not been generated.");
    cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(name));
    return true;
  }

  if (allow_named_dim) { // for named dims in parameters
    if (!SSTab().DeclaredInScope(n.name)) {
      DefineASymbol(n.name, MakeIntegerType());
      cur_vn = vn.GenerateValueNumberFromSignature(SSTab().InScopeName(n.name));
    } else {
      cur_vn = vn.GetValueNumberOfSignature(SSTab().InScopeName(n.name));
    }
    return true;
  }

  if (vn.HasValueNumberForNode(n)) {
    Error(n.LOC(), "value number has been generated for `" + n.name + "'.");
    error_count++;
    return false;
  }

  // sometime we need value a symbol (symbolic value)
  // TODO: improve it - only generate valno for integer types
  if (!ValidVN(cur_mdspan_vn)) cur_vn = vn.GenerateValueNumberForNode(n);

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
      assert(ValidVN(cur_mdspan_vn) && "unexpected value number for mdspan.");

      // Put alias names of mdspan into the value number table
      vn.AssociateSignatureWithValueNumber(
          SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
      SetNodeType(*n.type,
                  MakeSpannedType(n.type->base_type, span->GetTypeDetail()));

    } else if (IsValidRank(span->Rank())) {
      assert(ValidVN(cur_mdspan_vn) && "unexpected value number for mdspan.");
      // Put alias names of mdspan into the value number table
      vn.AssociateSignatureWithValueNumber(
          SSTab().ScopedName(n.sym->name + ".span"), cur_mdspan_vn);
      SetNodeType(*n.type,
                  MakeSpannedType(n.type->base_type, span->GetTypeDetail()));
    } else {
      // the value number is unknown at compile time
      Error(n.LOC(), "The type can not be inference at compile time.");
      error_count++;
      return false;
    }

    if (n.sym) {
      DefineASymbol(n.sym->name + ".span",
                    cast<SpannedType>(n.type->GetType())->GetMDSpanType());
      DefineASymbol(n.sym->name, n.type->GetType());
    }

    InvalidateVisitorValNOs();
    return true;
  }

  if (n.sym && n.type->isScalar()) {
    assert(!ValidVN(cur_mdspan_vn) && "unexpected current mdspan value.");

    // get the value number and make it defined
    vn.GetValueNumberOfSignature(SSTab().ScopedName(n.sym->name));
    if (n.sym) DefineASymbol(n.sym->name, n.GetType());

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

  Shape s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
  SetNodeType(n, MakeMDSpanType(s)); // useful for the sema check

  std::string iv_name = SSTab().ScopedName("@" + n.bpv->name);
  vn.AssociateSignatureWithValueNumber(iv_name, cur_vn);
  SetNodeType(*n.bpv, MakeBoundedITupleType(s, "pv"));
  DefineASymbol("@" + n.bpv->name, MakeMDSpanType(s));
  DefineASymbol(n.bpv->name, n.bpv->GetType());

  std::map<size_t, std::string> idx2dim;
  idx2dim[0] = "x";
  idx2dim[1] = "y";
  idx2dim[2] = "z";
  for (size_t i = 0; i < n.SubCount(); ++i) {
    const auto& [sym, b] = n.GetIV(i);
    std::string bound;
    if (auto il = dyn_cast<AST::IntLiteral>(b))
      bound = "const_" + std::to_string(il->Val());
    else if (auto id = dyn_cast<AST::Identifier>(b)) {
      if (auto name_in_scope = SSTab().NameInScopeOrNull(id->name)) {
        if (vn.HasValueNumberOfSignature(*name_in_scope))
          bound = *name_in_scope;
        else
          choreo_unreachable("symbol `" + *name_in_scope +
                             "' is not associated with a value number.");
      } else {
        choreo_unreachable("expect symbol `" + *name_in_scope +
                           "' to be defined in symtab!");
      }
    } else
      choreo_unreachable("unexpected type of parallelby bound item");
    int valno = vn.GetOrInsertValueNumberFromSignature(bound);
    std::string iv_name = SSTab().ScopedName("@" + sym->name);
    vn.AssociateSignatureWithValueNumber(iv_name, valno);
    Shape s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(valno));
    SetNodeType(*sym, MakeBoundedITupleType(s, "pi:" + idx2dim[i]));
    DefineASymbol("@" + sym->name, MakeMDSpanType(s));
    DefineASymbol(sym->name, sym->GetType());
  }
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
  return true;
}

bool ShapeInference::Visit(AST::WithIn& n) {
  TraceEachVisit(n);

  if (cannot_proceed) return true;

  if (auto mds = dyn_cast<AST::MultiDimSpans>(n.in)) {
    assert(ValidVN(cur_mdspan_vn) &&
           "no valid value number generated for the mdspan.");
    if (n.with_matchers)
      if (n.with_matchers->Count() != mds->Rank()) {
        Error(n.LOC(), "inconsistent with-in values and cmpt_bounds.");
        error_count++;
        return false;
      }

  } else if (isa<AST::Expr>(n.in)) {
    if (auto id = AST::GetIdentifier(*n.in))
      cur_mdspan_vn =
          vn.GetValueNumberOfSignature(SSTab().InScopeName(id->name));
    else
      cur_mdspan_vn = cur_vn;
    assert(ValidVN(cur_mdspan_vn) && "no valid vn for with-in.");
    InvalidateVN(cur_vn);
  } else {
    choreo_unreachable("unexpected with-in statement.");
  }

  auto vn_sig = vn.GetSignatureFromValueNumber(cur_mdspan_vn);

  // requires the elements inside mdspan to be non-zero values
  bool found_zero = false;
  ProcessValueNumberString(vn_sig,
                           [this, &vn_sig, &n, &found_zero](int valno, size_t) {
                             auto sig = vn.GetSignatureFromValueNumber(valno);
                             if (sig == "const_0") { found_zero = true; }
                           });
  if (found_zero) {
    Error(n.LOC(),
          "zero value is deduced for the mdspan inside the with-in statement.");
    error_count++;
    cannot_proceed = true;
    Error(n.LOC(),
          "unable to apply shape inference for function '" + fname + "'.");
    return false;
  }

  auto GenSignatureAndDoValno = [this, &vn_sig, &n](int valno, size_t index) {
    if (UnknownVN(valno)) return; // do not associate it with vn of "?"
    if (n.with) {
      std::string name = SSTab().ScopedName("@" + n.with->name) + "(" +
                         std::to_string(index) + ")";
      vn.AssociateSignatureWithValueNumber(name, valno);
    }

    if (n.with_matchers) {
      auto sym = cast<AST::Identifier>((*n.with_matchers)[index]);
      std::string name = SSTab().ScopedName("@" + sym->name);
      vn.AssociateSignatureWithValueNumber(name, valno);
      Shape s = GenShapeFromSignature(vn.GetSignatureFromValueNumber(valno));
      SetNodeType(*sym, MakeBoundedITupleType(s));
      DefineASymbol("@" + sym->name, MakeMDSpanType(s));

      // because we use bounded integer var as identifier
      name = SSTab().ScopedName(sym->name);
      DefineASymbol(sym->name, sym->GetType());
      // vn.GetOrInsertValueNumberFromSignature(name);
      // TODO(wsj): deal with expression contains bounded integers
    }
  };
  if (CountElementsInSignature(vn_sig) ==
      1) // support `with idx={m} in [xx] {}`
    GenSignatureAndDoValno(vn.GetValueNumberOfSignature(vn_sig), 0);
  else
    ProcessValueNumberString(vn_sig, GenSignatureAndDoValno);

  if (n.with) {
    vn.AssociateSignatureWithValueNumber(SSTab().ScopedName("@" + n.with->name),
                                         cur_mdspan_vn);
    Shape s = GenShapeFromSignature(vn_sig);
    SetNodeType(*n.with, MakeBoundedITupleType(s));
    DefineASymbol("@" + n.with->name, MakeMDSpanType(s));
    DefineASymbol(n.with->name, n.with->GetType());
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
  assert(ValidVN(cur_vn) && "failed to get the list value.");

  auto pty = SSTab().LookupSymbol(n.id->name);
  assert((isa<SpannedType>(pty) || isa<FutureType>(pty)) &&
         "unexpected data type.");

  auto sty = GetSpannedType(pty);
  if (!sty) {
    Error(n.LOC(), "internal error: span_as operates on non-spanned type.");
    return false;
  }

  vn.AssociateSignatureWithValueNumber(
      SSTab().ScopedName(n.nid->name + ".span"), cur_vn);

  cur_mdspan_vn = cur_vn;

  auto shape = GenShapeFromSignature(vn.GetSignatureFromValueNumber(cur_vn));
  auto nty = MakeSpannedType(sty->ElementType(), shape, sty->GetStorage());

  SetNodeType(n, nty);

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
      return "#" + std::to_string(vn.GetOrInsertValueNumberFromSignature(cv));
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
    cur_vn = vn.GetOrInsertValueNumberFromSignature(add_sig);
  } else if (auto tcfg = dyn_cast<TransposeConfig>(n.config)) {
    // gen new vn if and only if n.to is AST::Memory
    if (isa<AST::Memory>(n.to)) {
      auto& dim_values = tcfg->dim_values;
      auto orig_sig = vn.GetSignatureFromValueNumber(cur_vn);
      auto shape_components = SplitStringByDelimiter(orig_sig);
      auto sig = shape_components[dim_values[0]];
      for (size_t i = 1; i < dim_values.size(); ++i)
        sig += "," + shape_components[dim_values[i]];
      cur_vn = vn.GetOrInsertValueNumberFromSignature(sig);
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
    vn.AssociateSignatureWithValueNumber(SSTab().ScopedName(f_span), cur_vn);
    DefineASymbol(n.future, n.GetType());
    DefineASymbol(f_span, MakeMDSpanType(s)); // implicit symbol
  }

  auto vn_sig = vn.GetSignatureFromValueNumber(cur_vn);
  // set alias expressions with proper value numbers
  if (CountElementsInSignature(vn_sig) > 1) {
    ProcessValueNumberString(vn_sig, [this, &vn_sig](int valno, size_t index) {
      if (UnknownVN(valno)) return; // do not associate it with vn of "?"
      vn.GetOrInsertValueNumberFromSignature("index_const_" +
                                             std::to_string(index));
      auto elem_sig = vn_sig + "(" + std::to_string(index) + ")";
      if (!vn.HasValueNumberOfSignature(elem_sig))
        vn.AssociateSignatureWithValueNumber(
            vn_sig + "(" + std::to_string(index) + ")", valno);
    });
  }

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

    // make sure all expressions get the value numbers
    tsi->Positions()->accept(*this);
    if (tsi->MultipleExprs()) tsi->GetTFSSExpr()->accept(*this);

    auto pos_vns = Collapse(tsi->Positions(), true);
    assert(cur_vns.size() == pos_vns.size());

    // when the code provides explicit tiling factors or subspan
    std::vector<int> tfs_vns;

    if (tsi->MultipleExprs()) {
      tfs_vns = Collapse(tsi->GetTFSSExpr());
      assert(tfs_vns.size() == pos_vns.size());
    }

    for (size_t index = 0; index < pos_vns.size(); ++index) {
      if (tsi->HasSubSpanExpr()) {
        // block.span = subspan
        auto lvi = vn.GenValueItemFromValueNumber(cur_vns[index]);
        auto rvi = vn.GenValueItemFromValueNumber(tfs_vns[index]);
        if (sbe::nu_lt(lvi, rvi)) {
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
        if (sbe::nu_lt(lvi, rvi)) {
          Error(tsi->LOC(),
                "the subspan dimension (dim: " + std::to_string(index) +
                    ") is larger than the data (" + STR(rvi) + " > " +
                    PSTR(lvi) + ").");
          error_count++;
        }
        auto mod_sig = SignatureOfBinOp("%", cur_vns[index], tfs_vns[index]);
        mod_vns.push_back(vn.GetOrInsertValueNumberFromSignature(mod_sig));
        res_vns.push_back(tfs_vns[index]);
      } else if (tsi->HasTilingExpr()) {
        // block.span = data.span / tiling_factor
        auto lvi = vn.GenValueItemFromValueNumber(cur_vns[index]);
        auto rvi = vn.GenValueItemFromValueNumber(tfs_vns[index]);
        if (sbe::nu_lt(lvi, rvi)) {
          Error(tsi->LOC(), "the tiling factor (dim: " + std::to_string(index) +
                                ") is larger than the data dimension (" +
                                STR(rvi) + " > " + PSTR(lvi) + ").");
          error_count++;
        }
        auto res_sig = SignatureOfBinOp("/", cur_vns[index], tfs_vns[index]);
        res_vns.push_back(vn.GetOrInsertValueNumberFromSignature(res_sig));
      } else {
        // block.span = data.span / #pos
        auto lvi = vn.GenValueItemFromValueNumber(cur_vns[index]);
        auto rvi = vn.GenValueItemFromValueNumber(pos_vns[index]);
        if (sbe::nu_lt(lvi, rvi)) {
          Error(tsi->LOC(), "the tiling factor (dim: " + std::to_string(index) +
                                ") is larger than the data dimension (" +
                                STR(rvi) + " > " + STR(lvi) + ").");
          error_count++;
        }
        auto res_sig = SignatureOfBinOp("/", cur_vns[index], pos_vns[index]);
        res_vns.push_back(vn.GetOrInsertValueNumberFromSignature(res_sig));
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
      ca_valno = vn.GetOrInsertValueNumberFromSignature(m_sign);
      auto b_valno = vn.GetOrInsertValueNumberFromSignature(b_sign);
      block_shape =
          GenShapeFromSignature(vn.GetSignatureFromValueNumber(b_valno));
    } else
      ca_valno = vn.GetOrInsertValueNumberFromSignature(b_sign);
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
      expr->s = GenShapeFromSignature(vn.GetSignatureForNode(*s));
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
    if (s0ty && s0ty->HasSufficientInfo())
      SetNodeType(n, s0ty);
    else {
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

    cur_mdspan_vn = vn.GenerateValueNumberForNode(n);
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
    auto ln = SSTab().InScopeName(vn.VNSymbolName(*id));

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
      auto symbol = SSTab().InScopeName(vn.VNSymbolName(*id));
      // the VN is considered to be identical if none exist
      if (!ValidVN(vn.GetValueNumberOfSignature(symbol))) {
        vn.RebindSignatureWithValueNumber(symbol, valno);
        auto equals = type_equals.GetEquals(SSTab().InScopeName(id->name));
        for (auto& e : equals.value().get()) {
          auto asym = e + ".span";
          if (!vn.HasValidValueNumberOfSignature(asym))
            vn.RebindSignatureWithValueNumber(asym, valno);
          else
            assert(valno == vn.GetValueNumberOfSignature(asym));
        }
      }
    } else
      choreo_unreachable("expect an identifier.");
  }
}

bool ShapeInference::CanBeValueNumbered(AST::Node* n) const {
  if (!n) return true;

  if (auto mv = dyn_cast<AST::MultiValues>(n)) {
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
    dbgs() << "[symtab] add: " << SSTab().InScopeName(name)
           << ", type: " << PSTR(ty) << "\n";
}

} // end namespace Choreo
