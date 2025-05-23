#include "typeinfer.hpp"

#include <iostream>

#include "ast.hpp"
#include "types.hpp"

using namespace Choreo;

bool TypeInference::BeforeBeforeVisit(AST::Node& n) {
  if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    // the type will be modified after parameters/return are processed
    AssignSymbolWithType(n.LOC(), f->name, MakeUnknownType());
  }
  return true;
}

bool TypeInference::BeforeVisitImpl(AST::Node& n) {
  Visitor::BeforeVisit(n);
  if (isa<AST::Program>(&n)) {
    type_equals.Reset();
  } else if (isa<AST::DMA>(&n)) {
    dma_fmty = BaseType::UNKNOWN;
    dma_mem = Storage::NONE;
  } else if (isa<AST::Parameter>(&n)) {
    allow_named_dim = true;
  }
  return true;
}

bool TypeInference::AfterVisitImpl(AST::Node& n) {
  if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    auto sym_ty = GetSymbolType(f->LOC(), f->name);
    assert(!isa<UnknownType>(sym_ty) && "symbol type is not deduced.");

    auto func_ty = cast<FunctionType>(sym_ty);
    if (AST::istypeof<UnknownType>(f) || isa<SpannedType>(func_ty->out_ty)) {
      // update the return type node since type inference could have changed the
      // function type already
      SetNodeType(*f->f_decl.ret_type, func_ty->out_ty);
      SetNodeType(f->f_decl, sym_ty);
      f->SetType(sym_ty);
    }
    if (CCtx().ShowInferredTypes()) {
      dbgs() << "Function:  " << SSTab().InScopeName(f->name)
             << ", Type: " << AST::TYPE_STR(*f) << "\n";
    }
  } else if (isa<AST::DMA>(&n)) {
    dma_fmty = BaseType::UNKNOWN;
    dma_mem = Storage::NONE;
  } else if (isa<AST::Parameter>(&n)) {
    allow_named_dim = false;
  }

  Visitor::AfterVisit(n);
  return true;
}

bool TypeInference::AssignSymbolWithType(const location& loc,
                                         const std::string& sym,
                                         const ptr<Type>& ty) {
  if (!SSTab().DefineSymbol(sym, ty)) {
    Error(loc, "symbol `" + sym + "' has already been associated with a type.");
    error_count++;
    return false;
  }

  VST_DEBUG(dbgs() << "Assign symbol `" << sym << "` with type: " << STR(*ty)
                   << "\n");

  return true;
}

ptr<Type> TypeInference::GetSymbolType(const location& loc,
                                       const std::string& name) {
  if (!SSTab().IsDeclared(name)) {
    Error(loc, "The symbol `" + name + "' has not been defined.");
    error_count++;
    return nullptr;
  }
  if (auto pty = SSTab().LookupSymbol(name)) {
    return pty;
  } else {
    Error(loc, "symbol `" + name + "' is not associated with a type.");
    error_count++;
    return nullptr;
  }
}

bool TypeInference::ModifySymbolType(const location& loc,
                                     const std::string& name,
                                     const ptr<Type>& ty) {
  if (!SSTab().IsDeclared(name)) {
    Error(loc, "The symbol `" + name + "' has not been defined.");
    error_count++;
    return false;
  }
  if (!SSTab().ModifySymbolType(name, ty)) {
    Error(loc, "symbol `" + name + "' is not associated with a type.");
    error_count++;
    return false;
  }

  VST_DEBUG(dbgs() << "Modify symbol `" << name << "` with type: " << STR(*ty)
                   << "\n");

  return true;
}

bool TypeInference::SetAsCurrentType(AST::Node& nd, const std::string& n) {
  const auto nty = nd.GetType();
  if (nty->HasSufficientInfo()) {
    // already has a type with sufficient info, check for consistence.
    if (cur_type->HasSufficientInfo()) {
      if (BetterQuality(cur_type, nty)) {
        SetNodeType(nd, ShadowTypeStorage(cur_type));
        return true;
      } else {
        assert(!BetterQuality(nty, cur_type) &&
               "the inference type should be better qualified.");
        if (!(cur_type->LogicalEqual(*nty))) {
          Error(nd.LOC(), "can not infer the type of `" + n + "'.");
          error_count++;
          return false;
        }
      }
    }
  }

  // Or else we need to set the type with current
  // Check for inference failures
  if (isa<UnknownType>(cur_type)) {
    Error(nd.LOC(), "can not infer the type of `" + n + "'.");
    error_count++;
    return false;
  }

  if (!cur_type->HasSufficientInfo()) {
    Error(nd.LOC(), "can not infer '" + cur_type->Name() +
                        "' type detail of symbol `" + n + "'.");
    error_count++;
    return false;
  }

  // complement the storage information when exists
  if (auto st = dyn_cast<SpannedType>(cur_type))
    if (auto n = dyn_cast<AST::NamedVariableDecl>(&nd))
      if (n->mem) st->SetStorage(n->mem->st);

  // The type is successfully inferred, set the node
  SetNodeType(nd, ShadowTypeStorage(cur_type));

  return true;
}

bool TypeInference::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeIntegerType());
  return true;
}

bool TypeInference::Visit(AST::FloatLiteral& n) {
  TraceEachVisit(n);
  if (std::holds_alternative<float>(n.value))
    SetNodeType(n, MakeFloatType());
  else if (std::holds_alternative<double>(n.value))
    SetNodeType(n, MakeDoubleType());
  else
    choreo_unreachable("unhandled floating-point type.");
  return true;
}

bool TypeInference::Visit(AST::StringLiteral& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeStringType());
  return true;
}

bool TypeInference::Visit(AST::Boolean& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeBooleanType());
  return true;
}

bool TypeInference::Visit(AST::DataType& n) {
  TraceEachVisit(n);

  allow_named_dim = false;

  if (n.getBaseType() == BaseType::UNKNOWN)
    return true; // ignore the annotation that needs inference

  assert((cur_type == nullptr) && "Expecting null type.");

  if (!n.mdspan_type) {
    cur_type = n.GetType(); // simple types
    return true;
  }

  // compound type
  if (auto mdspan = dyn_cast<AST::MultiDimSpans>(n.mdspan_type)) {
    auto shape = cast<MDSpanType>(mdspan->GetType())->GetShape();
    if (n.isArray())
      SetNodeType(n, MakeSpannedArrayType(n.base_type, shape, n.array_dims));
    else
      SetNodeType(n, MakeSpannedType(n.getFundamentalType(), shape));
    cur_type = n.GetType();
  }

  return true;
}

bool TypeInference::Visit(AST::Identifier& n) {
  TraceEachVisit(n);

  // for named dims in parameters
  if (allow_named_dim && !SSTab().DeclaredInScope(n.name))
    AssignSymbolWithType(n.LOC(), n.name, MakeIntegerType());

  return true;
}

bool TypeInference::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  // annotate it should not take storage
  if (n.init_expr && (isa<AST::Select>(AST::Ref(n.init_expr)) ||
                      isa<AST::SpanAs>(AST::Ref(n.init_expr))))
    n.SetNote("ref");

  if (cur_type && !SetAsCurrentType(n, n.name_str)) {
    cur_type.reset();
    return false;
  }

  cur_type.reset();

  if (AST::istypeof<UnknownType>(&n)) {
    Error(n.LOC(), "can not infer the type of `" + n.name_str + "'.");
    error_count++;
    return false;
  }

  auto nty = NodeType(n);
  AssignSymbolWithType(n.LOC(), n.name_str, nty);

  if (AST::istypeof<SpannedType>(&n)) {
    AssignSymbolWithType(n.LOC(), n.name_str + ".span",
                         cast<SpannedType>(nty)->GetMDSpanType());
  }

  if (AST::istypeof<FutureType>(&n)) {
    AssignSymbolWithType(n.LOC(), n.name_str + ".data",
                         cast<FutureType>(nty)->GetSpannedType());
    AssignSymbolWithType(
        n.LOC(), n.name_str + ".span",
        cast<FutureType>(nty)->GetSpannedType()->GetMDSpanType());
  }

  if (CCtx().ShowInferredTypes()) {
    dbgs() << ((AST::istypeof<FutureType>(&n)) ? "Future" : "Symbol");
    dbgs() << ":    " << InScopeName(n.name_str) << ", Type: " << PSTR(nty);
    dbgs() << "\n";
  }

  return true;
}

bool TypeInference::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);

  if (n.init_expr) {
    if (AST::istypeof<UnknownType>(n.init_expr)) {
      Error(n.LOC(), "unable to inference the type of `" + n.name_str + "'.");
      error_count++;
      return false;
    }

    if (!n.init_expr->GetType()->HasSufficientInfo()) {
      Error(n.LOC(),
            "unable to inference the type detail of `" + n.name_str + "'.");
      error_count++;
      return false;
    }

    SetNodeType(n, n.init_expr->GetType());
  } else if (AST::istypeof<UnknownType>(&n)) {
    // need type inference
    Error(n.LOC(),
          "`" + n.name_str +
              "' is declared without type annotation or initialization.");
    error_count++;
    return false;
  }

  AssignSymbolWithType(n.LOC(), n.name_str, n.GetType());

  if (CCtx().ShowInferredTypes()) {
    dbgs() << "Partial:   " << InScopeName(n.name_str)
           << ", Type: " << AST::TYPE_STR(n) << "\n";
  }
  return true;
}

bool TypeInference::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);

#if 0
  auto dty = GetSymbolType(n.LOC(), n.GetDataName());
  if (n.AccessElement()) {
    auto sty = cast<SpannedType>(dty);
    SetNodeType(n, MakeElemScalarType(sty->ElementType()));
  } else
    SetNodeType(n, dty);
#endif

  return true;
}

// ituple override operator "=" for definition
bool TypeInference::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  if (n.AssignToDataElement()) {
    // should be assigned already by DataAccess
    assert(isa<ScalarType>(NodeType(*n.da)));
    auto dty = GetSymbolType(n.LOC(), n.GetDataArrayName());
    auto ety = MakeElemScalarType(cast<SpannedType>(dty)->ElementType());
    SetNodeType(*n.da, ety);
    SetNodeType(n, ety);
    return true;
  }

  if (n.value && (isa<AST::Select>(n.value) || isa<AST::SpanAs>(n.value)))
    n.SetNote("ref");

  if (SSTab().IsDeclared(n.GetName())) {
    auto ety = NodeType(*n.value);
    if (isa<FutureType>(ety) || IsMutable(*ety)) {
      // no type inference is necessary
      SetNodeType(n, ety);
      SetNodeType(*n.da, ety);
      cur_type.reset();
      return true;
    } else if (auto vty = GetSymbolType(n.da->LOC(), n.GetName());
               IsMutable(*vty)) {
      SetNodeType(n, vty);
      SetNodeType(*n.da, vty);
      cur_type.reset();
      return true;
    } else {
      Error(n.LOC(),
            "current choreo does not support symbol re-assignment except for "
            "future/mutable type. Current type: " +
                PSTR(ety));
      error_count++;
      SetNodeType(n, MakeUnknownType());
      cur_type.reset();
      return false;
    }
  }

  assert(!n.da->AccessElement());

  if (isa<UnknownType>(NodeType(*n.value))) {
    Error(n.LOC(), "fail to deduce type of `" + n.GetName() + "'.");
    error_count++;
    cur_type.reset();
    return false;
  }

  auto ty = ShadowTypeStorage(NodeType(*n.value));

  AssignSymbolWithType(n.LOC(), n.GetName(), ty);
  SetNodeType(n, ty);
  SetNodeType(*n.da, ty);

  if (auto fty = dyn_cast<FutureType>(ty)) {
    AssignSymbolWithType(n.LOC(), n.GetName() + ".data", fty->GetSpannedType());
    AssignSymbolWithType(n.LOC(), n.GetName() + ".span",
                         fty->GetSpannedType()->GetMDSpanType());
  }

  if (CCtx().ShowInferredTypes()) {
    dbgs() << "Symbol:    " << InScopeName(n.GetName())
           << ", Type: " << PSTR(ty) << "\n";
  }

  cur_type.reset();
  return true;
}

bool TypeInference::Visit(AST::IntIndex& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeIntegerType());
  return true;
}

bool TypeInference::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);
  cur_type.reset();
  std::vector<ptr<Type>> param_tys;
  for (auto& param : n.params->values) param_tys.emplace_back(param->GetType());

  n.SetType(MakeFunctionType(n.ret_type->GetType(), param_tys));
  if (!ModifySymbolType(n.LOC(), n.name, n.GetType())) return false;

  return true;
}

bool TypeInference::Visit(AST::Parameter& p) {
  TraceEachVisit(p);
  // obtain its type
  SetNodeType(p, p.type->GetType());

  if (p.HasSymbol()) {
    if (isa<UnknownType>(p.type->GetType()) || isa<UnknownType>(p.GetType())) {
      Error(p.LOC(),
            "fail to deduce the type of parameter `" + p.sym->name + "'.");
      error_count++;
      return false;
    }

    AssignSymbolWithType(p.LOC(), p.sym->name, p.GetType());
    if (auto sty = dyn_cast<SpannedType>(p.GetType()))
      AssignSymbolWithType(p.LOC(), p.sym->name + ".span", sty->s_type);
  }

  // collect the parameter types
  cur_param_types.push_back(p.GetType());

  if (CCtx().ShowInferredTypes()) {
    dbgs() << "Parameter: ";
    if (p.HasSymbol())
      dbgs() << InScopeName(p.sym->name);
    else
      dbgs() << "(unnamed)";
    dbgs() << ", Type: " << AST::TYPE_STR(p) << "\n";
  }

  cur_type.reset();
  return true;
}

bool TypeInference::Visit(AST::ParamList& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::MultiDimSpans& n) {
  TraceEachVisit(n);
  cur_type.reset();
  return true;
}

bool TypeInference::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  if (auto ref = n.GetReference()) {
    if (auto id = dyn_cast<AST::Identifier>(ref)) {
      if (auto pty = GetSymbolType(n.LOC(), id->name)) {
        // special handling of the span-of spanned type
        if (SuffixedWith(id->name, ".span")) {
          assert(isa<MDSpanType>(pty) && "incorrect type annotated.");
        }
        SetNodeType(n, pty);
        return true;
      } else {
        Warning(n.LOC(),
                "symbol `" + id->name + "' is not associated with a type.");
        return false;
      }
    }

    // must have de-sugared early
    assert(!isa<AST::IntIndex>(ref));

    if (AST::istypeof<UnknownType>(ref)) {
      Error(n.LOC(), "unable to infer the type of expression.");
      error_count++;
      return false;
    }

    SetNodeType(n, ref->GetType());
    return true;
  }

  if (n.GetForm() == AST::Expr::Unary) {
    if (n.op == "ubound") {
      auto id = cast<AST::Identifier>(n.GetR());
      if (auto bty =
              dyn_cast<BoundedITupleType>(GetSymbolType(id->LOC(), id->name)))
        SetNodeType(n, MakeITupleType(bty->Dims()));
      else if (isa<BoundedIntegerType>(GetSymbolType(id->LOC(), id->name)))
        SetNodeType(n, MakeIntegerType());
      else
        choreo_unreachable("ubound type '" + AST::TYPE_STR(n.GetR()) +
                           "' is unexpected.");
      return true;
    } else if (n.op == "sizeof") {
      SetNodeType(n, MakeIntegerType());
      return true;
    } else if (n.op == "dataof") {
      auto ref = cast<AST::Expr>(n.GetR())->GetReference();
      auto id = cast<AST::Identifier>(ref);
      SetNodeType(n, GetSymbolType(id->LOC(), id->name + ".data"));
      return true;
    } else if (n.op == "!") {
      SetNodeType(n, MakeBooleanType());
      return true;
    } else if (n.op == "++" || n.op == "--") {
      SetNodeType(n, NodeType(*n.GetR()));
      return true;
    }
    choreo_unreachable("type inference is yet to implement for '" + n.op +
                       "'.");
  }

  if (n.GetForm() == AST::Expr::Binary) {
    if (n.op == "dimof") {
      SetNodeType(n, MakeIntegerType());
      return true;
    } else if (n.op == "elemof") {
      assert(isa<EventType>(NodeType(n)) && "only support elemof event array.");
      return true;
    }

    auto& pty_lhs = n.GetL()->GetType();
    auto& pty_rhs = n.GetR()->GetType();

    bool is_mutable = IsMutable(*pty_lhs) || IsMutable(*pty_rhs);

    if (n.IsLogical()) {
      if ((IsActualBoundedIntegerType(pty_lhs) && ConvertibleToInt(pty_rhs)) ||
          (IsActualBoundedIntegerType(pty_rhs) && ConvertibleToInt(pty_lhs)) ||
          (ConvertibleToInt(pty_lhs) && ConvertibleToInt(pty_rhs))) {
        SetNodeType(n, MakeBooleanType());
        return true;
      } else {
        Error(n.LOC(), "The operands of the expression cannot undergo '" +
                           n.op + "' logical operation.");
        error_count++;
        return false;
      }
    }

    if ((isa<MDSpanType>(pty_lhs) && isa<ITupleType>(pty_rhs)) ||
        (isa<MDSpanType>(pty_rhs) && isa<ITupleType>(pty_lhs))) {
      if (n.op == "concat") {
        SetNodeType(n, MakeMDSpanType(n.s));
        return true;
      }
      if (pty_lhs->Dims() == pty_rhs->Dims()) {
        SetNodeType(n,
                    MakeMDSpanType(n.s)); // note: the shape has been inferred
        cur_type = n.GetType();
        return true;
      }

      for (const auto& pty : {pty_lhs, pty_rhs})
        if (isa<ITupleType>(pty))
          if (ConvertibleToInt(pty)) {
            SetNodeType(n, MakeMDSpanType(n.s));
            cur_type = n.GetType();
            return true;
          }

      Error(n.LOC(),
            "The operands of the expression be performed for inconsistent "
            "shape dimension: " +
                std::to_string(pty_lhs->Dims()) + " vs. " +
                std::to_string(pty_rhs->Dims()));
      error_count++;
      return false;
    } else if (isa<MDSpanType>(pty_lhs) && isa<MDSpanType>(pty_rhs)) {
      if (n.op == "concat") {
        SetNodeType(n, MakeMDSpanType(n.s));
        return true;
      }
      if (!((n.op == "/") || (n.op == "%") || (n.op == "cdiv"))) {
        Error(n.LOC(),
              "The operands of the div/mod expression cannot undergo '" + n.op +
                  "' operation.");
        error_count++;
        SetNodeType(n, MakeUnknownType());
        return false;
      } else if (pty_lhs->Dims() == pty_rhs->Dims()) {
        SetNodeType(n, MakeITupleType(pty_lhs->Dims()));
        cur_type = n.GetType();
        return true;
      } else {
        Error(n.LOC(),
              "The operands of the expression be performed for inconsistent "
              "shape dimension.");
        error_count++;
        return false;
      }
    } else if (isa<ITupleType>(pty_rhs) && isa<ITupleType>(pty_lhs)) {
      if (n.op == "concat") {
        if (!cast<ITupleType>(pty_rhs)->IsDimValid() ||
            !cast<ITupleType>(pty_lhs)->IsDimValid())
          SetNodeType(n, MakeUninitITupleType());
        SetNodeType(n, MakeITupleType(pty_rhs->Dims() + pty_lhs->Dims()));
        return true;
      }
      if (pty_lhs->Dims() == pty_rhs->Dims()) {
        SetNodeType(n, pty_rhs);
        cur_type = n.GetType();
        return true;
      } else {
        Error(n.LOC(),
              "The operands of the expression be performed for inconsistent "
              "shape dimension.");
        error_count++;
        return false;
      }
    } else if (isa<ITupleType>(pty_rhs) && isa<IntegerType>(pty_lhs)) {
      SetNodeType(n, pty_rhs);
      cur_type = n.GetType();
    } else if (isa<ITupleType>(pty_lhs) && isa<IntegerType>(pty_rhs)) {
      SetNodeType(n, pty_lhs);
      cur_type = n.GetType();
    } else if ((isa<MDSpanType>(pty_rhs) && isa<IntegerType>(pty_lhs)) ||
               (isa<MDSpanType>(pty_lhs) && isa<IntegerType>(pty_rhs))) {
      SetNodeType(n, MakeMDSpanType(n.s));
      cur_type = n.GetType();
    } else if (isa<BoundedITupleType>(pty_lhs) && isa<IntegerType>(pty_rhs)) {
      if (n.op == "#-" || n.op == "#+")
        SetNodeType(n, MakeBoundedITupleType(n.s));
      else if (n.op == "#" || n.op == "#*" || n.op == "#/" || n.op == "#%") {
        Error(n.LOC(), "The operands of the expression cannot undergo '" +
                           n.op + "' binary operation.");
        error_count++;
      } else
        SetNodeType(n, pty_lhs);

      cur_type = n.GetType();
    } else if (isa<BoundedITupleType>(pty_lhs) &&
               isa<BoundedITupleType>(pty_rhs)) {
      // to support `chunkat(x, y#z)`
      if (n.op == "#") {
        auto bitt_lhs = cast<BoundedITupleType>(pty_lhs);
        auto bitt_rhs = cast<BoundedITupleType>(pty_rhs);
        // bounded integer in within will be transformed to bounded ituple in
        // valno.hpp
        assert(
            bitt_lhs->Dims() == 1 && bitt_rhs->Dims() == 1 &&
            "for now only support multiplication of one dim bounded ituples.");
        auto ub = bitt_lhs->GetUpperBound(0) * bitt_rhs->GetUpperBound(0);
        SetNodeType(n, MakeBoundedITupleType(Shape(1, ub)));
        cur_type = n.GetType();
      } else
        SetNodeType(n, MakeUnknownType());
    } else if (n.IsArith() && !n.IsUBArith() && CanYieldAnInteger(pty_lhs) &&
               CanYieldAnInteger(pty_rhs)) {
      if (isa<ScalarFloatType>(pty_lhs) || isa<ScalarFloatType>(pty_rhs)) {
        if (isa<DoubleType>(pty_lhs) || isa<DoubleType>(pty_rhs))
          SetNodeType(n, MakeDoubleType());
        else
          SetNodeType(n, MakeFloatType());
      } else {
        // it is ok to make compatiable types to do arith
        if (IsActualBoundedIntegerType(pty_lhs) && isa<IntegerType>(pty_rhs))
          SetNodeType(n, pty_lhs);
        else if (IsActualBoundedIntegerType(pty_rhs) &&
                 isa<IntegerType>(pty_lhs))
          SetNodeType(n, pty_rhs);
        else
          SetNodeType(n, MakeIntegerType(is_mutable));
      }
    } else if (*pty_lhs != *pty_rhs) {
      Error(n.LOC(), "The operands of the expression cannot undergo '" + n.op +
                         "' binary operation.");
      error_count++;
      return false;
    } else {
      SetNodeType(n, n.GetR()->GetType());
      cur_type = n.GetType();
      return true;
    }
  } // AST::Expr::Binary

  if (n.GetForm() == AST::Expr::Ternary) {
    if (n.op == "?") {
      auto& pty_lhs = n.GetL()->GetType();
      auto& pty_rhs = n.GetR()->GetType();

      if (pty_lhs->HasSufficientInfo() && pty_rhs->HasSufficientInfo()) {
        if (*pty_lhs != *pty_rhs) {
          Error(n.LOC(), "The operands of the expression cannot undergo '" +
                             n.op + "' operation.");
          error_count++;
          return false;
        }
        SetNodeType(n, pty_lhs);
        cur_type = n.GetType();
        return true;
      }
    } else {
      choreo_unreachable(
          "inference of the current ternary operation is not implemented.");
    }
  } // AST::Expr::Ternary
  return true;
}

bool TypeInference::Visit(AST::IntTuple& n) {
  TraceEachVisit(n);
  cur_type = n.GetType();
  return true;
}

bool TypeInference::Visit(AST::SpanAs& n) {
  TraceEachVisit(n);

  auto ity = NodeType(*n.id);
  if (!isa<SpannedType>(ity) && !isa<FutureType>(ity)) {
    Error(n.LOC(), "fail to infer the type of `" + STR(n.id) + "'.");
    error_count++;
    return false;
  }

  auto nty = NodeType(n);
  auto sty = cast<SpannedType>(nty);

  if (isa<SpannedType>(ity)) {
    SetNodeType(*n.nid, ShadowTypeStorage(sty));
    cur_type = nty;
  } else {
    auto fty =
        cast<SpannedType>(GetSymbolType(n.id->LOC(), n.id->name + ".data"));
    SetNodeType(n, ShadowTypeStorage(MakeSpannedType(
                       fty->f_type, sty->GetShape(), fty->GetStorage())));
    cur_type = n.GetType();
  }

  return true;
}

bool TypeInference::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  // future's type has been obtained by shape inference
  if (AST::istypeof<UnknownType>(&n)) {
    Error(n.LOC(), "fail to infer the FUTURE type of `" + n.future + "'.");
    error_count++;
    return false;
  }

  if (n.operation == ".any") {
    n.SetType(MakePlaceHolderFutureType());
    AssignSymbolWithType(n.LOC(), n.future + ".span",
                         MakePlaceHolderMDSpanType());
    AssignSymbolWithType(n.LOC(), n.future + ".data",
                         MakePlaceHolderSpannedType());
    AssignSymbolWithType(n.LOC(), n.future, MakePlaceHolderFutureType());
    return true;
  }

  // update the future type. fill info including storage, fundanmental type
  auto fty = cast<FutureType>(n.GetType());
  auto sty = MakeSpannedType(dma_fmty, fty->GetShape(), dma_mem);
  auto nty = MakeFutureType(sty, fty->IsAsync());
  n.SetType(nty);

  if (!n.future.empty()) {
    if (SSTab().IsDeclared(n.future)) {
      ModifySymbolType(n.LOC(), n.future + ".span", sty->GetMDSpanType());
      ModifySymbolType(n.LOC(), n.future + ".data", sty);
      ModifySymbolType(n.LOC(), n.future, nty);
    } else {
      AssignSymbolWithType(n.LOC(), n.future + ".span", sty->GetMDSpanType());
      AssignSymbolWithType(n.LOC(), n.future + ".data", sty);
      AssignSymbolWithType(n.LOC(), n.future, nty);
    }
  }

  if (CCtx().ShowInferredTypes()) {
    dbgs() << "Future:    "
           << ((n.future.empty()) ? SSTab().ScopeName() + "(anon)"
                                  : InScopeName(n.future))
           << ", Type: " << AST::TYPE_STR(n) << "\n";
  }

  cur_type.reset();
  return true;
}

bool TypeInference::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  AssignSymbolWithType(n.LOC(), n.bpv->name, n.bpv->GetType());
  if (CCtx().ShowInferredTypes()) {
    dbgs() << "Bounded:   " << InScopeName(n.bpv->name)
           << ", Type: " << AST::TYPE_STR(n.bpv) << "\n";
  }

  for (auto sym : n.cmpt_bpvs->AllValues()) {
    auto id = cast<AST::Identifier>(sym);
    AssignSymbolWithType(sym->LOC(), id->name, sym->GetType());
    if (CCtx().ShowInferredTypes()) {
      dbgs() << "Bounded:   " << InScopeName(id->name)
             << ", Type: " << AST::TYPE_STR(sym) << "\n";
    }
  }
  return true;
}

bool TypeInference::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  if (n.with) AssignSymbolWithType(n.LOC(), n.with->name, n.with->GetType());

  if (n.with_matchers) {
    for (auto pid : n.with_matchers->values) {
      auto id = cast<AST::Identifier>(pid);
      AssignSymbolWithType(n.LOC(), id->name, id->GetType());
    }
  }

  if (CCtx().ShowInferredTypes()) {
    if (n.with) {
      dbgs() << "Bounded:   ";
      dbgs() << InScopeName(n.with->name)
             << ", Type: " << AST::TYPE_STR(*n.with) << "\n";
    }
    if (n.with_matchers) {
      for (auto pid : n.with_matchers->values) {
        auto id = cast<AST::Identifier>(pid);
        dbgs() << "Bounded:   " << InScopeName(id->name)
               << ", Type: " << AST::TYPE_STR(*id) << "\n";
      }
    }
  }

  return true;
}

bool TypeInference::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::Memory& n) {
  TraceEachVisit(n);
  dma_mem = n.Get();
  return true;
}

bool TypeInference::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);
  auto ty = GetSymbolType(n.data->LOC(), n.data->name);
  if (isa<FutureType>(ty))
    ty = GetSymbolType(n.data->LOC(), n.data->name + ".data");
  auto sty = cast<SpannedType>(ty);
  auto fmty = sty->ElementType();
  auto sto = sty->GetStorage();
  assert(fmty != BaseType::UNKNOWN);
  if ((dma_fmty != BaseType::UNKNOWN) && (fmty != dma_fmty)) {
    Error(n.LOC(), "transfer data type with different types: " + STR(fmty) +
                       " vs. " + STR(dma_fmty));
    error_count++;
  }
  dma_fmty = fmty;
  dma_mem = sto;

  if (n.positions) {
    // update all the nodes with correct types
    for (auto& v : n.positions->AllValues()) { SetNodeType(*v, NodeType(*v)); }
  }
  // also update current node
  SetNodeType(n, MakeSpannedType(
                     fmty, cast<SpannedType>(n.GetType())->GetShape(), sto));

  return true;
}

bool TypeInference::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::Trigger& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::Call& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::Rotate& n) {
  TraceEachVisit(n);

  auto ty = type_equals.ResolveEqualFutures(*n.ids);

  if (!ty) {
    Error(n.LOC(), "Failed to deduce types inside ROTATE.");
    error_count++;
    return false;
  }

  return true;
}

bool TypeInference::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::Select& n) {
  TraceEachVisit(n);

  if (CanYieldAnInteger(NodeType(*n.select_factor))) {
    // normalize the shape
    SetNodeType(*n.select_factor, MakeIntegerType(n.select_factor->s));
  }

  if (cur_type = type_equals.ResolveEqualFutures(*n.expr_list)) {
    SetNodeType(n, cur_type);
    return true;
  }

  // TODO: inference type
  auto val = n.expr_list->ValueAt(0);
  auto sty = dyn_cast<SpannedType>(NodeType(*val));
  assert(sty);
  dma_mem = sty->GetStorage();
  dma_fmty = sty->ElementType();
  SetNodeType(n, MakeSpannedType(dma_fmty, sty->GetShape(), dma_mem));
  cur_type = n.GetType();

  return true;
}

bool TypeInference::Visit(AST::Return& n) {
  TraceEachVisit(n);

  if (!n.value) return true; // void return;

  ptr<Type> vty = n.value->GetType();
  if (auto ref = cast<AST::Expr>(n.value)->GetReference())
    if (auto id = dyn_cast<AST::Identifier>(ref))
      vty = GetSymbolType(n.LOC(), id->name);

  // get the return value's type
  if (isa<UnknownType>(vty)) {
    Error(n.LOC(), "failed to inference the type of " + AST::STR(*n.value));
    error_count++;
    return false;
  }

  auto ty = GetSymbolType(n.LOC(), fname);
  if (auto fty = dyn_cast<FunctionType>(ty)) {
    if (auto rty = dyn_cast<SpannedType>(fty->out_ty)) {
      auto tty = cast<SpannedType>(vty);
      if (!tty->HasSufficientInfo()) {
        Error(n.LOC(),
              "failed to inference the type detail of " + AST::STR(*n.value));
        error_count++;
        return false;
      }
      if (rty->Dims() != tty->Dims()) {
        Error(n.LOC(),
              "return type inconsistant: " + STR(*rty) + " vs. " + STR(*tty));
        error_count++;
        return false;
      }

      // already has sufficient info, make a comparison to avoid inconsistent
      // return type
      if (rty->HasSufficientInfo() && !rty->RuntimeShaped()) {
        if (*rty->GetMDSpanType() != *tty->GetMDSpanType() ||
            rty->ElementType() != tty->ElementType()) {
          Error(n.LOC(),
                "return type inconsistant: " + STR(*rty) + " vs. " + STR(*tty));
          error_count++;
          return false;
        } else if (tty->m_type != Storage::DEFAULT &&
                   tty->m_type != Storage::GLOBAL) {
          Error(n.LOC(),
                "can not return type with non-default/global storage.");
          error_count++;
          return false;
        }
      } else {
        // supplement information, note global should be mapped back
        auto nty = MakeSpannedType(tty->ElementType(), tty->GetShape());
        ModifySymbolType(n.LOC(), fname, MakeFunctionType(nty, fty->in_tys));
      }
    } else if (isa<UnknownType>(fty->out_ty)) {
      // the type must be inferred
      if (auto tty = dyn_cast<SpannedType>(vty)) {
        // global should be mapped back
        auto nty = MakeSpannedType(tty->ElementType(), tty->GetShape());
        ModifySymbolType(n.LOC(), fname, MakeFunctionType(nty, fty->in_tys));
      } else
        ModifySymbolType(n.LOC(), fname, MakeFunctionType(vty, fty->in_tys));
    }
  }
  return true;
}

bool TypeInference::Visit(AST::LoopRange& n) {
  TraceEachVisit(n);
  return true;
}

bool TypeInference::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  cur_type.reset(); // no current type to annotate the stmts inside
  return true;
}

bool TypeInference::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  cur_type.reset(); // no current type to annotate the stmts inside
  return true;
}

bool TypeInference::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  cur_type.reset(); // no current type to annotate the stmts inside
  return true;
}

bool TypeInference::Visit(AST::IncrementBlock& n) {
  TraceEachVisit(n);
  cur_type.reset(); // no current type to annotate the stmts inside
  return true;
}

bool TypeInference::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);
  cur_type.reset(); // no current type to annotate the stmts inside
  return true;
}
bool TypeInference::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);
  return true;
}
bool TypeInference::Visit(AST::Program& n) {
  TraceEachVisit(n);
  return true;
}
