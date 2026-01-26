#include "typeinfer.hpp"

#include "ast.hpp"
#include "target_utils.hpp"
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
  } else if (auto call = dyn_cast<AST::Call>(&n)) {
    if (call->template_args) {
      in_template_param = true; // enter template param visit
    }
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
      // update the return information nodes
      if (auto spty = dyn_cast<SpannedType>(func_ty->out_ty)) {
        f->f_decl.ret_type->SetType(spty);
        if (f->f_decl.ret_type->mdspan_type) {
          assert(spty->s_type);
          f->f_decl.ret_type->mdspan_type->SetType(spty->s_type->Clone());
        }
      }
    }
    if (CCtx().ShowInferredTypes()) {
      dbgs() << "Function:  " << SSTab().InScopeName(f->name)
             << ", Type: " << AST::TYPE_STR(*f) << "\n";
    }
  } else if (isa<AST::DMA>(&n) || isa<AST::NamedVariableDecl>(&n) ||
             isa<AST::Assignment>(&n)) {
    dma_fmty = BaseType::UNKNOWN;
    dma_mem = Storage::NONE;
  } else if (isa<AST::Parameter>(&n)) {
    allow_named_dim = false;
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    for (auto& rn : fb->GetRanges()) {
      auto range = cast<AST::LoopRange>(rn);
      auto sym_ty = GetSymbolType(n.LOC(), range->IVName());
      SetNodeType(*range->iv, sym_ty);
    }
  }

  Visitor::AfterVisit(n);
  return true;
}

bool TypeInference::AssignSymbolWithType(const location& loc,
                                         const std::string& sym,
                                         const ptr<Type>& ty) {
  if (!SSTab().DefineSymbol(sym, ty)) {
    Error1(loc,
           "symbol `" + sym + "' has already been associated with a type.");
    return false;
  }

  VST_DEBUG(dbgs() << " |- symbol type: `" << sym << "` -> '" << STR(*ty)
                   << "''\n");

  return true;
}

ptr<Type> TypeInference::GetSymbolType(const location& loc,
                                       const std::string& name) {
  if (!SSTab().IsDeclared(name)) {

    Error1(loc, "The symbol `" + name + "' has not been defined.");
    return nullptr;
  }
  if (auto pty = SSTab().LookupSymbol(name)) {
    return pty;
  } else {
    Error1(loc, "symbol `" + name + "' is not associated with a type.");
    return nullptr;
  }
}

bool TypeInference::ModifySymbolType(const location& loc,
                                     const std::string& name,
                                     const ptr<Type>& ty) {
  if (!SSTab().IsDeclared(name)) {
    Error1(loc, "The symbol `" + name + "' has not been defined.");
    return false;
  }
  if (!SSTab().ModifySymbolType(name, ty)) {
    Error1(loc, "symbol `" + name + "' is not associated with a type.");
    return false;
  }

  VST_DEBUG(dbgs() << " |- modify type: `" << name << "` -> '" << STR(*ty)
                   << "'\n");

  return true;
}

bool TypeInference::SetAsCurrentType(AST::Node& nd, const std::string& n) {
  const auto nty = nd.GetType();
  if (nty->HasSufficientInfo()) {
    // already has a type with sufficient info, check for consistence.

    // ignore mutable integer type
    if (IsMutable(*nty)) return true;

    if (cur_type->HasSufficientInfo()) {
      if (BetterQuality(cur_type, nty)) {
        SetNodeType(nd, ShadowTypeStorage(cur_type));
        return true;
      } else {
        assert(!BetterQuality(nty, cur_type) &&
               "the inference type should be better qualified.");
        if (!(cur_type->LogicalEqual(*nty))) {
          Error1(nd.LOC(), "can not infer the type of `" + n + "'.");
          return false;
        }
      }
    }
  }

  // Or else we need to set the type with current
  // Check for inference failures
  if (isa<UnknownType>(cur_type)) {
    Error1(nd.LOC(), "can not infer the type of `" + n + "'.");
    return false;
  }

  if (!cur_type->HasSufficientInfo()) {
    Error1(nd.LOC(), "can not infer '" + cur_type->Name() +
                         "' type detail of symbol `" + n + "'.");
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

  if (in_template_param) in_template_param = false; // exit template param visit
  return true;
}

bool TypeInference::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);
  BaseType res = n.GetType()->GetBaseType();
  assert(IsIntegerType(res));
  SetNodeType(n, MakeScalarIntegerType(res, false));
  return true;
}

bool TypeInference::Visit(AST::FloatLiteral& n) {
  TraceEachVisit(n);
  if (std::holds_alternative<float>(n.value))
    SetNodeType(n, MakeF32Type());
  else if (std::holds_alternative<double>(n.value))
    SetNodeType(n, MakeF64Type());
  else
    choreo_unreachable("unhandled floating-point type.");
  return true;
}

bool TypeInference::Visit(AST::StringLiteral& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeStringType());
  return true;
}

bool TypeInference::Visit(AST::BoolLiteral& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeBooleanType());
  return true;
}

bool TypeInference::Visit(AST::DataType& n) {
  TraceEachVisit(n);

  allow_named_dim = false;

  if (n.getBaseType() == BaseType::UNKNOWN)
    return true; // ignore the annotation that needs inference

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
      SetNodeType(n, MakeSpannedType(n.getBaseType(), shape));
    cur_type = n.GetType();
  }

  return true;
}

bool TypeInference::Visit(AST::Identifier& n) {
  TraceEachVisit(n);

  // for named dims in parameters
  if (allow_named_dim && !SSTab().DeclaredInScope(n.name))
    AssignSymbolWithType(n.LOC(), n.name, MakeIntegerType());

  if (in_template_param && !SSTab().IsDeclared(n.name))
    AssignSymbolWithType(n.LOC(), n.name, MakeIntegerType());

  return true;
}

bool TypeInference::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  // annotate it should not take storage
  if (n.init_expr && (isa<AST::Select>(AST::Ref(n.init_expr)) ||
                      isa<AST::SpanAs>(AST::Ref(n.init_expr))))
    n.AddNote("ref");

  if (cur_type && !SetAsCurrentType(n, n.name_str)) {
    cur_type.reset();
    assert(false);
    return false;
  }

  cur_type.reset();

  if (AST::istypeof<UnknownType>(&n)) {
    Error1(n.LOC(), "can not infer the type of `" + n.name_str + "'.");
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
      Error1(n.LOC(), "unable to inference the type of `" + n.name_str + "'.");
      return false;
    }

    if (!n.init_expr->GetType()->HasSufficientInfo()) {
      Error1(n.LOC(),
             "unable to inference the type detail of `" + n.name_str + "'.");
      return false;
    }

    SetNodeType(n, n.init_expr->GetType());
  } else if (AST::istypeof<UnknownType>(&n)) {
    // need type inference
    Error1(n.LOC(),
           "`" + n.name_str +
               "' is declared without type annotation or initialization.");
    return false;
  }

  AssignSymbolWithType(n.LOC(), n.name_str, n.GetType());

  if (CCtx().ShowInferredTypes()) {
    dbgs() << "Partial:   " << InScopeName(n.name_str)
           << ", Type: " << AST::TYPE_STR(n) << "\n";
  }

  // The node only occurs when decl named mdspan.
  // So the type is unnecessary to propagate.
  cur_type.reset();

  return true;
}

bool TypeInference::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);

  // type is not ready for a decl
  if (n.IsDecl()) return true;

  // It is a reference
  auto dty = GetSymbolType(n.LOC(), n.GetDataName());
  SetNodeType(*n.data, dty);

  if (n.AccessElement()) {
    auto sty = cast<SpannedType>(dty);
    SetNodeType(n, MakeScalarType(sty->ElementType()));
  } else
    SetNodeType(n, dty);

  return true;
}

// ituple override operator "=" for definition
bool TypeInference::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  if (n.AssignToDataElement()) {
    // should be assigned already by DataAccess
    assert(isa<ScalarType>(NodeType(*n.da)));
    auto dty = GetSymbolType(n.LOC(), n.GetDataArrayName());
    auto ety = MakeScalarType(cast<SpannedType>(dty)->ElementType());
    SetNodeType(*n.da, ety);
    SetNodeType(n, ety);
    return true;
  }

  if (n.value && (isa<AST::Select>(n.value) || isa<AST::SpanAs>(n.value)))
    n.AddNote("ref");

  if (!n.IsDecl()) {
    auto vty = GetSymbolType(n.LOC(), n.GetName());
    auto ety = NodeType(*n.value);
    if (isa<FutureType>(vty)) {
      // no type inference is necessary
      SetNodeType(n, ety);
      SetNodeType(*n.da, ety);
      cur_type.reset();
      return true;
    } else if (IsMutable(*vty)) {
      // no type inference is necessary
      SetNodeType(n, vty);
      SetNodeType(*n.da, vty);
      cur_type.reset();
      return true;
    } else {
      Error1(n.LOC(),
             "current choreo does not support symbol re-assignment except for "
             "future/mutable type. Current type: " +
                 PSTR(vty));
      SetNodeType(n, MakeUnknownType());
      cur_type.reset();
      return false;
    }
  }

  assert(!n.da->AccessElement());

  if (isa<UnknownType>(NodeType(*n.value))) {
    Error1(n.LOC(), "fail to deduce type of `" + n.GetName() + "'.");
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

  if (auto sty = dyn_cast<SpannedType>(p.type->GetType())) {
    if (p.GetAttr() == ParamAttr::GLOBAL_INPUT)
      sty->SetStorage(Storage::GLOBAL);
  }

  // obtain its type
  SetNodeType(p, p.type->GetType());

  if (p.HasSymbol()) {
    if (isa<UnknownType>(p.type->GetType()) || isa<UnknownType>(p.GetType())) {
      Error1(p.LOC(),
             "fail to deduce the type of parameter `" + p.sym->name + "'.");
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
        if (SuffixedWith(id->name, ".span"))
          assert(isa<MDSpanType>(pty) && "incorrect type annotated.");
        SetNodeType(n, pty);
        cur_type = n.GetType();
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
      Error1(n.LOC(), "unable to infer the type of expression.");
      return false;
    }

    SetNodeType(n, NodeType(*ref));
    cur_type = n.GetType();
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
    } else if (n.op == "sizeof") {
      SetNodeType(n, MakeIntegerType());
    } else if (n.op == "dataof" || n.op == "mdataof") {
      auto ref = cast<AST::Expr>(n.GetR())->GetReference();
      auto id = cast<AST::Identifier>(ref);
      SetNodeType(n, GetSymbolType(
                         id->LOC(),
                         id->name + (n.op == "mdataof" ? ".mdata" : ".data")));
    } else if (n.op == "addrof") {
      // earlysema has set it already
      assert(isa<AddrType>(NodeType(n)));
      return true;
    } else if (n.op == "!") {
      SetNodeType(n, MakeBooleanType());
    } else if (n.op == "++" || n.op == "--") {
      SetNodeType(n, NodeType(*n.GetR()));
    } else if (n.op == "~") {
      assert(CanYieldAnInteger(NodeType(*n.GetR())));
      SetNodeType(n, MakeIntegerType(true));
    } else if (n.op == "cast") {
      auto cexpr = cast<AST::CastExpr>(&n);
      SetNodeType(n, MakeScalarType(cexpr->ToType(), true));
    } else {
      choreo_unreachable("type inference is yet to implement for '" + n.op +
                         "'.");
    }
    cur_type = n.GetType();
    return true;
  } // AST::Expr::Unary

  if (n.GetForm() == AST::Expr::Binary) {
    if (n.op == "dimof") {
      SetNodeType(n, MakeIntegerType());
      cur_type = n.GetType();
      return true;
    } else if (n.op == "elemof") {
      assert(isa<EventType>(NodeType(n)) && "only support elemof event array.");
      cur_type = n.GetType();
      return true;
    }

    auto pty_lhs = NodeType(*n.GetL());
    auto pty_rhs = NodeType(*n.GetR());

    if (n.IsCompare()) {
      if ((IsActualBoundedIntegerType(pty_lhs) && ConvertibleToInt(pty_rhs)) ||
          (IsActualBoundedIntegerType(pty_rhs) && ConvertibleToInt(pty_lhs)) ||
          (CanYieldAnInteger(pty_lhs) && CanYieldAnInteger(pty_rhs))) {
        SetNodeType(n, MakeBooleanType());
        cur_type = n.GetType();
        return true;
      } else {
        Error1(n.LOC(), "The operands of the expression cannot undergo '" +
                            n.op + "' logical operation, the types are '" +
                            PSTR(pty_lhs) + "' and '" + PSTR(pty_rhs) + "'");
        return false;
      }
    }

    if (n.IsLogical()) {
      if (isa<BooleanType>(pty_lhs) && isa<BooleanType>(pty_rhs)) {
        SetNodeType(n, MakeBooleanType());
        cur_type = n.GetType();
        return true;
      } else {
        Error1(n.LOC(), "The operands of the expression cannot undergo '" +
                            n.op + "' logical operation.");
        return false;
      }
    }

    if ((isa<MDSpanType>(pty_lhs) && isa<ITupleType>(pty_rhs)) ||
        (isa<MDSpanType>(pty_rhs) && isa<ITupleType>(pty_lhs))) {
      if (n.op == "concat") {
        SetNodeType(n, MakeMDSpanType(n.s));
        cur_type = n.GetType();
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

      Error1(n.LOC(),
             "The operands of the expression be performed for inconsistent "
             "shape dimension: " +
                 std::to_string(pty_lhs->Dims()) + " vs. " +
                 std::to_string(pty_rhs->Dims()));
      return false;
    } else if (isa<MDSpanType>(pty_lhs) && isa<MDSpanType>(pty_rhs)) {
      if (n.op == "concat") {
        SetNodeType(n, MakeMDSpanType(n.s));
        cur_type = n.GetType();
        return true;
      }
      if (!((n.op == "/") || (n.op == "%") || (n.op == "cdiv"))) {
        Error1(n.LOC(),
               "The operands of the div/mod expression cannot undergo '" +
                   n.op + "' operation.");
        SetNodeType(n, MakeUnknownType());
        return false;
      } else if (pty_lhs->Dims() == pty_rhs->Dims()) {
        SetNodeType(n, MakeITupleType(pty_lhs->Dims()));
        cur_type = n.GetType();
        return true;
      } else {
        Error1(n.LOC(),
               "The operands of the expression be performed for inconsistent "
               "shape dimension.");
        return false;
      }
    } else if (isa<ITupleType>(pty_rhs) && isa<ITupleType>(pty_lhs)) {
      if (n.op == "concat") {
        if (!cast<ITupleType>(pty_rhs)->IsDimValid() ||
            !cast<ITupleType>(pty_lhs)->IsDimValid())
          SetNodeType(n, MakeUninitITupleType());
        SetNodeType(n, MakeITupleType(pty_rhs->Dims() + pty_lhs->Dims()));
        cur_type = n.GetType();
        return true;
      }
      if (pty_lhs->Dims() == pty_rhs->Dims()) {
        SetNodeType(n, pty_rhs);
        cur_type = n.GetType();
        return true;
      } else {
        Error1(n.LOC(),
               "The operands of the expression be performed for inconsistent "
               "shape dimension.");
        return false;
      }
    } else if (isa<ITupleType>(pty_rhs) && isa<ScalarIntegerType>(pty_lhs)) {
      SetNodeType(n, pty_rhs);
    } else if (isa<ITupleType>(pty_lhs) && isa<ScalarIntegerType>(pty_rhs)) {
      SetNodeType(n, pty_lhs);
    } else if ((isa<MDSpanType>(pty_rhs) && isa<ScalarIntegerType>(pty_lhs)) ||
               (isa<MDSpanType>(pty_lhs) && isa<ScalarIntegerType>(pty_rhs))) {
      SetNodeType(n, MakeMDSpanType(n.s));
    } else if (isa<BoundedITupleType>(pty_lhs) &&
               isa<ScalarIntegerType>(pty_rhs)) {
      if (n.op == "#-" || n.op == "#+")
        SetNodeType(n, MakeBoundedITupleType(n.s));
      else if (n.op == "#" || n.op == "#*" || n.op == "#/" || n.op == "#%") {
        Error1(n.LOC(), "The operands of the expression cannot undergo '" +
                            n.op + "' binary operation.");
      } else if (n.op == "&" || n.op == "|" || n.op == "^" || n.op == "<<" ||
                 n.op == ">>") {
        SetNodeType(n, MakeIntegerType(true));
      } else
        SetNodeType(n, pty_lhs);
    } else if (isa<BoundedITupleType>(pty_rhs) &&
               isa<ScalarIntegerType>(pty_lhs)) {
      if (n.op == "#-" || n.op == "#+" || n.op == "#" || n.op == "#*" ||
          n.op == "#/" || n.op == "#%")
        Error1(n.LOC(), "The operands of the expression cannot undergo '" +
                            n.op + "' binary operation.");
      else if (n.op == "&" || n.op == "|" || n.op == "^" || n.op == "<<" ||
               n.op == ">>")
        SetNodeType(n, MakeIntegerType(true));
      else
        SetNodeType(n, pty_rhs);
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
      }
      // else the type is decayed. use the type of earlysema's
    } else if (n.IsArith() && !n.IsUBArith() && CanYieldAnInteger(pty_lhs) &&
               CanYieldAnInteger(pty_rhs)) {
      // use the type inferred by early sema
      assert(n.GetType() != nullptr);
    } else if (n.isBitwise() && CanYieldAnInteger(pty_rhs) &&
               CanYieldAnInteger(pty_lhs)) {
      bool is_mutable = IsMutable(*pty_lhs) || IsMutable(*pty_rhs);
      SetNodeType(n, MakeIntegerType(is_mutable));
    } else if (isa<SpannedType>(pty_lhs) || isa<SpannedType>(pty_rhs)) {
      // set to any operand for later check
      if (isa<SpannedType>(pty_lhs))
        SetNodeType(n, pty_lhs);
      else if (isa<SpannedType>(pty_rhs))
        SetNodeType(n, pty_rhs);
    } else if (*pty_lhs != *pty_rhs) {
      Error1(n.LOC(), "The operands of the expression cannot undergo '" + n.op +
                          "' binary operation.");
      return false;
    } else {
      SetNodeType(n, NodeType(*n.GetR()));
    }
    cur_type = n.GetType();
    return true;
  } // AST::Expr::Binary

  if (n.GetForm() == AST::Expr::Ternary) {
    if (n.op == "?") {
      auto pty_lhs = NodeType(*n.GetL());
      auto pty_rhs = NodeType(*n.GetR());

      if (pty_lhs->HasSufficientInfo() && pty_rhs->HasSufficientInfo()) {
        if (CanYieldAnInteger(pty_lhs) && CanYieldAnInteger(pty_rhs)) {
          SetNodeType(n, pty_lhs->Clone());
          if (isa<ScalarType>(pty_rhs)) SetNodeType(n, pty_rhs->Clone());
        } else {
          if (*pty_lhs != *pty_rhs) {
            if (!(IsMutable(*pty_lhs) || IsMutable(*pty_rhs))) {
              Error1(n.LOC(),
                     "The operands of the expression cannot undergo '" + n.op +
                         "' operation, the types are '" + PSTR(pty_lhs) +
                         "' and '" + PSTR(pty_rhs) + "'");
              return false;
            }
          }
          SetNodeType(n, pty_lhs);
        }
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

bool TypeInference::Visit(AST::CastExpr& n) {
  TraceEachVisit(n);
  auto cast_from = NodeType(*n.GetR());
  SetNodeType(n, MakeScalarType(n.ToType(), true));
  VST_DEBUG(dbgs() << " |- type node cast: `" << PSTR(n.GetR()) << "`:\n\t`"
                   << PSTR(cast_from) << "` to `" << PSTR(n.GetType())
                   << "`\n");
  cur_type = n.GetType();
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
    Error1(n.LOC(), "fail to infer the type of `" + STR(n.id) + "'.");
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
                       fty->e_type, sty->GetShape(), fty->GetStorage())));
    cur_type = n.GetType();
  }

  return true;
}

bool TypeInference::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  // future's type has been obtained by shape inference
  if (AST::istypeof<UnknownType>(&n)) {
    Error1(n.LOC(), "fail to infer the FUTURE type of `" + n.future + "'.");
    return false;
  }

  if (n.operation == ".any") {
    n.SetType(MakePlaceHolderFutureType());
    AssignSymbolWithType(n.LOC(), n.future + ".span",
                         MakePlaceHolderMDSpanType());
    AssignSymbolWithType(n.LOC(), n.future + ".data",
                         MakePlaceHolderSpannedType());
    if (n.IsSparse())
      AssignSymbolWithType(n.LOC(), n.future + ".mdata",
                           MakePlaceHolderSpannedType());
    AssignSymbolWithType(n.LOC(), n.future, MakePlaceHolderFutureType());
    return true;
  }

  // update the future type. fill info including storage, fundamental type
  auto fty = cast<FutureType>(n.GetType());
  auto sty = MakeSpannedType(dma_fmty, fty->GetShape(), dma_mem);
  auto nty = MakeFutureType(sty, fty->IsAsync());
  n.SetType(nty);

  if (!n.future.empty()) {
    if (SSTab().IsDeclared(n.future)) {
      ModifySymbolType(n.LOC(), n.future + ".span", sty->GetMDSpanType());
      ModifySymbolType(n.LOC(), n.future + ".data", sty);
      if (n.IsSparse()) ModifySymbolType(n.LOC(), n.future + ".mdata", sty);
      ModifySymbolType(n.LOC(), n.future, nty);
    } else {
      AssignSymbolWithType(n.LOC(), n.future + ".span", sty->GetMDSpanType());
      AssignSymbolWithType(n.LOC(), n.future + ".data", sty);
      if (n.IsSparse()) AssignSymbolWithType(n.LOC(), n.future + ".mdata", sty);
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

bool TypeInference::Visit(AST::MMA& n) {
  TraceEachVisit(n);

  auto& op = *n.GetOperation();
  switch (op.Tag()) {
  case AST::MMAOperation::Fill: {
    auto fill_ty = MakeSpannedType(op.FillingType(), GenUninitShape());
    // any usage of this symbol is illegal util the inference happens
    AssignSymbolWithType(n.LOC(), op.FillingSymbol(), fill_ty);
    AssignSymbolWithType(n.LOC(), op.FillingSymbol() + ".span",
                         fill_ty->GetMDSpanType());
  } break;
  case AST::MMAOperation::Load: {
    AssignSymbolWithType(n.LOC(), op.GetFuture(), n.GetType()->Clone());
    auto sty = GetSpannedType(n.GetType());
    assert(sty && "expect a spanned type.");
    AssignSymbolWithType(n.LOC(), op.GetFuture() + ".span",
                         sty->GetMDSpanType()->Clone());
    if (CCtx().ShowInferredTypes()) {
      dbgs() << "Future:    " << InScopeName(op.GetFuture())
             << ", Type: " << AST::TYPE_STR(n) << "\n";
    }
  } break;
  case AST::MMAOperation::Exec: {
    auto acc_ty = GetSymbolType(n.LOC(), op.ExecOperand(0));
    auto sty = GetSpannedType(acc_ty);
    auto ety = sty->ElementType();
    ptr<SpannedType> mc_ty = nullptr;
    auto nsty = cast<SpannedType>(n.GetType());
    // mc type is explicit annotated, set it
    if (ety != BaseType::UNKSCALAR) {
      auto shape = nsty->GetShape();
      auto storage = sty->GetStorage();
      mc_ty = MakeSpannedType(ety, shape, storage);
    } else {
      mc_ty = cast<SpannedType>(n.GetType());
    }

    // element type must be inferred
    if (mc_ty->ElementType() == BaseType::UNKSCALAR) {
      auto candidate_tys = MMALimit::InferResultType();
      if (candidate_tys.empty())
        Error1(n.LOC(), "Failed to infer the element type of `" +
                            op.ExecOperand(0) +
                            ". Please explicitly annotate.");
      else {
        auto shape = mc_ty->GetShape();
        auto storage = mc_ty->GetStorage();
        mc_ty = MakeSpannedType(candidate_tys.front(), shape, storage);
      }
    }

    ModifySymbolType(n.LOC(), op.ExecOperand(0), mc_ty);
    SetNodeType(n, mc_ty);
    auto mcs_ty = mc_ty->GetMDSpanType()->Clone();
    ModifySymbolType(n.LOC(), op.ExecOperand(0) + ".span", mcs_ty);
    if (CCtx().ShowInferredTypes()) {
      dbgs() << "Symbol:    " << InScopeName(op.ExecOperand(0))
             << ", Type: " << AST::TYPE_STR(n) << "\n";
    }
  } break;
  case AST::MMAOperation::Store: {
  } break;
  default: break;
  }
  return true;
}

bool TypeInference::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  AssignSymbolWithType(n.LOC(), n.BPV()->name, n.BPV()->GetType());
  if (CCtx().ShowInferredTypes()) {
    dbgs() << "Bounded:   " << InScopeName(n.BPV()->name)
           << ", Type: " << AST::TYPE_STR(n.BPV()) << "\n";
  }

  for (auto sym : n.AllSubPVs()) {
    auto id = cast<AST::Identifier>(sym);
    AssignSymbolWithType(sym->LOC(), id->name, id->GetType());
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
  // if ((dma_fmty != BaseType::UNKNOWN) && (fmty != dma_fmty)) {
  //   Error1(n.LOC(), "assign/transfer data with a different type: " +
  //   STR(fmty) +
  //                       " vs. " + STR(dma_fmty));
  // }
  dma_fmty = fmty;
  dma_mem = sto;

  // update all the positions with correct types
  for (auto tsi : n.AllOperations()) {
    for (auto& v : tsi->GetIndices()) { SetNodeType(*v, NodeType(*v)); }
    if (auto s = tsi->GetStrides())
      for (auto& v : s->AllValues()) SetNodeType(*v, NodeType(*v));
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
  cur_type = n.GetType(); // use early-sema's type
  assert(cur_type);

  return true;
}

bool TypeInference::Visit(AST::Rotate& n) {
  TraceEachVisit(n);

  auto ty = type_equals.ResolveEqualFutures(*n.ids);

  if (!ty) {
    Error1(n.LOC(), "Failed to deduce types inside ROTATE.");
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

  // if (CanYieldAnInteger(NodeType(*n.select_factor))) {
  //   // normalize the shape
  //   SetNodeType(*n.select_factor, MakeIntegerType());
  // }

  if ((cur_type = type_equals.ResolveEqualFutures(*n.expr_list))) {
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
    Error1(n.LOC(), "failed to inference the type of " + AST::STR(*n.value));
    return false;
  }

  auto ty = GetSymbolType(n.LOC(), fname);
  if (auto fty = dyn_cast<FunctionType>(ty)) {
    if (auto rty = dyn_cast<SpannedType>(fty->out_ty)) {
      auto tty = cast<SpannedType>(vty);
      if (!tty->HasSufficientInfo()) {
        Error1(n.LOC(),
               "failed to inference the type detail of " + AST::STR(*n.value));
        return false;
      }
      if (rty->Dims() != tty->Dims()) {
        Error1(n.LOC(),
               "return type inconsistent: " + STR(*rty) + " vs. " + STR(*tty));
        return false;
      }

      // already has sufficient info, make a comparison to avoid inconsistent
      // return type
      if (rty->HasSufficientInfo() && !rty->RuntimeShaped()) {
        if (*rty->GetMDSpanType() != *tty->GetMDSpanType() ||
            rty->ElementType() != tty->ElementType()) {
          Error1(n.LOC(), "return type inconsistent: " + STR(*rty) + " vs. " +
                              STR(*tty));
          return false;
        } else if (tty->m_type != Storage::DEFAULT &&
                   tty->m_type != Storage::GLOBAL) {
          Error1(n.LOC(),
                 "can not return type with non-default/global storage.");
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
