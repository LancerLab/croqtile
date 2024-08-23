#include "typeinfer.hpp"

#include <iostream>

#include "ast.hpp"
#include "types.hpp"

using namespace Choreo;

bool TypeInference::BeforeVisit(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
    SSTab().EnterScope("");  // global scope
  } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    AssignSymbolWithType(n.LOC(), f->name,
                         MakeUnknownType());  // the type will be modified after
                                              // parameters/return are processed
    SSTab().EnterScope(f->name);
    cur_func_name = f->name;
  } else if (isa<AST::ParallelBy>(&n)) {
    static size_t count = 0;
    SSTab().EnterScope("paraby_" + std::to_string(count++));
  } else if (isa<AST::WithBlock>(&n)) {
    static size_t count = 0;
    SSTab().EnterScope("within_" + std::to_string(count++));
  } else if (isa<AST::ForeachBlock>(&n)) {
    static size_t count = 0;
    SSTab().EnterScope("foreach_" + std::to_string(count++));
  } else if (isa<AST::DMA>(&n)) {
    dma_fmty = BaseType::UNKNOWN;
    dma_mem = Storage::NONE;
  } else if (isa<AST::Parameter>(&n)) {
    allow_named_dim = true;
  }
  return true;
}

bool TypeInference::AfterVisit(AST::Node &n) {
  if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
      isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
      isa<AST::ForeachBlock>(&n)) {
    SSTab().LeaveScope();
  }
  if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    cur_func_name = "";
    auto sym_ty = GetSymbolType(f->LOC(), f->name);
    assert(!isa<UnknownType>(sym_ty) && "symbol type is not deduced.");

    auto func_ty = cast<FunctionType>(sym_ty);
    if (AST::typeof<UnknownType>(f) || isa<SpannedType>(func_ty->out_ty)) {
      // update the return type node since type inference could have changed the
      // function type already
      f->f_decl.ret_type->SetType(func_ty->out_ty);
      f->f_decl.SetType(sym_ty);
      f->SetType(sym_ty);
    }
    if (Dump)
      os << "Function:  " << SSTab().InScopeName(f->name)
         << ", Type: " << AST::TYPE_STR(*f) << "\n";
  } else if (isa<AST::DMA>(&n)) {
    dma_fmty = BaseType::UNKNOWN;
    dma_mem = Storage::NONE;
  } else if (isa<AST::Parameter>(&n)) {
    allow_named_dim = false;
  }
  return true;
}

bool TypeInference::AssignSymbolWithType(const location &loc,
                                         const std::string &sym,
                                         const ptr<Type> &ty) {
  if (!SSTab().DefineSymbol(sym, ty)) {
    Error(loc, "symbol `" + sym + "' has already been associated with a type.");
    error_count++;
    return false;
  }

  if (trace_visit)
    os << "Assign symbol `" << sym << "` with type: " << STR(*ty) << "\n";

  return true;
}

ptr<Type> TypeInference::GetSymbolType(const location &loc,
                                       const std::string &name) {
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

bool TypeInference::ModifySymbolType(const location &loc,
                                     const std::string &name,
                                     const ptr<Type> &ty) {
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

  if (trace_visit)
    os << "Modify symbol `" << name << "` with type: " << STR(*ty) << "\n";

  return true;
}

bool TypeInference::SetAsCurrentType(AST::Node &nd, const std::string &n) {
  const auto ty = nd.GetType();
  if (ty->HasSufficientInfo()) {
    // already has a type with sufficient info, check for consistence.
    if (cur_type->HasSufficientInfo() && !(*cur_type == *ty)) {
      Error(nd.LOC(), "can not infer the type of `" + n + "'.");
      error_count++;
      return false;
    } else
      return true;
  }

  // Or else we need to set the type with current
  // Check for inference failures
  if (isa<UnknownType>(cur_type.get())) {
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
      if (n->mem)
        st->SetStorage(n->mem->st);

  // The type is successfully inferred, set the node
  nd.SetType(cur_type);

  return true;
}

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.TypeNameString() << ": "; \
    os << "\n";                       \
  }

bool TypeInference::Visit(AST::MultiNodes &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::MultiValues &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::IntLiteral &n) {
  n.SetType(MakeIntegerType());
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::Boolean &n) {
  n.SetType(MakeBooleanType());
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::DataType &n) {
  __TRACE_EACH_VISIT__(n)

  allow_named_dim = false;

  if (n.getBaseType() == BaseType::UNKNOWN)
    return true;  // ignore the annotation that needs inference

  assert((cur_type == nullptr) && "Expecting null type.");

  if (!n.mdspan_type) {
    cur_type = n.GetType();  // simple types
    return true;
  }

  // compound type
  if (auto mdspan = dyn_cast<AST::MultiDimSpans>(n.mdspan_type)) {
    n.SetType(MakeSpannedType(n.getFundamentalType(),
                              cast<MDSpanType>(mdspan->GetType())->GetShape()));
    cur_type = n.GetType();
  }

  return true;
}

bool TypeInference::Visit(AST::Identifier &n) {
  __TRACE_EACH_VISIT__(n)

  // for named dims in parameters
  if (allow_named_dim && !SSTab().DeclaredInScope(n.name))
    SSTab().DefineSymbol(n.name, MakeIntegerType());

  return true;
}

bool TypeInference::Visit(AST::NamedVariableDecl &n) {
  __TRACE_EACH_VISIT__(n)

  if (!SetAsCurrentType(n, n.name_str)) {
    cur_type.reset();
    return false;
  }

  cur_type.reset();

  if (AST::typeof<UnknownType>(&n)) {
    Error(n.LOC(), "can not infer the type of `" + n.name_str + "'.");
    error_count++;
    return false;
  }

  AssignSymbolWithType(n.LOC(), n.name_str, n.GetType());

  if (AST::typeof<SpannedType>(&n))
    AssignSymbolWithType(n.LOC(), n.name_str + ".span",
                         cast<SpannedType>(n.GetType())->GetMDSpanType());

  if (Dump) {
    os << "Symbol:    " << SSTab().InScopeName(n.name_str)
       << ", Type: " << AST::TYPE_STR(n);
    os << "\n";
  }

  return true;
}

bool TypeInference::Visit(AST::NamedTypeDecl &n) {
  __TRACE_EACH_VISIT__(n)

  if (n.init_expr) {
    if (AST::typeof<UnknownType>(n.init_expr)) {
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

    n.SetType(n.init_expr->GetType());
  } else if (AST::typeof<UnknownType>(&n)) {
    // need type inference
    Error(n.LOC(),
          "`" + n.name_str +
              "' is declared without type annotation or initialization.");
    error_count++;
    return false;
  }

  AssignSymbolWithType(n.LOC(), n.name_str, n.GetType());

  if (Dump) {
    os << "Partial:   " << SSTab().InScopeName(n.name_str)
       << ", Type: " << AST::TYPE_STR(n) << "\n";
  }
  return true;
}

// ituple override operator "=" for definition
bool TypeInference::Visit(AST::Assignment &n) {
  __TRACE_EACH_VISIT__(n)
  if (SSTab().IsDeclared(n.name)) {
    Error(n.LOC(), "current choreo does not support symbol re-assignment.");
    error_count++;
  }
  if (isa<UnknownType>(n.value->GetType())) {
    Error(n.LOC(), "fail to deduce type of `" + n.name + "'.");
    error_count++;
  } else {
    auto ty = n.value->GetType();
    AssignSymbolWithType(n.LOC(), n.name, ty);

    if (Dump) {
      os << "Symbol:    " << SSTab().InScopeName(n.name)
         << ", Type: " << PSTR(ty) << "\n";
    }
  }
  return true;
}

bool TypeInference::Visit(AST::IntIndex &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::FunctionDecl &n) {
  __TRACE_EACH_VISIT__(n)
  cur_type.reset();
  std::vector<ptr<Type>> param_tys;
  for (auto &param : n.params->values) param_tys.emplace_back(param->GetType());

  n.SetType(MakeFunctionType(n.ret_type->GetType(), param_tys));
  if (!ModifySymbolType(n.LOC(), n.name, n.GetType())) return false;

  return true;
}

bool TypeInference::Visit(AST::Parameter &p) {
  __TRACE_EACH_VISIT__(p)
  // obtain its type
  p.SetType(p.type->GetType());

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

  if (Dump) {
    os << "Parameter: ";
    if (p.HasSymbol())
      os << SSTab().InScopeName(p.sym->name);
    else
      os << "(unnamed)";
    os << ", Type: " << AST::TYPE_STR(p) << "\n";
  }

  cur_type.reset();
  return true;
}

bool TypeInference::Visit(AST::ParamList &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::MultiDimSpans &n) {
  __TRACE_EACH_VISIT__(n)
  cur_type.reset();
  return true;
}

bool TypeInference::Visit(AST::Expr &n) {
  __TRACE_EACH_VISIT__(n)
  if (auto ref = n.GetReference()) {
    if (auto id = dyn_cast<AST::Identifier>(ref.get())) {
      if (auto pty = GetSymbolType(n.LOC(), id->name)) {
        // special handling of the span-of spanned type
        if (SuffixedWith(id->name, ".span")) {
          assert(isa<MDSpanType>(pty) && "incorrect type annotated.");
        }
        n.SetType(pty);
        return true;
      } else {
        Warning(n.LOC(),
                "symbol `" + id->name + "' is not associated with a type.");
        return false;
      }
    }

    // must have de-sugared early
    assert(!isa<AST::IntIndex>(ref.get()));

    if (AST::typeof<UnknownType>(ref.get())) {
      Error(n.LOC(), "unable to infer the type of expression.");
      error_count++;
      return false;
    }

    n.SetType(ref->GetType());
    return true;
  }

  if (n.GetForm() == AST::Expr::Unary) {
    if (n.op == "ubound") {
      auto id = cast<AST::Identifier>(n.GetR());
      if (auto bty =
              dyn_cast<BoundedITupleType>(GetSymbolType(id->LOC(), id->name)))
        n.SetType(MakeITupleType(bty->Dims()));
      else if (isa<BoundedIntegerType>(GetSymbolType(id->LOC(), id->name)))
        n.SetType(MakeIntegerType());
      else
        choreo_unreachable("ubound type '" + AST::TYPE_STR(n.GetR()) +
                           "' is unexpected.");
      return true;
    } else if (n.op == "sizeof") {
      n.SetType(MakeIntegerType());
      return true;
    } else if (n.op == "dataof") {
      auto ref = cast<AST::Expr>(n.GetR())->GetReference();
      auto id = cast<AST::Identifier>(ref);
      n.SetType(GetSymbolType(id->LOC(), id->name + ".data"));
      return true;
    }
    choreo_unreachable("type inference is yet to implement.");
  }

  if (n.GetForm() == AST::Expr::Binary) {
    if (n.op == "dimof") {
      n.SetType(MakeIntegerType());
      return true;
    }

    auto &pty_lhs = n.GetL()->GetType();
    auto &pty_rhs = n.GetR()->GetType();
    if ((isa<MDSpanType>(pty_lhs) && isa<ITupleType>(pty_rhs)) ||
        (isa<MDSpanType>(pty_rhs) && isa<ITupleType>(pty_lhs))) {
      if (n.op == "concat") {
        n.SetType(MakeMDSpanType(n.s));
        return true;
      }
      if (pty_lhs->Dims() == pty_rhs->Dims()) {
        n.SetType(MakeMDSpanType(n.s));  // note: the shape has been inferenced
        cur_type = n.GetType();
        return true;
      } else {
        Error(n.LOC(),
              "The operands of the expression be performed for inconsistant "
              "shape dimension.");
        error_count++;
        return false;
      }
    } else if (isa<MDSpanType>(pty_lhs) && isa<MDSpanType>(pty_rhs)) {
      if (n.op == "concat") {
        n.SetType(MakeMDSpanType(n.s));
        return true;
      }
      if (!((n.op == "/") || (n.op == "%") || (n.op == "cdiv"))) {
        Error(n.LOC(), "The operands of the div/mod expression cannot undergo '" +
                           n.op + "' operation.");
        error_count++;
        return false;
      } else if (pty_lhs->Dims() == pty_rhs->Dims()) {
        n.SetType(MakeITupleType(pty_lhs->Dims()));
        cur_type = n.GetType();
        return true;
      } else {
        Error(n.LOC(),
              "The operands of the expression be performed for inconsistant "
              "shape dimension.");
        error_count++;
        return false;
      }
    } else if (isa<ITupleType>(pty_rhs) && isa<ITupleType>(pty_lhs)) {
      if (n.op == "concat") {
        if (!cast<ITupleType>(pty_rhs)->IsDimValid() ||
            !cast<ITupleType>(pty_lhs)->IsDimValid())
          n.SetType(MakeUninitITupleType());
        n.SetType(MakeITupleType(pty_rhs->Dims() + pty_lhs->Dims()));
        return true;
      }
      if (pty_lhs->Dims() == pty_rhs->Dims()) {
        n.SetType(pty_rhs);
        cur_type = n.GetType();
        return true;
      } else {
        Error(n.LOC(),
              "The operands of the expression be performed for inconsistant "
              "shape dimension.");
        error_count++;
        return false;
      }
    } else if (isa<ITupleType>(pty_rhs) && isa<IntegerType>(pty_lhs)) {
      n.SetType(pty_rhs);
      cur_type = n.GetType();
    } else if (isa<ITupleType>(pty_lhs) && isa<IntegerType>(pty_rhs)) {
      n.SetType(pty_lhs);
      cur_type = n.GetType();
    } else if ((isa<MDSpanType>(pty_rhs) && isa<IntegerType>(pty_lhs)) ||
               (isa<MDSpanType>(pty_lhs) && isa<IntegerType>(pty_rhs))) {
      n.SetType(MakeMDSpanType(n.s));
      cur_type = n.GetType();
    } else if (isa<BoundedITupleType>(pty_lhs) && isa<IntegerType>(pty_rhs)) {
      n.SetType(pty_lhs);
      cur_type = n.GetType();
      // TODO(wsj) result type is?
      // bounded integer, lb and ub changed!
    } else if (*pty_lhs != *pty_rhs) {
      Error(n.LOC(), "The operands of the expression cannot undergo '" + n.op +
                         "' operation.");
      error_count++;
      return false;
    } else {
      n.SetType(n.GetR()->GetType());
      cur_type = n.GetType();
      return true;
    }
  }  // AST::Expr::Binary

  if (n.GetForm() == AST::Expr::Ternary) {
    choreo_unreachable("inference of ternary operation is not implemented.");
  }
  return true;
}

bool TypeInference::Visit(AST::IntTuple &n) {
  __TRACE_EACH_VISIT__(n)
  cur_type = n.GetType();
  return true;
}

bool TypeInference::Visit(AST::DMA &n) {
  __TRACE_EACH_VISIT__(n)

  // future's type has been obtained by shape inference
  if (AST::typeof<UnknownType>(&n)) {
    Error(n.LOC(), "fail to infer the FUTURE type of `" + n.future + "'.");
    error_count++;
    return false;
  }

  // update the future type. fill info including storage, fundanmental type
  auto fty = cast<FutureType>(n.GetType());
  auto sty = MakeSpannedType(dma_fmty, fty->GetShape(), dma_mem);
  auto nty = MakeFutureType(sty, fty->IsAsync());
  n.SetType(nty);

  if (!n.future.empty()) {
    AssignSymbolWithType(n.LOC(), n.future + ".span", sty->GetMDSpanType());
    AssignSymbolWithType(n.LOC(), n.future + ".data", sty);
    AssignSymbolWithType(n.LOC(), n.future, nty);
  }

  if (Dump) {
    os << "Future:    "
       << ((n.future.empty()) ? SSTab().ScopeName() + "(anon)"
                              : SSTab().InScopeName(n.future))
       << ", Type: " << AST::TYPE_STR(n) << "\n";
  }

  cur_type.reset();
  return true;
}

bool TypeInference::Visit(AST::ParallelBy &n) {
  __TRACE_EACH_VISIT__(n)
  AssignSymbolWithType(n.LOC(), n.biv, n.GetType());
  if (Dump) {
    os << "Bounded:   " << SSTab().InScopeName(n.biv)
       << ", Type: " << AST::TYPE_STR(n) << "\n";
  }
  return true;
}

bool TypeInference::Visit(AST::WhereBind &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::WithIn &n) {
  __TRACE_EACH_VISIT__(n)
  if (n.with) AssignSymbolWithType(n.LOC(), n.with->name, n.with->GetType());

  if (n.with_matchers) {
    for (auto pid : n.with_matchers->values) {
      auto id = cast<AST::Identifier>(pid.get());
      AssignSymbolWithType(n.LOC(), id->name, id->GetType());
    }
  }

  if (Dump) {
    if (n.with) {
      os << "Bounded:   ";
      os << SSTab().InScopeName(n.with->name)
         << ", Type: " << AST::TYPE_STR(*n.with) << "\n";
    }
    if (n.with_matchers) {
      for (auto pid : n.with_matchers->values) {
        auto id = cast<AST::Identifier>(pid);
        os << "Bounded:   " << SSTab().InScopeName(id->name)
           << ", Type: " << AST::TYPE_STR(*id) << "\n";
      }
    }
  }

  return true;
}

bool TypeInference::Visit(AST::WithBlock &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::Memory &n) {
  __TRACE_EACH_VISIT__(n)
  dma_mem = n.Get();
  return true;
}

bool TypeInference::Visit(AST::ChunkAt &n) {
  __TRACE_EACH_VISIT__(n)
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
    for (auto &v : n.positions->AllValues()) {
      v->SetType(NodeType(*v));
    }
  }
  // also update current node
  n.SetType(
      MakeSpannedType(fmty, cast<SpannedType>(n.GetType())->GetShape(), sto));

  return true;
}

bool TypeInference::Visit(AST::Wait &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::Call &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::Select &n) {
  __TRACE_EACH_VISIT__(n)
  auto& val = n.span_expr_list->AllValues()[0];
  auto sty = dyn_cast<SpannedType>(val->GetType());
  assert(sty);
  auto fmty = sty->ElementType();
  auto sto = sty->GetStorage();
  dma_mem = sto;
  dma_fmty = fmty;
  n.SetType(MakeSpannedType(fmty, sty->GetShape(), sto));
  cur_type = n.GetType();
  return true;
}

bool TypeInference::Visit(AST::Return &n) {
  __TRACE_EACH_VISIT__(n)

  if (!n.value) return true;  // void return;

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

  auto ty = GetSymbolType(n.LOC(), cur_func_name);
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
        if (*rty != *tty) {
          Error(n.LOC(),
                "return type inconsistant: " + STR(*rty) + " vs. " + STR(*tty));
          error_count++;
          return false;
        }
      } else {
        // supplement information, note global should be mapped back
        auto nty = MakeSpannedType(tty->ElementType(), tty->GetShape());
        ModifySymbolType(n.LOC(), cur_func_name,
                         MakeFunctionType(nty, fty->in_tys));
      }
    } else if (isa<UnknownType>(fty->out_ty)) {
      // the type must be inferenced
      if (auto tty = dyn_cast<SpannedType>(vty)) {
        // global should be mapped back
        auto nty = MakeSpannedType(tty->ElementType(), tty->GetShape());
        ModifySymbolType(n.LOC(), cur_func_name,
                         MakeFunctionType(nty, fty->in_tys));
      } else
        ModifySymbolType(n.LOC(), cur_func_name,
                         MakeFunctionType(vty, fty->in_tys));
    }
  }
  return true;
}

bool TypeInference::Visit(AST::LoopRange &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::ForeachBlock &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeInference::Visit(AST::ChoreoFunction &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeInference::Visit(AST::CppSourceCode &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeInference::Visit(AST::Program &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
