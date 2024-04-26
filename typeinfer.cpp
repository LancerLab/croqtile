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
    auto func_ty = cast<FunctionType>(sym_ty);
    if (isa<SpannedType>(func_ty->out_ty)) {
      // update the return type node since type inference could have changed the
      // function type already
      f->f_decl.ret_type->SetType(func_ty->out_ty);
      f->f_decl.SetType(sym_ty);
      f->SetType(sym_ty);
    }
  }
  return true;
}

bool TypeInference::AssignSymbolWithType(const location &loc,
                                         const std::string &sym,
                                         const ptr<Type> &ty) {
  if (!SSTab().DefineSymbol(sym, ty)) {
    Error(loc, "symbol `" + sym + "' has already been associated with a type.");
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
    return nullptr;
  }
  if (auto pty = SSTab().LookupSymbol(name)) {
    return pty;
  } else {
    Error(loc, "symbol `" + name + "' is not associated with a type.");
    return nullptr;
  }
}

bool TypeInference::ModifySymbolType(const location &loc,
                                     const std::string &name,
                                     const ptr<Type> &ty) {
  if (!SSTab().IsDeclared(name)) {
    Error(loc, "The symbol `" + name + "' has not been defined.");
    return false;
  }
  if (!SSTab().ModifySymbolType(name, ty)) {
    Error(loc, "symbol `" + name + "' is not associated with a type.");
    return false;
  }

  if (trace_visit)
    os << "Modify symbol `" << name << "` with type: " << STR(*ty) << "\n";

  return true;
}

bool TypeInference::SetCurrentType(AST::Node &nd, const std::string &n) {
  const auto ty = nd.GetType();
  if (ty->HasSufficientInfo()) {
    // already has a type with sufficient info, check for consistence.
    if (cur_type->HasSufficientInfo() && !(*cur_type == *ty)) {
      Error(nd.LOC(), "can not infer the type of `" + n + "'.");
      return false;
    } else
      return true;
  }

  // Or else we need to set the type with current
  // Check for inference failures
  if (isa<UnknownType>(cur_type.get())) {
    Error(nd.LOC(), "can not infer the type of `" + n + "'.");
    return false;
  }

  if (!cur_type->HasSufficientInfo()) {
    Error(nd.LOC(), "can not infer '" + cur_type->Name() +
                        "' type detail of symbol `" + n + "'.");
    return false;
  }

  // complement the storage information when exists
  if (auto st = dyn_cast<SpannedType>(cur_type))
    if (auto n = dyn_cast<AST::NamedVariableDecl>(&nd))
      st->SetStorage(n->mem->st);

  // The type is successfully inferred, set the node
  nd.SetType(cur_type);

  return true;
}

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
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
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::DataType &n) {
  __TRACE_EACH_VISIT__(n)
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
  return true;
}

bool TypeInference::Visit(AST::NamedVariableDecl &n) {
  __TRACE_EACH_VISIT__(n)

  if (!SetCurrentType(n, n.name_str)) {
    cur_type.reset();
    return false;
  }

  cur_type.reset();

  if (AST::typeof<UnknownType>(&n)) {
    Error(n.LOC(), "can not infer the type of `" + n.name_str + "'.");
    return false;
  }

  AssignSymbolWithType(n.LOC(), n.name_str, n.GetType());

  if (AST::typeof<SpannedType>(&n))
    AssignSymbolWithType(n.LOC(), n.name_str + ".span",
                         cast<SpannedType>(n.GetType())->GetMDSpanType());

  if (Dump) {
    os << "Symbol:    " << SSTab().InScopeName(n.name_str)
       << ", Type: " << AST::TYPE_STR(n);
    if (n.mem) os << ", Storage: " << AST::STR(*n.mem);
    os << "\n";
  }

  return true;
}

bool TypeInference::Visit(AST::NamedTypeDecl &n) {
  __TRACE_EACH_VISIT__(n)
  if (AST::typeof<UnknownType>(&n)) {
    // need type inference
    if (!n.init_expr) {
      Error(n.LOC(),
            "`" + n.name_str +
                "' is declared without type annotation or initialization.");
      return false;
    }

    if (AST::typeof<UnknownType>(n.init_expr.get())) {
      Error(n.LOC(), "unable to inference the type of `" + n.name_str + "'.");
      return false;
    }

    if (!n.init_expr->GetType()->HasSufficientInfo()) {
      Error(n.LOC(),
            "unable to inference the type detail of `" + n.name_str + "'.");
      return false;
    }

    n.SetType(n.init_expr->GetType());
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
      return false;
    }

    AssignSymbolWithType(p.LOC(), p.sym->name, p.GetType());
    AssignSymbolWithType(p.LOC(), p.sym->name + ".span", p.type->GetType());
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
  //  assert(!cur_mdspan_value.IsValid() && "Expecting null mdspan value.");
  //  cur_mdspan_value = mds.MakeValueList();
  return true;
}

bool TypeInference::Visit(AST::Expr &n) {
  __TRACE_EACH_VISIT__(n)
  if (auto ref = n.GetReference()) {
    if (auto id = dyn_cast<AST::Identifier>(ref.get())) {
      if (auto pty = GetSymbolType(n.LOC(), id->name)) {
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
      return false;
    }

    n.SetType(ref->GetType());
    return true;
  }

  if (n.t == AST::Expr::Binary) {
    if (n.op == "dimof" || n.op == "sizeof") {
      n.SetType(MakeIntegerType());
      return true;
    }

    if (*n.value_r->GetType() != *n.value_l->GetType()) {
      Error(n.LOC(), "binary expression with different operand type.");
      return false;
    }
    n.SetType(n.value_r->GetType());
    return true;
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
    return false;
  }

  AssignSymbolWithType(n.LOC(), n.future, n.GetType());
  auto s = cast<FutureType>(n.GetType())->GetShape();
  AssignSymbolWithType(n.LOC(), n.future + ".span", MakeMDSpanType(s));

  if (Dump) {
    os << "Future:    " << SSTab().InScopeName(n.future)
       << ", Type: " << AST::TYPE_STR(n) << "\n";
  }

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

bool TypeInference::Visit(AST::RequireBind &n) {
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
        auto id = cast<AST::Identifier>(pid.get());
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
  return true;
}

bool TypeInference::Visit(AST::ChunkAt &n) {
  __TRACE_EACH_VISIT__(n)
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

bool TypeInference::Visit(AST::Return &n) {
  __TRACE_EACH_VISIT__(n)

  if (!n.value) return true;

  ptr<Type> vty = n.value->GetType();
  if (auto ref = cast<AST::Expr>(n.value)->GetReference())
    if (auto id = dyn_cast<AST::Identifier>(ref))
      vty = GetSymbolType(n.LOC(), id->name);

  // get the value's type
  if (isa<UnknownType>(vty)) {
    Error(n.LOC(), "failed to inference the type of " + AST::STR(*n.value));
    return false;
  }

  auto ty = GetSymbolType(n.LOC(), cur_func_name);
  if (auto fty = dyn_cast<FunctionType>(ty)) {
    if (auto rty = dyn_cast<SpannedType>(fty->out_ty)) {
      auto tty = cast<SpannedType>(vty);
      if (!tty->HasSufficientInfo()) {
        Error(n.LOC(),
              "failed to inference the type detail of " + AST::STR(*n.value));
        return false;
      }
      if (rty->Dims() != tty->Dims()) {
        Error(n.LOC(),
              "return type inconsistant: " + STR(*rty) + " vs. " + STR(*tty));
        return false;
      }

#if 0
      // TODO: why triggers?
      // already has sufficient info, make a comparison to avoid inconsistent return type
      if (rty->HasSufficientInfo()) {
        if (*rty != *tty) {
          Error(n.LOC(), "return type inconsistant: " + STR(*rty) + " vs. " + STR(*tty));
          return false;
        }
      } else
#endif
      ModifySymbolType(n.LOC(), cur_func_name,
                       MakeFunctionType(vty, fty->in_tys));
    }
  }
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
