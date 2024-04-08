#include "typeinfer.hpp"

#include <iostream>

#include "ast.hpp"
#include "types.hpp"

using namespace Choreo;

bool TypeInference::BeforeVisit(AST::Node &n) {
  if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    EnterScope(f->name);
  } else if (isa<AST::ParallelBy>(&n)) {
    static size_t count = 0;
    EnterScope("paraby_" + std::to_string(count++));
  } else if (isa<AST::WithBlock>(&n)) {
    static size_t count = 0;
    EnterScope("within_" + std::to_string(count++));
  } else if (isa<AST::ForeachBlock>(&n)) {
    static size_t count = 0;
    EnterScope("foreach_" + std::to_string(count++));
  }

  return true;
}

bool TypeInference::AfterVisit(AST::Node &n) {
  if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
      isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
    LeaveScope();
  }
  return true;
}

bool TypeInference::AssignSymbolWithType(const location &loc,
                                         const std::string &sym,
                                         const ptr<Type> &ty) {
  if (!DefineSymbol(sym, ty)) {
    Error(loc, "symbol `" + sym + "' has already been associated with a type.");
    return false;
  }
  return true;
}

ptr<Type> TypeInference::GetSymbolType(const location &loc,
                                       const std::string &name) {
  if (!IsDeclared(name)) {
    Error(loc, "The symbol `" + name + "' has not been defined.");
    return nullptr;
  }
  if (auto *sym = LookupSymbol(name)) {
    return sym->GetType();
  } else {
    Error(loc, "symbol `" + name + "' is not associated with a type.");
    return nullptr;
  }
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
    cur_type = n.GetType();
    return true;
  }

  if (auto mdspan = dyn_cast<AST::MultiDimSpans>(n.mdspan_type.get())) {
    n.SetType(std::make_shared<SpannedType>(
        n.getFundamentalType(), *(cast<MDSpanType>(mdspan->GetType().get()))));
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

  AssignSymbolWithType(n.LOC(), n.name_str, n.GetType());

  if (Dump) {
    os << "Symbol:    " << *InScopeName(n.name_str) << ", Type: ";
    n.PrintType(os);
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
    os << "Partial:   " << *InScopeName(n.name_str) << ", Type: ";
    n.GetType()->Print(os);
    os << "\n";
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
  return true;
}

bool TypeInference::Visit(AST::Parameter &p) {
  __TRACE_EACH_VISIT__(p)
  // obtain its type
  p.SetType(p.type->GetType());

  if (p.HasSymbol()) {
    AssignSymbolWithType(p.LOC(), p.sym->name, p.GetType());
    if (!isa<UnknownType>(p.type->GetType().get()))
      AssignSymbolWithType(p.LOC(), p.sym->name + ".span", p.GetType());
  }

  // collect the parameter types
  cur_param_types.push_back(p.GetType());

  if (Dump) {
    os << "Parameter: ";
    if (p.HasSymbol())
      os << *InScopeName(p.sym->name);
    else
      os << "(unnamed)";
    os << ", Type: ";
    p.GetType()->Print(os);
    os << "\n";
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
  DefineSymbol(n.future->name, MakeFutureType());
  DefineSymbol(n.future->name + ".span", MakeUnknownType());
  return true;
}

bool TypeInference::Visit(AST::ParallelBy &n) {
  __TRACE_EACH_VISIT__(n)
  DefineSymbol(n.biv, n.GetType());
  if (Dump) {
    os << "Bounded:   ";
    os << *InScopeName(n.biv);
    os << ", Type: ";
    n.GetType()->Print(os);
    os << "\n";
  }
  return true;
}

bool TypeInference::Visit(AST::RequireBind &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeInference::Visit(AST::WithIn &n) {
  __TRACE_EACH_VISIT__(n)
  if (n.with) DefineSymbol(n.with->name, n.with->GetType());

  if (n.with_matchers) {
    for (auto pid : n.with_matchers->values) {
      auto id = cast<AST::Identifier>(pid.get());
      DefineSymbol(id->name, id->GetType());
    }
  }

  if (Dump) {
    if (n.with) {
      os << "Bounded:   ";
      os << *InScopeName(n.with->name);
      os << ", Type: ";
      n.with->GetType()->Print(os);
      os << "\n";
    }
    if (n.with_matchers) {
      for (auto pid : n.with_matchers->values) {
        auto id = cast<AST::Identifier>(pid.get());
        os << "Bounded:   ";
        os << *InScopeName(id->name);
        os << ", Type: ";
        id->GetType()->Print(os);
        os << "\n";
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
