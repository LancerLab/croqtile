#include "typecheck.hpp"

#include "aux.hpp"

using namespace Choreo;

#define __TRACE_EACH_VISIT__(n)                                                \
  if (trace_visit) {                                                           \
    os << n.TypeNameString() << ": ";                                          \
    os << "\n";                                                                \
  }

bool TypeChecker::BeforeVisitImpl(AST::Node&) { return true; }

bool TypeChecker::AfterVisitImpl(AST::Node&) { return true; }

bool TypeChecker::Visit(AST::MultiNodes& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::MultiValues& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::IntLiteral& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::Boolean& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::Expr& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::MultiDimSpans& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::NamedTypeDecl& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::NamedVariableDecl& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::IntTuple& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool TypeChecker::Visit(AST::Assignment& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (!ReportUnknownSymbol(n.name, n.LOC(), __FILE__, __LINE__)) return false;

  if ((*GetSymbolType(n.name) != *NodeType(*n.value)) ||
      (*NodeType(n) != *NodeType(*n.value))) {
    Error(n.LOC(), "inconsistent types are found in the assignment.");
    error_count++;
    return false;
  }

  return true;
}

bool TypeChecker::Visit(AST::IntIndex& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (!isa<IntegerType>(n.value->GetType())) {
    Error(n.LOC(), "Expect `" + PSTR(n.value) + "' to be a integer type.");
    error_count++;
    return false;
  }
  return true;
}
bool TypeChecker::Visit(AST::DataType& n) {
  __TRACE_EACH_VISIT__(n)
  // TODO: figure out if we could check SufficientInfo
  if (!ReportUnknown(n, __FILE__, __LINE__, true)) return false;
  return true;
}
bool TypeChecker::Visit(AST::Identifier& n) {
  __TRACE_EACH_VISIT__(n)
  if (PrefixedWith(n.name, "$")) return true; // do not check internal symbols
  if (!ReportUnknownSymbol(n.name, n.LOC(), __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::Parameter& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::ParamList& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::ParallelBy& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::WhereBind& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::WithIn& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::WithBlock& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Memory& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeChecker::Visit(AST::SpanAs& n) {
  __TRACE_EACH_VISIT__(n)

  if (!ReportUnknownSymbol(n.id->name, n.LOC(), __FILE__, __LINE__))
    return false;

  auto ity = GetSymbolType(n.id->name);

  if (!(isa<SpannedType>(ity) || isa<FutureType>(ity))) {
    Error(n.LOC(), "Expect symbol `" + n.id->name + "' to be a spanned type.");
    error_count++;
    return false;
  }

  if (!(AST::typeof<SpannedType>(&n))) {
    Error(n.LOC(), "Invalid type of span_as expression.");
    error_count++;
    return false;
  }

  SpannedType* sty = nullptr;
  if (auto fty = dyn_cast<FutureType>(ity))
    sty = fty->GetSpannedType().get();
  else
    sty = cast<SpannedType>(ity);

  auto nty = cast<SpannedType>(NodeType(n));

  if (sty->ElementType() != nty->ElementType()) {
    Error(n.LOC(), "Inconsistent element type: (" + STR(nty->ElementType()) +
                       " = span_as(" + STR(sty->ElementType()) + ".");
    error_count++;
    return false;
  }

  if (!sty->RuntimeShaped() && !nty->RuntimeShaped()) {
    // check if the shape size are same
    if (sty->ShapeSize() != nty->ShapeSize()) {
      Error(n.LOC(), "Inconsistent mdspan size: " + n.id->name + "(" +
                         STR(sty->ShapeSize()) + ") = spanas (" + n.nid->name +
                         "(" + STR(nty->ShapeSize()) + ")).");
      error_count++;
      return false;
    }
  }

  return true;
}

bool TypeChecker::Visit(AST::DMA& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  bool IsDummy = (n.operation == ".any");
  auto ty = n.GetType();

  if (IsDummy) {
    if (n.future.empty()) {
      Error(n.LOC(), "A dummy/async DMA must be named.");
      error_count++;
      return false;
    }
    if (!isa<PlaceHolderType>(ty) ||
        (cast<PlaceHolderType>(ty)->Category() != TypeCategory::FUTURE)) {
      Error(n.LOC(), "Expect a placeholder type but got '" + PSTR(ty) + "'.");
      error_count++;
      return false;
    }
    return true;
  }

  if (!isa<FutureType>(ty)) {
    Error(n.LOC(), "Expect the DMA to produce a FutureType, but got '" +
                       PSTR(n.GetType()) + "'.");
    error_count++;
  }

  if (cast<FutureType>(ty)->IsAsync() && n.future.empty()) {
    Error(n.LOC(), "A dummy/async DMA must be named.");
    error_count++;
  }

  if (!isa<AST::ChunkAt>(n.from) || !isa<SpannedType>(n.from->GetType())) {
    Error(n.LOC(),
          "The 'from' of DMA is not as expected: " + n.from->TypeNameString() +
              "(" + PSTR(n.from->GetType()) + ").");
    error_count++;
  }

  if (!isa<AST::ChunkAt>(n.to) || !isa<SpannedType>(n.to->GetType())) {
    Error(n.LOC(),
          "The 'to' of DMA is not as expected: " + n.to->TypeNameString() +
              "(" + PSTR(n.from->GetType()) + ").");
    error_count++;
  }

  auto& fty = n.from->GetType();
  auto& tty = n.to->GetType();

  if (n.operation == ".transp") {
    // no transposed shape need to be generated
    // do DataEqual() manually
    auto tc = cast<TransposeConfig>(n.config);
    auto sfty = cast<SpannedType>(fty);
    auto stty = cast<SpannedType>(tty);
    auto f_shape = sfty->GetShape();
    auto t_shape = stty->GetShape();
    if (sfty->f_type != stty->f_type ||
        f_shape.DimCount() != t_shape.DimCount()) {
      Error(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                         ") with " + PSTR(tc) + " and 'to'(" + PSTR(tty) +
                         ").");
      error_count++;
    } else {
      auto& dim_values = tc->dim_values;
      for (size_t i = 0; i < dim_values.size(); ++i) {
        if (!IsValueItemEqual(f_shape.ValueAt(dim_values[i]),
                              t_shape.ValueAt(i))) {
          Error(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                             ") with " + PSTR(tc) + " and 'to'(" + PSTR(tty) +
                             ").");
          error_count++;
          break;
        }
      }
    }
  } else if (!(cast<SpannedType>(fty)->DataEqual(*tty))) {
    Error(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                       ") and 'to'(" + PSTR(tty) + ").");
    error_count++;
  }

  return true;
}
bool TypeChecker::Visit(AST::ChunkAt& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::Wait& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Call& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeChecker::Visit(AST::Swap& n) {
  __TRACE_EACH_VISIT__(n)
  auto lty = NodeType(*n.lhs);
  auto rty = NodeType(*n.rhs);

  if (*lty != *rty) {
    Error(n.LOC(), "swapping data of different types (" + PSTR(lty) + " vs. " +
                       PSTR(rty));
    error_count++;
  }

  return true;
}

bool TypeChecker::Visit(AST::Select& n) {
  __TRACE_EACH_VISIT__(n)
  size_t ec = error_count;

  if (!isa<IntegerType>(NodeType(*n.select_factor))) {
    ++error_count;
    Error(n.select_factor->LOC(), "Expect " + PSTR(n.select_factor) +
                                      " to be an integer type but got " +
                                      PSTR(NodeType(*n.select_factor)) + ".");
  }

  auto expr_list = n.expr_list;
  auto expr0 = expr_list->ValueAt(0);
  if (!isa<FutureType>(NodeType(*expr0)) &&
      !isa<SpannedType>(NodeType(*expr0))) {
    ++error_count;
    Error(expr0->LOC(),
          "Expect " + PSTR(expr0) + " to be a future/spanned type.");
    return false;
  }

  for (auto expr : expr_list->AllValues()) {
    if (*NodeType(*expr) == *NodeType(*expr0)) continue;

    ++error_count;
    Error(expr->LOC(), "Type mismatch inside SELECT: " + PSTR(expr) + "(" +
                           TYPE_STR(expr) + ") vs. " + PSTR(expr0) + "(" +
                           TYPE_STR(expr0) + ").");
  }

  return ec == error_count;
}

bool TypeChecker::Visit(AST::Return& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::LoopRange& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::ForeachBlock& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::FunctionDecl& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::ChoreoFunction& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::CppSourceCode& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Program& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool TypeChecker::ReportUnknownSymbol(const std::string& name,
                                      const location& loc, const char* file,
                                      int line) {
  if (isa<UnknownType>(GetSymbolType(name))) {
    ++error_count;
    Error(loc, "failed to obtain the type of " + name + ".");
    if (trace_visit) os << file << ":" << line << "\n";
    return false;
  }
  return true;
}

bool TypeChecker::ReportUnknown(AST::Node& n, const char* file, int line,
                                bool ignore_detail) {
  if (isa<UnknownType>(NodeType(n))) {
    ++error_count;
    Error(n.LOC(), "failed to obtain a type.");
    if (trace_visit) os << file << ":" << line << ", " << STR(n) << "\n";
    return false;
  }

  if (!ignore_detail && !NodeType(n)->HasSufficientInfo()) {
    ++error_count;
    Error(n.LOC(), "failed to obtain a type with sufficient info.");
    if (trace_visit)
      os << file << ":" << line << ", " << STR(n) << "(" << PSTR(NodeType(n))
         << ")\n";
    return false;
  }

  return true;
}

bool TypeChecker::HasError() {
  if (error_count) {
    os << "Totally " << error_count << " errors have been detected.\n";
    return true;
  }
  return false;
}
