#include "semacheck.hpp"
#include "types.hpp"

using namespace Choreo;

bool SemaChecker::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) pending_futures.clear();
  return true;
}

bool SemaChecker::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) {
    if (!pending_futures.empty()) {
      Error(n.LOC(), "some futures are not explicitly waited: " +
                         DelimitedString(pending_futures) + ".");
      error_count++;
    }
  }
  return true;
}

bool SemaChecker::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::FloatLiteral& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::Boolean& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::MultiDimSpans& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::IntTuple& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::Visit(AST::Assignment& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (!ReportUnknownSymbol(n.name, n.LOC(), __FILE__, __LINE__)) return false;

  if ((*GetSymbolType(n.name) != *NodeType(*n.value)) ||
      (*NodeType(n) != *NodeType(*n.value))) {
    dbgs() << STR(*GetSymbolType(n.name)) << STR(*NodeType(*n.value))
           << STR(*NodeType(n));

    Error(n.LOC(), "inconsistent types are found in the assignment.");
    error_count++;
    return false;
  }

  return true;
}

bool SemaChecker::Visit(AST::IntIndex& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (!isa<IntegerType>(n.value->GetType())) {
    Error(n.LOC(), "Expect `" + PSTR(n.value) + "' to be a integer type.");
    error_count++;
    return false;
  }
  return true;
}
bool SemaChecker::Visit(AST::DataType& n) {
  TraceEachVisit(n);
  // TODO: figure out if we could check SufficientInfo
  if (!ReportUnknown(n, __FILE__, __LINE__, true)) return false;
  return true;
}
bool SemaChecker::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  if (PrefixedWith(n.name, "$")) return true; // do not check internal symbols
  if (n.name == "_") return true;             // ignore unit biv
  if (!ReportUnknownSymbol(n.name, n.LOC(), __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::Parameter& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::Visit(AST::ParamList& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  if (auto shape = GetShape(NodeType(n)); shape.IsDynamic()) {
    std::string mds = STR(shape);
    auto mds_vals = SplitStringByDelimiter(mds.substr(1, mds.size() - 2), ", ");
    int idx = 1;
    for (auto& mds_val : mds_vals) {
      std::string lhs, op, rhs, message;
      lhs = mds_val;
      op = ">";
      rhs = "0";
      message =
          "The " + Ordinal(idx) +
          " bound item of parallelby is invalid: should be greater than 0";
      FCtx(fname).AppendRtCheck(
          {lhs, op, rhs, n.bounds->ValueAt(idx - 1)->LOC(), message, {}});
      ++idx;
    }
  }

  return true;
}
bool SemaChecker::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  if (auto shape = GetShape(NodeType(*n.in)); shape.IsDynamic()) {
    std::string mds = STR(shape);
    auto mds_vals = SplitStringByDelimiter(mds.substr(1, mds.size() - 2), ", ");
    int idx = 1;
    for (auto& mds_val : mds_vals) {
      std::string lhs, op, rhs, message;
      lhs = mds_val;
      op = "!=";
      rhs = "0";
      message = "zero is detected for the " + Ordinal(idx) +
                " dim of the mdspan inside the with-in statement",
      FCtx(fname).AppendRtCheck({lhs, op, rhs, n.LOC(), message, {}});
      ++idx;
    }
  }
  return true;
}
bool SemaChecker::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::Memory& n) {
  TraceEachVisit(n);
  return true;
}

bool SemaChecker::Visit(AST::SpanAs& n) {
  TraceEachVisit(n);

  if (!ReportUnknownSymbol(n.id->name, n.LOC(), __FILE__, __LINE__))
    return false;

  auto ity = GetSymbolType(n.id->name);

  if (!(isa<SpannedType>(ity) || isa<FutureType>(ity))) {
    Error(n.LOC(), "Expect symbol `" + n.id->name + "' to be a spanned type.");
    error_count++;
    return false;
  }

  if (!(AST::istypeof<SpannedType>(&n))) {
    Error(n.LOC(), "Invalid type of span_as expression.");
    error_count++;
    return false;
  }

  auto sty = GetSpannedType(ity);
  auto nty = cast<SpannedType>(NodeType(n));

  if (sty->ElementType() != nty->ElementType()) {
    Error(n.LOC(), "Inconsistent element type: (" + STR(nty->ElementType()) +
                       " = span_as(" + STR(sty->ElementType()) + ".");
    error_count++;
    return false;
  }

  if (!sty->RuntimeShaped() && !nty->RuntimeShaped()) {
    // check if the shape size are same
    if (sty->ElementCount() != nty->ElementCount()) {
      Error(n.LOC(), "Inconsistent mdspan size: " + n.id->name + "(" +
                         STR(sty->ElementCount()) + ") = spanas (" +
                         n.nid->name + "(" + STR(nty->ElementCount()) + ")).");
      error_count++;
      return false;
    }
  }

  return true;
}

bool SemaChecker::Visit(AST::DMA& n) {
  TraceEachVisit(n);

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

  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  if (!isa<FutureType>(ty)) {
    Error(n.LOC(), "Expect the DMA to produce a FutureType, but got '" +
                       PSTR(n.GetType()) + "'.");
    error_count++;
  }

  if (cast<FutureType>(ty)->IsAsync() && n.future.empty()) {
    Error(n.LOC(), "A dummy/async DMA must be named.");
    error_count++;
  }

  if (!n.future.empty() && cast<FutureType>(ty)->IsAsync())
    pending_futures.insert(InScopeName(n.future));
  if (!n.chain_from.empty()) pending_futures.erase(InScopeName(n.chain_from));

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
    // do LogicalEqual() manually
    auto tc = cast<TransposeConfig>(n.config);
    auto sfty = cast<SpannedType>(fty);
    auto stty = cast<SpannedType>(tty);
    auto f_shape = sfty->GetShape();
    auto t_shape = stty->GetShape();
    if (sfty->f_type != stty->f_type || !f_shape.SameRankAs(t_shape)) {
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
  } else if (n.operation == ".pad") {
    // no padded shape need to be generated
    // do LogicalEqual() manually
    auto pc = cast<PadConfig>(n.config);
    auto sfty = cast<SpannedType>(fty);
    auto stty = cast<SpannedType>(tty);
    auto f_shape = sfty->GetShape();
    auto t_shape = stty->GetShape();
    if (sfty->f_type != stty->f_type || !f_shape.SameRankAs(t_shape)) {
      Error(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                         ") with " + PSTR(pc) + " and 'to'(" + PSTR(tty) +
                         ").");
      error_count++;
    } else {
      size_t dim_count = f_shape.DimCount();
      auto clampLongToInt = [](long value) {
        return static_cast<int>(std::clamp(
            value, static_cast<long>(std::numeric_limits<int>::min()),
            static_cast<long>(std::numeric_limits<int>::max())));
      };
      for (size_t i = 0; i < dim_count; ++i) {
        size_t pad_length = pc->pad_high[i] + pc->pad_low[i] + pc->pad_mid[i];
        if (!IsValueItemEqual(f_shape.ValueAt(i) +
                                  ValueItem(clampLongToInt(pad_length)),
                              t_shape.ValueAt(i))) {
          Error(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                             ") with " + PSTR(pc) + " and 'to'(" + PSTR(tty) +
                             ").");
          error_count++;
          break;
        }
      }
    }
  } else if (!(cast<SpannedType>(fty)->LogicalEqual(*tty)) &&
             !allow_auto_threading) {
    Error(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                       ") and 'to'(" + PSTR(tty) + ").");
    error_count++;
  }

  return true;
}

bool SemaChecker::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  for (auto& f : n.GetFutures()) {
    auto fty = NodeType(*f);
    if (!isa<FutureType>(fty)) {
      Error(n.LOC(),
            "Wait for a non-future type " + PSTR(f) + "(" + PSTR(fty) + ").");
      error_count++;
    } else if (auto id = AST::GetIdentifier(*f))
      pending_futures.erase(InScopeName(id->name));
  }

  return true;
}

bool SemaChecker::Visit(AST::Call& n) {
  TraceEachVisit(n);

  if (n.template_args) {
    size_t count = 0;
    for (auto& v : n.template_args->AllValues()) {
      count++;
      auto ty = NodeType(*v);
      // must be a scalar type
      if (!CanYieldAnInteger(ty)) {
        Error(n.LOC(),
              "The " + Ordinal(count) + " template argument of type '" +
                  PSTR(ty) +
                  "` can not be used to instantiate the kernel function.");
        error_count++;
      }
      auto val_expr = cast<AST::Expr>(v)->opt_vals.int_expr;
      // fail if the template argument can not be evaluated as a compile-time
      // constant
      if (!IsValidValueItem(val_expr) || !isa<int>(&val_expr)) {
        Error(n.LOC(), "The " + Ordinal(count) +
                           " template argument of type '" + PSTR(ty) +
                           "` can not be evaluated at choreo compile time.");
        error_count++;
      }
    }
  }

  return true;
}

bool SemaChecker::Visit(AST::Rotate& n) {
  TraceEachVisit(n);
  size_t index = 0;
  for (auto s : n.ids->AllValues()) {
    if (auto id = AST::GetIdentifier(*s))
      pending_futures.erase(InScopeName(id->name));

    if (index == 0) {
      index++;
      continue;
    }
    auto lty = NodeType(*n.ids->ValueAt(index - 1));
    auto rty = NodeType(*n.ids->ValueAt(index));
    if (*lty != *rty) {
      Error(n.LOC(), "swapping values of different types (" + PSTR(lty) +
                         " vs. " + PSTR(rty));
      error_count++;
    }

    auto lid = AST::GetIdentifier(*n.ids->ValueAt(index - 1));
    auto rid = AST::GetIdentifier(*n.ids->ValueAt(index));
    assert(lid && rid && "no idendifier is found.");
    auto l_scope = GetScope(InScopeName(lid->name));
    auto r_scope = GetScope(InScopeName(rid->name));
    if (l_scope != r_scope) {
      Error(n.LOC(),
            "swapping values defined in different scopes is forbidden (" +
                InScopeName(lid->name) + " vs. " + InScopeName(rid->name));
      error_count++;
    }

    index++;
  }
  return true;
}

bool SemaChecker::Visit(AST::Select& n) {
  TraceEachVisit(n);
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
    if (auto id = AST::GetIdentifier(*expr))
      pending_futures.erase(InScopeName(id->name)); // can not check statically

    if (*NodeType(*expr) == *NodeType(*expr0)) continue;

    ++error_count;
    Error(expr->LOC(), "Type mismatch inside SELECT: " + PSTR(expr) + "(" +
                           TYPE_STR(expr) + ") vs. " + PSTR(expr0) + "(" +
                           TYPE_STR(expr0) + ").");
  }

  return ec == error_count;
}

bool SemaChecker::Visit(AST::Return& n) {
  TraceEachVisit(n);

  if (n.value) {
    auto vty = NodeType(*n.value);
    if (!(isa<SpannedType>(vty) || isa<ScalarType>(vty))) {
      Error(n.LOC(),
            "returning value with type '" + PSTR(vty) + "' is not supproted.");
      error_count++;
      return false;
    }
  }

  return true;
}
bool SemaChecker::Visit(AST::LoopRange& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);
  return true;
}
bool SemaChecker::Visit(AST::Program& n) {
  TraceEachVisit(n);
  return true;
}

bool SemaChecker::ReportUnknownSymbol(const std::string& name,
                                      const location& loc, const char* file,
                                      int line) {
  if (isa<UnknownType>(GetSymbolType(name))) {
    ++error_count;
    Error(loc, "failed to obtain the type of " + name + ".");
    if (debug_visit) dbgs() << file << ":" << line << "\n";
    return false;
  }
  return true;
}

bool SemaChecker::ReportUnknown(AST::Node& n, const char* file, int line,
                                bool ignore_detail) {
  if (STR(n) == "_") return true; // ignore built-in unit iv.
  if (isa<UnknownType>(NodeType(n))) {
    ++error_count;
    Error(n.LOC(), "failed to obtain a type.");
    if (debug_visit) dbgs() << file << ":" << line << ", " << STR(n) << "\n";
    return false;
  }

  // dbgs() << "checking node = " << STR(n) << "\n";
  // dbgs() << "checking node = " << PSTR(NodeType(n)) << "\n";
  // dbgs() << "ignore_detail = " << ignore_detail << "\n";
  // dbgs() << "has-sufficient-info = " << NodeType(n)->HasSufficientInfo() <<
  // "\n";
  if (!ignore_detail && !NodeType(n)->HasSufficientInfo()) {
    ++error_count;
    Error(n.LOC(), "failed to obtain a type with sufficient info.");
    if (debug_visit)
      dbgs() << file << ":" << line << ", " << STR(n) << "("
             << PSTR(NodeType(n)) << ")\n";
    return false;
  }

  return true;
}

bool SemaChecker::HasError() {
  if (error_count) {
    dbgs() << "Totally " << error_count << " errors have been detected.\n";
    return true;
  }
  return false;
}
