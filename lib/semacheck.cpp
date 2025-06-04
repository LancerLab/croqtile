#include "semacheck.hpp"
#include "types.hpp"

using namespace Choreo;

bool SemaChecker::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) {
    pending_async.clear();
    waited_async.clear();
  }
  return true;
}

bool SemaChecker::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) {
    for (auto n : waited_async) pending_async.erase(n);

    if (!pending_async.empty()) {
      Error(n.LOC(), "some asyncs are not explicitly waited: " +
                         DelimitedString(pending_async) + ".");
      error_count++;
    }
  }
  return true;
}

bool SemaChecker::VisitNode(AST::IntLiteral& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::VisitNode(AST::FloatLiteral& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::VisitNode(AST::BoolLiteral& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::VisitNode(AST::Expr& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  // check out-of-bound for the elemof operation in wait or trigger.
  // note: elemof in chunkat is not Expr node, so we do not check it here.
  if (n.op == "elemof") {
    auto arr_sym = GetArrayBaseSymbol(n);
    size_t subscription_level = GetSubScriptLevel(n);
    // access: events[a][b][c]
    // level:         1  2  3
    auto ty = NodeType(*arr_sym);
    auto arr_ty = cast<ArrayType>(ty);
    assert(arr_ty && "expect the array symbol to be an array type.");

    size_t arr_rank = arr_ty->ArrayRank();
    if (subscription_level > arr_rank) {
      Error(n.LOC(), "Invalid array access: expected " +
                         std::to_string(arr_rank) + " dimensions, but " +
                         std::to_string(subscription_level) + " were used.");
      error_count++;
      return false;
    }

    auto dims = arr_ty->Dimensions();
    int bound = arr_ty->Dimension(subscription_level - 1);
    auto idx = n.GetR();

    // TODO: parallel p by 2 { xxx; dma arr[p] => local; }
    // if the index is a bounded var, hard to determine if it is out of bound
    if (isa<BoundedType>(NodeType(*idx))) return true;

    // using shape info to check the index
    auto shape = cast<AST::Expr>(idx)->s;
    // if the shape is not valid, it means the index is mutable!
    if (!shape.IsValid()) return true;
    assert(shape.DimCount() == 1);

    if (shape.IsDynamic()) {
      std::string shape_str = STR(shape);
      std::string idx_str = shape_str.substr(1, shape_str.size() - 2);
      std::string lhs, op, rhs, message;
      lhs = idx_str;
      op = "<";
      rhs = std::to_string(bound);
      message = "Index " + lhs + " is out of bounds of the " +
                Ordinal(subscription_level) + " dimension of array '" +
                PSTR(arr_sym) + "', where the valid range is [0, " + rhs + ").";
      FCtx(fname).AppendRtCheck({lhs, op, rhs, idx->LOC(), message, {}});
      FCtx(fname).AppendRtCheck({lhs, ">=", "0", idx->LOC(), message, {}});
    } else {
      auto idx_val = VIInt(shape.ValueAt(0));
      assert(idx_val);
      if (*idx_val < 0 || *idx_val >= bound) {
        Error(idx->LOC(), "Index " + std::to_string(*idx_val) +
                              " is out of bounds of the " +
                              Ordinal(subscription_level) +
                              " dimension of array '" + PSTR(arr_sym) +
                              "', where the valid range is [0, " +
                              std::to_string(bound) + ").");
        error_count++;
        return false;
      }
    }
  }

  return true;
}
bool SemaChecker::VisitNode(AST::MultiDimSpans& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::VisitNode(AST::NamedTypeDecl& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool SemaChecker::VisitNode(AST::NamedVariableDecl& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  auto ty = NodeType(n);
  auto s = GetShape(ty);
  if (s.IsValid()) {
    for (auto sv : s.Value())
      if (*sv == *sbe::nu(0)) {
        Error(n.LOC(), "found 0-dimension within the shape of variable `" +
                           n.name_str + "'.");
        error_count++;
      }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::IntTuple& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::DataAccess& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (n.AccessElement() &&
      !ReportUnknownSymbol(n.GetDataName(), n.LOC(), __FILE__, __LINE__))
    return false;

  // TODO: static out-of-bound check
  // data.at(xxx, xxx) or future.data.at(xxx, xxx)

  return true;
}

bool SemaChecker::VisitNode(AST::Assignment& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return false;

  if ((*NodeType(n) != *NodeType(*n.value))) {
    Error(n.LOC(), "inconsistent types are found in the assignment: " +
                       STR(*NodeType(*n.value)) + " vs. " + STR(*NodeType(n)) +
                       ".");
    error_count++;
    return false;
  }

  return true;
}

bool SemaChecker::VisitNode(AST::IntIndex& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (!isa<IntegerType>(n.value->GetType())) {
    Error(n.LOC(), "Expect `" + PSTR(n.value) + "' to be a integer type.");
    error_count++;
    return false;
  }
  return true;
}
bool SemaChecker::VisitNode(AST::DataType& n) {
  // TODO: figure out if we could check SufficientInfo
  if (!ReportUnknown(n, __FILE__, __LINE__, true)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::Identifier& n) {
  if (PrefixedWith(n.name, "$")) return true; // do not check internal symbols
  if (n.name == "_") return true;             // ignore unit bpv
  if (!ReportUnknownSymbol(n.name, n.LOC(), __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::Parameter& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::ParallelBy& n) {
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
          {lhs, op, rhs, n.cmpt_bounds->ValueAt(idx - 1)->LOC(), message, {}});
      ++idx;
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::WithIn& n) {
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

bool SemaChecker::VisitNode(AST::SpanAs& n) {
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

bool SemaChecker::VisitNode(AST::DMA& n) {
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
    pending_async.insert(InScopeName(n.future));
  if (!n.chain_from.empty()) waited_async.insert(InScopeName(n.chain_from));

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
                                  sbe::nu(clampLongToInt(pad_length)),
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

bool SemaChecker::VisitNode(AST::ChunkAt& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  if (n.indices && n.indices->Count()) {
    auto ty = NodeType(*n.data);
    auto arr_ty = cast<ArrayType>(ty);
    assert(arr_ty && "expect the array symbol to be an array type.");

    size_t rank = arr_ty->ArrayRank();
    size_t idx_cnt = n.indices->Count();
    // need exactly `rank` indices to access the array!
    if (idx_cnt != rank) {
      Error(n.LOC(), "Invalid array access: expected " + std::to_string(rank) +
                         " dimensions, but " + std::to_string(idx_cnt) +
                         " were used.");
      error_count++;
      return false;
    }

    // check if the indices are out of bound
    for (size_t i = 0; i < rank; ++i) {
      int bound = arr_ty->Dimension(i);

      auto idx = n.indices->ValueAt(i);
      // TODO: improve the out-of-bound check for bounded vars
      if (isa<BoundedType>(NodeType(*idx))) continue;

      auto shape = cast<AST::Expr>(idx)->s;
      // if the shape is not valid, it means the index is mutable!
      if (!shape.IsValid()) return true;
      assert(shape.DimCount() == 1);

      if (shape.IsDynamic()) {
        std::string shape_str = STR(shape);
        std::string idx_str = shape_str.substr(1, shape_str.size() - 2);
        std::string lhs, op, rhs, message;
        lhs = idx_str;
        op = "<";
        rhs = std::to_string(bound);
        message = "Index " + lhs + " is out of bounds of the " +
                  Ordinal(i + 1) + " dimension of array '" + PSTR(n.data) +
                  "', where the valid range is [0, " + rhs + ").";
        FCtx(fname).AppendRtCheck({lhs, op, rhs, idx->LOC(), message, {}});
        FCtx(fname).AppendRtCheck({lhs, ">=", "0", idx->LOC(), message, {}});
      } else {
        auto idx_val = VIInt(shape.ValueAt(0));
        assert(idx_val);
        if (*idx_val < 0 || *idx_val >= bound) {
          Error(idx->LOC(), "Index " + std::to_string(*idx_val) +
                                " is out of bounds of the " + Ordinal(i + 1) +
                                " dimension of array '" + PSTR(n.data) +
                                "', where the valid range is [0, " +
                                std::to_string(bound) + ").");
          error_count++;
          return false;
        }
      }
    }
  }

  // TODO: fix normalize to make it work
#if 0
  if (!n.s.IsValid()) {
    Error(n.LOC(), "The tiled block shape is invalid.");
    error_count++;
  }
#endif

  return true;
}

bool SemaChecker::VisitNode(AST::Trigger& n) {
  for (auto& f : n.GetEvents()) {
    auto fty = NodeType(*f);
    if (!isa<EventType>(fty)) {
      Error(n.LOC(),
            "trigger a non-event type " + PSTR(f) + " (" + PSTR(fty) + ").");
      error_count++;
      continue;
    }
    if (auto id = AST::GetIdentifier(*f))
      pending_async.insert(InScopeName(id->name));
    else if (auto e = dyn_cast<AST::Expr>(f)) {
      if (e->op != "elemof") {
        Error(n.LOC(), "expect a element-of operation but got " + e->op + ").");
        error_count++;
        continue;
      }
      auto bid = GetArrayBaseSymbol(*e);
      pending_async.insert(InScopeName(bid->name));
    }
  }
  return true;
}

bool SemaChecker::VisitNode(AST::Wait& n) {
  for (auto& f : n.GetTargets()) {
    auto fty = NodeType(*f);
    if (!isa<FutureType>(fty) && !isa<EventType>(fty)) {
      Error(n.LOC(),
            "wait for a non-async type " + PSTR(f) + " (" + PSTR(fty) + ").");
      error_count++;
      continue;
    }
    if (auto id = AST::GetIdentifier(*f))
      waited_async.insert(InScopeName(id->name));
    else if (auto e = dyn_cast<AST::Expr>(f)) {
      if (e->op != "elemof") {
        Error(n.LOC(), "expect a element-of operation but got " + e->op + ").");
        error_count++;
        continue;
      }
      auto bid = GetArrayBaseSymbol(*e);
      waited_async.insert(InScopeName(bid->name));
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::Call& n) {
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
      auto expr = cast<AST::Expr>(v);
      // fail if the template argument can not be evaluated as a compile-time
      // constant
      if (!expr->Opts().HasVal() || !expr->Opts().GetVal()->IsNumeric()) {
        Error(n.LOC(), "The " + Ordinal(count) +
                           " template argument of type '" + PSTR(ty) +
                           "` can not be evaluated at choreo compile time.");
        error_count++;
      }
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::Rotate& n) {
  size_t index = 0;
  for (auto s : n.ids->AllValues()) {
    if (auto id = AST::GetIdentifier(*s))
      waited_async.insert(InScopeName(id->name));

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
    assert(lid && rid && "no identifier is found.");
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

bool SemaChecker::VisitNode(AST::Select& n) {
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
      waited_async.insert(InScopeName(id->name)); // can not check statically

    if (*NodeType(*expr) == *NodeType(*expr0)) continue;

    ++error_count;
    Error(expr->LOC(), "Type mismatch inside SELECT: " + PSTR(expr) + "(" +
                           TYPE_STR(expr) + ") vs. " + PSTR(expr0) + "(" +
                           TYPE_STR(expr0) + ").");
  }

  return ec == error_count;
}

bool SemaChecker::VisitNode(AST::Return& n) {
  if (n.value) {
    auto vty = NodeType(*n.value);
    if (!(isa<SpannedType>(vty) || isa<ScalarType>(vty))) {
      Error(n.LOC(),
            "returning value with type '" + PSTR(vty) + "' is not supported.");
      error_count++;
      return false;
    }
  }

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
