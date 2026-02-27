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

    if (!pending_async.empty())
      Error1(n.LOC(), "some asyncs are not explicitly waited: " +
                          DelimitedString(pending_async) + ".");
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

  if (input_deps.Contains(n.GetR()))
    input_deps.Add(n);
  else if (input_deps.Contains(n.GetL()))
    input_deps.Add(n);
  else if (input_deps.Contains(n.GetC()))
    input_deps.Add(n);

  if (local_deps.Contains(n.GetR()))
    local_deps.Add(n);
  else if (local_deps.Contains(n.GetL()))
    local_deps.Add(n);
  else if (local_deps.Contains(n.GetC()))
    local_deps.Add(n);

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
      Error1(n.LOC(), "Invalid array access: expected " +
                          std::to_string(arr_rank) + " dimensions, but " +
                          std::to_string(subscription_level) + " were used.");
      return false;
    }

    auto dims = arr_ty->Dimensions();
    const ValueItem& bound_vi = arr_ty->Dimension(subscription_level - 1);
    int bound = *VIInt(bound_vi);
    auto idx = n.GetR();

    // TODO: parallel p by 2 { xxx; dma arr[p] => local; }
    // if the index is a bounded var, hard to determine if it is out of bound
    if (isa<BoundedType>(NodeType(*idx))) return true;

    auto expr = cast<AST::Expr>(idx);
    if (!expr->Opts().HasVal()) {
      VST_DEBUG(dbgs() << "Expression: " << PSTR(expr)
                       << " does not have a value!\n");
      return false;
    }
    auto index = expr->Opts().GetVal();

    // skip checking the one with 'nil' value though
    if (!IsComputable(index)) {
      Error1(expr->LOC(), "The " + Ordinal(subscription_level) +
                              " subscription index can not be evaluated.");
      return true;
    }

    // 0 <= index < bound
    auto asrt0 = sbe::bop(OpCode::LT, index, bound_vi)->Normalize();
    auto asrt1 = sbe::bop(OpCode::GE, index, sbe::nu(0))->Normalize();
    assert(IsValidValueItem(asrt0) && IsValidValueItem(asrt1));

    auto message = "Index " + STR(index) + " is out of bounds of the " +
                   Ordinal(subscription_level) + " dimension of array '" +
                   PSTR(arr_sym) + "', where the valid range is [0, " +
                   std::to_string(bound) + ")";

    EmitAssertion(asrt0, message, expr->LOC(), expr);
    EmitAssertion(asrt1, message, expr->LOC(), expr);
  } else if (n.IsBinary() && n.IsArith()) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    auto lsty = dyn_cast<SpannedType>(lty);
    auto rsty = dyn_cast<SpannedType>(rty);

    // non-spanned types are checked already
    if (!lsty && !rsty) return true;

    // Allows `tensor <op> scalar`
    if ((lsty && isa<ScalarType>(rty)) || (rsty && isa<ScalarType>(lty)))
      return true;

    auto lshape = lsty->GetShape();
    auto rshape = rsty->GetShape();

    assert(lshape.IsValid());
    assert(rshape.IsValid());

    bool compatible = true;
    bool warn = false;
    if (lshape.Rank() == rshape.Rank()) {
      if (lshape != rshape) compatible = false;
    } else {
      // only allow broadcasting of msb dimenisons
      auto min_rank = std::min(lshape.Rank(), rshape.Rank());
      assert(min_rank >= 1);
      for (size_t i = 1; i <= min_rank; ++i) {
        auto lidx = lshape.Rank() - i;
        auto ridx = rshape.Rank() - i;
        if (sbe::must_ne(lshape.ValueAt(lidx), rshape.ValueAt(ridx))) {
          compatible = false;
          break;
        } else if (sbe::may_ne(lshape.ValueAt(lidx), rshape.ValueAt(ridx))) {
          warn = true;
          break;
        }
      }
    }
    if (!compatible) {
      Error1(n.LOC(), "inconsistent shapes for spanned-operation `" + n.op +
                          "`(" + STR(lshape) + " v.s. " + STR(rshape) + ").");
    } else if (warn) {
      Warning(n.LOC(), "shapes may be inconsistent for spanned-operation `" +
                           n.op + "`(" + STR(lshape) + " v.s. " + STR(rshape) +
                           ").");
      // TODO: emit runtime check
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::MultiDimSpans& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (n.list)
    if (auto mv = dyn_cast<AST::MultiValues>(n.list))
      for (const auto& v : mv->AllValues())
        if (local_deps.Contains(v)) {
          local_deps.Add(*n.list);
          break;
        }
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
  if (s.IsValid())
    for (auto sv : s.Value())
      if (*sv == *sbe::nu(0))
        Error1(n.LOC(), "found 0-dimension within the shape of variable `" +
                            n.name_str + "'.");

  if (n.init_expr && input_deps.Contains(n.init_expr)) input_deps.Add(n);

  local_deps.Add(InScopeName(n.name_str));

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
    Error1(n.LOC(), "inconsistent types are found in the assignment: " +
                        STR(*NodeType(*n.value)) + " vs. " + STR(*NodeType(n)) +
                        ".");
    return false;
  }

  if (n.value && input_deps.Contains(n.value)) input_deps.Add(n);

  if (n.IsDecl()) local_deps.Add(InScopeName(n.da->GetDataName()));

  return true;
}

bool SemaChecker::VisitNode(AST::IntIndex& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (!isa<ScalarIntegerType>(n.value->GetType())) {
    Error1(n.LOC(), "Expect `" + PSTR(n.value) + "' to be a integer type.");
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

  if (n.sym) input_deps.Add(InScopeName(n.sym->name));

  return true;
}

bool SemaChecker::VisitNode(AST::ParallelBy& n) {
  if (auto shape = GetShape(NodeType(n)); shape.IsDynamic()) {
    int index = 1;
    for (auto& dim : shape.Value()) {
      auto spv = n.SubPVs()->ValueAt(index - 1);
      auto& loc = spv->LOC();
      if (!IsComputable(dim)) {
        Error1(loc, "The parallel count (" + Ordinal(index) +
                        "th) can not be evaluated.");
        continue;
      }
      auto message =
          "The " + Ordinal(index) +
          " bound item of parallelby is invalid: should be greater than 0";
      auto asrt = sbe::cmp(">", dim, sbe::nu(0))->Normalize();
      assert(IsValidValueItem(asrt));

      EmitAssertion(asrt, message, loc, spv);
      ++index;
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::WithIn& n) {
  if (auto shape = GetShape(NodeType(*n.in)); shape.IsDynamic()) {
    int index = 1;
    for (auto& dim : shape.Value()) {
      if (!IsComputable(dim))
        continue; // not reporting error since there could be no use of the
                  // value
      std::string message = "zero is detected for the " + Ordinal(index) +
                            " dim of the mdspan inside the with-in statement";
      auto asrt = sbe::cmp("!=", dim, sbe::nu(0))->Normalize();
      assert(IsValidValueItem(asrt));

      EmitAssertion(asrt, message, n.in->LOC(), n.in);
      ++index;
    }
  }
  return true;
}

bool SemaChecker::VisitNode(AST::SpanAs& n) {
  if (!ReportUnknownSymbol(n.id->name, n.LOC(), __FILE__, __LINE__))
    return false;

  auto ity = GetSymbolType(n.id->name);

  if (!(isa<SpannedType>(ity) || isa<FutureType>(ity))) {
    Error1(n.LOC(), "Expect symbol `" + n.id->name + "' to be a spanned type.");
    return false;
  }

  if (!(AST::istypeof<SpannedType>(&n))) {
    Error1(n.LOC(), "Invalid type of span_as expression.");
    return false;
  }

  auto sty = GetSpannedType(ity);
  auto nty = cast<SpannedType>(NodeType(n));

  if (sty->ElementType() != nty->ElementType()) {
    Error1(n.LOC(), "Inconsistent element type: (" + STR(nty->ElementType()) +
                        " = span_as(" + STR(sty->ElementType()) + ".");
    return false;
  }

  auto p = (sty->IsDense() & nty->IsDense());
  if (p == Modality::NOT)
    Error1(n.LOC(), "span_as is applied on non-contiguous spanned data.");
  else if (p == Modality::MAY)
    Warning(n.LOC(),
            "span_as could be applied on non-contiguous spanned data.");

  if (!sty->RuntimeShaped() && !nty->RuntimeShaped()) {
    // check if the shape size are same
    if (sty->ElementCount() != nty->ElementCount()) {
      Error1(n.LOC(), "Inconsistent mdspan size: " + n.id->name + "(" +
                          STR(sty->ElementCount()) + ") = spanas (" +
                          n.nid->name + "(" + STR(nty->ElementCount()) + ")).");
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
      Error1(n.LOC(), "A dummy/async DMA must be named.");
      return false;
    }
    if (!isa<PlaceHolderType>(ty) ||
        (cast<PlaceHolderType>(ty)->GetBaseType() != BaseType::FUTURE)) {
      Error1(n.LOC(), "Expect a placeholder type but got '" + PSTR(ty) + "'.");
      return false;
    }
    return true;
  }

  // Check swizzle: only report error if swizzle is explicitly specified
  // and we're not in a WGMMA context
  auto swiz_set = CCtx().TargetSwizzleModes();
  if (!swiz_set.empty() && !swiz_set.count(n.GetSwizzleMode())) {
    // For now, we just validate the swizzle value is valid (128, 64, or 32)
    // The WGMMA context check will be done in codegen phase
    Error1(n.LOC(), "Invalid swizzle mode: " + STR(n.GetSwizzleMode()) + ".");
    return false;
  }

  if (n.IsSparse()) {
    extern Option<bool> sim_sparse;
    if (!sim_sparse) {
      Warning(n.LOC(),
              "Sparse DMA is enabled without -sim; this path is experimental.");
    }
    if (n.operation != ".copy") {
      Error1(n.LOC(), "Sparse DMA only supports dma.copy.sp currently.");
      return false;
    }
    auto sp = n.GetSparsePattern();
    if (!(sp.first == 2 && sp.second == 4)) {
      Error1(n.LOC(), "Only 2:4 structured sparsity is supported; got " +
                          std::to_string(sp.first) + ":" +
                          std::to_string(sp.second) + ".");
      return false;
    }
  }

  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  if (!isa<FutureType>(ty))
    Error1(n.LOC(), "Expect the DMA to produce a FutureType, but got '" +
                        PSTR(n.GetType()) + "'.");

  if (cast<FutureType>(ty)->IsAsync() && n.future.empty())
    Error1(n.LOC(), "A dummy/async DMA must be named.");

  if (!n.future.empty() && cast<FutureType>(ty)->IsAsync())
    pending_async.insert(InScopeName(n.future));
  if (!n.chain_from.empty()) waited_async.insert(InScopeName(n.chain_from));

  if (!isa<AST::ChunkAt>(n.from) || !isa<SpannedType>(n.from->GetType()))
    Error1(n.LOC(),
           "The 'from' of DMA is not as expected: " + n.from->TypeNameString() +
               "(" + PSTR(n.from->GetType()) + ").");

  if (!isa<AST::ChunkAt>(n.to) || !isa<SpannedType>(n.to->GetType()))
    Error1(n.LOC(),
           "The 'to' of DMA is not as expected: " + n.to->TypeNameString() +
               "(" + PSTR(n.from->GetType()) + ").");

  auto& fty = n.from->GetType();
  auto& tty = n.to->GetType();

  auto sfty = cast<SpannedType>(fty);
  auto stty = cast<SpannedType>(tty);
  auto f_shape = sfty->GetShape();
  auto t_shape = stty->GetShape();

  if (n.IsSparse()) {
    if (!n.GetSrc()->NoTilingOperation() || !n.GetDst()->NoTilingOperation()) {
      Error1(n.LOC(), "Sparse DMA currently requires symbol-to-symbol copy "
                      "with no tiling.");
      return false;
    }
    if ((f_shape.Rank() != 2 && f_shape.Rank() != 3) ||
        (t_shape.Rank() != 2 && t_shape.Rank() != 3) || f_shape.IsDynamic() ||
        t_shape.IsDynamic()) {
      Error1(n.LOC(),
             "Sparse DMA currently requires static rank-2 or rank-3 tensors.");
      return false;
    }
  }

  if (n.operation == ".transp") {
    // no transposed shape need to be generated
    // do LogicalEqual() manually
    auto tc = cast<TransposeConfig>(n.config);
    if (sfty->e_type != stty->e_type || !f_shape.SameRankAs(t_shape)) {
      Error1(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                          ") with " + PSTR(tc) + " and 'to'(" + PSTR(tty) +
                          ").");
    } else {
      auto& dim_values = tc->dim_values;
      for (size_t i = 0; i < dim_values.size(); ++i) {
        auto eq =
            sbe::oc_eq(f_shape.ValueAt(dim_values[i]), t_shape.ValueAt(i));
        assert(IsValidValueItem(eq));
        auto message = "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                       ") with " + PSTR(tc) + " and 'to'(" + PSTR(tty) +
                       ") at the " + Ordinal(i + 1) + " dim.";
        EmitAssertion(eq, message, n.from->LOC(), n.from);
      }
    }
  } else if (n.operation == ".pad") {
    // no padded shape need to be generated
    // do LogicalEqual() manually
    auto pc = cast<PadConfig>(n.config);
    if (sfty->e_type != stty->e_type || !f_shape.SameRankAs(t_shape)) {
      Error1(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                          ") with " + PSTR(pc) + " and 'to'(" + PSTR(tty) +
                          ").");
    } else {
      size_t dim_count = f_shape.DimCount();
      for (size_t i = 0; i < dim_count; ++i) {
        auto h_i = cast<AST::Expr>(pc->pad_high->ValueAt(i));
        auto l_i = cast<AST::Expr>(pc->pad_low->ValueAt(i));
        auto m_i = cast<AST::Expr>(pc->pad_mid->ValueAt(i));
        if (!h_i->Opts().HasVal()) {
          VST_DEBUG(dbgs() << "Expression: " << PSTR(pc->pad_high->ValueAt(i))
                           << " does not have a value!\n");
          return false;
        }
        if (!l_i->Opts().HasVal()) {
          VST_DEBUG(dbgs() << "Expression: " << PSTR(pc->pad_low->ValueAt(i))
                           << " does not have a value!\n");
          return false;
        }
        if (!m_i->Opts().HasVal()) {
          VST_DEBUG(dbgs() << "Expression: " << PSTR(pc->pad_mid->ValueAt(i))
                           << " does not have a value!\n");
          return false;
        }

        auto pad_length =
            h_i->Opts().GetVal() + l_i->Opts().GetVal() + m_i->Opts().GetVal();
        if (!IsValueItemEqual(f_shape.ValueAt(i) + pad_length,
                              t_shape.ValueAt(i))) {
          Error1(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                              ") with " + PSTR(pc) + " and 'to'(" + PSTR(tty) +
                              ").");
          break;
        }
      }
    }
  } else if (!(cast<SpannedType>(fty)->LogicalEqual(*tty))) {
    // check: to-buffer size must be larger or equal than from
    auto asrt = sbe::bop(OpCode::LE, f_shape.ElementCountValue(),
                         t_shape.ElementCountValue())
                    ->Normalize();
    assert(IsValidValueItem(asrt));

    auto message = "DMA to-buffer is too small (" +
                   STR(f_shape.ElementCountValue()) + " > " +
                   STR(t_shape.ElementCountValue()) + ")";
    EmitAssertion(asrt, message, n.LOC(), n.from);

    bool emit_error = true;
    std::string msg;
    if (f_shape != t_shape && f_shape.IsValid() && t_shape.IsValid()) {
      // ignore the symbolic value inconsistance
      for (size_t i = 0; i < f_shape.Rank(); ++i) {
        auto& fv = f_shape.ValueAt(i);
        auto& tv = t_shape.ValueAt(i);
        if (!IsValueItemEqual(fv, tv)) {
          if (tv->IsSymbolic() || fv->IsSymbolic()) {
            emit_error = false;
            msg += " 'from[" + std::to_string(i) + "](" + STR(fv) +
                   ")' v.s. 'to[" + std::to_string(i) + "](" + STR(tv) + ")";
            continue;
          } else {
            emit_error = true;
            break;
          }
        }
      }
    }
    if (emit_error) {
      Error1(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                          ") and 'to'(" + PSTR(tty) + ").");
    } else {
      Warning(n.LOC(),
              "Dimensions could be inconsistent between DMA" + msg + ").");
    }
  }

  for (const auto& ca_node : {n.from, n.to}) {
    const auto& ca = cast<AST::ChunkAt>(ca_node);
    auto sty = GetSpannedType(GetSymbolType(ca->RefSymbol()));
    assert(sty);
    Shape s = sty->GetShape();
    ValueList strd = sty->GetStrides();

    for (auto& op : ca->AllOperations()) {
      if (auto rop = dyn_cast<AST::SOP::Reshape>(op)) {
        switch (IsContiguous(s, strd)) {
        case Modality::NOT:
          Warning(rop->LOC(),
                  "'span_as' is applied to non-contiguous spanned data (" +
                      STR(s) + " {" + STR(strd) + "}).");
          break;
        case Modality::MAY:
          Warning(rop->LOC(),
                  "'span_as' may be applied to non-contiguous spanned data (" +
                      STR(s) + " {" + STR(strd) + "}).");
          break;
        case Modality::MUST:
        default: break;
        }
      }
      s = op->GetBlockShape();
      strd = op->GetBlockStrides();
    }
  }
#if 0
  // Check if the spanned operations are valid
  for (const auto& ca_node : {n.from, n.to}) {
    const auto& ca = cast<AST::ChunkAt>(ca_node);
    if (ca->NoTilingOperation()) continue;
    Shape original_shape = GetShape(GetSymbolType(ca->RefSymbol()));
    bool has_noncontiguous = false;
    size_t last_tiling = 0;
    for (size_t i = 0; i < ca->OpCount(); ++i)
      if (!isa<AST::SOP::Reshape>(ca->OpAt(i))) last_tiling = i;
    bool has_reshape = false;
    for (size_t i = 0; i < ca->OpCount(); ++i) {
      const auto& sop = ca->OpAt(i);
      if (isa<AST::SOP::Reshape>(ca->OpAt(ca->OpCount() - i - 1)))
        has_reshape = true;
      auto is_contiguous = AST::IsContiguousSOp(*sop, original_shape);
      if (auto val = std::get_if<bool>(&is_contiguous); val && *val == false) {
        has_noncontiguous = true;
        if (i != last_tiling)
          Warning(sop->LOC(), "There are more than one tiling executed in "
                              "noncontiguous manner.");
      } else if (!val) {
        if (i != last_tiling)
          Warning(sop->LOC(), "There may have more than one tiling executed in "
                              "noncontiguous manner.");
        if (has_reshape)
          Warning(sop->LOC(),
                  "The reshape operation inside DMA expression maybe is "
                  "executed on a noncontiguous tiling result.");
      }
      if (has_noncontiguous && isa<AST::SOP::Reshape>(sop))
        Warning(sop->LOC(), "The reshape operation inside DMA expression is "
                            "executed on a noncontiguous tiling result.");
      original_shape = sop->GetBlockShape();
    }
  }
#endif

  return true;
}

bool SemaChecker::VisitNode(AST::MMA& n) {
  auto& op = *n.GetOperation();
  switch (op.Tag()) {
  case AST::MMAOperation::Fill: break;
  case AST::MMAOperation::Load: {
    // Check swizzle consistency between DMA and MMA load
    // Find the corresponding DMA operation that loads to shared memory
    auto load_from = op.LoadFrom();
    if (load_from && isa<AST::ChunkAt>(load_from)) {
      auto ref_sym = load_from->RefSymbol();
      // Try to find a DMA that writes to this symbol
      // This is a simplified check - in a full implementation, we'd track all
      // DMAs For now, we just validate that the swizzle value is valid
      auto mma_swizzle = op.GetSwizzleMode();
      auto swiz_set = CCtx().TargetSwizzleModes();
      if (!swiz_set.empty() && !swiz_set.count(mma_swizzle)) {
        Error1(n.LOC(),
               "Invalid swizzle value in MMA load: " + STR(mma_swizzle) + ".");
        return false;
      }

      // Provide guidance on TILE_K constraints
      // Note: These are recommendations, not enforced constraints
      // Users should set TILE_K accordingly:
      // - swizzle(128): TILE_K = 64 (default)
      // - swizzle(64):  TILE_K = 32
      // - swizzle(32):  TILE_K = 16
      if (mma_swizzle == SwizMode::B64) {
        VST_DEBUG(dbgs() << "MMA load with swizzle(64): Consider setting "
                            "TILE_K = 32 for optimal performance\n");
      } else if (mma_swizzle == SwizMode::B32) {
        VST_DEBUG(dbgs() << "MMA load with swizzle(32): Consider setting "
                            "TILE_K = 16 for optimal performance\n");
      }
    }
    break;
  }
  case AST::MMAOperation::Exec: {
    auto& a_sym = AST::FragName(op.ExecOperand(1));
    auto& b_sym = AST::FragName(op.ExecOperand(2));
    auto& c_sym = AST::FragName(op.ExecOperand(0));
    auto a_ty = GetSpannedType(GetSymbolType(a_sym));
    auto b_ty = GetSpannedType(GetSymbolType(b_sym));
    auto c_ty = GetSpannedType(GetSymbolType(c_sym));
    bool old_ec = error_count;
    if (a_ty == nullptr)
      Error1(n.LOC(), "Expect `" + a_sym + "' to contain a spanned data.");
    if (b_ty == nullptr)
      Error1(n.LOC(), "Expect `" + b_sym + "' to contain a spanned data.");
    if (c_ty == nullptr)
      Error1(n.LOC(), "Expect `" + c_sym + "' to contain a spanned data.");
    if (old_ec != error_count) return false;

    auto a_shape = a_ty->GetShape();
    auto b_shape = b_ty->GetShape();
    auto c_shape = c_ty->GetShape();
    if ((a_shape.Rank() != 2) || a_shape.IsDynamic())
      Error1(n.LOC(), "Expect `" + a_sym +
                          "' to be a matrix with fixed size, but got: " +
                          STR(a_shape) + ".");
    if ((b_shape.Rank() != 2) || b_shape.IsDynamic())
      Error1(n.LOC(), "Expect `" + b_sym +
                          "' to be a matrix with fixed size, but got: " +
                          STR(b_shape) + ".");
    if ((c_shape.Rank() != 2) || c_shape.IsDynamic())
      Error1(n.LOC(), "Expect `" + c_sym +
                          "' to be a matrix with fixed size, but got: " +
                          STR(c_shape) + ".");
    if (old_ec != error_count) return false;

    bool shape_match = true;
    bool sparse_packed_match = false;
    ValueList cs_vals;
    switch (op.GetMethod()) {
    case AST::MMAOperation::ROW_ROW:
      if (!sbe::ceq(a_shape.ValueAt(1), b_shape.ValueAt(1)))
        shape_match = false;
      if (op.IsSparse() && !shape_match) {
        auto a_k = a_shape.ValueAt(1);
        auto b_k = b_shape.ValueAt(1);
        auto a_k2 = a_k * sbe::nu(2);
        if (sbe::ceq(a_k2, b_k)) sparse_packed_match = true;
      }
      cs_vals.push_back(a_shape.ValueAt(0));
      cs_vals.push_back(b_shape.ValueAt(0));
      break;
    case AST::MMAOperation::ROW_COL:
      if (!sbe::ceq(a_shape.ValueAt(1), b_shape.ValueAt(0)))
        shape_match = false;
      if (op.IsSparse() && !shape_match) {
        auto a_k = a_shape.ValueAt(1);
        auto b_k = b_shape.ValueAt(0);
        auto a_k2 = a_k * sbe::nu(2);
        if (sbe::ceq(a_k2, b_k)) sparse_packed_match = true;
      }
      cs_vals.push_back(a_shape.ValueAt(0));
      cs_vals.push_back(b_shape.ValueAt(1));
      break;
    case AST::MMAOperation::COL_ROW:
      if (!sbe::ceq(a_shape.ValueAt(0), b_shape.ValueAt(1)))
        shape_match = false;
      if (op.IsSparse() && !shape_match) {
        auto a_k = a_shape.ValueAt(0);
        auto b_k = b_shape.ValueAt(1);
        auto a_k2 = a_k * sbe::nu(2);
        if (sbe::ceq(a_k2, b_k)) sparse_packed_match = true;
      }
      cs_vals.push_back(a_shape.ValueAt(1));
      cs_vals.push_back(b_shape.ValueAt(0));
      break;
    case AST::MMAOperation::COL_COL:
      if (!sbe::ceq(a_shape.ValueAt(0), b_shape.ValueAt(0)))
        shape_match = false;
      if (op.IsSparse() && !shape_match) {
        auto a_k = a_shape.ValueAt(0);
        auto b_k = b_shape.ValueAt(0);
        auto a_k2 = a_k * sbe::nu(2);
        if (sbe::ceq(a_k2, b_k)) sparse_packed_match = true;
      }
      cs_vals.push_back(a_shape.ValueAt(1));
      cs_vals.push_back(b_shape.ValueAt(1));
      break;
    default: choreo_unreachable("unsupported mma execution method.");
    }
    if (!shape_match && !sparse_packed_match) {
      Error1(n.LOC(), "MMA: matrix shapes do not match: `" + a_sym + "'(" +
                          STR(a_shape) + ") v.s. `" + b_sym + "'(" +
                          STR(b_shape) + ").");
      return false;
    }

    if (op.IsSparse()) {
      ValueItem k_dim;
      switch (op.GetMethod()) {
      case AST::MMAOperation::ROW_ROW:
      case AST::MMAOperation::ROW_COL: k_dim = a_shape.ValueAt(1); break;
      case AST::MMAOperation::COL_ROW:
      case AST::MMAOperation::COL_COL: k_dim = a_shape.ValueAt(0); break;
      default: break;
      }
      if (auto kv = VIInt(k_dim)) {
        if ((*kv % 4) != 0) {
          Error1(
              n.LOC(),
              "Sparse MMA requires K dimension to be a multiple of 4. Got: " +
                  STR(k_dim));
          return false;
        }
      } else {
        Warning(n.LOC(), "Sparse MMA expects K dimension multiple of 4; unable "
                         "to verify at compile time.");
      }
    }
  } break;
  case AST::MMAOperation::Store: break;
  case AST::MMAOperation::Commit: break;
  default: choreo_unreachable("unsupported mma operation.");
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
      Error1(n.LOC(), "Invalid array access: expected " + std::to_string(rank) +
                          " dimensions, but " + std::to_string(idx_cnt) +
                          " were used.");
      return false;
    }

    // check if any indices are out of bound
    for (size_t i = 0; i < rank; ++i) {
      int bound = *VIInt(arr_ty->Dimension(i));

      auto expr = n.indices->ValueAt(i);
      // TODO: improve the out-of-bound check for bounded vars
      if (isa<BoundedType>(NodeType(*expr))) continue;

      auto e = cast<AST::Expr>(expr);
      if (!e->Opts().HasVal()) {
        VST_DEBUG(dbgs() << "Expression: " << PSTR(expr)
                         << " does not have a value!\n");
        return false;
      }
      auto index = e->Opts().GetVal();

      // skip checking the one with 'nil' value though
      if (!IsComputable(index)) {
        Error1(expr->LOC(), "The " + Ordinal(i) +
                                " subscription index can not be evaluated.");
        return true;
      }

      // 0 <= index < bound
      auto asrt0 =
          sbe::bop(OpCode::LT, index, arr_ty->Dimension(i))->Normalize();
      auto asrt1 = sbe::bop(OpCode::GE, index, sbe::nu(0))->Normalize();
      assert(IsValidValueItem(asrt0) && IsValidValueItem(asrt1));

      auto message = "Index " + STR(index) + " is out of bounds of the " +
                     Ordinal(i + 1) + " dimension of array '" + PSTR(n.data) +
                     "', where the valid range is [0, " +
                     std::to_string(bound) + ")";
      EmitAssertion(asrt0, message, expr->LOC(), expr);
      EmitAssertion(asrt1, message, expr->LOC(), expr);
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::Trigger& n) {
  for (auto& f : n.GetEvents()) {
    auto fty = NodeType(*f);
    if (!isa<EventType>(fty)) {
      Error1(n.LOC(),
             "trigger a non-event type " + PSTR(f) + " (" + PSTR(fty) + ").");
      continue;
    }
    if (auto id = AST::GetIdentifier(*f))
      pending_async.insert(InScopeName(id->name));
    else if (auto e = dyn_cast<AST::Expr>(f)) {
      if (e->op != "elemof") {
        Error1(n.LOC(),
               "expect a element-of operation but got " + e->op + ").");
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
      Error1(n.LOC(),
             "wait for a non-async type " + PSTR(f) + " (" + PSTR(fty) + ").");
      continue;
    }
    if (auto id = AST::GetIdentifier(*f))
      waited_async.insert(InScopeName(id->name));
    else if (auto e = dyn_cast<AST::Expr>(f)) {
      if (e->op != "elemof") {
        Error1(n.LOC(),
               "expect a element-of operation but got " + e->op + ").");
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
      auto expr = cast<AST::Expr>(v);
      if (expr->IsReference() && isa<AST::DataType>(expr->GetReference())) {
        continue;
      }
      // must be a scalar type
      if (!CanYieldAnInteger(ty))
        Error1(n.LOC(),
               "The " + Ordinal(count) + " template argument of type '" +
                   PSTR(ty) +
                   "` can not be used to instantiate the kernel function.");
      // fail if the template argument can not be evaluated as a compile-time
      // constant
      if (!expr->Opts().HasVal() || !expr->Opts().GetVal()->IsNumeric())
        Warning(n.LOC(), "The " + Ordinal(count) +
                             " template argument of type '" + PSTR(ty) +
                             "` can not be evaluated at choreo compile time.");
    }
  }

  if (n.IsBIF()) {
    const auto func_name = n.function->name;
    if (func_name == "assert") {
      auto cmp = n.arguments->ValueAt(0);
      if (auto cexpr = dyn_cast<AST::Expr>(cmp);
          cexpr && cexpr->Opts().HasVal()) {
        if (auto bv = VIBool(cexpr->Opts().GetVal());
            bv && (bv.value() == false)) {
          std::string msg;
          if (auto str = dyn_cast<AST::StringLiteral>(n.arguments->ValueAt(1)))
            msg = str->value;
          else
            choreo_unreachable(
                "choreo assertion requires a string message as the second.");
          Error1(n.LOC(), "choreo assertion abort: " + msg);
        }
      }
    }
    if (n.IsArith()) {
      auto pty = NodeType(*n.arguments->ValueAt(0));
      if (!(isa<ScalarFloatType>(pty) || isa<VectorType>(pty)))
        Error1(
            n.LOC(),
            "expect the argument to be a float type or vector type but got '" +
                PSTR(pty) + "'.");

      for (size_t i = 1; i < n.arguments->Count(); ++i) {
        auto sty = NodeType(*n.arguments->ValueAt(i));
        if (!sty->ApprxEqual(*pty))
          Error1(n.LOC(),
                 "expect the " + std::to_string(i) +
                     "th argument to be the same type as the first one.");
      }
    }
  }

  if (n.IsBIF()) return true;

  // resolve device functions
  if (!resolve_fns) return true;
  if (n.device_functions.empty()) return true;

  bool function_found = false;
  auto function_name = n.function->name;
  ptr<AST::DeviceFunctionDecl> matched_function = nullptr;
  std::string mismatch_msg = "";
  for (size_t i = 0; i < n.device_functions.size(); i++) {
    auto device_function = n.device_functions[i];
    bool arg_match = true;
    for (size_t param_index = 0; param_index < n.arguments->Count();
         param_index++) {
      auto pnode = n.arguments->ValueAt(param_index);
      auto arg_ty = NodeType(*pnode);
      auto param_ty = device_function->param_types[param_index];

      std::string attr = param_ty->attr;
      if (auto spanned_ty = dyn_cast<SpannedType>(arg_ty)) {
        auto m_ty = spanned_ty->GetStorage();
        if ((attr.find("__private__") != std::string::npos) ||
            (attr.find("__attribute__((address_space(5)))") !=
             std::string::npos)) {
          if (m_ty != Storage::LOCAL)
            arg_match = false;
          else
            pnode->AddNote("annotate_as"); // annotate the addrspace
        } else if ((attr.find("__shared__") != std::string::npos) ||
                   (attr.find("__attribute__((shared))") !=
                    std::string::npos)) {
          if (m_ty != Storage::SHARED)
            arg_match = false;
          else
            pnode->AddNote("annotate_as");
        }

        if (!arg_match) {
          mismatch_msg = "the type of " + std::to_string(param_index + 1) +
                         "th argument '" + PSTR(arg_ty) + "' is not " +
                         STR(m_ty) + ".";
          break;
        }
      }

      if (arg_match) {
        matched_function = device_function;
        function_found = true;
      }
    }
  } // end of device function loop

  if (!function_found) {
    Warning(n.LOC(), "unable to find a device function '" + function_name +
                         "', because " + mismatch_msg);
  } else if (debug_visit) {
    if (matched_function->IsTemplated())
      dbgs() << "Find instantiated device function '";
    else
      dbgs() << "Find matched device function '";
    dbgs() << matched_function->name << "' -> "
           << PSTR(matched_function->ret_type) << " (";
    for (size_t i = 0; i < matched_function->param_types.size(); ++i) {
      auto& pt = matched_function->param_types[i];
      if (i > 0) dbgs() << ", ";
      dbgs() << PSTR(pt);
    }
    dbgs() << ")\n";
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
    if (*lty != *rty)
      Error1(n.LOC(), "swapping values of different types (" + PSTR(lty) +
                          " vs. " + PSTR(rty));

    auto lid = AST::GetIdentifier(*n.ids->ValueAt(index - 1));
    auto rid = AST::GetIdentifier(*n.ids->ValueAt(index));
    assert(lid && rid && "no identifier is found.");
    auto l_scope = GetScope(InScopeName(lid->name));
    auto r_scope = GetScope(InScopeName(rid->name));
    if (l_scope != r_scope)
      Error1(n.LOC(),
             "swapping values defined in different scopes is forbidden (" +
                 InScopeName(lid->name) + " vs. " + InScopeName(rid->name));

    index++;
  }
  return true;
}

bool SemaChecker::VisitNode(AST::Select& n) {
  size_t ec = error_count;

  if (!CanYieldAnInteger(n.select_factor->GetType()))
    Error1(n.select_factor->LOC(),
           "Expect " + PSTR(n.select_factor) +
               " to be an integer type but got " +
               NodeType(*n.select_factor)->TypeNameString() + ".");

  auto expr_list = n.expr_list;
  auto expr0 = expr_list->ValueAt(0);
  if (!isa<FutureType>(NodeType(*expr0)) &&
      !isa<SpannedType>(NodeType(*expr0))) {
    Error1(expr0->LOC(),
           "Expect " + PSTR(expr0) + " to be a future/spanned type.");
    return false;
  }

  for (auto expr : expr_list->AllValues()) {
    if (auto id = AST::GetIdentifier(*expr))
      waited_async.insert(InScopeName(id->name)); // can not check statically

    if (*NodeType(*expr) == *NodeType(*expr0)) continue;

    Error1(expr->LOC(), "Type mismatch inside SELECT: " + PSTR(expr) + "(" +
                            TYPE_STR(expr) + ") vs. " + PSTR(expr0) + "(" +
                            TYPE_STR(expr0) + ").");
  }

  int64_t select_value_cnt = static_cast<int64_t>(expr_list->Count());
  if (auto il = AST::GetIntLiteral(*n.select_factor)) {
    if (il->Val() < 0 || il->Val() >= select_value_cnt)
      Error1(il->LOC(), "The select factor `" + PSTR(il) +
                            "` is not in bound [0, " +
                            std::to_string(select_value_cnt) + ")");
  } else {
    if (n.select_factor->Opts().HasVal()) {
      auto v = n.select_factor->Opts().GetVal();
      EmitAssertion(sbe::oc_ge(v, sbe::nu(0)),
                    "The select factor `" + PSTR(n.select_factor) +
                        "` should be greater than or equal to 0",
                    n.select_factor->LOC(), n.select_factor);
      EmitAssertion(
          sbe::oc_lt(v, sbe::nu(select_value_cnt)),
          "The select factor `" + PSTR(n.select_factor) +
              "` should be less than " + std::to_string(select_value_cnt) +
              ", which is the count of values in the select statement",
          n.select_factor->LOC(), n.select_factor);
    }
  }

  for (auto& v : n.expr_list->AllValues())
    if (input_deps.Contains(v)) {
      input_deps.Add(n);
      break;
    }

  return ec == error_count;
}

bool SemaChecker::VisitNode(AST::Return& n) {
  if (n.value) {
    auto vty = NodeType(*n.value);
    if (!(isa<SpannedType>(vty) || isa<ScalarType>(vty))) {
      Error1(n.LOC(),
             "returning value with type '" + PSTR(vty) + "' is not supported.");
      return false;
    }
  }

  return true;
}

bool SemaChecker::ReportUnknownSymbol(const std::string& name,
                                      const location& loc, const char* file,
                                      int line) {
  if (isa<UnknownType>(GetSymbolType(name))) {
    Error1(loc, "failed to obtain the type of `" + name + "'.");
    VST_DEBUG(dbgs() << file << ":" << line << "\n");
    return false;
  }
  return true;
}

bool SemaChecker::ReportUnknown(AST::Node& n, const char* file, int line,
                                bool ignore_detail) {
  if (STR(n) == "_") return true; // ignore built-in unit iv.
  if (isa<UnknownType>(NodeType(n))) {
    Error1(n.LOC(), "failed to obtain a type.");
    VST_DEBUG(dbgs() << file << ":" << line << ", " << STR(n) << "\n");
    return false;
  }

  if (!ignore_detail && !NodeType(n)->HasSufficientInfo()) {
    Error1(n.LOC(), "failed to obtain a type with sufficient info.");
    VST_DEBUG(dbgs() << file << ":" << line << ", " << STR(n) << "("
                     << PSTR(NodeType(n)) << ")\n");
    return false;
  }

  return true;
}

void SemaChecker::EmitAssertion(const ValueItem& pred,
                                const std::string& message, const location& l,
                                const ptr<AST::Node>& n) {
  if (auto b = VIBool(pred)) {
    if (b.value() == false) Error1(l, message);
    // else no assertion is triggered
  } else {
    if (local_deps.Contains(n)) {
      // TODO: emit device check that is related to the local values
      if (isa<AST::Expr>(n) || isa<AST::NamedVariableDecl>(n) ||
          isa<AST::Assignment>(n))
        VST_DEBUG(dbgs() << "failed to generate check for " << STR(n) << ".\n");
    } else {
      if (isa<AST::Expr>(n) || isa<AST::NamedVariableDecl>(n) ||
          isa<AST::Assignment>(n))
        if (!input_deps.Contains(n))
          VST_DEBUG(dbgs() << "questionable: check is not related to input: "
                           << PSTR(n) << ".\n");

      FCtx(fname).InsertAssertion(pred, l, message);
    }
  }
}
