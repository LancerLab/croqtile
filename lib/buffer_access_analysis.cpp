#include "buffer_access_analysis.hpp"

#include "ast.hpp"
#include "aux.hpp"
#include "io.hpp"
#include "types.hpp"

using namespace Choreo;

// ---------- Scope / parallel-level tracking ----------

bool BufferAccessAnalyzer::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n))
    pl_stack_.push_back(ParallelLevel::SEQ);
  else if (isa<AST::ChoreoFunction>(&n))
    pl_stack_.push_back(ParallelLevel::SEQ);
  else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    ParallelLevel lvl = pb->GetLevel();
    pl_stack_.push_back(lvl == ParallelLevel::NONE ? CurrentLevel() : lvl);
  }
  return true;
}

bool BufferAccessAnalyzer::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    auto& log = CCtx().GetBufferAccessLog();
    log.events = std::move(events_);
    log.future_buffers = tracker_.future_buffers;
    log.future_producers = std::move(future_producers_);
  } else if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n)) {
    pl_stack_.pop_back();
  }
  return true;
}

// ---------- Name resolution ----------

BufferAccessAnalyzer::VarSet
BufferAccessAnalyzer::ResolveBuffers(const std::string& name) const {
  std::string scoped = Scoped(name);
  VarSet result;

  // A DMA future's `.data` denotes the destination buffer only. The source
  // buffer is read directly by the DMA and never through the future.
  if (tracker_.future_buffers.count(scoped)) {
    for (const auto& [src, dst] : tracker_.future_buffers.at(scoped)) {
      if (!dst.empty()) result.insert(tracker_.ResolveNoStorageAlias(dst));
    }
    return result;
  }

  // An alias denotes its source buffer.
  if (tracker_.alias_.count(scoped)) {
    result.insert(tracker_.ResolveNoStorageAlias(tracker_.alias_.at(scoped)));
    return result;
  }

  // A binding (select / rotate) fans out to its member buffers.
  if (tracker_.bindings_.count(scoped)) {
    for (const auto& member : tracker_.bindings_.at(scoped))
      result.insert(tracker_.ResolveNoStorageAlias(member));
    return result;
  }

  result.insert(tracker_.ResolveNoStorageAlias(scoped));
  return result;
}

Storage BufferAccessAnalyzer::StorageOf(const std::string& scoped_name) const {
  if (!SymTab()->Exists(scoped_name)) return Storage::NONE;
  auto sty = GetSpannedType(SymTab()->GetSymbol(scoped_name)->GetType());
  if (!sty) return Storage::NONE;
  return sty->GetStorage();
}

void BufferAccessAnalyzer::Record(AccessKind kind, AST::Node* stmt,
                                  const std::string& name,
                                  AccessEntity entity) {
  if (name.empty()) return;
  // `.data` is the accessor of a future's destination buffer, not part of the
  // symbol name (e.g. `f.data.at(i)` resolves to the future `f`).
  std::string stripped = RemoveSuffix(name, ".data");
  for (const auto& buf : ResolveBuffers(stripped)) {
    Storage sto = StorageOf(buf);
    if (sto == Storage::NONE || sto == Storage::REG) continue;
    events_.push_back({kind, stmt, buf, sto, entity, CurrentLevel(), order_++});
  }
}

void BufferAccessAnalyzer::RecordAll(AccessKind kind, AST::Node* stmt,
                                     const VarSet& names, AccessEntity entity) {
  for (const auto& name : names) Record(kind, stmt, name, entity);
}

// ---------- Visit handlers ----------

bool BufferAccessAnalyzer::Visit(AST::NamedVariableDecl& n) {
  auto ty = GetSymbolType(n.name_str);
  if (auto sty = dyn_cast<SpannedType>(ty)) {
    if (!IsRef(n)) {
      // A storage-bearing buffer is produced (allocated/initialized) here.
      Record(AccessKind::WRITE, &n, n.name_str, AccessEntity::THREADS);
      return true;
    }
    // Reference: resolve to the aliased source buffer.
    assert(n.init_expr && "expecting the init_expr is not nullptr.");
    auto e = dyn_cast<AST::Expr>(n.init_expr);
    assert(e && "expecting the init_expr is an expr.");
    if (auto sa = dyn_cast<AST::SpanAs>(e->GetR())) {
      tracker_.AddAlias(n.name_str, sa->id->name);
    } else if (e->GetOp() == Op::ElemOf) {
      std::string base_array = AST::GetArrayBaseSymbol(*e)->name;
      tracker_.AddAlias(n.name_str, base_array);
    }
  }
  return true;
}

bool BufferAccessAnalyzer::Visit(AST::Assignment& n) {
  if (auto sa = dyn_cast<AST::SpanAs>(n.value)) {
    tracker_.AddAlias(n.GetName(), sa->id->name);
    return true;
  }
  if (n.AssignToDataElement()) {
    // buffer.at(i) = expr : a WRITE to the buffer.
    Record(AccessKind::WRITE, &n, n.GetDataArrayName(), AccessEntity::THREADS);
  }
  // The value expression is read (covers `x = buffer.at(i)` and any buffer
  // operands of the right-hand side).
  if (isa<AST::Expr>(n.value))
    RecordAll(AccessKind::READ, &n, GetAllSymbolicOperands(n.value.get()),
              AccessEntity::THREADS);
  return true;
}

bool BufferAccessAnalyzer::Visit(AST::DMA& n) {
  if (n.IsDummy()) return true; // `.any` future has no source/destination.

  AccessEntity entity = n.IsTMA() ? AccessEntity::TMA : AccessEntity::DMA;
  std::string src = n.FromSymbol();
  std::string dst = n.ToSymbol();
  // An auto-inferred destination (e.g. `=> shared` / `=> local`) has no named
  // buffer: the future itself is the handle for the destination, and its
  // storage comes from the future's type.
  if (dst.empty() && !n.future.empty()) dst = n.future;

  Record(AccessKind::READ, &n, src, entity);
  if (!dst.empty()) Record(AccessKind::WRITE, &n, dst, entity);

  if (!n.future.empty()) {
    // Only a named destination needs a future -> buffer mapping; an
    // auto-inferred destination already resolves to the future itself.
    if (!n.ToSymbol().empty())
      tracker_.AddFut2Buffers(n.future, DMABufInfo{src, n.ToSymbol()});
    future_producers_[InScopeNameForRef(n.future)] = &n;
  }
  return true;
}

bool BufferAccessAnalyzer::Visit(AST::BufferMap& n) {
  if (!n.result.empty() && n.source) {
    auto srcOps = GetAllSymbolicOperands(n.source.get());
    for (const auto& src : srcOps) tracker_.AddNoStorageAlias(n.result, src);
  }
  return true;
}

bool BufferAccessAnalyzer::Visit(AST::MMA& n) {
  // Only the buffer <-> fragment transfers touch a storage-bearing buffer:
  // Load/LoadR read the source buffer, Store writes the destination buffer.
  // Fill / Exec / Scale operate on REG fragments, which are filtered out.
  auto op = n.GetOperation();
  switch (op->Tag()) {
  case AST::MMAOperation::Scale: {
    RecordAll(AccessKind::READ, &n, GetAllSymbolicOperands(op->ScaleA().get()),
              AccessEntity::MMA);
    RecordAll(AccessKind::READ, &n, GetAllSymbolicOperands(op->ScaleB().get()),
              AccessEntity::MMA);
  } break;
  case AST::MMAOperation::Exec: {
    RecordAll(AccessKind::READ, &n,
              GetAllSymbolicOperands(op->ExecOperand(1).get()),
              AccessEntity::MMA);
    RecordAll(AccessKind::READ, &n,
              GetAllSymbolicOperands(op->ExecOperand(2).get()),
              AccessEntity::MMA);
  } break;
  case AST::MMAOperation::Store: {
    RecordAll(AccessKind::WRITE, &n,
              GetAllSymbolicOperands(op->StoreTo().get()), AccessEntity::MMA);
  } break;
  case AST::MMAOperation::Load:
  case AST::MMAOperation::LoadR: {
    RecordAll(AccessKind::READ, &n,
              GetAllSymbolicOperands(op->LoadFrom().get()), AccessEntity::MMA);
  } break;
  default: break;
  }
  return true;
}

bool BufferAccessAnalyzer::Visit(AST::Call& n) {
  for (const auto& arg : n.GetArguments())
    RecordAll(AccessKind::READ, &n, GetAllSymbolicOperands(arg.get()),
              AccessEntity::THREADS);
  return true;
}

bool BufferAccessAnalyzer::Visit(AST::Rotate& n) {
  for (const auto& item : n.GetIds()) {
    auto id = cast<AST::Identifier>(item);
    auto sname = InScopeName(id->name);
    for (const auto& other : n.GetIds()) {
      auto other_id = cast<AST::Identifier>(other);
      auto other_sname = InScopeName(other_id->name);
      if (other_sname == sname) continue;
      for (const auto& buf_info : tracker_.future_buffers[other_sname])
        tracker_.AddFut2Buffers(id->name, buf_info);
    }
  }
  return true;
}
