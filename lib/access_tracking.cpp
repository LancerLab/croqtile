#include "access_tracking.hpp"
#include "ast.hpp"
#include "aux.hpp"
#include "io.hpp"
#include "types.hpp"

namespace Choreo {

std::string AccessTracker::GetScopedName(const std::string& name) const {
  assert(!name.empty() && "expecting a valid name.");
  assert(scoper && "AccessTracker requires a scoper callback.");
  return scoper(name);
}

AccessTracker::VarSet
AccessTracker::GetAllSymbolicOperands(const AST::Node* n) const {
  if (auto id = dyn_cast<AST::Identifier>(n)) {
    if (id->name == "__choreo_no_tiling__") return {};
    return {scoper(id->name)};
  }
  if (auto expr = dyn_cast<AST::Expr>(n)) {
    VarSet res;
    if (auto c = expr->GetC()) {
      VarSet sub = GetAllSymbolicOperands(c.get());
      res.insert(sub.begin(), sub.end());
    }
    for (const auto& e : {expr->GetL(), expr->GetR()})
      if (e) {
        VarSet sub = GetAllSymbolicOperands(e.get());
        res.insert(sub.begin(), sub.end());
      }
    return res;
  }
  if (isa<AST::IntLiteral>(n) || isa<AST::FloatLiteral>(n) ||
      isa<AST::StringLiteral>(n) || isa<AST::BoolLiteral>(n))
    return {};
  if (auto ii = dyn_cast<AST::IntIndex>(n))
    return GetAllSymbolicOperands(ii->value.get());
  if (auto mds = dyn_cast<AST::MultiDimSpans>(n)) {
    if (!mds->ref_name.empty()) return {sym_scoper(mds->ref_name)};
    auto mv = dyn_cast<AST::MultiValues>(mds->list);
    if (!mv) {
      if (debug)
        dbgs() << "the list of mds is not multivalues: " << PSTR(n) << "\n";
      return {};
    }
    VarSet res;
    for (const auto& v : mv->AllValues()) {
      VarSet sub = GetAllSymbolicOperands(v.get());
      res.insert(sub.begin(), sub.end());
    }
    return res;
  }
  if (auto da = dyn_cast<AST::DataAccess>(n)) return {da->GetDataName()};
  if (auto it = dyn_cast<AST::IntTuple>(n)) {
    VarSet res;
    for (const auto& v : it->GetValues()->AllValues()) {
      VarSet sub = GetAllSymbolicOperands(v.get());
      res.insert(sub.begin(), sub.end());
    }
    return res;
  }
  if (isa<AST::Call>(n)) return {};
  if (auto chunkat = dyn_cast<AST::ChunkAt>(n)) {
    VarSet res;
    res.insert(sym_scoper(chunkat->RefSymbol()));
    for (auto tsi : chunkat->AllOperations())
      for (const auto& rfn : tsi->ReferredNodes()) {
        VarSet sub = GetAllSymbolicOperands(rfn.get());
        res.insert(sub.begin(), sub.end());
      }
    if (chunkat->indices)
      for (const auto& idx : chunkat->indices->AllValues()) {
        VarSet sub = GetAllSymbolicOperands(idx.get());
        res.insert(sub.begin(), sub.end());
      }
    return res;
  }
  if (isa<AST::Nullptr>(n)) return {};
  choreo_unreachable("node type: " + n->TypeNameString() +
                     " is not handled yet.");
  return {};
}

std::string AccessTracker::ResolveNoStorageAlias(const std::string& var) const {
  // A no-storage alias (e.g. buffer.map/remap result) owns no storage of its
  // own.  Redirect its uses to the source buffer so the alias itself never
  // gets a live range or a local allocation.
  std::string svar = var;
  while (no_storage_alias_.count(svar)) svar = no_storage_alias_.at(svar);
  return svar;
}

// y = x.spanas(...), then y is alias to x
void AccessTracker::AddAlias(const std::string& alias_var,
                             const std::string& original_var) {
  std::string salias = GetScopedName(alias_var);
  std::string soriginal = GetScopedName(original_var);
  if (debug) dbgs() << "Add alias: " << salias << " <-> " << soriginal << "\n";
  alias_[salias] = soriginal;
}

// m = input.map/remap(...), then m owns no storage of its own.  Uses of m are
// redirected to input so that m never gets a live range or local allocation.
void AccessTracker::AddNoStorageAlias(const std::string& alias_var,
                                      const std::string& original_var) {
  std::string salias = GetScopedName(alias_var);
  std::string soriginal = GetScopedName(original_var);
  if (debug)
    dbgs() << "Add no-storage alias: " << salias << " <-> " << soriginal
           << "\n";
  no_storage_alias_[salias] = soriginal;
}

// `x = dma.copy y => z`
// then `y` and `z` are bound with `x`
void AccessTracker::AddBinding(const std::string& bind_res,
                               const std::string& bind_src) {
  std::string sres = GetScopedName(bind_res);
  std::string ssrc = GetScopedName(bind_src);
  if (debug) dbgs() << "AddBinding: " << sres << " <- " << ssrc << "\n";
  bindings_[sres].insert(ssrc);
}

void AccessTracker::RemoveBinding(const std::string& bind_res,
                                  const std::string& bind_src) {
  std::string sres = GetScopedName(bind_res);
  std::string ssrc = GetScopedName(bind_src);
  if (debug) dbgs() << "RemoveBinding: " << sres << " <- " << ssrc << "\n";
  bindings_[sres].erase(ssrc);
}

void AccessTracker::AddFut2Buffers(const std::string& fut,
                                   const DMABufInfo& buf_info) {
  std::string sfut = GetScopedName(fut);
  std::string ssrc = GetScopedName(buf_info.first);
  std::string sdst = GetScopedName(buf_info.second);
  if (debug)
    dbgs() << "AddFut2Buffers: " << sfut << " -> " << ssrc << ", " << sdst
           << "\n";
  future_buffers[sfut].insert({ssrc, sdst});
}

} // namespace Choreo
