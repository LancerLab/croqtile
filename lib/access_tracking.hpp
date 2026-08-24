#ifndef __CHOREO_ACCESS_TRACKING_HPP__
#define __CHOREO_ACCESS_TRACKING_HPP__

#include <cassert>
#include <functional>
#include <queue>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace Choreo {

namespace AST {
struct Node;
}

// Shared symbol-resolution primitives for buffer-access analyses.
//
// Owns the alias / binding / no-storage-alias / future-buffer maps that tell a
// buffer-access analysis how names relate: an alias resolves to its source
// buffer, a binding resolves to its member buffers, and a DMA future resolves
// to the (src, dst) buffers it transfers.  Extracted from LivenessAnalyzer so
// that LivenessAnalyzer and BufferAccessAnalyzer share one definition of "what
// an alias is", while each pass keeps its own event stream and fixpoint.
struct AccessTracker {
  using VarSet = std::set<std::string>;
  using DMABufInfo = std::pair<std::string, std::string>;

  // Maps a (possibly unscoped) reference name to its scoped name, respecting
  // declaration order.  Provided by the owning visitor, which knows the
  // current symbol table.
  std::function<std::string(const std::string&)> scoper;
  // Maps a (possibly unscoped) symbol name to its scoped name via a full
  // symbol-table walk.  Provided by the owning visitor.
  std::function<std::string(const std::string&)> sym_scoper;
  // Mirrors the owning visitor's debug flag for trace logging.
  bool debug = false;

  // one to one. Alias of buffer. Could happen in spanas, etc.
  std::unordered_map<std::string, std::string> alias_;
  // one to one. Alias that owns no storage of its own (e.g. buffer.map/remap
  // result). Uses of such an alias are attributed directly to the source
  // buffer, so the alias itself never gets a live range or local allocation.
  std::unordered_map<std::string, std::string> no_storage_alias_;
  // one to many. Bind var to other vars. Could happen in select, dma, etc.
  std::unordered_map<std::string, VarSet> bindings_;
  // future name -> set of (src, dst) buffer pairs it transfers.
  std::unordered_map<std::string, std::set<DMABufInfo>> future_buffers;

  explicit AccessTracker(
      std::function<std::string(const std::string&)> scoper = {},
      std::function<std::string(const std::string&)> sym_scoper = {})
      : scoper(std::move(scoper)), sym_scoper(std::move(sym_scoper)) {}

  // Scope a name through the owning visitor's symbol table.
  std::string GetScopedName(const std::string& name) const;

  // Collect every symbol referenced by an expression subtree (recursing
  // through sub-expressions, indices, spans, and chunk-ats), each resolved to
  // its scoped name.
  VarSet GetAllSymbolicOperands(const AST::Node* n) const;

  // Resolve a var through the no-storage-alias chain.
  std::string ResolveNoStorageAlias(const std::string& var) const;

  void AddAlias(const std::string& alias_var, const std::string& original_var);
  void AddNoStorageAlias(const std::string& alias_var,
                         const std::string& original_var);
  void AddBinding(const std::string& bind_res, const std::string& bind_src);
  void RemoveBinding(const std::string& bind_res, const std::string& bind_src);
  void AddFut2Buffers(const std::string& fut, const DMABufInfo& buf_info);

  // Transitive closure of `vars` over the given maps (each a std::string- or
  // VarSet-valued map).
  template <typename... MapTypes>
  VarSet TransitiveClosure(const VarSet& vars, const MapTypes&... maps);
};

namespace access_detail {

template <typename MapType>
void ProcessMap(const std::string& current, const MapType& mp,
                AccessTracker::VarSet& result, AccessTracker::VarSet& processed,
                std::queue<std::string>& queue) {
  if (!mp.count(current)) return;
  const auto& next = mp.at(current);

  if constexpr (std::is_same_v<std::string,
                               typename std::decay<decltype(next)>::type>) {
    if (!processed.count(next)) {
      result.insert(next);
      queue.push(next);
    }
  } else if constexpr (std::is_same_v<
                           AccessTracker::VarSet,
                           typename std::decay<decltype(next)>::type>) {
    for (const auto& next_var : next)
      if (!processed.count(next_var)) {
        result.insert(next_var);
        queue.push(next_var);
      }
  } else {
    assert(false && "expecting the value of mp is std::string or "
                    "std::set<std::string>.");
  }
}

} // namespace access_detail

template <typename... MapTypes>
AccessTracker::VarSet
AccessTracker::TransitiveClosure(const VarSet& vars, const MapTypes&... maps) {
  VarSet result = vars;
  VarSet processed;
  std::queue<std::string> queue;
  for (const auto& item : vars) queue.push(item);

  while (!queue.empty()) {
    std::string current = queue.front();
    queue.pop();
    if (processed.count(current)) continue;
    processed.insert(current);

    (access_detail::ProcessMap(current, maps, result, processed, queue), ...);
  }
  return result;
}

} // namespace Choreo

#endif // __CHOREO_ACCESS_TRACKING_HPP__
